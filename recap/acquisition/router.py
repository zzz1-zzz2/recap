"""P1-3D: AcquisitionRouter — dispatch SearchAction to executor."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

from recap.acquisition.definition_search import find_definition
from recap.acquisition.related_test_search import find_related_tests
from recap.acquisition.schema import (
    AcquisitionHit,
    AcquisitionResult,
    AcquisitionStatus,
)
from recap.diagnosis.search_contract import (
    SearchAction,
    SearchActionType,
    SearchContract,
    SearchTarget,
    SearchTargetKind,
)


class AcquisitionRouter:
    """Dispatch SearchActions to executors.

    v1 supports only FIND_DEFINITION and FIND_RELATED_TESTS. Other
    action types return UNSUPPORTED. The router NEVER modifies the
    repo and NEVER reads gold patch / gold context.
    """

    def __init__(
        self,
        repo_root: Path | str,
        r1_viewed_files: Iterable[str] | None = None,
        failed_test_names: Iterable[str] | None = None,
        max_files_examined: int = 200,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.r1_viewed_files = set(r1_viewed_files or [])
        self.failed_test_names = list(failed_test_names or [])
        self.max_files_examined = max_files_examined

    def dispatch(self, action: SearchAction) -> AcquisitionResult:
        """Execute one SearchAction against the repo."""
        if action.action_type == SearchActionType.FIND_DEFINITION:
            return find_definition(
                action, self.repo_root,
                max_files_examined=self.max_files_examined,
            )
        if action.action_type == SearchActionType.FIND_RELATED_TESTS:
            return find_related_tests(
                action,
                self.repo_root,
                r1_viewed_files=self.r1_viewed_files,
                failed_test_names=self.failed_test_names,
                max_files_examined=self.max_files_examined,
            )
        return AcquisitionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            target=action.target,
            status=AcquisitionStatus.UNSUPPORTED,
            errors=[f"action_type={action.action_type.value} not implemented in v1"],
        )

    def dispatch_contract(
        self, contract: SearchContract,
    ) -> list[AcquisitionResult]:
        """Execute every action in a contract, returning one result per action."""
        results = [self.dispatch(a) for a in contract.actions]
        # Post-processing: test-to-source expansion for FIND_RELATED_TESTS hits
        expanded = self._expand_test_results(results, contract)
        results.extend(expanded)
        return results

    # ── Test-to-Source Expansion ────────────────────────────────────

    _REPO_MODULE_PREFIXES = {"astropy", "sympy", "django", "sphinx", "pydicom",
                              "sqlalchemy", "matplotlib", "requests", "flask"}

    def _expand_test_results(
        self,
        results: list[AcquisitionResult],
        contract: SearchContract,
    ) -> list[AcquisitionResult]:
        """Post-process FIND_RELATED_TESTS results to extract repo symbol definitions.

        For each test file hit, parse the test file AST, extract repo-local
        symbols (imports, calls, fixtures), and look up their definitions.
        Returns additional AcquisitionResult objects with source-level hits.
        """
        expanded: list[AcquisitionResult] = []
        seen_defs: set[tuple[str, int]] = set()  # (file_path, start_line)

        for r in results:
            if r.action_type != SearchActionType.FIND_RELATED_TESTS:
                continue
            if r.status != AcquisitionStatus.FOUND:
                continue
            for hit in r.hits[:2]:  # max 2 test files
                test_path = Path(self.repo_root) / hit.file_path if not Path(hit.file_path).is_absolute() else Path(hit.file_path)
                if not test_path.exists():
                    continue
                try:
                    symbols = self._extract_repo_symbols(test_path)
                except Exception:
                    continue

                for sym in symbols[:4]:  # max 4 symbols
                    # Look up definition
                    def_action = SearchAction(
                        action_id=f"t2s_{contract.contract_id}_{sym.replace('.', '_')}",
                        action_type=SearchActionType.FIND_DEFINITION,
                        target=SearchTarget(value=sym, kind=SearchTargetKind.SYMBOL),
                        budget=2,
                    )
                    def_result = self.dispatch(def_action)
                    if def_result.status == AcquisitionStatus.FOUND:
                        for dh in def_result.hits[:2]:
                            key = (dh.file_path, dh.start_line)
                            if key not in seen_defs:
                                seen_defs.add(key)
                                # Tag these hits so they can be identified as expanded
                                dh.retrieval_method = "test_to_source_expansion"
                                dh.relevance_reason = (
                                    f"Symbol '{sym}' extracted from test file "
                                    f"{hit.file_path} and resolved to definition"
                                )
                        expanded.append(def_result)

        return expanded

    def _extract_repo_symbols(self, test_file: Path) -> list[str]:
        """Parse a test file's AST to extract repository-local symbols.

        Returns deduplicated, ordered list of symbol names suitable for
        FIND_DEFINITION lookup.
        """
        source = test_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        symbols: list[str] = []
        seen: set[str] = set()

        for node in ast.walk(tree):
            # import X.Y.Z → extract X.Y or X (if repo prefix matches)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    # Extract top-level module prefix
                    prefix = name.split(".")[0]
                    if prefix in self._REPO_MODULE_PREFIXES:
                        if name not in seen:
                            seen.add(name)
                            symbols.append(name)
                    elif alias.asname:
                        # Store alias for later resolution
                        pass

            # from X.Y import Z → extract Z, also check X.Y for repo prefix
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    prefix = node.module.split(".")[0]
                    if prefix in self._REPO_MODULE_PREFIXES:
                        for alias in node.names:
                            name = alias.name if not alias.asname else alias.asname
                            if name not in seen:
                                seen.add(name)
                                symbols.append(name)

        # Second pass: extract function calls with repo-object prefixes
        # e.g., ufunc.__call__(x) where ufunc is from repo
        # This is heuristic: look for names not in stdlib or third-party
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                # obj.method() — check if obj matches a repo symbol
                if isinstance(node.func.value, ast.Name):
                    obj_name = node.func.value.id
                    attr_name = node.func.attr
                    if obj_name in seen:
                        # obj.method is a repo symbol call
                        full = f"{obj_name}.{attr_name}"
                        if full not in seen:
                            seen.add(full)
                            symbols.append(full)

        return symbols[:10]
