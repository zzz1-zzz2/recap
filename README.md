# ReCAP: Diagnosis-Guided Repair-State Reconstruction

**ReCAP** is a framework for post-validation recovery in repository-level program repair agents. When an LLM-based agent generates a candidate patch that fails validation, ReCAP diagnoses *what context was missing* and reconstructs the continuation state for a focused second attempt.

## Paper

_ReCAP: Diagnosis-Guided Repair-State Reconstruction for Post-Validation Recovery in Repository-Level Repair Agents_ (AAAI 2027 submission)

## Architecture

```
Patch → Test → Failure Signal
                ↓
         [ConDiag Diagnosis]
   Multi-feature Fusion to Infer
      Missing Context Types
                ↓
         [Reconstruction Plan] (K⁺, K⁻, 𝒬)
         ┌────┬────┬────┐
         ↓    ↓    ↓    ↓
      Preserve Rehydrate Suppress Acquire
         │    │         │       │
         └────┴─────────┴───────┘
                ↓
         [Pack: K₂]
 Diagnosis-aware Context Budget Allocation
                ↓
         [Guide: ρ₂]
      Structured Revision Brief
                ↓
         [Continue: R2]
 Continue Repairing from the Failed Workspace
```

## 2×2 Evaluation Protocol

| Condition | Diagnosis | Reconstruction | Description |
|-----------|-----------|---------------|-------------|
| SF | ❌ | ❌ | Stateful Feedback — baseline |
| GR | ❌ | ✅ | Generic Reconstruction — compression only |
| SG | ✅ | ❌ | Structured Guidance — diagnosis only |
| ReCAP | ✅ | ✅ | Full framework |

## Project Structure

```
recap/
├── diagnosis/          — Failure event analysis, evidence alignment,
│                         hypothesis construction, causal refinement
│   ├── failure_event.py
│   ├── clustering.py
│   ├── alignment.py
│   ├── hypothesis.py
│   ├── causal_refinement.py
│   ├── revision_contract.py
│   ├── search_contract.py (evidence ledger)
│   └── signals/        — Test log, patch, trajectory extractors
│
├── reconstruction/     — Plan → Acquire → Pack → Guide pipeline
│   ├── context_unit.py  — Unified evidence representation
│   ├── planner.py       — (K⁺, K⁻, 𝒬) ← Plan(C₁, H, B)
│   ├── packer.py        — K₂ under budget with priorities
│   ├── rehydrator.py    — Restore R1-seen but dropped evidence
│   ├── suppressor.py    — Remove contradicted/redundant context
│   ├── revision_brief.py — Structured ρ₂
│   └── pipeline.py      — Full pipeline orchestration
│
├── acquisition/         — Grounded search contract execution
│   ├── router.py        — Action dispatch
│   ├── definitions.py   — Symbol definition lookup
│   └── tests.py         — Related test discovery
│
├── continuation/        — R2 execution environment
│   ├── state.py         — z₂ = (K₂, ρ₂)
│   └── runner.py        — R2 agent loop
│
├── conditions/          — 2×2 ablation experiment conditions
│   ├── config.py        — Feature flags (SF, GR, SG, ReCAP)
│   ├── stateful_feedback.py
│   └── recap.py
│
├── evaluation/          — SWE-bench harness integration
│   └── evaluator.py
│
├── checkpoint/          — C₁ freeze, load, validate
│   └── loader.py
│
└── utils/               — Hashing, token counting
    ├── hashing.py
    └── token_count.py
```

## Quick Start

### Requirements

- Python 3.11+
- Docker (for SWE-bench evaluation)
- LLM API key (DeepSeek, Claude, etc.)

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your API key
```

### Run a Recovery Episode

```bash
# Run R1 (first attempt)
python experiments/run_round1.py --instance django__django-11400

# Run ReCAP recovery (R2)
python experiments/run_recovery.py \
  --instance django__django-11400 \
  --checkpoint run_<timestamp> \
  --condition recap

# Run full 4-condition ablation
python experiments/run_ablation.py \
  --instance django__django-11400 \
  --checkpoint run_<timestamp> \
  --output-dir artifacts/ablation_v1/
```

### Evaluate Results

```bash
python experiments/evaluate_runs.py \
  --results-dir artifacts/ablation_v1/
```

## Key Concepts

**Checkpoint C₁**: Frozen first-failure state = (workspace, patch, trajectory, failure witness). Shared across all recovery conditions.

**ContextUnit**: Unified evidence representation carrying content, provenance, operation type (PRESERVE/REHYDRATE/ACQUIRE/SUPPRESS), and diagnosis linkage. Every reconstruction operation produces or consumes ContextUnits.

**K₂**: The continuation context — diagnosis-conditioned representation of the evidence required for the next repair decision. Strictly bounded by token budget.

**ρ₂**: Structured revision brief — (c*, E_inspect, O_edit, N_forbid, V_target) rendered as actionable text.

**Abstention**: When diagnosis confidence is too low, ReCAP falls back to conservative mode without diagnosis-guided suppression or acquisition.

## Citation

```bibtex
@inproceedings{recap2027,
  title={ReCAP: Diagnosis-Guided Repair-State Reconstruction
         for Post-Validation Recovery in Repository-Level Repair Agents},
  author={Anonymous},
  booktitle={KDD},
  year={2027}
}
```
