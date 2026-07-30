"""Stable hashing utilities for content-addressed IDs."""
import hashlib
import json


def stable_hash(*parts: str) -> str:
    """Content-addressed hash from concatenated parts."""
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def dict_hash(d: dict) -> str:
    """Stable hash of a dict (sorted keys)."""
    raw = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
