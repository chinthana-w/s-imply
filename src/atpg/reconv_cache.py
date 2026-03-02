"""
Disk-persisted cache for reconvergent path pair topology.

Reconvergent pairs encode only circuit topology (node IDs and paths)
and are a pure function of the .bench file — they never change between runs.
Persisting them avoids expensive BFS traversals on subsequent collections.
"""

import os
import pickle


def _cache_path(bench_path: str) -> str:
    """Return path to the disk cache for a given bench file."""
    cache_dir = os.path.join(os.path.dirname(bench_path), ".reconv_cache")
    os.makedirs(cache_dir, exist_ok=True)
    name = os.path.basename(bench_path) + ".pairs.pkl"
    return os.path.join(cache_dir, name)


def load_pair_cache(bench_path: str) -> dict | None:
    """Load the reconvergent pair cache from disk. Returns None on miss or error."""
    path = _cache_path(bench_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            cache = pickle.load(f)
        if isinstance(cache, dict):
            return cache
    except Exception as e:
        print(f"[Warning] Failed to load reconv pair cache for {bench_path}: {e}")
    return None


def persist_pair_cache(bench_path: str, pair_cache: dict) -> None:
    """Persist the reconvergent pair cache to disk."""
    if not pair_cache:
        return
    path = _cache_path(bench_path)
    try:
        with open(path, "wb") as f:
            pickle.dump(pair_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"[Warning] Failed to persist reconv pair cache for {bench_path}: {e}")
