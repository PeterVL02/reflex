"""
Feature store for frozen, label-free features, cached on disk by text hash.

- "embedding": Qwen3-Embedding-0.6B vectors (1024 dims, see model/embeddings.py).
- "laya": zero-shot Laya "noul" probability for each instruction in
  model/sys_one_prompt.json, one column per key in JSON key order.

Neither feature set looks at labels and neither model is fitted on this dataset, so
computing them once for every row (train and test alike) leaks nothing across folds.
Rows are keyed by the SHA-1 of the text, so duplicates, row order and any subset
(a CV fold, new prompts at inference) resolve to the same cached vectors.
"""

import hashlib
import json
import os
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from reflex.data.loader import CACHE_PATH, VERSION, load_data

FeatureName = Literal["embedding", "laya"]
FEATURE_NAMES: tuple[FeatureName, ...] = ("embedding", "laya")


def get_laya_columns() -> list[str]:
    """Return the Laya feature column names: the sys_one_prompt.json keys, in order."""
    from reflex.model.sys_one import PROMPTS_PATH

    return list(json.loads(PROMPTS_PATH.read_text(encoding="utf-8")))


def get_laya_instructions() -> list[str]:
    """Return the Laya instruction texts, in the same order as `get_laya_columns`."""
    from reflex.model.sys_one import load_instructions

    return [load_instructions(key) for key in get_laya_columns()]


def get_cache_path(name: FeatureName) -> Path:
    """Return the .npz cache file for a feature set and the current dataset version."""
    return CACHE_PATH.parent / f"features_{name}_{VERSION}.npz"


def hash_text(text: str) -> str:
    """Return the SHA-1 hex digest of a text, used as its cache key."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def compute_features(texts: list[str], name: FeatureName) -> npt.NDArray[np.float32]:
    """
    Compute a feature set directly with the underlying model, bypassing the cache.

    Args:
        texts (list[str]): Texts to featurise.
        name (FeatureName): Which feature set to compute.

    Returns:
        npt.NDArray[np.float32]: Shape (len(texts), 1024) for "embedding" or
        (len(texts), len(get_laya_columns())) for "laya".
    """
    if name == "embedding":
        from reflex.model.embeddings import embed_prompts

        return embed_prompts(texts).astype(np.float32)

    from reflex.model.sys_one import classify_prompts

    columns = [classify_prompts(texts, prompt_name=key) for key in get_laya_columns()]
    return np.column_stack(columns).astype(np.float32)


@cache
def load_feature_cache(
    name: FeatureName,
) -> tuple[dict[str, int], npt.NDArray[np.float32]]:
    """
    Load a feature cache from disk (memoised; cleared by `build_feature_cache`).

    Args:
        name (FeatureName): Which feature set to load.

    Returns:
        tuple[dict[str, int], npt.NDArray[np.float32]]: Map from text hash to row
        index, and the feature matrix. Both are empty if no cache file exists.

    Raises:
        ValueError: If the Laya cache was built with different instructions than
            sys_one_prompt.json now holds.
    """
    path = get_cache_path(name)
    if not path.exists():
        return {}, np.empty((0, 0), dtype=np.float32)
    with np.load(path) as npz:
        if name == "laya" and npz["instructions"].tolist() != get_laya_instructions():
            raise ValueError(
                f"{path} is stale (sys_one_prompt.json changed); "
                "run build_feature_cache(refresh=True)."
            )
        keys: list[str] = npz["keys"].tolist()
        X: npt.NDArray[np.float32] = npz["X"]
    return {key: i for i, key in enumerate(keys)}, X


def get_features(
    texts: list[str] | pd.Series[str], name: FeatureName
) -> npt.NDArray[np.float32]:
    """
    Look up features for texts, in the order given, computing any missing from the cache.

    Texts missing from the cache (e.g. new prompts at inference) are computed on the
    fly but not persisted.

    Args:
        texts (list[str] | pd.Series[str]): Texts to featurise; duplicates are fine.
        name (FeatureName): Which feature set to return.

    Returns:
        npt.NDArray[np.float32]: One row per text, in the order of `texts`.

    Raises:
        FileNotFoundError: If the cache file doesn't exist. Without it every training
            frame would be re-embedded on the GPU on every fit, silently.
    """
    if not get_cache_path(name).exists():
        raise FileNotFoundError(
            f"{get_cache_path(name)} missing; run build_feature_cache() first."
        )
    keys = [hash_text(text) for text in texts]
    index, X = load_feature_cache(name)
    missing = {key: text for key, text in zip(keys, texts) if key not in index}
    if not missing:
        return X[[index[key] for key in keys]]
    # ponytail: misses are recomputed every call; persist them if inference repeats prompts
    X_missing = compute_features(list(missing.values()), name)
    rows = {key: X_missing[i] for i, key in enumerate(missing)}
    return np.stack([rows[key] if key in rows else X[index[key]] for key in keys])


def build_feature_cache(refresh: bool = False) -> None:
    """
    Compute every feature set for each unique text in `load_data()` and save it.

    Args:
        refresh (bool): Recompute feature sets whose cache file already exists.
    """
    texts: list[str] = load_data()["text"].drop_duplicates().tolist()
    keys = np.array([hash_text(text) for text in texts])
    for name in FEATURE_NAMES:
        path = get_cache_path(name)
        if path.exists() and not refresh:
            continue
        X = compute_features(texts, name)
        extra = (
            {"instructions": np.array(get_laya_instructions())}
            if name == "laya"
            else {}
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write then rename, so an interrupted rebuild never leaves a truncated cache.
        tmp = path.with_name(f"{path.stem}.tmp.npz")
        np.savez(tmp, keys=keys, X=X, **extra)
        os.replace(tmp, path)
    load_feature_cache.cache_clear()
