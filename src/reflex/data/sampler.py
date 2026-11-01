import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from reflex.data.data_transformation import normalize_text
from reflex.data.loader import CONFIG

SEED: int = CONFIG["data"]["seed"]
NON_ALNUM_PATTERN: re.Pattern[str] = re.compile(r"[\W_]+")


def make_dedup_key(text: str) -> str:
    """Return a key shared by texts that differ only in case, punctuation, spacing or invisible chars."""
    return NON_ALNUM_PATTERN.sub("", normalize_text(text))


def assign_split_groups(df: pd.DataFrame) -> pd.Series[str]:
    """
    Return the unit each row is split by: its group, merged with near-duplicate groups.

    The HF `group_id` links augmented variants to their original, but 182 clusters of
    originals differ only in punctuation/spacing and still have separate group_ids, so
    they could land on both sides of a fold. Groups whose originals share a
    `make_dedup_key` are merged. Groups without an original (or with an empty key) keep
    their own group_id.

    Args:
        df (pd.DataFrame): Rows with 'text', 'group_id' and 'augmented' columns.

    Returns:
        pd.Series[str]: One split-group label per row, aligned with `df`.

    Raises:
        ValueError: If one group holds originals with different dedup keys.
    """
    originals = df[~df["augmented"]]
    keys = originals["text"].map(make_dedup_key).replace("", np.nan)
    # A group with two differently keyed originals would merge depending on row order.
    if (keys.groupby(originals["group_id"]).nunique() > 1).any():
        raise ValueError("a group has originals with different dedup keys")
    key_of_group = dict(zip(originals["group_id"], "key:" + keys))
    return df["group_id"].map(key_of_group).fillna(df["group_id"])


def make_splits(
    df: pd.DataFrame, n_folds: int, seed: int = SEED
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Split the DataFrame into group-level, label-stratified cross-validation folds.

    Groups are first merged with near-duplicate groups (`assign_split_groups`).

    Folds are drawn over the original (non-augmented) rows only, so each validation
    fold holds originals only and every original lands in exactly one validation
    fold. Training gets every row (originals and augmented) whose group_id is not in
    that fold's validation set; augmented variants of validation groups are dropped,
    so no group appears on both sides. Groups without an original are always in
    training. Safe to call again on a training fold for nested CV.

    Args:
        df (pd.DataFrame): Rows with 'label', 'group_id' and 'augmented' columns.
        n_folds (int): The number of folds for cross-validation.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[list[pd.DataFrame], list[pd.DataFrame]]: A tuple containing two lists:
            - List of training DataFrames for each fold.
            - List of validation DataFrames for each fold.
    """
    df = df.sample(frac=1, random_state=seed)
    split_groups = assign_split_groups(df)
    is_original = ~df["augmented"]
    originals = df[is_original]
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_splits: list[pd.DataFrame] = []
    val_splits: list[pd.DataFrame] = []

    for _, val_idx in cv.split(
        originals, originals["label"], split_groups[is_original]
    ):
        val_split = originals.iloc[val_idx]
        held_out = split_groups[is_original].iloc[val_idx]
        train_splits.append(df[~split_groups.isin(held_out)])
        val_splits.append(val_split)

    return train_splits, val_splits
