"""
Leakage invariants on a tiny synthetic frame, so CI needs no dataset, model or GPU.

The frame mimics the HF data: originals with augmented variants sharing a group_id, an
official test split, two sources, and near-duplicate originals in different groups (one of
them a near-twin of a test prompt).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from reflex.data.sampler import assign_split_groups, make_dedup_key, make_splits
from reflex.evaluation.analysis import (
    choose_threshold,
    compute_rates,
    evaluate_leave_one_source_out,
)
from reflex.evaluation.k_fold_cross import evaluate_holdout


def make_row(
    text: str, label: int, group: str, augmented: bool, source: str, split: str
) -> dict[str, str | int | bool]:
    """Return one synthetic dataset row with the columns the pipeline reads."""
    return {
        "text": text,
        "label": label,
        "group_id": group,
        "augmented": augmented,
        "source": source,
        "split": split,
        "severity": "medium" if label else "not assigned",
        "category": "direct_injection" if label else "benign",
    }


def make_frame() -> pd.DataFrame:
    """Return 60 groups (test every 10th), augmented variants for train, and 2 near-twins."""
    rows = []
    for i in range(60):
        label, source = i % 2, "a" if i < 30 else "b"
        split = "test" if i % 10 == 0 else "train"
        text = f"prompt number {i} about topic {i % 7}"
        rows.append(make_row(text, label, f"g{i}", False, source, split))
        if split == "train":
            rows.append(make_row(text + " please", label, f"g{i}", True, source, split))
    # Near-duplicates in other groups: a twin of train prompt 1, and of test prompt 10.
    rows.append(
        make_row("Prompt number 1 : about topic 1!", 1, "g_dup1", False, "a", "train")
    )
    rows.append(
        make_row("prompt number 10, about topic 3", 0, "g_dup10", False, "a", "train")
    )
    return pd.DataFrame(rows)


def test_near_duplicate_groups_are_merged() -> None:
    """Originals differing only in punctuation/case share a split group."""
    df = make_frame()
    groups = assign_split_groups(df)
    assert (
        groups[df["group_id"] == "g1"].iloc[0]
        == groups[df["group_id"] == "g_dup1"].iloc[0]
    )
    assert (
        groups[df["group_id"] == "g2"].iloc[0] != groups[df["group_id"] == "g1"].iloc[0]
    )


def test_group_with_conflicting_originals_is_rejected() -> None:
    """A group whose originals have different dedup keys must not merge silently."""
    df = pd.concat(
        [
            make_frame(),
            pd.DataFrame([make_row("something else", 1, "g1", False, "a", "train")]),
        ]
    )
    with pytest.raises(ValueError, match="different dedup keys"):
        assign_split_groups(df)


@pytest.mark.parametrize("nested", [False, True])
def test_make_splits_has_no_leakage(nested: bool) -> None:
    """No split group or dedup key spans train and val; val is originals only, each once."""
    df = make_frame()
    if nested:  # an outer training fold, as nested CV and OOF stacking split it again
        df = make_splits(df, 3)[0][0]
    keys = df["text"].map(make_dedup_key)
    train_splits, val_splits = make_splits(df, 3)
    seen: list[int] = []
    for train, val in zip(train_splits, val_splits):
        assert not val["augmented"].any()
        assert not set(train["group_id"]) & set(val["group_id"])
        assert not set(keys[train.index]) & set(keys[val.index])
        seen += val.index.tolist()
    assert sorted(seen) == sorted(df.index[~df["augmented"]])


def test_holdout_dedup_drops_test_near_twins(tmp_path: Path) -> None:
    """The dedup holdout scores the test split minus prompts with a near-twin outside it."""
    model = make_pipeline(CountVectorizer(), LogisticRegression())
    official, dedup = evaluate_holdout(
        {"lr": model}, make_frame(), results_path=tmp_path / "r.jsonl"
    )
    assert official["n_val"] == 6
    assert dedup["n_val"] == 5  # test prompt 10 has a near-twin in train
    assert official["n_train"] == dedup["n_train"]
    assert len((tmp_path / "r.jsonl").read_text().splitlines()) == 2


def test_holdout_rejects_augmented_test_rows(tmp_path: Path) -> None:
    """Augmented rows in the test split would leak variants of training prompts."""
    df = make_frame()
    df.loc[df["split"] == "test", "augmented"] = True
    with pytest.raises(ValueError, match="augmented"):
        evaluate_holdout(
            {"lr": make_pipeline(CountVectorizer(), LogisticRegression())},
            df,
            results_path=tmp_path / "r.jsonl",
        )


def test_leave_one_source_out_rejects_cross_source_groups() -> None:
    """Holding out a source must not leave part of a split group in training."""
    df = make_frame()
    df.loc[df["group_id"] == "g_dup1", "source"] = (
        "b"  # g_dup1 merges with g1 (source a)
    )
    with pytest.raises(ValueError, match="span several sources"):
        evaluate_leave_one_source_out(
            make_pipeline(CountVectorizer(), LogisticRegression()), df
        )


@pytest.mark.parametrize("target", [0.0, 0.01, 0.1, 0.5])
def test_choose_threshold_respects_target_fpr(target: float) -> None:
    """The chosen threshold's FPR never exceeds the target, including with ties."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    scores = np.round(rng.random(500) * 0.5 + 0.5 * y, 2)  # rounded, so many ties
    fpr, _ = compute_rates(y, scores, choose_threshold(y, scores, target))
    assert fpr <= target + 1e-12
