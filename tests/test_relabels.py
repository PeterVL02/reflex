"""Relabel map invariants, checked on the committed CSV and a synthetic frame (no dataset)."""

import pandas as pd
import pytest

from reflex.data.loader import RELABELS_PATH, RelabelVariant, apply_relabels
from reflex.evaluation.k_fold_cross import SEVERITY_CODES


def make_relabel_frame(relabels: pd.DataFrame) -> pd.DataFrame:
    """Return an original plus an augmented row per relabelled group, and one untouched group."""
    rows = [
        {
            "group_id": group,
            "augmented": augmented,
            "label": old,
            "severity": "high" if old else "not assigned",
        }
        for group, old in zip(relabels["group_id"], relabels["old_label"])
        for augmented in (False, True)
    ]
    rows.append(
        {"group_id": "other", "augmented": False, "label": 1, "severity": "medium"}
    )
    return pd.DataFrame(rows).astype({"label": "int32"})  # as in the HF data


def test_relabel_csv_is_well_formed() -> None:
    """One row per group, real flips, consistent severity, short reasons without quotes."""
    relabels = pd.read_csv(RELABELS_PATH)
    assert relabels["group_id"].is_unique
    assert set(relabels["variant"]) <= {"A", "B"}
    assert (relabels["old_label"] != relabels["new_label"]).all()
    benign = relabels["new_label"] == 0
    assert (relabels.loc[benign, "new_severity"] == "not assigned").all()
    assert relabels.loc[~benign, "new_severity"].isin(SEVERITY_CODES).all()
    assert relabels["reason"].str.len().le(100).all()
    assert not relabels["reason"].str.contains("[\"']").any()
    assert relabels["human_verified"].dtype == bool


@pytest.mark.parametrize("variant", ["A", "B"])
def test_relabels_apply_to_whole_groups(variant: RelabelVariant) -> None:
    """Every row of a relabelled group changes; other rows and the input stay as they were."""
    relabels = pd.read_csv(RELABELS_PATH)
    df = make_relabel_frame(relabels)
    out = apply_relabels(df, variant)
    active = relabels[relabels["variant"] <= variant].set_index("group_id")
    changed = df["group_id"].isin(active.index)
    assert out.groupby("group_id")["label"].nunique().eq(1).all()
    assert (
        out.loc[changed, "label"]
        == df.loc[changed, "group_id"].map(active["new_label"])
    ).all()
    assert out[~changed].equals(df[~changed])
    assert df.equals(make_relabel_frame(relabels))  # input not mutated


def test_relabels_reject_mismatched_labels() -> None:
    """A map that no longer matches the data (e.g. a new revision) must fail loudly."""
    df = make_relabel_frame(pd.read_csv(RELABELS_PATH))
    df.loc[0, "label"] = 1 - df.loc[0, "label"]
    with pytest.raises(ValueError, match="does not match"):
        apply_relabels(df, "B")
