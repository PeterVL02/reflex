import json
from pathlib import Path
from typing import Literal, TypedDict

import pandas as pd


class DataConfig(TypedDict):
    repo: str
    version: str
    seed: int
    cache_dir: str
    revision: str


class Config(TypedDict):
    data: DataConfig


class HFRow(TypedDict):
    text: str
    label: int
    category: str
    source: str
    severity: str
    group_id: str
    augmented: bool
    tags: list[str]


REPO_ROOT: Path = Path(__file__).resolve().parents[3]
CONFIG: Config = json.loads(
    (Path(__file__).with_name("config.json")).read_text(encoding="utf-8")
)

REPO: str = CONFIG["data"]["repo"]
VERSION: str = CONFIG["data"]["version"]
REVISION: str = CONFIG["data"]["revision"]
# The revision is in the name, so bumping it in config.json never reuses a stale cache.
CACHE_PATH: Path = (
    REPO_ROOT / CONFIG["data"]["cache_dir"] / f"{VERSION}_{REVISION[:7]}.parquet"
)
# Our relabel map (group_id only, no texts; `human_verified` tracks the user's review).
# "A" sets the `control` category to benign. "B" adds WildGuard rows labelled malicious
# whose text is completely benign (a blind text-only review found none so far, so B == A).
# Variant "B" applies both layers.
RELABELS_PATH: Path = Path(__file__).with_name("relabels.csv")
type RelabelVariant = Literal["A", "B"]
RELABEL_LAYERS: dict[RelabelVariant, tuple[str, ...]] = {"A": ("A",), "B": ("A", "B")}


def apply_relabels(df: pd.DataFrame, variant: RelabelVariant) -> pd.DataFrame:
    """
    Return a copy of df with our relabels for `variant` applied to whole groups.

    Every row whose `group_id` is in the map (original and augmented variants alike) gets
    the new label and severity. These are our judgement, not the published labels.

    Raises:
        ValueError: If a relabelled group is missing or its label is not `old_label`
            (e.g. after a dataset revision bump).
    """
    relabels = pd.read_csv(RELABELS_PATH)
    relabels = relabels[relabels["variant"].isin(RELABEL_LAYERS[variant])].set_index(
        "group_id"
    )
    in_map = df["group_id"].isin(relabels.index)
    old = df.loc[in_map, "group_id"].map(relabels["old_label"])
    if (
        set(df.loc[in_map, "group_id"]) != set(relabels.index)
        or (df.loc[in_map, "label"] != old).any()
    ):
        raise ValueError("relabel map does not match the data; check the revision")
    out = df.copy()
    groups = out.loc[in_map, "group_id"]
    out.loc[in_map, "label"] = groups.map(relabels["new_label"]).astype(
        out["label"].dtype
    )
    out.loc[in_map, "severity"] = groups.map(relabels["new_severity"])
    return out


def load_data(
    refresh: bool = False, relabel: RelabelVariant | None = None
) -> pd.DataFrame:
    """
    Load the dataset (all Hugging Face splits concatenated) from the local parquet
    cache, downloading and caching it first if absent. Assigns 'not assigned' to
    values with missing severity.

    Args:
        refresh (bool): Re-download from Hugging Face and overwrite the cache.
        relabel (RelabelVariant | None): Apply our relabel variant ("A" or "B", see
            `apply_relabels`). None (default) keeps the published labels.

    Returns:
        pd.DataFrame: The concatenated dataset as a DataFrame.
    """
    if CACHE_PATH.exists() and not refresh:
        df = pd.read_parquet(CACHE_PATH)
        return df if relabel is None else apply_relabels(df, relabel)

    from datasets import load_dataset  # slow import, only needed to download

    ds = load_dataset(REPO, VERSION, revision=REVISION)
    # Keep the official HF split so we can also score on its test set, like the data card.
    df = pd.concat(
        [ds[key].to_pandas().assign(split=key) for key in ds], ignore_index=True
    )
    df["severity"] = df["severity"].replace("", "not assigned")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH)
    return df if relabel is None else apply_relabels(df, relabel)


def make_X_y(df: pd.DataFrame) -> tuple[pd.Series[str], pd.Series[int]]:
    """
    Split the DataFrame into features (X) and labels (y).

    Args:
        df (pd.DataFrame): The input DataFrame to be split.

    Returns:
        tuple[pd.Series[str], pd.Series[int]]: The texts (X) and labels (y).
    """
    X = df["text"]
    y = df["label"]
    return X, y
