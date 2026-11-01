import json
from pathlib import Path
from typing import TypedDict

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


def load_data(refresh: bool = False) -> pd.DataFrame:
    """
    Load the dataset (all Hugging Face splits concatenated) from the local parquet
    cache, downloading and caching it first if absent. Assigns 'not assigned' to
    values with missing severity.

    Args:
        refresh (bool): Re-download from Hugging Face and overwrite the cache.

    Returns:
        pd.DataFrame: The concatenated dataset as a DataFrame.
    """
    if CACHE_PATH.exists() and not refresh:
        return pd.read_parquet(CACHE_PATH)

    from datasets import load_dataset  # slow import, only needed to download

    ds = load_dataset(REPO, VERSION, revision=REVISION)
    # Keep the official HF split so we can also score on its test set, like the data card.
    df = pd.concat(
        [ds[key].to_pandas().assign(split=key) for key in ds], ignore_index=True
    )
    df["severity"] = df["severity"].replace("", "not assigned")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH)
    return df


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
