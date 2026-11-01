"""
Deployment analyses on the standard outer folds: out-of-fold predictions, operating
points at fixed FPR (oracle vs honest thresholds), error breakdowns and
leave-one-source-out robustness.

Fitting and model inputs go through `k_fold_cross.fit_estimator` / `make_X`, and every
split through `make_splits`, so the leakage rules of `nested_cv` carry over.
"""

from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score

from reflex.data.sampler import SEED, make_splits
from reflex.evaluation.k_fold_cross import fit_estimator, make_X

type ErrorKind = Literal["fp", "fn"]


class OperatingPoint(TypedDict):
    fold: int
    target_fpr: float
    oracle_threshold: float
    oracle_fpr: float
    oracle_recall: float
    honest_threshold: float
    honest_fpr: float
    honest_recall: float


class SourceResult(TypedDict):
    source: str
    n: int
    n_malicious: int
    recall: float
    fpr: float
    auroc: float


def predict_fitted(
    estimator: BaseEstimator, train: pd.DataFrame, val: pd.DataFrame
) -> npt.NDArray[np.float64]:
    """Fit a fresh clone on train and return its P(malicious) for each row of val."""
    model = fit_estimator(estimator, train)
    return model.predict_proba(make_X(val, model))[:, 1].astype(np.float64)


def predict_oof(
    estimator: BaseEstimator, df: pd.DataFrame, n_folds: int = 5, seed: int = SEED
) -> pd.DataFrame:
    """
    Return out-of-fold predictions over `make_splits(df, n_folds, seed)`.

    Returns:
        pd.DataFrame: Every validation row (originals only) with its columns plus
        `fold` and `score` (P(malicious) from the model fit on that fold's train split).
    """
    frames = [
        val.assign(fold=k, score=predict_fitted(estimator, train, val))
        for k, (train, val) in enumerate(zip(*make_splits(df, n_folds, seed)))
    ]
    return pd.concat(frames)


def predict_inner_oof(
    estimator: BaseEstimator,
    df: pd.DataFrame,
    n_folds: int = 5,
    n_inner: int = 3,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Return, per outer fold, out-of-fold predictions over inner folds of its train split.

    These never touch the outer validation fold, so thresholds chosen on them are honest.

    Returns:
        pd.DataFrame: Inner validation rows with `fold` (the outer fold), `inner_fold`
        and `score`.
    """
    frames = [
        predict_oof(estimator, train, n_inner, seed)
        .rename(columns={"fold": "inner_fold"})
        .assign(fold=k)
        for k, train in enumerate(make_splits(df, n_folds, seed)[0])
    ]
    return pd.concat(frames)


def choose_threshold(
    y: npt.NDArray[np.int64], scores: npt.NDArray[np.float64], target_fpr: float
) -> float:
    """
    Return the lowest threshold whose FPR (flagging `score >= threshold`) is <= target_fpr.

    This is the max-recall ROC point under the FPR budget for (y, scores).
    """
    negatives = np.sort(scores[y == 0])[::-1]
    k = int(np.floor(target_fpr * len(negatives) + 1e-9))
    if k >= len(negatives):
        return float("-inf")
    return float(np.nextafter(negatives[k], np.inf))


def compute_rates(
    y: npt.NDArray[np.int64], scores: npt.NDArray[np.float64], threshold: float
) -> tuple[float, float]:
    """Return (FPR, recall) when flagging `score >= threshold`; NaN where a class is absent."""
    flagged = scores >= threshold
    fpr = float(flagged[y == 0].mean()) if (y == 0).any() else float("nan")
    recall = float(flagged[y == 1].mean()) if (y == 1).any() else float("nan")
    return fpr, recall


def evaluate_operating_points(
    oof: pd.DataFrame, inner_oof: pd.DataFrame, target_fprs: list[float]
) -> list[OperatingPoint]:
    """
    Score each outer fold at each target FPR with an oracle and an honest threshold.

    The oracle threshold is read off the validation fold itself (optimistic); the honest
    one is chosen on that fold's inner out-of-fold scores and then applied to it.

    Args:
        oof (pd.DataFrame): `predict_oof` output.
        inner_oof (pd.DataFrame): `predict_inner_oof` output for the same model and folds.
        target_fprs (list[float]): FPR budgets, e.g. [0.01, 0.02, 0.05].
    """
    points: list[OperatingPoint] = []
    for fold, val in oof.groupby("fold"):
        inner = inner_oof[inner_oof["fold"] == fold]
        y, scores = val["label"].to_numpy(), val["score"].to_numpy()
        for target in target_fprs:
            oracle = choose_threshold(y, scores, target)
            honest = choose_threshold(
                inner["label"].to_numpy(), inner["score"].to_numpy(), target
            )
            oracle_fpr, oracle_recall = compute_rates(y, scores, oracle)
            honest_fpr, honest_recall = compute_rates(y, scores, honest)
            points.append(
                OperatingPoint(
                    fold=int(fold),
                    target_fpr=target,
                    oracle_threshold=oracle,
                    oracle_fpr=oracle_fpr,
                    oracle_recall=oracle_recall,
                    honest_threshold=honest,
                    honest_fpr=honest_fpr,
                    honest_recall=honest_recall,
                )
            )
    return points


def label_outcomes(
    oof: pd.DataFrame, thresholds: float | dict[int, float]
) -> pd.DataFrame:
    """
    Add an `outcome` column ("tp", "fp", "tn", "fn") to out-of-fold predictions.

    Args:
        oof (pd.DataFrame): `predict_oof` output.
        thresholds (float | dict[int, float]): One threshold, or one per outer fold.
    """
    threshold = (
        oof["fold"].map(thresholds) if isinstance(thresholds, dict) else thresholds
    )
    flagged, malicious = oof["score"] >= threshold, oof["label"] == 1
    outcome = np.where(
        flagged, np.where(malicious, "tp", "fp"), np.where(malicious, "fn", "tn")
    )
    return oof.assign(outcome=outcome)


def count_errors(outcomes: pd.DataFrame, kind: ErrorKind, by: str) -> pd.DataFrame:
    """
    Count errors of one kind per value of column `by`.

    Returns:
        pd.DataFrame: Per group `n` (malicious rows for "fn", benign for "fp"), `errors`
        and `rate`, sorted by `errors` descending.
    """
    pool = outcomes[outcomes["label"] == (1 if kind == "fn" else 0)]
    is_error = (pool["outcome"] == kind).groupby(pool[by])
    table = pd.DataFrame(
        {"n": is_error.size(), "errors": is_error.sum(), "rate": is_error.mean()}
    )
    return table.sort_values("errors", ascending=False)


def get_confident_errors(
    outcomes: pd.DataFrame, kind: ErrorKind, n: int = 15, width: int = 200
) -> pd.DataFrame:
    """Return the n most confident errors of one kind, with texts cut to `width` chars."""
    errors = outcomes[outcomes["outcome"] == kind]
    errors = errors.sort_values("score", ascending=kind == "fn").head(n)
    return errors[["score", "category", "source"]].assign(
        text=errors["text"].str.slice(0, width).str.replace("\n", " ")
    )


def find_cross_source_groups(df: pd.DataFrame) -> list[str]:
    """Return the group_ids whose rows come from more than one `source`."""
    n_sources = df.groupby("group_id")["source"].nunique()
    return n_sources[n_sources > 1].index.tolist()


def evaluate_leave_one_source_out(
    estimator: BaseEstimator, df: pd.DataFrame
) -> list[SourceResult]:
    """
    For each source, fit on all other sources and score that source's originals at 0.5.

    Raises:
        ValueError: If a group spans two sources (holding one out would leak the group).
    """
    if cross := find_cross_source_groups(df):
        raise ValueError(f"{len(cross)} groups span several sources")
    results: list[SourceResult] = []
    for source in sorted(df["source"].unique()):
        held = df["source"] == source
        val = df[held & ~df["augmented"]]
        y = val["label"].to_numpy()
        scores = predict_fitted(estimator, df[~held], val)
        fpr, recall = compute_rates(y, scores, 0.5)
        results.append(
            SourceResult(
                source=str(source),
                n=len(val),
                n_malicious=int(y.sum()),
                recall=recall,
                fpr=fpr,
                auroc=float(roc_auc_score(y, scores))
                if len(np.unique(y)) > 1
                else float("nan"),
            )
        )
    return results
