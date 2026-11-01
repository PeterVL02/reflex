"""
Nested k-fold cross-validation and the official-holdout evaluation, with one record per
fold appended to results/cv_results.jsonl.

Estimator inputs: text-only estimators get `df["text"]`. Estimators with a truthy
`uses_frame` class attribute get `df[FRAME_COLUMNS]` (text plus the group columns needed
for their own inner splits); label, severity, category, source and tags are never passed
in X. Estimators with `predict_severity` also get the encoded severity in `fit` and are
scored on it over malicious validation rows.
"""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NotRequired, Self, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from reflex.data.features import get_features, get_laya_columns
from reflex.data.sampler import SEED, assign_split_groups, make_splits

# parents: evaluation -> reflex -> src -> repo root
RESULTS_PATH: Path = (
    Path(__file__).resolve().parents[3] / "results" / "cv_results.jsonl"
)
FRAME_COLUMNS: list[str] = ["text", "group_id", "augmented"]
SEVERITY_CODES: dict[str, int] = {"low": 1, "medium": 2, "high": 3, "critical": 4}

type MetricName = Literal["auroc", "accuracy", "precision", "recall", "f1"]
type EvalKind = Literal["cv", "holdout", "holdout_dedup"]


class Metrics(TypedDict):
    auroc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float


class SeverityMetrics(TypedDict):
    n: int
    mae: float
    rmse: float
    mae_by_source: dict[str, float]


class FoldRecord(TypedDict):
    timestamp: str
    run_id: str
    model: str
    params: str
    fold: int
    n_train: int
    n_val: int
    metrics: Metrics
    inner_scores: dict[str, float]
    seed: int
    eval: NotRequired[EvalKind]  # missing in older records, which are all "cv"
    severity: NotRequired[SeverityMetrics | None]


class LayaClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, prompt_name: str = "v1_attack_harm") -> None:
        self.prompt_name = prompt_name

    def fit(self, X: pd.Series[str], y: pd.Series[int]) -> LayaClassifier:
        """No-op: the Laya model is pretrained."""
        return self

    def predict_proba(self, X: pd.Series[str]) -> npt.NDArray[np.float64]:
        """Return [P(label 0), P(label 1)] per prompt from Laya."""
        # Heavy import, so load on first use.
        from reflex.model.sys_one import classify_prompts

        p = classify_prompts(X, prompt_name=self.prompt_name)
        return np.column_stack([1 - p, p])


class CachedLayaClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, prompt_name: str = "v2_attack_min") -> None:
        self.prompt_name = prompt_name

    def fit(self, X: pd.Series[str], y: pd.Series[int]) -> Self:
        """No-op: zero-shot Laya scores come from the frozen feature cache."""
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.Series[str]) -> npt.NDArray[np.float64]:
        """Return [P(label 0), P(label 1)] per prompt from the cached zero-shot Laya column."""
        column = get_laya_columns().index(self.prompt_name)
        p = get_features(X, "laya")[:, column].astype(np.float64)
        return np.column_stack([1 - p, p])


def encode_severity(severity: pd.Series[str]) -> pd.Series[float]:
    """Map low/medium/high/critical to 1-4; anything else (e.g. 'not assigned') to NaN."""
    return severity.map(SEVERITY_CODES).astype(float)


def make_X(df: pd.DataFrame, estimator: BaseEstimator) -> pd.Series[str] | pd.DataFrame:
    """Return the estimator's input: `df[FRAME_COLUMNS]` if it sets `uses_frame`, else the texts."""
    if getattr(estimator, "uses_frame", False):
        return df[FRAME_COLUMNS]
    return df["text"]


def score_estimator(
    estimator: BaseEstimator, X: pd.Series[str] | pd.DataFrame, y: pd.Series[int]
) -> Metrics:
    """
    Score a fitted binary classifier on (X, y).

    Uses predict_proba (threshold 0.5) when available, else predict plus decision_function
    for AUROC. AUROC is NaN when y holds a single class, FPR when y has no negatives.
    """
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(X)[:, 1]
        labels = (scores >= 0.5).astype(int)
    else:
        labels = estimator.predict(X)
        scores = (
            estimator.decision_function(X)
            if hasattr(estimator, "decision_function")
            else labels
        )
    tn, fp, _, _ = confusion_matrix(y, labels, labels=[0, 1]).ravel()
    return Metrics(
        auroc=float(roc_auc_score(y, scores)) if y.nunique() > 1 else float("nan"),
        accuracy=float(accuracy_score(y, labels)),
        precision=float(precision_score(y, labels, zero_division=0)),
        recall=float(recall_score(y, labels, zero_division=0)),
        f1=float(f1_score(y, labels, zero_division=0)),
        fpr=float(fp / (fp + tn)) if fp + tn else float("nan"),
    )


def score_severity(estimator: BaseEstimator, df: pd.DataFrame) -> SeverityMetrics:
    """
    Score a fitted estimator's `predict_severity` on the malicious rows of df only.

    Returns:
        SeverityMetrics: MAE and RMSE on the 1-4 scale, plus MAE per `source`.
    """
    target = encode_severity(df["severity"])
    malicious = df[(df["label"] == 1) & target.notna()]
    predicted = estimator.predict_severity(make_X(malicious, estimator))
    abs_error = pd.Series(
        np.abs(predicted - encode_severity(malicious["severity"]).to_numpy()),
        index=malicious.index,
    )
    by_source = abs_error.groupby(malicious["source"]).mean()
    return SeverityMetrics(
        n=len(malicious),
        mae=float(abs_error.mean()),
        rmse=float(np.sqrt((abs_error**2).mean())),
        mae_by_source={str(k): float(v) for k, v in by_source.items()},
    )


def fit_estimator(estimator: BaseEstimator, train: pd.DataFrame) -> BaseEstimator:
    """Fit a fresh clone on train, passing encoded severity if it has `predict_severity`."""
    model = clone(estimator)
    X, y = make_X(train, model), train["label"]
    if hasattr(model, "predict_severity"):
        return model.fit(X, y, severity=encode_severity(train["severity"]))
    return model.fit(X, y)


def fit_and_score(
    estimator: BaseEstimator, train: pd.DataFrame, val: pd.DataFrame
) -> Metrics:
    """Fit a fresh clone of the estimator on train and score it on val."""
    model = fit_estimator(estimator, train)
    return score_estimator(model, make_X(val, model), val["label"])


def save_record(record: FoldRecord, path: Path = RESULTS_PATH) -> None:
    """Append one record as a JSON line, creating the results directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def select_candidate(
    candidates: dict[str, BaseEstimator],
    train: pd.DataFrame,
    n_inner: int,
    metric: MetricName,
    seed: int,
) -> tuple[str, dict[str, float]]:
    """
    Pick the candidate with the best mean `metric` over inner `make_splits` folds of train.

    With a single candidate no inner folds are run and its inner scores are empty.

    Returns:
        tuple[str, dict[str, float]]: The chosen name and each candidate's inner score.
    """
    if len(candidates) == 1:
        return next(iter(candidates)), {}
    inner_train, inner_val = make_splits(train, n_inner, seed)
    inner_scores = {
        name: float(
            np.nanmean(
                [
                    fit_and_score(estimator, tr, va)[metric]
                    for tr, va in zip(inner_train, inner_val)
                ]
            )
        )
        for name, estimator in candidates.items()
    }
    return max(inner_scores, key=inner_scores.__getitem__), inner_scores


def evaluate_fold(
    candidates: dict[str, BaseEstimator],
    train: pd.DataFrame,
    val: pd.DataFrame,
    n_inner: int,
    metric: MetricName,
    seed: int,
    run_id: str,
    fold: int,
    eval_kind: EvalKind,
) -> FoldRecord:
    """Select on inner folds of train, refit the winner on train, score it once on val."""
    best, inner_scores = select_candidate(candidates, train, n_inner, metric, seed)
    model = fit_estimator(candidates[best], train)
    return FoldRecord(
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        run_id=run_id,
        model=best,
        params=repr(candidates[best]),
        fold=fold,
        n_train=len(train),
        n_val=len(val),
        metrics=score_estimator(model, make_X(val, model), val["label"]),
        inner_scores=inner_scores,
        seed=seed,
        eval=eval_kind,
        severity=score_severity(model, val)
        if hasattr(model, "predict_severity")
        else None,
    )


def nested_cv(
    candidates: dict[str, BaseEstimator],
    df: pd.DataFrame,
    n_outer: int = 5,
    n_inner: int = 3,
    metric: MetricName = "auroc",
    seed: int = SEED,
    results_path: Path = RESULTS_PATH,
) -> list[FoldRecord]:
    """
    Run nested CV: per outer fold, pick the candidate with the best mean inner-fold `metric`,
    refit it on the full outer train split and score it on the outer val split.

    With a single candidate the inner loop is skipped (plain k-fold CV), which is how a
    pretrained model such as LayaClassifier is evaluated on the same outer folds.

    Each outer-fold record (eval="cv") is appended to `results_path` as soon as it is computed.

    Returns:
        list[FoldRecord]: One record per outer fold.
    """
    run_id = uuid.uuid4().hex[:8]
    records: list[FoldRecord] = []
    outer_train, outer_val = make_splits(df, n_outer, seed)
    for fold, (train, val) in enumerate(zip(outer_train, outer_val)):
        record = evaluate_fold(
            candidates, train, val, n_inner, metric, seed, run_id, fold, "cv"
        )
        save_record(record, results_path)
        records.append(record)
    return records


def evaluate_holdout(
    candidates: dict[str, BaseEstimator],
    df: pd.DataFrame,
    n_inner: int = 3,
    metric: MetricName = "auroc",
    seed: int = SEED,
    results_path: Path = RESULTS_PATH,
    drop_near_duplicates: bool = False,
) -> FoldRecord:
    """
    Evaluate on the official HF test split, comparable with the data card.

    Selection (only with more than one candidate) uses inner `make_splits` folds of the
    non-test rows (train + validation); the winner is refit on all non-test rows and scored
    once on `split == "test"`. The test split is never used for selection or tuning.

    With `drop_near_duplicates`, test prompts with a near-duplicate outside the test split
    (54 of 942 in `full`, see `assign_split_groups`) are left out of scoring, which removes the
    leak the HF split itself has. Training is unchanged.

    Returns:
        FoldRecord: The holdout record (eval="holdout" or "holdout_dedup", fold=-1), also
        appended to `results_path`.
    """
    is_test = df["split"] == "test"
    # Holds for the HF configs; guard it so another dataset version cannot leak.
    assert not df.loc[is_test, "augmented"].any(), "test split has augmented rows"
    assert not set(df.loc[is_test, "group_id"]) & set(df.loc[~is_test, "group_id"]), (
        "a group spans the test split and the rest"
    )
    scored = is_test
    if drop_near_duplicates:
        split_groups = assign_split_groups(df)
        scored = is_test & ~split_groups.isin(split_groups[~is_test])
    record = evaluate_fold(
        candidates,
        df[~is_test],
        df[scored],
        n_inner,
        metric,
        seed,
        uuid.uuid4().hex[:8],
        -1,
        "holdout_dedup" if drop_near_duplicates else "holdout",
    )
    save_record(record, results_path)
    return record


def summarize(records: list[FoldRecord]) -> pd.DataFrame:
    """Return the mean and std of each metric across the given fold records."""
    return pd.DataFrame([r["metrics"] for r in records]).agg(["mean", "std"])


def summarize_severity(records: list[FoldRecord]) -> pd.DataFrame:
    """
    Return the mean and std across records of severity MAE, RMSE and MAE per source.

    Columns are 'mae', 'rmse' and 'mae_<source>'; empty if no record has severity.
    """
    rows = [
        {
            "mae": s["mae"],
            "rmse": s["rmse"],
            **{f"mae_{k}": v for k, v in s["mae_by_source"].items()},
        }
        for r in records
        if (s := r.get("severity")) is not None
    ]
    return pd.DataFrame(rows).agg(["mean", "std"]) if rows else pd.DataFrame()
