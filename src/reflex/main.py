import time
from typing import TypedDict

import pandas as pd
from sklearn.base import BaseEstimator

from reflex.data.loader import load_data
from reflex.evaluation.k_fold_cross import (
    RESULTS_PATH,
    CachedLayaClassifier,
    FoldRecord,
    evaluate_holdout,
    nested_cv,
    summarize,
    summarize_severity,
)
from reflex.model.combined import CombinedClassifier, make_base_model

N_OUTER: int = 5
TABLE_METRICS: tuple[str, ...] = ("auroc", "accuracy", "f1", "fpr")


class ModelResult(TypedDict):
    name: str
    cv: list[FoldRecord]
    holdout: FoldRecord
    holdout_dedup: FoldRecord
    seconds: float


def evaluate_model(
    name: str, estimator: BaseEstimator, df: pd.DataFrame
) -> ModelResult:
    """Run 5-fold CV, the official holdout and the near-dedup holdout for one model."""
    start = time.perf_counter()
    cv = nested_cv({name: estimator}, df, n_outer=N_OUTER)
    holdout = evaluate_holdout({name: estimator}, df)
    holdout_dedup = evaluate_holdout({name: estimator}, df, drop_near_duplicates=True)
    seconds = time.perf_counter() - start
    print(f"{name}: done in {seconds:.0f}s", flush=True)
    return ModelResult(
        name=name, cv=cv, holdout=holdout, holdout_dedup=holdout_dedup, seconds=seconds
    )


def build_metric_table(results: list[ModelResult]) -> pd.DataFrame:
    """Return one row per model: CV mean±std, holdout and dedup-holdout value per metric."""
    rows: dict[str, dict[str, str]] = {}
    for result in results:
        cv = summarize(result["cv"])
        row = {
            f"cv_{m}": f"{cv.loc['mean', m]:.4f} +- {cv.loc['std', m]:.4f}"
            for m in TABLE_METRICS
        }
        row |= {
            f"test_{m}": f"{result['holdout']['metrics'][m]:.4f}" for m in TABLE_METRICS
        }
        row |= {
            f"dedup_{m}": f"{result['holdout_dedup']['metrics'][m]:.4f}"
            for m in TABLE_METRICS
        }
        rows[result["name"]] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def build_severity_table(results: list[ModelResult]) -> pd.DataFrame:
    """Return severity MAE/RMSE (overall and per source) for models that predict severity."""
    rows: dict[str, pd.Series[float]] = {}
    for result in results:
        cv = summarize_severity(result["cv"])
        if cv.empty:
            continue
        rows[f"{result['name']} cv"] = cv.loc["mean"]
        rows[f"{result['name']} test"] = summarize_severity([result["holdout"]]).loc[
            "mean"
        ]
    return pd.DataFrame(rows).T


def main() -> None:
    """Compare the baseline, zero-shot Laya and the combined models (5-fold CV + HF test)."""
    df = load_data()
    candidates: dict[str, BaseEstimator] = {
        "tfidf_logreg": make_base_model(),
        "laya_zero_shot_v2": CachedLayaClassifier("v2_attack_min"),
        "combined_lr": CombinedClassifier("lr"),
        "combined_gbm": CombinedClassifier("gbm"),
        "combined_mlp": CombinedClassifier("mlp"),
    }
    results = [evaluate_model(name, est, df) for name, est in candidates.items()]

    # Ablate the combiner with the lowest mean CV FPR (the deployment metric).
    best = min(
        (r for r in results if r["name"].startswith("combined_")),
        key=lambda r: summarize(r["cv"]).loc["mean", "fpr"],
    )
    combiner = candidates[best["name"]].combiner
    ablations = {
        f"{best['name']}_no_embedding": CombinedClassifier(
            combiner, use_embedding=False
        ),
        f"{best['name']}_no_laya": CombinedClassifier(combiner, use_laya=False),
    }
    results += [evaluate_model(name, est, df) for name, est in ablations.items()]

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\nBinary label (cv = 5-fold mean +- std, test = official HF test split)")
        print(build_metric_table(results))
        print("\nSeverity on malicious rows (MAE / RMSE on the 1-4 scale)")
        print(build_severity_table(results).round(3))
    print(
        f"\nTotal {sum(r['seconds'] for r in results):.0f}s; records in {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
