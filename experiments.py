"""
Experiment grid. Every entry of `build_experiments()` is one model, run through the
project's standard 5-fold CV (group-level, leakage-free folds from `make_splits`).

    uv run experiments.py                  # run the whole grid
    uv run experiments.py --only combined  # only entries whose name contains "combined"
    uv run experiments.py --list           # print the entry names and exit
    uv run experiments.py --holdout        # also score the official HF test split

Choose between experiments on CV only. Use --holdout once, on the final candidates:
choosing by test score turns the test split into a validation set.
Every fold is appended to results/cv_results.jsonl, and a table ranked by CV FPR is printed.
"""

import argparse
import time
from functools import partial
from itertools import product
from typing import Callable  # noqa: UP035 (user convention: typing constructs from typing)

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from reflex.data.data_transformation import VECTORIZER_NAMES, make_vectorizer
from reflex.data.loader import load_data
from reflex.data.sampler import SEED
from reflex.evaluation.k_fold_cross import (
    RESULTS_PATH,
    CachedLayaClassifier,
    evaluate_holdout,
    nested_cv,
    summarize,
)
from reflex.model.combined import CombinedClassifier

N_OUTER: int = 5
METRICS: tuple[str, ...] = ("fpr", "auroc", "accuracy", "f1", "recall")

# Classifiers placed after a text vectorizer.
TEXT_MODELS: dict[str, Callable[[], BaseEstimator]] = {
    "lr": partial(LogisticRegression, max_iter=1000),
    "rf": partial(
        RandomForestClassifier,
        n_estimators=300,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEED,
    ),
    "nb": MultinomialNB,
}


def build_experiments() -> dict[str, BaseEstimator]:
    """Return the experiment grid as name -> unfitted estimator. Edit this to add experiments."""
    experiments: dict[str, BaseEstimator] = {
        f"{vec}+{model}": make_pipeline(make_vectorizer(vec), TEXT_MODELS[model]())
        for vec, model in product(VECTORIZER_NAMES, TEXT_MODELS)
    }
    # Full model: base text features x combiner (embedding + Laya + handcrafted always on).
    for combiner, vec in product(
        ("lr", "gbm", "rf"), ("tfidf_word", "bow_word", "tfidf_char")
    ):
        experiments[f"combined_{combiner}[{vec}]"] = CombinedClassifier(
            combiner, base_features=vec
        )
    experiments["combined_mlp[tfidf_word]"] = CombinedClassifier("mlp")
    experiments["laya_zero_shot_v2"] = CachedLayaClassifier("v2_attack_min")
    return experiments


def run_experiment(
    name: str, estimator: BaseEstimator, df: pd.DataFrame, holdout: bool
) -> dict[str, float]:
    """
    Run one experiment and return its table row.

    Returns:
        dict[str, float]: CV mean of each metric, the CV std of FPR, the test value of each
            metric when `holdout` is set, and the wall time in seconds.
    """
    start = time.perf_counter()
    cv = summarize(nested_cv({name: estimator}, df, n_outer=N_OUTER))
    row = {m: float(cv.loc["mean", m]) for m in METRICS}
    row["fpr_std"] = float(cv.loc["std", "fpr"])
    if holdout:
        test = evaluate_holdout({name: estimator}, df)["metrics"]
        row |= {f"test_{m}": float(test[m]) for m in METRICS}
    row["seconds"] = time.perf_counter() - start
    print(
        f"{name}: FPR {row['fpr']:.4f}, AUROC {row['auroc']:.4f} ({row['seconds']:.0f}s)"
    )
    return row


def main() -> None:
    """Parse arguments, run the selected experiments and print a table ranked by CV FPR."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--only", default="", help="run entries whose name contains this"
    )
    parser.add_argument(
        "--list", action="store_true", help="print entry names and exit"
    )
    parser.add_argument(
        "--holdout", action="store_true", help="also score the HF test split"
    )
    args = parser.parse_args()

    experiments = {n: e for n, e in build_experiments().items() if args.only in n}
    if args.list or not experiments:
        print("\n".join(experiments) or f"no experiment name contains {args.only!r}")
        return

    df = load_data()
    rows = {n: run_experiment(n, e, df, args.holdout) for n, e in experiments.items()}
    table = pd.DataFrame.from_dict(rows, orient="index").sort_values("fpr")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(f"\n5-fold CV means, ranked by FPR (records in {RESULTS_PATH})")
        print(table.round(4))


if __name__ == "__main__":
    main()
