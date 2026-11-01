"Fine-tuned Laya (convaiinnovations/laya, English root) as an sklearn classifier."

import gc
import json
import shutil
from pathlib import Path
from typing import Literal, Self, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from laya.agent import Agent
from laya.train import TrainConfig, finetune
from sklearn.base import BaseEstimator, ClassifierMixin

from reflex.data.loader import REPO_ROOT
from reflex.model.sys_one import load_instructions

BASE_MODEL: str = "convaiinnovations/laya"
QUESTION_ID: str = "malicious"
EFFECTIVE_BATCH: int = 64  # laya's default: micro_batch 8 x grad_accum 8
MODELS_DIR: Path = (REPO_ROOT / "data" / "models").resolve()
DEFAULT_OUTPUT_DIR: Path = MODELS_DIR / "laya_ft"


class NoulQuestion(TypedDict):
    type: Literal["noul"]
    instructions: str


class TrainRow(TypedDict):
    state: str
    questions: dict[str, NoulQuestion]
    expected: dict[str, int]


class LayaFinetunedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        prompt_name: str = "v1_attack_harm",
        epochs: int = 4,
        freeze_encoder: bool = False,
        micro_batch: int = 8,
        output_dir: str = str(DEFAULT_OUTPUT_DIR),
    ) -> None:
        self.prompt_name = prompt_name
        self.epochs = epochs
        self.freeze_encoder = freeze_encoder
        self.micro_batch = micro_batch
        self.output_dir = output_dir

    def _questions(self) -> dict[str, NoulQuestion]:
        """Return the single yes/no question asked of every prompt."""
        return {
            QUESTION_ID: NoulQuestion(
                type="noul", instructions=load_instructions(self.prompt_name)
            )
        }

    def fit(self, X: pd.Series[str], y: pd.Series[int]) -> Self:
        """
        Fine-tune Laya on (X, y) into `output_dir` and load the result.

        `output_dir` is wiped first, so successive fits (e.g. CV folds) keep one
        checkpoint on disk. Laya holds out its own calibration slice from (X, y).

        Raises:
            ValueError: If `output_dir` is not a subdirectory of data/models (it gets deleted).
            FileExistsError: If another fit holds the lock on `output_dir`.
        """
        out = Path(self.output_dir).resolve()
        if MODELS_DIR not in out.parents:
            raise ValueError(f"output_dir must be inside {MODELS_DIR}, got {out}")
        # A second fit on the same dir would delete a running fine-tune's checkpoint.
        lock = out.with_name(f"{out.name}.lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.touch(exist_ok=False)
        try:
            self._finetune(X, y, out)
        finally:
            lock.unlink()
        return self

    def _finetune(self, X: pd.Series[str], y: pd.Series[int], out: Path) -> None:
        """Write the training JSONL, fine-tune into `out` and load the resulting agent."""
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        data = out / "train.jsonl"
        questions = self._questions()
        with data.open("w", encoding="utf-8") as f:
            for text, label in zip(X, y):
                row = TrainRow(
                    state=text, questions=questions, expected={QUESTION_ID: int(label)}
                )
                f.write(json.dumps(row) + "\n")

        config = TrainConfig(
            epochs=self.epochs,
            micro_batch=self.micro_batch,
            grad_accum=EFFECTIVE_BATCH // self.micro_batch,
            freeze_encoder=self.freeze_encoder,
        )
        finetune(str(data), BASE_MODEL, str(out), config)
        # Per-epoch copy of the same weights: another ~1.7 GB, not needed.
        shutil.rmtree(out / "checkpoint_latest", ignore_errors=True)
        gc.collect()
        torch.cuda.empty_cache()  # free the training copy before loading the agent
        self.agent_ = Agent(str(out))
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.Series[str]) -> npt.NDArray[np.float64]:
        """Return [P(label 0), P(label 1)] per prompt from the fine-tuned model."""
        results = self.agent_.predict_batch(list(X), self._questions(), batch_size=32)
        p = np.array([r["answers"][QUESTION_ID]["noul"] for r in results])
        return np.column_stack([1 - p, p])
