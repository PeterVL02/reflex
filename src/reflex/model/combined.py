"""
Stacked prompt classifier: frozen features + out-of-fold text-LR -> LR / GBM / RF / torch MLP.

Per row the combiner sees the Qwen embedding (1024) and zero-shot Laya scores (3) from the
feature cache, the 9 handcrafted features, and the TF-IDF + LR base model's P(malicious).
During `fit` that base score is out-of-fold: `make_splits(train, 3, seed)` partitions the
training groups, and each group (augmented variants included) is scored by a base model
that never saw it. The base model is then refit on all training rows for `predict_proba`.

Every combiner also predicts severity (1-4), trained on malicious rows only: a separate
regressor for "lr" / "gbm" / "rf", a second head for "mlp".
"""

from typing import Literal, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from reflex.data.data_transformation import (
    VectorizerName,
    extract_handcrafted_features,
    make_vectorizer,
)
from reflex.data.features import get_features
from reflex.data.sampler import SEED, make_splits

type CombinerName = Literal["lr", "gbm", "rf", "mlp"]

N_OOF_FOLDS: int = 3
# One fold (~20% of training groups) is the MLP's early-stopping slice.
N_EARLY_STOP_FOLDS: int = 5
MLP_HIDDEN: tuple[int, int] = (256, 64)
MLP_DROPOUT: float = 0.2
MLP_LR: float = 1e-3
MLP_WEIGHT_DECAY: float = 1e-4
MLP_BATCH: int = 256
MLP_MAX_EPOCHS: int = 200
MLP_PATIENCE: int = 10
SEVERITY_LOSS_WEIGHT: float = 1.0


def make_base_model(features: VectorizerName = "tfidf_word") -> Pipeline:
    """Return the unfitted text vectorizer + logistic regression base model."""
    return make_pipeline(make_vectorizer(features), LogisticRegression(max_iter=1000))


def compute_oof_scores(
    frame: pd.DataFrame, seed: int, features: VectorizerName = "tfidf_word"
) -> npt.NDArray[np.float64]:
    """
    Score every training row with a base model that did not see the row's group.

    Args:
        frame (pd.DataFrame): Positionally indexed rows with 'text', 'label', 'group_id'
            and 'augmented' columns.
        seed (int): Seed for `make_splits`.
        features (VectorizerName): Text features of the base model.

    Returns:
        npt.NDArray[np.float64]: Out-of-fold P(malicious), one per row of `frame`.

    Raises:
        ValueError: If some rows got no score (groups without an original row, which
            `make_splits` never puts in validation).
    """
    oof = np.full(len(frame), np.nan)
    for train, val in zip(*make_splits(frame, N_OOF_FOLDS, seed)):
        # val holds originals only; score their augmented variants too.
        held = np.flatnonzero(frame["group_id"].isin(val["group_id"]))
        model = make_base_model(features).fit(train["text"], train["label"])
        oof[held] = model.predict_proba(frame["text"].iloc[held])[:, 1]
    if np.isnan(oof).any():
        raise ValueError(f"{np.isnan(oof).sum()} rows have no out-of-fold score")
    return oof


class MultiTaskNet(torch.nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        h1, h2 = MLP_HIDDEN
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(n_features, h1),
            torch.nn.ReLU(),
            torch.nn.Dropout(MLP_DROPOUT),
            torch.nn.Linear(h1, h2),
            torch.nn.ReLU(),
            torch.nn.Dropout(MLP_DROPOUT),
        )
        self.label_head = torch.nn.Linear(h2, 1)
        self.severity_head = torch.nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the label logit and the standardised severity per row."""
        h = self.trunk(x)
        return self.label_head(h).squeeze(1), self.severity_head(h).squeeze(1)


def compute_multitask_loss(
    net: MultiTaskNet, x: torch.Tensor, y: torch.Tensor, severity: torch.Tensor
) -> torch.Tensor:
    """
    Return BCE on the label plus weighted MSE on severity over rows with a severity target.

    `severity` is NaN for rows without a target (benign rows, or no severity given).
    """
    logit, severity_pred = net(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
    mask = ~torch.isnan(severity)
    if mask.any():
        loss = loss + SEVERITY_LOSS_WEIGHT * torch.nn.functional.mse_loss(
            severity_pred[mask], severity[mask]
        )
    return loss


def train_multitask_net(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    severity_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    severity_val: torch.Tensor,
    seed: int,
) -> MultiTaskNet:
    """
    Train a MultiTaskNet with AdamW, early-stopped on the validation loss.

    Returns:
        MultiTaskNet: The network with its best-validation-loss weights, in eval mode.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    net = MultiTaskNet(x_train.shape[1])
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY
    )
    best_loss, stale = float("inf"), 0
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    for _ in range(MLP_MAX_EPOCHS):
        net.train()
        for batch in torch.randperm(len(x_train), generator=generator).split(MLP_BATCH):
            optimizer.zero_grad()
            compute_multitask_loss(
                net, x_train[batch], y_train[batch], severity_train[batch]
            ).backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            val_loss = compute_multitask_loss(net, x_val, y_val, severity_val).item()
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= MLP_PATIENCE:
                break
    net.load_state_dict(best_state)
    return net.eval()


class CombinedClassifier(ClassifierMixin, BaseEstimator):
    # Opt-in for evaluation.k_fold_cross: pass X = df[["text", "group_id", "augmented"]].
    uses_frame: bool = True

    def __init__(
        self,
        combiner: CombinerName = "lr",
        use_embedding: bool = True,
        use_laya: bool = True,
        base_features: VectorizerName = "tfidf_word",
        seed: int = SEED,
    ) -> None:
        self.combiner = combiner
        self.base_features = base_features
        self.use_embedding = use_embedding
        self.use_laya = use_laya
        self.seed = seed

    def _build_features(
        self, texts: pd.Series[str], base_scores: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Stack the enabled frozen features, handcrafted features and base-model score."""
        parts: list[npt.NDArray[np.float32] | npt.NDArray[np.float64]] = []
        if self.use_embedding:
            parts.append(get_features(texts, "embedding"))
        if self.use_laya:
            parts.append(get_features(texts, "laya"))
        parts += [extract_handcrafted_features(texts), base_scores[:, None]]
        return np.hstack(parts).astype(np.float64)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series[int],
        severity: pd.Series[float] | None = None,
    ) -> Self:
        """
        Fit the base model (out-of-fold, then on all rows), the combiner and severity model.

        Args:
            X (pd.DataFrame): Rows with 'text', 'group_id' and 'augmented' columns.
            y (pd.Series[int]): Binary labels, aligned with X.
            severity (pd.Series[float] | None): Encoded severity (1-4, NaN where
                unknown), aligned with X. Only rows with y == 1 are used. Without it
                `predict_severity` is unavailable.
        """
        if self.combiner not in ("lr", "gbm", "rf", "mlp"):
            raise ValueError(f"unknown combiner {self.combiner!r}")
        frame = X.reset_index(drop=True).assign(label=np.asarray(y))
        labels = frame["label"].to_numpy()
        targets = (
            np.full(len(frame), np.nan)
            if severity is None
            else np.asarray(severity, dtype=np.float64).copy()
        )
        targets[labels != 1] = np.nan
        has_target = ~np.isnan(targets)
        self.has_severity_ = bool(has_target.any())

        features = self._build_features(
            frame["text"], compute_oof_scores(frame, self.seed, self.base_features)
        )
        self.base_model_ = make_base_model(self.base_features).fit(
            frame["text"], labels
        )
        self.classes_ = np.array([0, 1])

        if self.combiner == "mlp":
            self._fit_mlp(frame, features, labels, targets)
            return self
        if self.combiner == "lr":
            self.classifier_ = make_pipeline(
                StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)
            )
            self.severity_model_ = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        elif self.combiner == "rf":
            self.classifier_ = RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=self.seed
            )
            self.severity_model_ = RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=self.seed
            )
        elif self.combiner == "gbm":
            self.classifier_ = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=self.seed,
            )
            self.severity_model_ = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=300,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=self.seed,
            )
        self.classifier_.fit(features, labels)
        if self.has_severity_:
            self.severity_model_.fit(features[has_target], targets[has_target])
        return self

    def _fit_mlp(
        self,
        frame: pd.DataFrame,
        features: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int64],
        targets: npt.NDArray[np.float64],
    ) -> None:
        """Fit the scaler and multi-task MLP, early-stopping on a make_splits slice of `frame`."""
        self.scaler_ = StandardScaler().fit(features)
        observed = targets[~np.isnan(targets)]
        self.severity_mean_ = float(observed.mean()) if observed.size else 0.0
        self.severity_std_ = float(observed.std()) if observed.size else 1.0
        x = torch.tensor(self.scaler_.transform(features), dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        s = torch.tensor(
            (targets - self.severity_mean_) / (self.severity_std_ or 1.0),
            dtype=torch.float32,
        )
        # frame is positionally indexed, so make_splits' row labels are positions.
        es_train, es_val = make_splits(frame, N_EARLY_STOP_FOLDS, self.seed)
        tr = torch.tensor(es_train[0].index.to_numpy())
        va = torch.tensor(es_val[0].index.to_numpy())
        self.net_ = train_multitask_net(
            x[tr], y[tr], s[tr], x[va], y[va], s[va], self.seed
        )

    def _transform(self, X: pd.DataFrame | pd.Series[str]) -> npt.NDArray[np.float64]:
        """Return combiner inputs for new rows, using the base model fit on all training rows."""
        texts = X["text"] if isinstance(X, pd.DataFrame) else X
        return self._build_features(texts, self.base_model_.predict_proba(texts)[:, 1])

    def _predict_mlp(
        self, features: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the MLP's P(malicious) and severity (original 1-4 scale) per row."""
        x = torch.tensor(self.scaler_.transform(features), dtype=torch.float32)
        with torch.no_grad():
            logit, severity = self.net_(x)
        p = torch.sigmoid(logit).numpy().astype(np.float64)
        severity_np = severity.numpy().astype(np.float64)
        return p, severity_np * self.severity_std_ + self.severity_mean_

    def predict_proba(
        self, X: pd.DataFrame | pd.Series[str]
    ) -> npt.NDArray[np.float64]:
        """Return [P(label 0), P(label 1)] per row (X may be the frame or just the texts)."""
        features = self._transform(X)
        if self.combiner == "mlp":
            p = self._predict_mlp(features)[0]
        else:
            p = self.classifier_.predict_proba(features)[:, 1]
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame | pd.Series[str]) -> npt.NDArray[np.int64]:
        """Return the predicted label per row, at threshold 0.5."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

    def predict_severity(
        self, X: pd.DataFrame | pd.Series[str]
    ) -> npt.NDArray[np.float64]:
        """
        Return predicted severity (clipped to 1-4), assuming each prompt is malicious.

        Raises:
            ValueError: If `fit` got no severity targets.
        """
        if not self.has_severity_:
            raise ValueError("fit was called without severity targets")
        features = self._transform(X)
        if self.combiner == "mlp":
            severity = self._predict_mlp(features)[1]
        else:
            severity = self.severity_model_.predict(features)
        return np.clip(severity, 1.0, 4.0)
