from __future__ import annotations

import copy
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from daneel.parameters import Parameters

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Configures application-wide logging format and severity level.

    Args:
        log_level: Logging severity level (e.g., 'DEBUG', 'INFO', 'WARNING').
    """
    numeric_level = getattr(logging, str(log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=False,
    )


# Default configuration used as fallback if no YAML file is provided
DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "csv_path": "tess_data.csv",
        "n_bins": 1000,
        "test_size": 0.2,
        "balanced": True,
        "samples_per_class": 350,
        "random_state": 42,
    },
    "model": {
        # Architecture
        "conv_channels": [16, 32, 64],   # Out-channels for each Conv1d block
        "kernel_size": 7,                # Convolution kernel width (odd recommended)
        "pool_size": 2,                  # MaxPool1d window size after each conv block
        "fc_units": [128, 64],           # Fully-connected layer widths before output
        "dropout": 0.3,                  # Dropout probability applied before each FC layer
        # Training
        "batch_size": 32,
        "epochs": 30,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "random_state": 42,
    },
    "output": {
        "report_path": "cnn/results/report.txt",
        "model_path": None,              # Optional path to save trained model weights
    },
    "logging": {
        "level": "INFO",
    },
}


# Config helper functions

def _deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merges *override* into *base*, returning a new dictionary."""
    merged = copy.deepcopy(base)
    if not override:
        return merged
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_user_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalises various YAML schemas into the standard {data, model, output, logging} structure.

    Supported layouts:
      1. Standard: ``{data: {...}, model: {...}, output: {...}}``
      2. Wrapped under ``cnn``, ``classifier``, or ``detection`` keys.
      3. Flat parameters at root level.
    """
    cfg: Dict[str, Any] = {"data": {}, "model": {}, "output": {}, "logging": {}}

    # Unwrap optional top-level wrapper key
    target = raw
    for key in ("cnn", "classifier", "detection"):
        if key in raw and isinstance(raw[key], dict):
            target = raw[key]
            break

    # Pull structured sub-sections first
    for section in ("data", "model", "output", "logging"):
        if section in target and isinstance(target[section], dict):
            cfg[section].update(target[section])
        elif section in raw and isinstance(raw[section], dict):
            cfg[section].update(raw[section])

    # Map flat / synonym keys
    _data_keys = {
        "csv_path": "csv_path",
        "data_path": "csv_path",
        "dataset_path": "csv_path",
        "dataset": "csv_path",
        "n_bins": "n_bins",
        "test_size": "test_size",
        "balanced": "balanced",
        "samples_per_class": "samples_per_class",
        "data_random_state": "random_state",
    }
    _model_keys = {
        "conv_channels": "conv_channels",
        "kernel_size": "kernel_size",
        "pool_size": "pool_size",
        "fc_units": "fc_units",
        "dropout": "dropout",
        "batch_size": "batch_size",
        "epochs": "epochs",
        "learning_rate": "learning_rate",
        "lr": "learning_rate",
        "weight_decay": "weight_decay",
        "random_state": "random_state",
    }
    _output_keys = {
        "report_path": "report_path",
        "output_path": "report_path",
        "output_file": "report_path",
        "report_file": "report_path",
        "model_path": "model_path",
        "model_save_path": "model_path",
    }

    for src in (raw, target):
        for k, v in src.items():
            if k in _data_keys and _data_keys[k] not in cfg["data"]:
                cfg["data"][_data_keys[k]] = v
            elif k in _model_keys and _model_keys[k] not in cfg["model"]:
                cfg["model"][_model_keys[k]] = v
            elif k in _output_keys and _output_keys[k] not in cfg["output"]:
                cfg["output"][_output_keys[k]] = v

    clean = {k: v for k, v in cfg.items() if v}
    return _deep_merge(DEFAULT_CONFIG, clean)


def _resolve_file_path(path_str: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """Resolves a file path, checking base directory, CWD, and repo root."""
    path = Path(path_str)
    if path.is_file():
        return path.resolve()
    if base_dir and (base_dir / path).is_file():
        return (base_dir / path).resolve()
    repo_root = Path(__file__).resolve().parents[3]
    if (repo_root / path).is_file():
        return (repo_root / path).resolve()
    parent_fallback = Path("..") / path
    if parent_fallback.is_file():
        return parent_fallback.resolve()
    return path


# Data balancing
def _create_balanced_dataset(
    X: np.ndarray,
    y: np.ndarray,
    samples_per_class: int = 350,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Balances a binary dataset using random sampling and photometric augmentation.

    Augmentation strategies (applied randomly):
      1. Gaussian noise addition.
      2. Uniform amplitude scaling.
      3. Temporal phase shift (circular roll).
      4. Combined amplitude scaling and Gaussian noise.
    """
    rng = np.random.default_rng(random_state)
    X0, X1 = X[y == 0], X[y == 1]

    if len(X0) == 0 or len(X1) == 0:
        raise ValueError("Both classes (0 and 1) must be present in the dataset.")

    logger.info("Balancing dataset (original Class 0: %d, Class 1: %d)", len(X0), len(X1))

    def _augment(X_orig: np.ndarray, n_target: int) -> np.ndarray:
        if len(X_orig) >= n_target:
            return X_orig[rng.choice(len(X_orig), n_target, replace=False)]
        parts = [X_orig]
        current = len(X_orig)
        while current < n_target:
            n_need = n_target - current
            idx = rng.choice(len(X_orig), min(len(X_orig), n_need), replace=False)
            t = rng.random()
            if t < 0.25:
                aug = X_orig[idx] + rng.normal(0.0, 0.01, (len(idx), X_orig.shape[1]))
            elif t < 0.50:
                scale = 1.0 + rng.uniform(-0.03, 0.03, (len(idx), 1))
                aug = X_orig[idx] * scale
            elif t < 0.75:
                shifts = rng.integers(-20, 20, size=len(idx))
                aug = np.array([np.roll(X_orig[i], s) for i, s in zip(idx, shifts)])
            else:
                scale = 1.0 + rng.uniform(-0.02, 0.02, (len(idx), 1))
                aug = X_orig[idx] * scale + rng.normal(0.0, 0.008, (len(idx), X_orig.shape[1]))
            parts.append(aug)
            current += len(aug)
        return np.vstack(parts)[:n_target]

    X0_b = _augment(X0, samples_per_class)
    X1_b = _augment(X1, samples_per_class)
    X_bal = np.vstack([X0_b, X1_b])
    y_bal = np.concatenate([np.zeros(samples_per_class), np.ones(samples_per_class)])
    perm = rng.permutation(len(X_bal))
    logger.info("Balanced dataset: %d samples per class (Total: %d)", samples_per_class, len(X_bal))
    return X_bal[perm], y_bal[perm]


# 1-D CNN architecture
class _Conv1DBlock(nn.Module):
    """Single Conv1D block: Conv -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)


class _LightCurveCNN(nn.Module):
    """1-D Convolutional Neural Network for binary light-curve classification.

    Architecture
    ------------
    Input  : (batch, 1, n_bins)   — single-channel 1-D signal
    Blocks : ``len(conv_channels)`` Conv1DBlocks (Conv -> BN -> ReLU -> MaxPool)
    Head   : Global Average Pool -> Flatten -> FC layers with Dropout -> sigmoid output
    """

    def __init__(
        self,
        input_length: int,
        conv_channels: List[int],
        kernel_size: int,
        pool_size: int,
        fc_units: List[int],
        dropout: float,
    ):
        super().__init__()

        # Stack Conv1D blocks; each block doubles the channel depth and halves the time dimension
        blocks: List[nn.Module] = []
        in_ch = 1  # single-channel input (one flux time-series)
        for out_ch in conv_channels:
            blocks.append(_Conv1DBlock(in_ch, out_ch, kernel_size, pool_size))
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*blocks)

        # Infer the size of the flattened tensor after all conv/pool layers
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            feat = self.conv_backbone(dummy)
            flat_dim = feat.shape[1] * feat.shape[2]

        # Classifier head: flatten -> alternating Linear/ReLU/Dropout layers -> single logit
        fc_layers: List[nn.Module] = [nn.Flatten()]
        prev = flat_dim
        for units in fc_units:
            fc_layers.extend([
                nn.Linear(prev, units),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = units
        fc_layers.append(nn.Linear(prev, 1))  # raw logit; sigmoid is applied at inference time
        self.fc_head = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logit (pre-sigmoid) scores."""
        # Output shape is (batch, 1) -> squeeze(1) removes that trailing size-1 dimension -> (batch,)
        return self.fc_head(self.conv_backbone(x)).squeeze(1)


# Main detection class
class CNNDetection:
    """CNN detection and classification class for exoplanet transit light curves.

    Parameters
    ----------
    params : dict, str, Path, or Parameters, optional
        Configuration dictionary, path to a YAML configuration file, or a
        :class:`~daneel.parameters.Parameters` instance.
    **kwargs : Any
        Additional keyword arguments that override configuration values.
    """

    def __init__(
        self,
        params: Optional[Union[Dict[str, Any], str, Path, Parameters]] = None,
        **kwargs: Any,
    ):
        self.config_path: Optional[Path] = None
        user_config: Dict[str, Any] = {}

        if isinstance(params, (str, Path)):
            self.config_path = Path(params)
            if not self.config_path.is_file():
                resolved = _resolve_file_path(self.config_path)
                if resolved.is_file():
                    self.config_path = resolved
                else:
                    raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            with open(self.config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
        elif isinstance(params, Parameters):
            user_config = copy.deepcopy(params.params)
        elif isinstance(params, dict):
            user_config = copy.deepcopy(params)

        if kwargs:
            user_config = _deep_merge(user_config, kwargs)

        self.config = _normalize_user_config(user_config)
        self.data_config = self.config["data"]
        self.model_config = self.config["model"]
        self.output_config = self.config["output"]

        log_level = self.config.get("logging", {}).get("level", "INFO")
        setup_logging(log_level)

        # Set global seeds for reproducibility
        _seed = self.model_config.get("random_state", 42)
        torch.manual_seed(_seed)
        np.random.seed(_seed)

        # Select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        # Data split holders
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.metadata_test: Optional[pd.DataFrame] = None
        self.X_err_test: Optional[np.ndarray] = None

        # Model & evaluation holders
        self.model: Optional[_LightCurveCNN] = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        self.optimal_threshold: float = 0.5
        self.metrics: Dict[str, Any] = {}
        self.report_text: str = ""
        self.roc_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    # Data loading
    def load_data(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        n_bins: Optional[int] = None,
        balanced: Optional[bool] = None,
        test_size: Optional[float] = None,
        samples_per_class: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
        """Loads dataset from CSV, performs a stratified split, and optionally balances.

        Returns
        -------
        X_train, X_test, y_train, y_test, metadata_test, X_err_test
        """
        csv_p = csv_path or self.data_config.get("csv_path", "tess_data.csv")
        base_dir = self.config_path.parent if self.config_path else None
        resolved_csv = _resolve_file_path(csv_p, base_dir=base_dir)

        if not resolved_csv.is_file():
            raise FileNotFoundError(f"Dataset CSV file not found: {resolved_csv}")

        n_bins_val = n_bins if n_bins is not None else self.data_config.get("n_bins", 1000)
        balanced_val = balanced if balanced is not None else self.data_config.get("balanced", True)
        test_size_val = test_size if test_size is not None else self.data_config.get("test_size", 0.2)
        samples_val = (
            samples_per_class
            if samples_per_class is not None
            else self.data_config.get("samples_per_class", 350)
        )
        rs_val = random_state if random_state is not None else self.data_config.get("random_state", 42)

        logger.info("Loading dataset from: %s", resolved_csv)
        df = pd.read_csv(resolved_csv)
        logger.info("Dataset shape: %s", df.shape)

        flux_cols = [f"flux_{i:04d}" for i in range(n_bins_val)]
        flux_err_cols = [f"flux_err_{i:04d}" for i in range(n_bins_val)]
        required_cols = flux_cols + ["label"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns (first 5: {missing[:5]})")

        X = df[flux_cols].values
        X_err = (
            df[flux_err_cols].values
            if set(flux_err_cols).issubset(df.columns)
            else np.zeros_like(X)
        )
        y = df["label"].values.astype(int)

        metadata_cols = [
            c for c in ["toi_name", "tic", "label", "disp", "period_d", "t0_bjd", "dur_hr", "sector"]
            if c in df.columns
        ]
        metadata = df[metadata_cols] if metadata_cols else pd.DataFrame(index=df.index)

        n_cls0, n_cls1 = int((y == 0).sum()), int((y == 1).sum())
        if n_cls0 < 2 or n_cls1 < 2:
            raise ValueError("Insufficient samples per class for stratified splitting.")

        X_train, X_test, y_train, y_test, _, X_err_test, _, idx_test = train_test_split(
            X, y, X_err, np.arange(len(y)),
            test_size=test_size_val,
            random_state=rs_val,
            stratify=y,
        )

        logger.info(
            "Stratified split -> Train: %d, Test: %d (test_size=%.2f, random_state=%d)",
            len(X_train), len(X_test), test_size_val, rs_val,
        )

        if balanced_val:
            X_train, y_train = _create_balanced_dataset(
                X_train, y_train,
                samples_per_class=samples_val,
                random_state=rs_val,
            )

        metadata_test = metadata.iloc[idx_test].reset_index(drop=True)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.metadata_test = metadata_test
        self.X_err_test = X_err_test

        return X_train, X_test, y_train, y_test, metadata_test, X_err_test

    # Model construction
    def build_model(self, input_length: Optional[int] = None) -> _LightCurveCNN:
        """Instantiates the 1-D CNN architecture from configuration.

        Args:
            input_length: Length of the 1-D input signal. If ``None``, it is inferred
                          from ``self.X_train`` or ``self.data_config['n_bins']``.

        Returns:
            The constructed :class:`_LightCurveCNN` moved to the active device.
        """
        if input_length is None:
            if self.X_train is not None:
                input_length = self.X_train.shape[1]
            else:
                input_length = self.data_config.get("n_bins", 1000)

        conv_channels = self.model_config.get("conv_channels", [16, 32, 64])
        kernel_size = self.model_config.get("kernel_size", 7)
        pool_size = self.model_config.get("pool_size", 2)
        fc_units = self.model_config.get("fc_units", [128, 64])
        dropout = self.model_config.get("dropout", 0.3)

        self.model = _LightCurveCNN(
            input_length=input_length,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            pool_size=pool_size,
            fc_units=fc_units,
            dropout=dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "CNN built | conv_channels=%s | kernel=%d | pool=%d | fc_units=%s | "
            "dropout=%.2f | trainable params: %d",
            conv_channels, kernel_size, pool_size, fc_units, dropout, n_params,
        )
        return self.model


    def fit(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> "CNNDetection":
        """Trains the CNN on (optionally) provided or previously loaded training data.

        Args:
            X: Feature matrix (n_samples, n_bins). Falls back to ``self.X_train``.
            y: Target labels (n_samples,). Falls back to ``self.y_train``.

        Returns:
            ``self`` for method chaining.
        """
        if X is None or y is None:
            if self.X_train is None or self.y_train is None:
                self.load_data()
            X = self.X_train
            y = self.y_train

        if self.model is None:
            self.build_model(input_length=X.shape[1])

        batch_size = self.model_config.get("batch_size", 32)
        epochs = self.model_config.get("epochs", 30)
        lr = self.model_config.get("learning_rate", 1e-3)
        wd = self.model_config.get("weight_decay", 1e-4)

        # Compute positive-class weight to handle any residual imbalance
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        pos_weight_val = float(n_neg / max(n_pos, 1))
        pos_weight = torch.tensor([pos_weight_val], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Build DataLoader — reshape X to (N, 1, L) for Conv1d
        X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)
        y_t = torch.from_numpy(y.astype(np.float32))
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        logger.info(
            "Training CNN | epochs=%d | batch_size=%d | lr=%g | weight_decay=%g | "
            "pos_weight=%.4f",
            epochs, batch_size, lr, wd, pos_weight_val,
        )

        self.train_losses = []
        self.train_accuracies = []

        self.model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(y_batch)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == y_batch.long()).sum().item()
                total += len(y_batch)

            scheduler.step()

            avg_loss = epoch_loss / max(total, 1)
            avg_acc = correct / max(total, 1)
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(avg_acc)

            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                logger.info(
                    "Epoch [%3d/%3d]  loss: %.4f  accuracy: %.2f%%",
                    epoch, epochs, avg_loss, avg_acc * 100,
                )

        logger.info("Training complete.")

        # Save model weights if requested
        model_path = self.output_config.get("model_path")
        if model_path:
            mp = Path(model_path)
            mp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), mp)
            logger.info("Model weights saved to: %s", mp)

        return self

    # Inference helpers
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted positive-class probabilities for *X*.

        Args:
            X: Feature matrix of shape (n_samples, n_bins).

        Returns:
            1-D array of probabilities in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        self.model.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(1).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            proba = torch.sigmoid(logits).cpu().numpy()
        return proba

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Returns binary class predictions for *X*.

        Args:
            X: Feature matrix of shape (n_samples, n_bins).
            threshold: Decision threshold (defaults to ``self.optimal_threshold``).

        Returns:
            Integer array of class labels (0 or 1).
        """
        thresh = threshold if threshold is not None else self.optimal_threshold
        return (self.predict_proba(X) >= thresh).astype(int)


    def evaluate(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Evaluates the trained CNN on test data.

        The optimal decision threshold is selected via Youden's J statistic
        (maximum TPR - FPR on the ROC curve).

        Args:
            X: Test features. Falls back to ``self.X_test``.
            y: Test labels.  Falls back to ``self.y_test``.

        Returns:
            Dictionary of evaluation metrics.
        """
        if X is None or y is None:
            if self.X_test is None or self.y_test is None:
                self.load_data()
            X = self.X_test
            y = self.y_test

        if self.model is None:
            self.fit()

        proba = self.predict_proba(X)
        fpr, tpr, thresholds = roc_curve(y, proba)

        # Youden's J statistic for optimal threshold
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_thresh = float(np.clip(thresholds[best_idx], 0.0, 1.0))
        self.optimal_threshold = best_thresh
        self.roc_data = (fpr, tpr, thresholds)

        y_pred_default = (proba >= 0.5).astype(int)
        y_pred_best = (proba >= best_thresh).astype(int)

        auc_score = float(roc_auc_score(y, proba))
        acc_default = float(accuracy_score(y, y_pred_default))
        acc_best = float(accuracy_score(y, y_pred_best))

        prec_planet = float(precision_score(y, y_pred_best, pos_label=1, zero_division=0))
        prec_non_planet = float(precision_score(y, y_pred_best, pos_label=0, zero_division=0))
        prec_macro = float(precision_score(y, y_pred_best, average="macro", zero_division=0))
        prec_weighted = float(precision_score(y, y_pred_best, average="weighted", zero_division=0))

        rec_planet = float(recall_score(y, y_pred_best, pos_label=1, zero_division=0))
        rec_non_planet = float(recall_score(y, y_pred_best, pos_label=0, zero_division=0))
        f1_planet = float(f1_score(y, y_pred_best, pos_label=1, zero_division=0))
        f1_non_planet = float(f1_score(y, y_pred_best, pos_label=0, zero_division=0))

        cm = confusion_matrix(y, y_pred_best)
        self.report_text = classification_report(
            y, y_pred_best,
            target_names=["Non-Planet", "Planet"],
            digits=4,
            zero_division=0,
        )

        logger.info(
            "Evaluation complete -> Optimal threshold: %.4f | AUC: %.4f | Accuracy: %.2f%%",
            best_thresh, auc_score, acc_best * 100,
        )

        self.metrics = {
            "auc_roc": auc_score,
            "optimal_threshold": best_thresh,
            "default_threshold": 0.5,
            "accuracy_default": acc_default,
            "accuracy_optimal": acc_best,
            "precision_planet": prec_planet,
            "precision_non_planet": prec_non_planet,
            "precision_macro": prec_macro,
            "precision_weighted": prec_weighted,
            "recall_planet": rec_planet,
            "recall_non_planet": rec_non_planet,
            "f1_planet": f1_planet,
            "f1_non_planet": f1_non_planet,
            "tpr_optimal": float(tpr[best_idx]),
            "fpr_optimal": float(fpr[best_idx]),
            "confusion_matrix": cm,
            # Training curves
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
        }

        return self.metrics

    # Display & save
    def display_statistics(self) -> None:
        """Prints formatted evaluation statistics to stdout via the logger."""
        if not self.metrics:
            self.evaluate()

        cm = self.metrics.get("confusion_matrix", np.array([[0, 0], [0, 0]]))
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        logger.info("=" * 80)
        logger.info("CNN EXOPLANET DETECTION STATISTICS")
        logger.info("=" * 80)
        logger.info("MODEL HYPERPARAMETERS:")
        logger.info("-" * 80)
        for k, v in self.model_config.items():
            logger.info("  %-22s: %s", k, v)
        logger.info("EVALUATION METRICS & PRECISION:")
        logger.info("-" * 80)
        logger.info(
            "  Precision (Planet / Positive Class)     : %.4f (%.2f%%)",
            self.metrics.get("precision_planet", 0.0),
            self.metrics.get("precision_planet", 0.0) * 100,
        )
        logger.info(
            "  Precision (Non-Planet / Negative Class) : %.4f (%.2f%%)",
            self.metrics.get("precision_non_planet", 0.0),
            self.metrics.get("precision_non_planet", 0.0) * 100,
        )
        logger.info(
            "  Precision (Macro Average)               : %.4f (%.2f%%)",
            self.metrics.get("precision_macro", 0.0),
            self.metrics.get("precision_macro", 0.0) * 100,
        )
        logger.info(
            "  Precision (Weighted Average)            : %.4f (%.2f%%)",
            self.metrics.get("precision_weighted", 0.0),
            self.metrics.get("precision_weighted", 0.0) * 100,
        )
        logger.info(
            "  Accuracy (at Optimal Threshold)         : %.2f%%",
            self.metrics.get("accuracy_optimal", 0.0) * 100,
        )
        logger.info(
            "  Accuracy (at Default Threshold 0.5000)  : %.2f%%",
            self.metrics.get("accuracy_default", 0.0) * 100,
        )
        logger.info(
            "  Optimal Decision Threshold (Youden's J) : %.4f (Default: 0.5000)",
            self.metrics.get("optimal_threshold", 0.5),
        )
        logger.info("  ROC-AUC Score                           : %.4f", self.metrics.get("auc_roc", 0.0))
        logger.info(
            "  Recall (Planet)                         : %.4f (%.2f%%)",
            self.metrics.get("recall_planet", 0.0),
            self.metrics.get("recall_planet", 0.0) * 100,
        )
        logger.info(
            "  Recall (Non-Planet)                     : %.4f (%.2f%%)",
            self.metrics.get("recall_non_planet", 0.0),
            self.metrics.get("recall_non_planet", 0.0) * 100,
        )
        logger.info("  F1-Score (Planet)                       : %.4f", self.metrics.get("f1_planet", 0.0))
        logger.info("  F1-Score (Non-Planet)                   : %.4f", self.metrics.get("f1_non_planet", 0.0))

        # Training-curve summary
        if self.train_losses:
            logger.info("TRAINING CURVE SUMMARY:")
            logger.info("-" * 80)
            logger.info(
                "  Epochs completed    : %d", len(self.train_losses),
            )
            logger.info(
                "  Final train loss    : %.4f", self.train_losses[-1],
            )
            logger.info(
                "  Final train acc     : %.2f%%", self.train_accuracies[-1] * 100 if self.train_accuracies else 0.0,
            )

        logger.info("CONFUSION MATRIX (Evaluated at Optimal Threshold):")
        logger.info("-" * 80)
        logger.info("                       Predicted Non-Planet    Predicted Planet")
        logger.info("  Actual Non-Planet :        %-18d    %-18d", tn, fp)
        logger.info("  Actual Planet     :        %-18d    %-18d", fn, tp)
        logger.info("DETAILED CLASSIFICATION REPORT (Optimal Threshold):")
        logger.info("-" * 80)
        for line in self.report_text.rstrip().split("\n"):
            logger.info("  %s", line)
        logger.info("=" * 80)

    def save_report(self, report_path: Optional[Union[str, Path]] = None) -> Path:
        """Saves a text evaluation report to *report_path*.

        Args:
            report_path: Destination file path. Falls back to ``output.report_path``
                         from the configuration.

        Returns:
            Resolved :class:`~pathlib.Path` of the written report file.
        """
        if not self.metrics:
            self.evaluate()

        dest_str = report_path or self.output_config.get("report_path", "cnn/results/report.txt")
        dest_path = Path(dest_str)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        cm = self.metrics.get("confusion_matrix", np.array([[0, 0], [0, 0]]))
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        lines = [
            "=" * 80,
            "CNN EVALUATION REPORT",
            "=" * 80,
            "",
            "MODEL HYPERPARAMETERS:",
            "-" * 80,
        ]
        for k, v in self.model_config.items():
            lines.append(f"  {k:<22}: {v}")
        lines.extend([
            "",
            "EVALUATION METRICS & PRECISION:",
            "-" * 80,
            f"  Precision (Planet / Positive Class)     : {self.metrics.get('precision_planet', 0.0):.4f} ({self.metrics.get('precision_planet', 0.0) * 100:.2f}%)",
            f"  Precision (Non-Planet / Negative Class) : {self.metrics.get('precision_non_planet', 0.0):.4f} ({self.metrics.get('precision_non_planet', 0.0) * 100:.2f}%)",
            f"  Precision (Macro Average)               : {self.metrics.get('precision_macro', 0.0):.4f} ({self.metrics.get('precision_macro', 0.0) * 100:.2f}%)",
            f"  Precision (Weighted Average)            : {self.metrics.get('precision_weighted', 0.0):.4f} ({self.metrics.get('precision_weighted', 0.0) * 100:.2f}%)",
            f"  Accuracy (at Optimal Threshold)         : {self.metrics.get('accuracy_optimal', 0.0) * 100:.2f}%",
            f"  Accuracy (at Default Threshold 0.5000)  : {self.metrics.get('accuracy_default', 0.0) * 100:.2f}%",
            f"  Optimal Decision Threshold (Youden's J) : {self.metrics.get('optimal_threshold', 0.5):.4f} (Default: 0.5000)",
            f"  ROC-AUC Score                           : {self.metrics.get('auc_roc', 0.0):.4f}",
            f"  Recall (Planet)                         : {self.metrics.get('recall_planet', 0.0):.4f} ({self.metrics.get('recall_planet', 0.0) * 100:.2f}%)",
            f"  Recall (Non-Planet)                     : {self.metrics.get('recall_non_planet', 0.0):.4f} ({self.metrics.get('recall_non_planet', 0.0) * 100:.2f}%)",
            f"  F1-Score (Planet)                       : {self.metrics.get('f1_planet', 0.0):.4f}",
            f"  F1-Score (Non-Planet)                   : {self.metrics.get('f1_non_planet', 0.0):.4f}",
            "",
        ])

        if self.train_losses:
            lines.extend([
                "TRAINING CURVE SUMMARY:",
                "-" * 80,
                f"  Epochs completed    : {len(self.train_losses)}",
                f"  Final train loss    : {self.train_losses[-1]:.4f}",
                f"  Final train acc     : {self.train_accuracies[-1] * 100 if self.train_accuracies else 0.0:.2f}%",
                "",
            ])

        lines.extend([
            "CONFUSION MATRIX (Evaluated at Optimal Threshold):",
            "-" * 80,
            "                       Predicted Non-Planet    Predicted Planet",
            f"  Actual Non-Planet :        {tn:<18}    {fp:<18}",
            f"  Actual Planet     :        {fn:<18}    {tp:<18}",
            "",
            "DETAILED CLASSIFICATION REPORT (Optimal Threshold):",
            "-" * 80,
            self.report_text.rstrip(),
            "=" * 80,
        ])

        dest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("CNN evaluation report saved to: %s", dest_path)
        return dest_path

    # Pipeline runner
    def run(
        self,
        output_file: Optional[Union[str, Path]] = None,
        display: bool = True,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Runs the complete CNN detection pipeline.

        Steps:
          1. :meth:`load_data`
          2. :meth:`build_model`
          3. :meth:`fit`
          4. :meth:`evaluate`
          5. :meth:`display_statistics` (if *display*)
          6. :meth:`save_report`         (if *save*)

        Args:
            output_file: Override path for the saved report.
            display: Whether to print statistics to stdout.
            save: Whether to save the report file.

        Returns:
            Dictionary of evaluation metrics (same as :meth:`evaluate`).
        """
        self.load_data()
        self.build_model()
        self.fit()
        self.evaluate()

        if display:
            self.display_statistics()

        if save:
            target = output_file or self.output_config.get("report_path")
            if target:
                self.save_report(target)

        return self.metrics


# Aliases
CNNClassifier = CNNDetection
CNN = CNNDetection
