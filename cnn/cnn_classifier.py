"""PyTorch 1D CNN detection pipeline for exoplanet transits (Task F).

Provides :class:`CNNDetection`, inheriting from
:class:`~daneel.detection.base.BaseTransitDetector` so that data loading,
stratified splitting, balanced augmentation, Youden's J threshold optimization,
evaluation metrics, logging, and report generation are completely unified with
the Random Forest (Task D) and Decision Tree (Task E) detectors.

Supports both Task F YAML schemas (with dedicated ``training:`` section,
``StandardScaler``, ``FocalLossWithLogits`` / ``BCEWithLogitsLoss``, warmup
``min_epochs``, early stopping on validation ROC-AUC, and ``ReduceLROnPlateau``)
and the compact ``config_default.yaml`` schema (``conv_channels``, ``kernel_size``,
``pool_size``, ``fc_units``, ``dropout``, ``weight_decay``).
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from daneel.parameters import Parameters
except ImportError:
    from src.daneel.parameters import Parameters  # pragma: no cover

from daneel.detection.base import BaseTransitDetector

logger = logging.getLogger(__name__)


# Centralized default configuration matching BaseTransitDetector schema
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
        "name": "Config 1: Default 1D CNN",
        "standardize": True,
        # Architecture
        "conv_channels": [64, 128, 256],
        "kernel_sizes": [3, 3, 3],
        "conv_dropouts": [0.3, 0.3, 0.4],
        "pool_size": 2,
        "use_batchnorm": True,
        "use_global_avg_pool": True,
        "dense_dims": [512, 256, 128],
        "dense_dropouts": [0.2, 0.2, 0.2],
        "activation": "relu",
        "l2_reg": 0.001,
        # Training
        "loss_type": "focal",
        "focal_gamma": 2.5,
        "focal_alpha": 0.75,
        "learning_rate": 0.0005,
        "batch_size": 32,
        "epochs": 120,
        "min_epochs": 15,
        "val_split": 0.20,
        "patience": 20,
        "lr_patience": 8,
        "lr_factor": 0.5,
        "min_lr": 1e-7,
        "random_state": 42,
        "device": None,
    },
    "output": {
        "report_path": "cnn/results/report_cnn.txt",
        "model_path": None,
    },
    "logging": {
        "level": "INFO",
    },
}

MODEL_KEYS: Dict[str, str] = {
    "name": "name",
    "standardize": "standardize",
    "conv_channels": "conv_channels",
    "kernel_sizes": "kernel_sizes",
    "kernel_size": "kernel_sizes",
    "conv_dropouts": "conv_dropouts",
    "pool_size": "pool_size",
    "use_batchnorm": "use_batchnorm",
    "use_global_avg_pool": "use_global_avg_pool",
    "dense_dims": "dense_dims",
    "fc_units": "dense_dims",
    "dense_dropouts": "dense_dropouts",
    "dropout": "dense_dropouts",
    "activation": "activation",
    "l2_reg": "l2_reg",
    "weight_decay": "l2_reg",
    "loss_type": "loss_type",
    "focal_gamma": "focal_gamma",
    "focal_alpha": "focal_alpha",
    "learning_rate": "learning_rate",
    "lr": "learning_rate",
    "batch_size": "batch_size",
    "epochs": "epochs",
    "min_epochs": "min_epochs",
    "val_split": "val_split",
    "patience": "patience",
    "lr_patience": "lr_patience",
    "lr_factor": "lr_factor",
    "min_lr": "min_lr",
    "random_state": "random_state",
    "device": "device",
}


def set_deterministic_seed(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_activation(name: str) -> nn.Module:
    """Return a PyTorch activation layer by normalized name."""
    key = name.strip().lower()
    mapping: Dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported activation '{name}'. Choose from {sorted(mapping)}")
    if key == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    return mapping[key]()


def _broadcast_list(value: Any, length: int, cast: type) -> List[Any]:
    """Normalize a scalar or sequence into a list of exact ``length``."""
    if isinstance(value, (int, float)):
        return [cast(value)] * length
    seq = [cast(v) for v in value]
    if len(seq) == length:
        return seq
    if len(seq) == 1:
        return seq * length
    if len(seq) > length:
        return seq[:length]
    return seq + [seq[-1]] * (length - len(seq))


class FocalLossWithLogits(nn.Module):
    """Binary focal loss computed stably from raw model logits."""

    def __init__(self, gamma: float = 2.5, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        alpha_factor = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_factor * ((1.0 - pt) ** self.gamma)
        return (focal_weight * bce).mean()


class TransitCNN1D(nn.Module):
    """Configurable 1D Convolutional Neural Network for light-curve classification."""

    def __init__(self, n_bins: int, model_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.n_bins = int(n_bins)
        self.conv_channels: List[int] = [int(c) for c in model_cfg.get("conv_channels", [64, 128, 256])]
        n_conv = len(self.conv_channels)
        self.kernel_sizes: List[int] = _broadcast_list(model_cfg.get("kernel_sizes", [3] * n_conv), n_conv, int)
        self.conv_dropouts: List[float] = _broadcast_list(model_cfg.get("conv_dropouts", [0.3] * n_conv), n_conv, float)
        self.pool_size: int = int(model_cfg.get("pool_size", 2))
        self.use_batchnorm: bool = bool(model_cfg.get("use_batchnorm", True))
        self.use_global_avg_pool: bool = bool(model_cfg.get("use_global_avg_pool", True))
        self.dense_dims: List[int] = [int(d) for d in model_cfg.get("dense_dims", [512, 256, 128])]
        self.dense_dropouts: List[float] = _broadcast_list(
            model_cfg.get("dense_dropouts", [0.2] * len(self.dense_dims)),
            len(self.dense_dims),
            float,
        )
        self.activation: str = str(model_cfg.get("activation", "relu"))
        self.l2_reg: float = float(model_cfg.get("l2_reg", 0.001))

        conv_layers: List[nn.Module] = []
        in_channels = 1
        current_length = self.n_bins

        for idx, (out_ch, k_size, drop_p) in enumerate(
            zip(self.conv_channels, self.kernel_sizes, self.conv_dropouts)
        ):
            padding = k_size // 2
            conv_layers.append(
                nn.Conv1d(in_channels, out_ch, kernel_size=k_size, padding=padding)
            )
            conv_layers.append(_make_activation(self.activation))
            if self.use_batchnorm:
                # Match Keras BatchNormalization defaults (eps=1e-3, momentum=0.99 -> 0.01 in PyTorch)
                conv_layers.append(nn.BatchNorm1d(out_ch, eps=1e-3, momentum=0.01))

            is_last_conv = idx == n_conv - 1
            if is_last_conv and self.use_global_avg_pool:
                conv_layers.append(nn.AdaptiveAvgPool1d(1))
                current_length = 1
            else:
                conv_layers.append(nn.MaxPool1d(kernel_size=self.pool_size))
                current_length = current_length // self.pool_size

            if drop_p > 0.0:
                conv_layers.append(nn.Dropout(p=drop_p))
            in_channels = out_ch

        self.features = nn.Sequential(*conv_layers)

        flat_dim = in_channels * current_length
        classifier_layers: List[nn.Module] = [nn.Flatten()]
        self.dense_linears: List[nn.Linear] = []
        in_features = flat_dim

        for hidden_dim, drop_p in zip(self.dense_dims, self.dense_dropouts):
            linear = nn.Linear(in_features, hidden_dim)
            self.dense_linears.append(linear)
            classifier_layers.append(linear)
            classifier_layers.append(_make_activation(self.activation))
            if drop_p > 0.0:
                classifier_layers.append(nn.Dropout(p=drop_p))
            in_features = hidden_dim

        self.output_linear = nn.Linear(in_features, 1)
        classifier_layers.append(self.output_linear)
        self.classifier = nn.Sequential(*classifier_layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Glorot/Xavier uniform (matching Keras defaults)."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x).squeeze(-1)

    def count_parameters(self) -> Tuple[int, int]:
        """Return ``(total_params, trainable_params)``."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return int(total), int(trainable)

    def describe_architecture(self) -> List[str]:
        """Return a concise layer-by-layer summary with parameter counts."""
        total_params, trainable_params = self.count_parameters()
        lines = [f"  Input           : (batch, 1, {self.n_bins})"]
        in_ch = 1
        seq_len = self.n_bins
        n_conv = len(self.conv_channels)

        for idx, (out_ch, k_size, drop_p) in enumerate(
            zip(self.conv_channels, self.kernel_sizes, self.conv_dropouts)
        ):
            is_last = idx == n_conv - 1
            pool_str = (
                "GlobalAvgPool1d"
                if (is_last and self.use_global_avg_pool)
                else f"MaxPool1d({self.pool_size})"
            )
            bn_str = f"BatchNorm1d({out_ch}) -> " if self.use_batchnorm else ""
            next_len = 1 if (is_last and self.use_global_avg_pool) else (seq_len // self.pool_size)
            lines.append(
                f"  Conv Block {idx + 1}    : Conv1d({in_ch}->{out_ch}, k={k_size}, pad={k_size // 2}, "
                f"{self.activation}) -> {bn_str}{pool_str} -> Dropout({drop_p:.2f}) "
                f"[out: ({out_ch}, {next_len})]"
            )
            in_ch = out_ch
            seq_len = next_len

        flat_dim = in_ch * seq_len
        lines.append(f"  Flatten         : ({flat_dim},)")
        prev_dim = flat_dim
        for idx, (h_dim, drop_p) in enumerate(zip(self.dense_dims, self.dense_dropouts)):
            lines.append(
                f"  Dense Block {idx + 1}   : Linear({prev_dim}->{h_dim}, {self.activation}, "
                f"L2={self.l2_reg}) -> Dropout({drop_p:.2f})"
            )
            prev_dim = h_dim

        lines.append(f"  Output Head     : Linear({prev_dim}->1) + Sigmoid")
        lines.append(
            f"  Parameters      : {trainable_params:,} trainable / {total_params:,} total"
        )
        return lines


class CNNDetection(BaseTransitDetector):
    """1-D Convolutional Neural Network detector for exoplanet transit light curves.

    Parameters
    ----------
    params : dict, str, Path, or Parameters, optional
        Configuration dictionary, path to YAML configuration file, or Parameters instance.
    **kwargs : Any
        Additional keyword arguments overriding configuration values.
    """

    model_name: str = "1D CNN"
    DEFAULT_CONFIG: Dict[str, Any] = DEFAULT_CONFIG
    MODEL_KEYS: Dict[str, str] = MODEL_KEYS
    WRAPPER_KEYS: Sequence[str] = ("classifier", "cnn", "convnet", "detection")

    def __init__(
        self,
        params: Optional[Union[Dict[str, Any], str, Path, Parameters]] = None,
        **kwargs: Any,
    ):
        super().__init__(params, **kwargs)
        req_device = self.model_config.get("device")
        if req_device:
            self.device = torch.device(req_device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[TransitCNN1D] = None
        self.scaler: Optional[StandardScaler] = None
        self._fitted: bool = False

        # Training history & summary state
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_aucs: List[float] = []
        self.best_val_auc: float = 0.0
        self.best_epoch: int = 0
        self.epochs_run: int = 0
        self.n_train_fit: int = 0
        self.n_val_fit: int = 0

    def normalize_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes Task F (``training:``) and compact (``config_default.yaml``) schemas."""
        prepared = copy.deepcopy(user_config) if user_config else {}

        # Merge optional `training:` section into `model:`
        if isinstance(prepared.get("training"), dict):
            model_sec = prepared.setdefault("model", {})
            if isinstance(model_sec, dict):
                for k, v in prepared.pop("training").items():
                    model_sec.setdefault(k, v)

        # Propagate `data.standardize` to `model.standardize` if not explicitly set under model
        data_sec = prepared.get("data")
        if isinstance(data_sec, dict) and "standardize" in data_sec:
            model_sec = prepared.setdefault("model", {})
            if isinstance(model_sec, dict) and "standardize" not in model_sec:
                model_sec["standardize"] = bool(data_sec["standardize"])

        # Resolve synonym keys inside `model:` section (e.g. from config_default.yaml)
        if isinstance(prepared.get("model"), dict):
            m_sec = prepared["model"]
            if "kernel_size" in m_sec and "kernel_sizes" not in m_sec:
                m_sec["kernel_sizes"] = m_sec.pop("kernel_size")
            if "fc_units" in m_sec and "dense_dims" not in m_sec:
                m_sec["dense_dims"] = m_sec.pop("fc_units")
            if "dropout" in m_sec:
                drop_val = m_sec.pop("dropout")
                if "dense_dropouts" not in m_sec:
                    m_sec["dense_dropouts"] = drop_val
                if "conv_dropouts" not in m_sec:
                    m_sec["conv_dropouts"] = drop_val
            if "weight_decay" in m_sec and "l2_reg" not in m_sec:
                m_sec["l2_reg"] = m_sec.pop("weight_decay")
            if "lr" in m_sec and "learning_rate" not in m_sec:
                m_sec["learning_rate"] = m_sec.pop("lr")

        normalized = super().normalize_config(prepared)

        # Ensure list-valued architecture hyperparameters match layer counts
        m_cfg = normalized["model"]
        conv_channels = [int(c) for c in m_cfg.get("conv_channels", [64, 128, 256])]
        m_cfg["conv_channels"] = conv_channels
        m_cfg["kernel_sizes"] = _broadcast_list(
            m_cfg.get("kernel_sizes", 3), len(conv_channels), int
        )
        m_cfg["conv_dropouts"] = _broadcast_list(
            m_cfg.get("conv_dropouts", 0.3), len(conv_channels), float
        )
        dense_dims = [int(d) for d in m_cfg.get("dense_dims", [512, 256, 128])]
        m_cfg["dense_dims"] = dense_dims
        m_cfg["dense_dropouts"] = _broadcast_list(
            m_cfg.get("dense_dropouts", 0.2), len(dense_dims), float
        )

        # Clean up any leftover alias keys that could have been merged
        for alias in ("kernel_size", "fc_units", "dropout", "weight_decay", "lr"):
            m_cfg.pop(alias, None)

        return normalized

    def _is_fitted(self) -> bool:
        """Returns True if the CNN model has been constructed and trained."""
        return self.model is not None and self._fitted

    def build_model(self) -> TransitCNN1D:
        """Constructs ``TransitCNN1D`` and moves it to the active compute device."""
        n_bins = int(self.data_config.get("n_bins", 1000))
        model = TransitCNN1D(n_bins=n_bins, model_cfg=self.model_config).to(self.device)
        total_params, trainable_params = model.count_parameters()
        logger.info(
            "Built TransitCNN1D on %s | trainable_params=%s (total=%s)",
            self.device,
            f"{trainable_params:,}",
            f"{total_params:,}",
        )
        self.model = model
        self._fitted = False
        return model

    def _build_loss_fn(self) -> nn.Module:
        loss_key = str(self.model_config.get("loss_type", "focal")).strip().lower()
        if loss_key == "focal":
            return FocalLossWithLogits(
                gamma=float(self.model_config.get("focal_gamma", 2.5)),
                alpha=float(self.model_config.get("focal_alpha", 0.75)),
            )
        if loss_key == "bce":
            return nn.BCEWithLogitsLoss()
        raise ValueError(f"Unsupported loss_type '{loss_key}'. Use 'focal' or 'bce'.")

    def _build_optimizer(self, model: TransitCNN1D) -> torch.optim.Optimizer:
        l2 = float(self.model_config.get("l2_reg", 0.001))
        lr = float(self.model_config.get("learning_rate", 0.0005))
        dense_weight_ids = {id(layer.weight) for layer in model.dense_linears}
        dense_weights = [p for p in model.parameters() if id(p) in dense_weight_ids]
        other_params = [p for p in model.parameters() if id(p) not in dense_weight_ids]
        param_groups = [
            {"params": dense_weights, "weight_decay": l2},
            {"params": other_params, "weight_decay": 0.0},
        ]
        return torch.optim.Adam(param_groups, lr=lr)

    def fit(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> CNNDetection:
        """Trains the 1-D CNN with internal stratified validation and early stopping."""
        if X is None or y is None:
            if self.X_train is None or self.y_train is None:
                self.load_data()
            X = self.X_train
            y = self.y_train

        rs = int(self.model_config.get("random_state", 42))
        set_deterministic_seed(rs)

        # Standardize inside fit() so load_data() remains model-agnostic
        if self.model_config.get("standardize", True):
            self.scaler = StandardScaler()
            X_proc = self.scaler.fit_transform(X)
        else:
            self.scaler = None
            X_proc = np.asarray(X, dtype=np.float32)

        val_split = float(self.model_config.get("val_split", 0.20))
        batch_size = int(self.model_config.get("batch_size", 32))
        epochs = int(self.model_config.get("epochs", 120))
        min_epochs = int(self.model_config.get("min_epochs", 15))
        patience = int(self.model_config.get("patience", 20))
        lr_patience = int(self.model_config.get("lr_patience", 8))
        lr_factor = float(self.model_config.get("lr_factor", 0.5))
        min_lr = float(self.model_config.get("min_lr", 1e-7))

        # Carve internal stratified validation split so X_test remains unseen
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_proc,
            y,
            test_size=val_split,
            random_state=rs,
            stratify=y,
        )
        self.n_train_fit = int(len(X_tr))
        self.n_val_fit = int(len(X_val))

        train_ds = TensorDataset(
            torch.tensor(X_tr[:, None, :], dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        dl_gen = torch.Generator()
        dl_gen.manual_seed(rs)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=dl_gen,
        )

        # Always rebuild cleanly with deterministic seed before training
        model = self.build_model()
        loss_fn = self._build_loss_fn()
        optimizer = self._build_optimizer(model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
        )

        X_val_tensor = torch.tensor(X_val[:, None, :], dtype=torch.float32, device=self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        best_val_auc = -1.0
        best_val_loss = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        effective_min_epochs = min(min_epochs, epochs)

        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.epochs_run = 0

        logger.info(
            "Training 1D CNN (%s) | epochs=%d | batch_size=%d | lr=%g | loss=%s | standardize=%s",
            self.model_config.get("name"),
            epochs,
            batch_size,
            float(self.model_config.get("learning_rate", 0.0005)),
            self.model_config.get("loss_type", "focal"),
            self.model_config.get("standardize", True),
        )

        for epoch in range(1, epochs + 1):
            self.epochs_run = epoch
            model.train()
            epoch_loss = 0.0
            n_seen = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(batch_y)
                n_seen += len(batch_y)

            avg_train_loss = epoch_loss / max(n_seen, 1)
            self.train_losses.append(avg_train_loss)

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_loss = float(loss_fn(val_logits, y_val_tensor).item())
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = float(roc_auc_score(y_val, val_probs))
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)

            if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                logger.info(
                    "Epoch [%3d/%3d]  train_loss: %.4f  val_loss: %.4f  val_auc: %.4f",
                    epoch,
                    epochs,
                    avg_train_loss,
                    val_loss,
                    val_auc,
                )

            if epoch < effective_min_epochs:
                continue

            scheduler.step(val_auc)

            improved = (val_auc > best_val_auc + 1e-4) or (
                abs(val_auc - best_val_auc) <= 1e-4 and val_loss < best_val_loss
            )
            if improved:
                best_val_auc = val_auc
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping triggered at epoch %d (best epoch=%d, val_auc=%.4f).",
                        epoch,
                        best_epoch,
                        best_val_auc,
                    )
                    break

        model.load_state_dict(best_state)
        self.best_val_auc = best_val_auc
        self.best_epoch = best_epoch
        self._fitted = True

        # Save model weights if requested
        model_path = self.output_config.get("model_path")
        if model_path:
            mp = Path(model_path)
            mp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), mp)
            logger.info("Model weights saved to: %s", mp)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted positive-class probabilities in [0, 1] for ``X``."""
        if self.model is None or not self._fitted:
            raise RuntimeError("Model has not been trained. Call fit() before predict_proba().")

        X_proc = self.scaler.transform(X) if self.scaler is not None else np.asarray(X, dtype=np.float32)
        batch_size = int(self.model_config.get("batch_size", 32))

        self.model.eval()
        tensor_x = torch.tensor(X_proc[:, None, :], dtype=torch.float32, device=self.device)
        probs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(tensor_x), batch_size):
                logits = self.model(tensor_x[start : start + batch_size])
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs, axis=0)

    # Subclass hooks for architecture, training summary & default-threshold breakdown
    def hook_extra_metrics(self, metrics: Dict[str, Any]) -> None:
        """Attaches CNN architecture, validation/training history, and default-threshold metrics."""
        total_params, trainable_params = (
            self.model.count_parameters() if self.model is not None else (0, 0)
        )
        arch_lines = self.model.describe_architecture() if self.model is not None else []

        metrics["config_name"] = str(self.model_config.get("name", self.model_name))
        metrics["val_auc_best"] = float(self.best_val_auc)
        metrics["best_epoch"] = int(self.best_epoch)
        metrics["epochs_run"] = int(self.epochs_run)
        metrics["total_params"] = int(total_params)
        metrics["trainable_params"] = int(trainable_params)
        metrics["architecture_lines"] = arch_lines
        metrics["n_train"] = int(self.n_train_fit)
        metrics["n_val"] = int(self.n_val_fit)
        metrics["n_test"] = int(len(self.X_test)) if self.X_test is not None else 0
        metrics["train_losses"] = self.train_losses
        metrics["val_losses"] = self.val_losses
        metrics["val_aucs"] = self.val_aucs

        if self.X_test is not None and self.y_test is not None and self._fitted:
            proba = self.predict_proba(self.X_test)
            y_pred_def = (proba >= 0.5).astype(int)
            metrics["precision_planet_default"] = float(
                precision_score(self.y_test, y_pred_def, pos_label=1, zero_division=0)
            )
            metrics["precision_non_planet_default"] = float(
                precision_score(self.y_test, y_pred_def, pos_label=0, zero_division=0)
            )
            metrics["precision_macro_default"] = float(
                precision_score(self.y_test, y_pred_def, average="macro", zero_division=0)
            )
            metrics["recall_planet_default"] = float(
                recall_score(self.y_test, y_pred_def, pos_label=1, zero_division=0)
            )
            metrics["f1_planet_default"] = float(
                f1_score(self.y_test, y_pred_def, pos_label=1, zero_division=0)
            )
            metrics["confusion_matrix_default"] = confusion_matrix(self.y_test, y_pred_def)

    def hook_display_extra_metrics(self) -> None:
        """Displays CNN validation AUC, best epoch, and parameter count in console statistics."""
        logger.info(
            "  Best Validation ROC-AUC                 : %.4f",
            self.metrics.get("val_auc_best", 0.0),
        )
        logger.info(
            "  Best Epoch / Epochs Run                 : %d / %d",
            self.metrics.get("best_epoch", 0),
            self.metrics.get("epochs_run", 0),
        )
        logger.info(
            "  Trainable Parameters                    : %s",
            f"{self.metrics.get('trainable_params', 0):,}",
        )

    def hook_display_extra_sections(self) -> None:
        """Displays CNN architecture and training curve summary in console statistics."""
        arch_lines = self.metrics.get("architecture_lines", [])
        if arch_lines:
            logger.info("MODEL ARCHITECTURE:")
            logger.info("-" * 80)
            for line in arch_lines:
                logger.info("%s", line)
        if self.train_losses:
            logger.info("TRAINING CURVE SUMMARY:")
            logger.info("-" * 80)
            logger.info("  Framework           : PyTorch (%s) on %s", torch.__version__, self.device)
            logger.info("  Epochs completed    : %d (best epoch: %d)", self.epochs_run, self.best_epoch)
            logger.info("  Final train loss    : %.4f", self.train_losses[-1])
            logger.info("  Best val ROC-AUC    : %.4f", self.best_val_auc)

    def hook_report_extra_metrics(self, lines: List[str]) -> None:
        """Appends CNN validation AUC, best epoch, and parameter count to report metrics."""
        lines.append(
            f"  Best Validation ROC-AUC                 : {self.metrics.get('val_auc_best', 0.0):.4f}"
        )
        lines.append(
            f"  Best Epoch / Epochs Run                 : {self.metrics.get('best_epoch', 0)} / {self.metrics.get('epochs_run', 0)}"
        )
        lines.append(
            f"  Trainable Parameters                    : {self.metrics.get('trainable_params', 0):,}"
        )

    def hook_report_extra_sections(self, lines: List[str]) -> None:
        """Appends CNN architecture, training curve, and default-threshold sections to report."""
        arch_lines = self.metrics.get("architecture_lines", [])
        if arch_lines:
            lines.extend(["MODEL ARCHITECTURE:", "-" * 80, *arch_lines, ""])

        if self.train_losses:
            lines.extend([
                "TRAINING CURVE SUMMARY:",
                "-" * 80,
                f"  Framework           : PyTorch ({torch.__version__}) on {self.device}",
                f"  Split sizes         : train={self.n_train_fit} / val={self.n_val_fit} / test={self.metrics.get('n_test', 0)}",
                f"  Epochs completed    : {self.epochs_run} (best epoch: {self.best_epoch})",
                f"  Final train loss    : {self.train_losses[-1]:.4f}",
                f"  Best val ROC-AUC    : {self.best_val_auc:.4f}",
                "",
            ])

        if "precision_planet_default" in self.metrics:
            cm_def = self.metrics.get("confusion_matrix_default", np.array([[0, 0], [0, 0]]))
            tn_d, fp_d, fn_d, tp_d = cm_def.ravel() if cm_def.size == 4 else (0, 0, 0, 0)
            lines.extend([
                "DEFAULT THRESHOLD SUMMARY (p >= 0.5000):",
                "-" * 80,
                f"  Precision (Planet / Positive Class)     : {self.metrics['precision_planet_default']:.4f} ({self.metrics['precision_planet_default'] * 100:.2f}%)",
                f"  Precision (Non-Planet / Negative Class) : {self.metrics['precision_non_planet_default']:.4f} ({self.metrics['precision_non_planet_default'] * 100:.2f}%)",
                f"  Precision (Macro Average)               : {self.metrics['precision_macro_default']:.4f} ({self.metrics['precision_macro_default'] * 100:.2f}%)",
                f"  Recall (Planet)                         : {self.metrics['recall_planet_default']:.4f} ({self.metrics['recall_planet_default'] * 100:.2f}%)",
                f"  F1-Score (Planet)                       : {self.metrics['f1_planet_default']:.4f}",
                f"  Confusion Matrix (TN / FP / FN / TP)    : {tn_d} / {fp_d} / {fn_d} / {tp_d}",
                "",
            ])


# Aliases for backwards compatibility and intuitive imports
CNNTransitDetector = CNNDetection
CNNClassifier = CNNDetection
CNN = CNNDetection
