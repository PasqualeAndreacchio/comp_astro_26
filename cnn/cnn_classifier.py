"""PyTorch 1D CNN pipeline for TESS exoplanet transit detection (Task F).

Reproduces the default 1D Convolutional Neural Network architecture and training
methodology from ``transit_analysis_deep_learning.ipynb`` in PyTorch while
sharing ``daneel.detection.data_process.load_data`` for an exact apples-to-apples
evaluation split with Task D (Random Forest) and Task E (Decision Tree).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any

import numpy as np
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from daneel.detection.data_process import RANDOM_STATE, load_data


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
    mapping: dict[str, type[nn.Module]] = {
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


@dataclass
class DataConfig:
    """Dataset loading, balancing, and standardization configuration."""

    csv_path: str = "tess_data.csv"
    n_bins: int = 1000
    samples_per_class: int = 350
    standardize: bool = True
    random_state: int = RANDOM_STATE


@dataclass
class CNNModelConfig:
    """Architecture hyperparameters for ``TransitCNN1D``."""

    name: str = "Config 1: Default 1D CNN"
    conv_channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 3, 3])
    conv_dropouts: list[float] = field(default_factory=lambda: [0.3, 0.3, 0.4])
    pool_size: int = 2
    use_batchnorm: bool = True
    use_global_avg_pool: bool = True
    dense_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    dense_dropouts: list[float] = field(default_factory=lambda: [0.2, 0.2, 0.2])
    activation: str = "relu"
    l2_reg: float = 0.001


@dataclass
class TrainingConfig:
    """Optimization, loss, and early-stopping hyperparameters."""

    loss_type: str = "focal"
    focal_gamma: float = 2.5
    focal_alpha: float = 0.75
    learning_rate: float = 0.0005
    batch_size: int = 32
    epochs: int = 120
    min_epochs: int = 15
    val_split: float = 0.20
    patience: int = 20
    lr_patience: int = 8
    lr_factor: float = 0.5
    min_lr: float = 1e-7
    random_state: int = RANDOM_STATE


@dataclass
class CNNExperimentConfig:
    """Complete YAML-serializable configuration for a Task F CNN experiment."""

    data: DataConfig = field(default_factory=DataConfig)
    model: CNNModelConfig = field(default_factory=CNNModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    report_path: str = "cnn/results/report_cnn.txt"

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> CNNExperimentConfig:
        """Load a CNN experiment configuration from a YAML file."""
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}

        data_raw = raw.get("data", {})
        model_raw = raw.get("model", {})
        train_raw = raw.get("training", {})
        output_raw = raw.get("output", {})

        data_cfg = DataConfig(
            csv_path=str(data_raw.get("csv_path", "tess_data.csv")),
            n_bins=int(data_raw.get("n_bins", 1000)),
            samples_per_class=int(data_raw.get("samples_per_class", 350)),
            standardize=bool(data_raw.get("standardize", True)),
            random_state=int(data_raw.get("random_state", RANDOM_STATE)),
        )

        conv_channels = [int(c) for c in model_raw.get("conv_channels", [64, 128, 256])]
        raw_kernels = model_raw.get("kernel_sizes", model_raw.get("kernel_size", 3))
        if isinstance(raw_kernels, int):
            kernel_sizes = [int(raw_kernels)] * len(conv_channels)
        else:
            kernel_sizes = [int(k) for k in raw_kernels]

        raw_conv_drop = model_raw.get("conv_dropouts", [0.3, 0.3, 0.4])
        if isinstance(raw_conv_drop, (int, float)):
            conv_dropouts = [float(raw_conv_drop)] * len(conv_channels)
        else:
            conv_dropouts = [float(d) for d in raw_conv_drop]

        dense_dims = [int(d) for d in model_raw.get("dense_dims", [512, 256, 128])]
        raw_dense_drop = model_raw.get("dense_dropouts", [0.2] * len(dense_dims))
        if isinstance(raw_dense_drop, (int, float)):
            dense_dropouts = [float(raw_dense_drop)] * len(dense_dims)
        else:
            dense_dropouts = [float(d) for d in raw_dense_drop]

        model_cfg = CNNModelConfig(
            name=str(model_raw.get("name", path.stem)),
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            conv_dropouts=conv_dropouts,
            pool_size=int(model_raw.get("pool_size", 2)),
            use_batchnorm=bool(model_raw.get("use_batchnorm", True)),
            use_global_avg_pool=bool(model_raw.get("use_global_avg_pool", True)),
            dense_dims=dense_dims,
            dense_dropouts=dense_dropouts,
            activation=str(model_raw.get("activation", "relu")),
            l2_reg=float(model_raw.get("l2_reg", 0.001)),
        )

        train_cfg = TrainingConfig(
            loss_type=str(train_raw.get("loss_type", "focal")),
            focal_gamma=float(train_raw.get("focal_gamma", 2.5)),
            focal_alpha=float(train_raw.get("focal_alpha", 0.75)),
            learning_rate=float(train_raw.get("learning_rate", 0.0005)),
            batch_size=int(train_raw.get("batch_size", 32)),
            epochs=int(train_raw.get("epochs", 120)),
            min_epochs=int(train_raw.get("min_epochs", 15)),
            val_split=float(train_raw.get("val_split", 0.20)),
            patience=int(train_raw.get("patience", 20)),
            lr_patience=int(train_raw.get("lr_patience", 8)),
            lr_factor=float(train_raw.get("lr_factor", 0.5)),
            min_lr=float(train_raw.get("min_lr", 1e-7)),
            random_state=int(train_raw.get("random_state", RANDOM_STATE)),
        )

        report_path = str(
            output_raw.get("report_path", f"cnn/results/report_{path.stem}.txt")
        )
        return cls(
            data=data_cfg,
            model=model_cfg,
            training=train_cfg,
            report_path=report_path,
        )


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

    def __init__(self, n_bins: int, cfg: CNNModelConfig) -> None:
        super().__init__()
        if not (len(cfg.conv_channels) == len(cfg.kernel_sizes) == len(cfg.conv_dropouts)):
            raise ValueError(
                "conv_channels, kernel_sizes, and conv_dropouts must have equal length"
            )
        if len(cfg.dense_dims) != len(cfg.dense_dropouts):
            raise ValueError("dense_dims and dense_dropouts must have equal length")

        self.n_bins = int(n_bins)
        self.cfg = cfg

        conv_layers: list[nn.Module] = []
        in_channels = 1
        current_length = self.n_bins
        n_conv = len(cfg.conv_channels)

        for idx, (out_ch, k_size, drop_p) in enumerate(
            zip(cfg.conv_channels, cfg.kernel_sizes, cfg.conv_dropouts)
        ):
            padding = k_size // 2
            conv_layers.append(
                nn.Conv1d(in_channels, out_ch, kernel_size=k_size, padding=padding)
            )
            conv_layers.append(_make_activation(cfg.activation))
            if cfg.use_batchnorm:
                # Match Keras BatchNormalization defaults (eps=1e-3, momentum=0.99 -> 0.01 in PyTorch)
                conv_layers.append(nn.BatchNorm1d(out_ch, eps=1e-3, momentum=0.01))

            is_last_conv = idx == n_conv - 1
            if is_last_conv and cfg.use_global_avg_pool:
                conv_layers.append(nn.AdaptiveAvgPool1d(1))
                current_length = 1
            else:
                conv_layers.append(nn.MaxPool1d(kernel_size=cfg.pool_size))
                current_length = current_length // cfg.pool_size

            if drop_p > 0.0:
                conv_layers.append(nn.Dropout(p=drop_p))
            in_channels = out_ch

        self.features = nn.Sequential(*conv_layers)

        flat_dim = in_channels * current_length
        classifier_layers: list[nn.Module] = [nn.Flatten()]
        self.dense_linears: list[nn.Linear] = []
        in_features = flat_dim

        for hidden_dim, drop_p in zip(cfg.dense_dims, cfg.dense_dropouts):
            linear = nn.Linear(in_features, hidden_dim)
            self.dense_linears.append(linear)
            classifier_layers.append(linear)
            classifier_layers.append(_make_activation(cfg.activation))
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

    def count_parameters(self) -> tuple[int, int]:
        """Return ``(total_params, trainable_params)``."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return int(total), int(trainable)

    def describe_architecture(self) -> list[str]:
        """Return a concise layer-by-layer summary with parameter counts."""
        total_params, trainable_params = self.count_parameters()
        lines = [f"  Input           : (batch, 1, {self.n_bins})"]
        in_ch = 1
        seq_len = self.n_bins
        n_conv = len(self.cfg.conv_channels)

        for idx, (out_ch, k_size, drop_p) in enumerate(
            zip(self.cfg.conv_channels, self.cfg.kernel_sizes, self.cfg.conv_dropouts)
        ):
            is_last = idx == n_conv - 1
            pool_str = (
                "GlobalAvgPool1d"
                if (is_last and self.cfg.use_global_avg_pool)
                else f"MaxPool1d({self.cfg.pool_size})"
            )
            bn_str = f"BatchNorm1d({out_ch}) -> " if self.cfg.use_batchnorm else ""
            next_len = 1 if (is_last and self.cfg.use_global_avg_pool) else (seq_len // self.cfg.pool_size)
            lines.append(
                f"  Conv Block {idx + 1}    : Conv1d({in_ch}->{out_ch}, k={k_size}, pad={k_size // 2}, "
                f"{self.cfg.activation}) -> {bn_str}{pool_str} -> Dropout({drop_p:.2f}) "
                f"[out: ({out_ch}, {next_len})]"
            )
            in_ch = out_ch
            seq_len = next_len

        flat_dim = in_ch * seq_len
        lines.append(f"  Flatten         : ({flat_dim},)")
        prev_dim = flat_dim
        for idx, (h_dim, drop_p) in enumerate(
            zip(self.cfg.dense_dims, self.cfg.dense_dropouts)
        ):
            lines.append(
                f"  Dense Block {idx + 1}   : Linear({prev_dim}->{h_dim}, {self.cfg.activation}, "
                f"L2={self.cfg.l2_reg}) -> Dropout({drop_p:.2f})"
            )
            prev_dim = h_dim

        lines.append(f"  Output Head     : Linear({prev_dim}->1) + Sigmoid")
        lines.append(
            f"  Parameters      : {trainable_params:,} trainable / {total_params:,} total"
        )
        return lines


@dataclass
class ThresholdMetrics:
    """Evaluation metrics at a given probability decision threshold."""

    threshold: float
    accuracy: float
    precision_planet: float
    precision_non_planet: float
    precision_macro: float
    precision_weighted: float
    recall_planet: float
    recall_non_planet: float
    f1_planet: float
    f1_non_planet: float
    confusion_matrix: np.ndarray
    classification_report: str


@dataclass
class EvaluationMetrics:
    """Full evaluation summary for a Task F CNN experiment."""

    config_name: str
    roc_auc: float
    val_auc_best: float
    best_epoch: int
    epochs_run: int
    total_params: int
    trainable_params: int
    architecture_lines: list[str]
    default_threshold: ThresholdMetrics
    optimal_threshold: ThresholdMetrics
    n_train: int
    n_val: int
    n_test: int


class CNNTransitDetector:
    """Encapsulates loading, PyTorch training, evaluation, and reporting for Task F."""

    def __init__(self, config: CNNExperimentConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: TransitCNN1D | None = None
        self.metrics: EvaluationMetrics | None = None

    @staticmethod
    def _resolve_csv_path(csv_path: str) -> Path:
        path = Path(csv_path)
        if path.is_file():
            return path
        fallback = Path("..") / csv_path
        if fallback.is_file():
            return fallback
        raise FileNotFoundError(f"Dataset CSV not found: {path.resolve()}")

    def load_dataset(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset via ``daneel.detection.data_process.load_data``."""
        csv_path = self._resolve_csv_path(self.config.data.csv_path)
        X_train, X_test, y_train, y_test, *_ = load_data(
            csv_path=str(csv_path),
            n_bins=self.config.data.n_bins,
            use_scaler=self.config.data.standardize,
            samples_per_class=self.config.data.samples_per_class,
        )
        return X_train, X_test, y_train, y_test

    def build_model(self) -> TransitCNN1D:
        """Instantiate ``TransitCNN1D`` and move it to the active device."""
        model = TransitCNN1D(
            n_bins=self.config.data.n_bins,
            cfg=self.config.model,
        ).to(self.device)
        self.model = model
        return model

    def _build_loss_fn(self) -> nn.Module:
        t_cfg = self.config.training
        loss_key = t_cfg.loss_type.strip().lower()
        if loss_key == "focal":
            return FocalLossWithLogits(gamma=t_cfg.focal_gamma, alpha=t_cfg.focal_alpha)
        if loss_key == "bce":
            return nn.BCEWithLogitsLoss()
        raise ValueError(f"Unsupported loss_type '{t_cfg.loss_type}'. Use 'focal' or 'bce'.")

    def _build_optimizer(self, model: TransitCNN1D) -> torch.optim.Optimizer:
        l2 = self.config.model.l2_reg
        dense_weight_ids = {id(layer.weight) for layer in model.dense_linears}
        dense_weights = [p for p in model.parameters() if id(p) in dense_weight_ids]
        other_params = [p for p in model.parameters() if id(p) not in dense_weight_ids]
        param_groups = [
            {"params": dense_weights, "weight_decay": l2},
            {"params": other_params, "weight_decay": 0.0},
        ]
        return torch.optim.Adam(param_groups, lr=self.config.training.learning_rate)

    @staticmethod
    def _compute_threshold_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float,
    ) -> ThresholdMetrics:
        return ThresholdMetrics(
            threshold=float(threshold),
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision_planet=float(
                precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            ),
            precision_non_planet=float(
                precision_score(y_true, y_pred, pos_label=0, zero_division=0)
            ),
            precision_macro=float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            precision_weighted=float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            recall_planet=float(
                recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            ),
            recall_non_planet=float(
                recall_score(y_true, y_pred, pos_label=0, zero_division=0)
            ),
            f1_planet=float(
                f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            ),
            f1_non_planet=float(
                f1_score(y_true, y_pred, pos_label=0, zero_division=0)
            ),
            confusion_matrix=confusion_matrix(y_true, y_pred),
            classification_report=classification_report(
                y_true,
                y_pred,
                target_names=["Non-Planet", "Planet"],
                digits=4,
                zero_division=0,
            ),
        )

    def _predict_proba(self, model: TransitCNN1D, X_arr: np.ndarray) -> np.ndarray:
        model.eval()
        tensor_x = torch.tensor(X_arr[:, None, :], dtype=torch.float32, device=self.device)
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(tensor_x), self.config.training.batch_size):
                logits = model(tensor_x[start : start + self.config.training.batch_size])
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def fit_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> EvaluationMetrics:
        """Train the 1D CNN with internal validation early stopping and evaluate on X_test."""
        t_cfg = self.config.training
        set_deterministic_seed(t_cfg.random_state)

        # Carve an internal stratified validation split so X_test stays unseen during training
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=t_cfg.val_split,
            random_state=t_cfg.random_state,
            stratify=y_train,
        )

        train_ds = TensorDataset(
            torch.tensor(X_tr[:, None, :], dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        dl_gen = torch.Generator()
        dl_gen.manual_seed(t_cfg.random_state)
        train_loader = DataLoader(
            train_ds,
            batch_size=t_cfg.batch_size,
            shuffle=True,
            generator=dl_gen,
        )

        model = self.build_model()
        loss_fn = self._build_loss_fn()
        optimizer = self._build_optimizer(model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=t_cfg.lr_factor,
            patience=t_cfg.lr_patience,
            min_lr=t_cfg.min_lr,
        )

        best_val_auc = -1.0
        best_val_loss = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        epochs_run = 0

        X_val_tensor = torch.tensor(X_val[:, None, :], dtype=torch.float32, device=self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        effective_min_epochs = min(t_cfg.min_epochs, t_cfg.epochs)

        for epoch in range(1, t_cfg.epochs + 1):
            epochs_run = epoch
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_loss = float(loss_fn(val_logits, y_val_tensor).item())
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = float(roc_auc_score(y_val, val_probs))

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
                if epochs_no_improve >= t_cfg.patience:
                    break

        model.load_state_dict(best_state)
        proba = self._predict_proba(model, X_test)

        y_pred_default = (proba >= 0.5).astype(int)
        roc_auc = float(roc_auc_score(y_test, proba))

        fpr, tpr, thresholds = roc_curve(y_test, proba)
        best_idx = int(np.argmax(tpr - fpr))
        best_thresh = float(np.clip(thresholds[best_idx], 0.0, 1.0))
        y_pred_opt = (proba >= best_thresh).astype(int)

        total_params, trainable_params = model.count_parameters()
        self.metrics = EvaluationMetrics(
            config_name=self.config.model.name,
            roc_auc=roc_auc,
            val_auc_best=best_val_auc,
            best_epoch=best_epoch,
            epochs_run=epochs_run,
            total_params=total_params,
            trainable_params=trainable_params,
            architecture_lines=model.describe_architecture(),
            default_threshold=self._compute_threshold_metrics(
                y_test, y_pred_default, 0.5
            ),
            optimal_threshold=self._compute_threshold_metrics(
                y_test, y_pred_opt, best_thresh
            ),
            n_train=int(len(X_tr)),
            n_val=int(len(X_val)),
            n_test=int(len(X_test)),
        )
        return self.metrics

    def format_report(self) -> str:
        """Render a clean evaluation report for a single CNN configuration."""
        if self.metrics is None:
            raise RuntimeError("Model must be evaluated before formatting report.")

        m = self.config.model
        t = self.config.training
        res = self.metrics
        d_m = res.default_threshold
        o_m = res.optimal_threshold

        def _cm_lines(cm: np.ndarray) -> list[str]:
            tn, fp, fn, tp = cm.ravel()
            return [
                "                       Predicted Non-Planet    Predicted Planet",
                f"  Actual Non-Planet :        {tn:<18d}    {fp:<18d}",
                f"  Actual Planet     :        {fn:<18d}    {tp:<18d}",
            ]

        loss_desc = (
            f"FocalLoss(gamma={t.focal_gamma}, alpha={t.focal_alpha})"
            if t.loss_type.lower() == "focal"
            else "BCEWithLogitsLoss"
        )

        lines = [
            "=" * 86,
            f"CNN EVALUATION REPORT - {res.config_name}",
            "=" * 86,
            "",
            "MODEL ARCHITECTURE:",
            "-" * 86,
            *res.architecture_lines,
            "",
            "TRAINING HYPERPARAMETERS:",
            "-" * 86,
            f"  framework         : PyTorch ({torch.__version__})",
            f"  standardize       : {self.config.data.standardize}",
            f"  loss_function     : {loss_desc}",
            f"  optimizer         : Adam (lr={t.learning_rate}, dense_l2={m.l2_reg})",
            f"  batch_size        : {t.batch_size}",
            f"  max_epochs        : {t.epochs} (stopped at {res.epochs_run}, best={res.best_epoch})",
            f"  split sizes       : train={res.n_train} / val={res.n_val} / test={res.n_test}",
            f"  random_state      : {t.random_state}",
            f"  Best Val ROC-AUC  : {res.val_auc_best:.4f}",
            f"  Test ROC-AUC      : {res.roc_auc:.4f}",
            "",
            "METRICS AT DEFAULT THRESHOLD (p >= 0.5000):",
            "-" * 86,
            f"  Precision (Planet)     : {d_m.precision_planet:.4f} ({d_m.precision_planet * 100:.2f}%)",
            f"  Precision (Non-Planet) : {d_m.precision_non_planet:.4f} ({d_m.precision_non_planet * 100:.2f}%)",
            f"  Precision (Macro Avg)  : {d_m.precision_macro:.4f} ({d_m.precision_macro * 100:.2f}%)",
            f"  Recall (Planet)        : {d_m.recall_planet:.4f} ({d_m.recall_planet * 100:.2f}%)",
            f"  Accuracy               : {d_m.accuracy * 100:.2f}%",
            "",
            "CONFUSION MATRIX (Default Threshold 0.5000):",
            *_cm_lines(d_m.confusion_matrix),
            "",
            "CLASSIFICATION REPORT (Default Threshold 0.5000):",
            d_m.classification_report.rstrip(),
            "",
            f"METRICS AT OPTIMAL THRESHOLD (Youden's J = {o_m.threshold:.4f}):",
            "-" * 86,
            f"  Precision (Planet)     : {o_m.precision_planet:.4f} ({o_m.precision_planet * 100:.2f}%)",
            f"  Precision (Non-Planet) : {o_m.precision_non_planet:.4f} ({o_m.precision_non_planet * 100:.2f}%)",
            f"  Precision (Macro Avg)  : {o_m.precision_macro:.4f} ({o_m.precision_macro * 100:.2f}%)",
            f"  Recall (Planet)        : {o_m.recall_planet:.4f} ({o_m.recall_planet * 100:.2f}%)",
            f"  Accuracy               : {o_m.accuracy * 100:.2f}%",
            "",
            "CONFUSION MATRIX (Optimal Youden's J Threshold):",
            *_cm_lines(o_m.confusion_matrix),
            "",
            "CLASSIFICATION REPORT (Optimal Youden's J Threshold):",
            o_m.classification_report.rstrip(),
            "=" * 86,
        ]
        return "\n".join(lines) + "\n"

    def save_report(self, output_path: str | Path | None = None) -> Path:
        """Write the individual experiment report to disk."""
        target = Path(output_path or self.config.report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.format_report(), encoding="utf-8")
        return target

    def run(self) -> EvaluationMetrics:
        """Execute loading, training, evaluation, and report export."""
        X_train, X_test, y_train, y_test = self.load_dataset()
        metrics = self.fit_and_evaluate(X_train, y_train, X_test, y_test)
        self.save_report()
        return metrics
