"""Convolutional Neural Network (CNN) detection module for exoplanet transits.

This module provides a 1-D CNN pipeline and detection class (:class:`CNNDetection`)
for binary classification of exoplanetary transit light curves (Planet vs Non-Planet).
It inherits from :class:`~daneel.detection.base.BaseTransitDetector`.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    from daneel.parameters import Parameters
except ImportError:
    from ..parameters import Parameters

from .base import BaseTransitDetector

logger = logging.getLogger(__name__)


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
        # Early Stopping
        "early_stopping": {
            "enabled": True,
            "patience": 15,
            "min_delta": 1e-4,
            "monitor": "val_loss",
            "mode": "min",
            "restore_best_weights": True,
        },
        # Learning Rate Scheduler
        "reduce_lr_on_plateau": {
            "enabled": True,
            "factor": 0.5,
            "patience": 7,
            "min_lr": 1e-6,
            "threshold": 1e-4,
            "monitor": "val_loss",
            "mode": "min",
        },
    },
    "output": {
        "report_path": "cnn/results/report.txt",
        "model_path": None,              # Optional path to save trained model weights
    },
    "logging": {
        "level": "INFO",
    },
}

MODEL_KEYS: Dict[str, str] = {
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
    "device": "device",
    "early_stopping": "early_stopping",
    "reduce_lr_on_plateau": "reduce_lr_on_plateau",
    "lr_scheduler": "reduce_lr_on_plateau",
}


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
class CNNDetection(BaseTransitDetector):
    """CNN detection and classification class for exoplanet transit light curves.

    Parameters
    ----------
    params : dict, str, Path, or Parameters, optional
        Configuration dictionary, path to a YAML configuration file, or a
        :class:`~daneel.parameters.Parameters` instance.
    **kwargs : Any
        Additional keyword arguments that override configuration values.
    """

    model_name: str = "CNN"
    DEFAULT_CONFIG: Dict[str, Any] = DEFAULT_CONFIG
    MODEL_KEYS: Dict[str, str] = MODEL_KEYS

    def __init__(
        self,
        params: Optional[Union[Dict[str, Any], str, Path, Parameters]] = None,
        **kwargs: Any,
    ):
        super().__init__(params, **kwargs)

        # Set global seeds for reproducibility
        _seed = self.model_config.get("random_state", 42)
        torch.manual_seed(_seed)
        np.random.seed(_seed)

        # Select device
        device_str = self.model_config.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
        logger.info("Using device: %s", self.device)

        # Model & training history holders
        self.model: Optional[_LightCurveCNN] = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []

    def _is_fitted(self) -> bool:
        """Returns True if the CNN model has been trained."""
        return self.model is not None

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
            conv_channels,
            kernel_size,
            pool_size,
            fc_units,
            dropout,
            n_params,
        )
        return self.model

    def fit(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> CNNDetection:
        """Trains the CNN on (optionally) provided or previously loaded training data.

        Supports YAML-configurable Early Stopping, ReduceLROnPlateau scheduler,
        and live tqdm epoch progress reporting.

        Args:
            X: Feature matrix (n_samples, n_bins). Falls back to ``self.X_train``.
            y: Target labels (n_samples,). Falls back to ``self.y_train``.
            X_val: Optional validation feature matrix. Falls back to ``self.X_test``.
            y_val: Optional validation target labels. Falls back to ``self.y_test``.

        Returns:
            ``self`` for method chaining.
        """
        if X is None or y is None:
            if self.X_train is None or self.y_train is None:
                self.load_data()
            X = self.X_train
            y = self.y_train

        if X_val is None or y_val is None:
            if self.X_test is not None and self.y_test is not None:
                X_val = self.X_test
                y_val = self.y_test

        if self.model is None:
            self.build_model(input_length=X.shape[1])

        batch_size = self.model_config.get("batch_size", 32)
        epochs = self.model_config.get("epochs", 30)
        lr = self.model_config.get("learning_rate", 1e-3)
        wd = self.model_config.get("weight_decay", 1e-4)

        # Early Stopping configuration
        es_cfg = self.model_config.get("early_stopping", {})
        if isinstance(es_cfg, bool):
            es_enabled = es_cfg
            es_cfg = {}
        elif isinstance(es_cfg, dict):
            es_enabled = es_cfg.get("enabled", True)
        else:
            es_enabled = False

        es_patience = int(es_cfg.get("patience", 15)) if es_enabled else 15
        es_min_delta = float(es_cfg.get("min_delta", 1e-4)) if es_enabled else 1e-4
        es_monitor = str(es_cfg.get("monitor", "val_loss")) if es_enabled else "val_loss"
        es_mode = str(es_cfg.get("mode", "min" if "loss" in es_monitor else "max")) if es_enabled else "min"
        es_restore_best = bool(es_cfg.get("restore_best_weights", True)) if es_enabled else False

        # ReduceLROnPlateau configuration
        rlr_cfg = self.model_config.get("reduce_lr_on_plateau", {})
        if isinstance(rlr_cfg, bool):
            rlr_enabled = rlr_cfg
            rlr_cfg = {}
        elif isinstance(rlr_cfg, dict):
            rlr_enabled = rlr_cfg.get("enabled", True)
        else:
            rlr_enabled = False

        rlr_factor = float(rlr_cfg.get("factor", 0.5)) if rlr_enabled else 0.5
        rlr_patience = int(rlr_cfg.get("patience", 7)) if rlr_enabled else 7
        rlr_min_lr = float(rlr_cfg.get("min_lr", 1e-6)) if rlr_enabled else 1e-6
        rlr_threshold = float(rlr_cfg.get("threshold", 1e-4)) if rlr_enabled else 1e-4
        rlr_monitor = str(rlr_cfg.get("monitor", "val_loss")) if rlr_enabled else "val_loss"
        rlr_mode = str(rlr_cfg.get("mode", "min" if "loss" in rlr_monitor else "max")) if rlr_enabled else "min"

        # Compute positive-class weight to handle any residual imbalance
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        pos_weight_val = float(n_neg / max(n_pos, 1))
        pos_weight = torch.tensor([pos_weight_val], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

        if rlr_enabled:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=rlr_mode,
                factor=rlr_factor,
                patience=rlr_patience,
                threshold=rlr_threshold,
                min_lr=rlr_min_lr,
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Build training DataLoader
        X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)
        y_t = torch.from_numpy(y.astype(np.float32))
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        # Build validation DataLoader if validation data is available
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_t = torch.from_numpy(X_val.astype(np.float32)).unsqueeze(1)
            y_val_t = torch.from_numpy(y_val.astype(np.float32))
            val_dataset = TensorDataset(X_val_t, y_val_t)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        logger.info(
            "Training CNN | epochs=%d | batch_size=%d | lr=%g | weight_decay=%g | "
            "pos_weight=%.4f | early_stopping=%s | reduce_lr=%s",
            epochs,
            batch_size,
            lr,
            wd,
            pos_weight_val,
            es_enabled,
            rlr_enabled,
        )

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        best_weights = copy.deepcopy(self.model.state_dict())
        best_metric = float("inf") if es_mode == "min" else float("-inf")
        best_epoch = 1
        es_patience_counter = 0

        epoch_pbar = tqdm(range(1, epochs + 1), desc="CNN Training", unit="epoch")
        for epoch in epoch_pbar:
            self.model.train()
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

            avg_train_loss = epoch_loss / max(total, 1)
            avg_train_acc = correct / max(total, 1)
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(avg_train_acc)

            # Evaluate on validation data if available
            avg_val_loss = None
            avg_val_acc = None
            if val_loader is not None:
                self.model.eval()
                val_epoch_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for X_val_b, y_val_b in val_loader:
                        X_val_b = X_val_b.to(self.device)
                        y_val_b = y_val_b.to(self.device)
                        val_logits = self.model(X_val_b)
                        val_batch_loss = criterion(val_logits, y_val_b)
                        val_epoch_loss += val_batch_loss.item() * len(y_val_b)
                        val_preds = (torch.sigmoid(val_logits) >= 0.5).long()
                        val_correct += (val_preds == y_val_b.long()).sum().item()
                        val_total += len(y_val_b)
                avg_val_loss = val_epoch_loss / max(val_total, 1)
                avg_val_acc = val_correct / max(val_total, 1)
                self.val_losses.append(avg_val_loss)
                self.val_accuracies.append(avg_val_acc)

            def _resolve_metric(m_name: str) -> float:
                if "val" in m_name and avg_val_loss is not None:
                    return avg_val_loss if "loss" in m_name else avg_val_acc
                return avg_train_loss if "loss" in m_name else avg_train_acc

            # Step scheduler
            if rlr_enabled:
                scheduler.step(_resolve_metric(rlr_monitor))
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            # Live metrics in tqdm
            postfix: Dict[str, Any] = {
                "train_loss": f"{avg_train_loss:.4f}",
                "train_acc": f"{avg_train_acc * 100:.1f}%",
            }
            if avg_val_loss is not None:
                postfix["val_loss"] = f"{avg_val_loss:.4f}"
                postfix["val_acc"] = f"{avg_val_acc * 100:.1f}%"
            postfix["lr"] = f"{current_lr:.1e}"

            # Early Stopping evaluation
            early_stop_triggered = False
            if es_enabled:
                current_es_val = _resolve_metric(es_monitor)
                if es_mode == "min":
                    improved = current_es_val < (best_metric - es_min_delta)
                else:
                    improved = current_es_val > (best_metric + es_min_delta)

                if improved:
                    best_metric = current_es_val
                    best_epoch = epoch
                    best_weights = copy.deepcopy(self.model.state_dict())
                    es_patience_counter = 0
                else:
                    es_patience_counter += 1

                postfix["es_pat"] = f"{es_patience_counter}/{es_patience}"

                if es_patience_counter >= es_patience:
                    early_stop_triggered = True

            epoch_pbar.set_postfix(postfix)

            if epoch == 1 or epoch % 10 == 0 or epoch == epochs or early_stop_triggered:
                val_str = (
                    f"  val_loss: {avg_val_loss:.4f}  val_acc: {avg_val_acc * 100:.2f}%"
                    if avg_val_loss is not None
                    else ""
                )
                logger.info(
                    "Epoch [%3d/%3d]  loss: %.4f  accuracy: %.2f%%%s  lr: %.1e",
                    epoch,
                    epochs,
                    avg_train_loss,
                    avg_train_acc * 100,
                    val_str,
                    current_lr,
                )

            if early_stop_triggered:
                logger.info(
                    "Early stopping triggered at epoch %d: '%s' did not improve for %d consecutive epochs (best %.4f at epoch %d).",
                    epoch,
                    es_monitor,
                    es_patience,
                    best_metric,
                    best_epoch,
                )
                epoch_pbar.close()
                break

        if es_enabled and es_restore_best and best_weights is not None:
            self.model.load_state_dict(best_weights)
            logger.info(
                "Restored best model weights from epoch %d (%s: %.4f).",
                best_epoch,
                es_monitor,
                best_metric,
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

    # Subclass hooks for training curve summary
    def hook_extra_metrics(self, metrics: Dict[str, Any]) -> None:
        """Attaches training curves to metrics dictionary."""
        metrics["train_losses"] = self.train_losses
        metrics["train_accuracies"] = self.train_accuracies
        metrics["val_losses"] = self.val_losses
        metrics["val_accuracies"] = self.val_accuracies

    def hook_display_extra_sections(self) -> None:
        """Displays training curve summary in console statistics."""
        if self.train_losses:
            logger.info("TRAINING CURVE SUMMARY:")
            logger.info("-" * 80)
            logger.info("  Epochs completed    : %d", len(self.train_losses))
            logger.info("  Final train loss    : %.4f", self.train_losses[-1])
            logger.info(
                "  Final train acc     : %.2f%%",
                self.train_accuracies[-1] * 100 if self.train_accuracies else 0.0,
            )
            if self.val_losses:
                logger.info("  Final val loss      : %.4f", self.val_losses[-1])
                logger.info(
                    "  Final val acc       : %.2f%%",
                    self.val_accuracies[-1] * 100 if self.val_accuracies else 0.0,
                )

    def hook_report_extra_sections(self, lines: List[str]) -> None:
        """Appends training curve summary to report text."""
        if self.train_losses:
            lines.extend([
                "TRAINING CURVE SUMMARY:",
                "-" * 80,
                f"  Epochs completed    : {len(self.train_losses)}",
                f"  Final train loss    : {self.train_losses[-1]:.4f}",
                f"  Final train acc     : {self.train_accuracies[-1] * 100 if self.train_accuracies else 0.0:.2f}%",
            ])
            if self.val_losses:
                lines.extend([
                    f"  Final val loss      : {self.val_losses[-1]:.4f}",
                    f"  Final val acc       : {self.val_accuracies[-1] * 100 if self.val_accuracies else 0.0:.2f}%",
                ])
            lines.append("")


# Aliases for backwards compatibility
CNNClassifier = CNNDetection
CNN = CNNDetection
