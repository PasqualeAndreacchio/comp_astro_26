"""Convolutional Neural Network (CNN) detection module for exoplanet transits.

This module provides a 1-D CNN pipeline and detection class (:class:`CNNDetection`)
for binary classification of exoplanetary transit light curves (Planet vs Non-Planet).
It inherits from :class:`~daneel.detection.base.BaseTransitDetector`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
    ) -> CNNDetection:
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
            epochs,
            batch_size,
            lr,
            wd,
            pos_weight_val,
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
                    epoch,
                    epochs,
                    avg_loss,
                    avg_acc * 100,
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

    def hook_report_extra_sections(self, lines: List[str]) -> None:
        """Appends training curve summary to report text."""
        if self.train_losses:
            lines.extend([
                "TRAINING CURVE SUMMARY:",
                "-" * 80,
                f"  Epochs completed    : {len(self.train_losses)}",
                f"  Final train loss    : {self.train_losses[-1]:.4f}",
                f"  Final train acc     : {self.train_accuracies[-1] * 100 if self.train_accuracies else 0.0:.2f}%",
                "",
            ])


# Aliases for backwards compatibility
CNNClassifier = CNNDetection
CNN = CNNDetection
