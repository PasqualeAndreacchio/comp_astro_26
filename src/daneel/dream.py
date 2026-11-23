"""Generative Adversarial Network (GAN) for synthesizing transit light curves.

This module creates a DCGAN architecture to generate synthetic 1-D exoplanet transit
light curves by reshaping them into 44x44 square inputs (1936 points).

Limitations of 44x44 square representation of 1D light curves:
1. Temporal Discontinuity: Folding a 1D temporal signal into a 2D matrix breaks
   time continuity at row boundaries. The end of row k (index 44k - 1) is artificially
   adjacent along the vertical axis to row k+1 (index 44k + 43), introducing spurious
   spatial correlations across 44-step time separations.
2. 2D Spatial Invariance Violation: 2D convolutions assume translation invariance
   in both horizontal and vertical directions. In folded light curves, the vertical axis
   is not a spatial dimension but periodic 44-step strides; 2D filters cannot distinguish
   continuous transit dips from folded boundary artifacts.
3. Length Truncation: 2000-sample sequences truncated to 1936 (44^2) points discard
   64 samples, potentially shaving baseline or transit wings.
4. Checkerboard Artifacts: 2D transposed convolutions often create checkerboard patterns
   in 2D space that unroll into high-frequency periodic ringing in the 1D light curve.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.utils as vutils
from tqdm import tqdm
import yaml

from daneel.parameters import Parameters
from daneel.detection.data_process import resolve_file_path
from daneel.utils import deep_merge, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_DREAM_CONFIG: Dict[str, Any] = {
    "data_path": "tess_data.csv",
    "filter_transits": True,
    "target_length": 1936,
    "side_length": 44,
    "epochs": 30,
    "batch_size": 32,
    "lr": 0.0002,
    "beta1": 0.5,
    "nz": 100,
    "ngf": 32,
    "ndf": 32,
    "dropout": 0.25,
    "random_state": 42,
    "num_generated": 16,
    "output_plot": "dream_synthetic_transits.png",
}


def weights_init(m: nn.Module) -> None:
    """DCGAN weight initialization: normal(0, 0.02) for conv/linear, normal(1, 0.02) for BN."""
    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    """DCGAN Generator mapping latent vectors z in R^nz to 1x44x44 images in [-1, 1]."""

    def __init__(self, nz: int = 100, ngf: int = 32, nc: int = 1):
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.fc = nn.Linear(nz, ngf * 4 * 11 * 11)
        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4) x 11 x 11 -> (ngf*2) x 22 x 22
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2) x 22 x 22 -> (ngf) x 22 x 22
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf) x 22 x 22 -> (nc) x 44 x 44
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Flatten 4D latent noise (B, nz, 1, 1) to 2D (B, nz) if needed
        if z.ndim == 4:
            z = z.view(z.size(0), -1)
        # Project latent vector and reshape into initial 2D feature maps (B, ngf*4, 11, 11)
        x = self.fc(z).view(-1, self.ngf * 4, 11, 11)
        # Upsample 11x11 -> 22x22 -> 44x44 via ConvTranspose2d, output in [-1, 1]
        return self.main(x)


class Discriminator(nn.Module):
    """DCGAN Discriminator classifying 1x44x44 inputs into real (1) vs synthetic (0)."""

    def __init__(self, nc: int = 1, ndf: int = 32, dropout: float = 0.25):
        super().__init__()
        self.main = nn.Sequential(
            # (nc) x 44 x 44 -> (ndf) x 22 x 22
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            # (ndf) x 22 x 22 -> (ndf*2) x 11 x 11
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            # (ndf*2) x 11 x 11 -> (ndf*4) x 5 x 5
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Global Average Pooling -> 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(ndf * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class LightCurveNormalizer:
    """Robust scaling between physical flux values and [-1, 1] for Tanh activation."""

    def __init__(self, lower_percentile: float = 0.5, upper_percentile: float = 99.5):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.low: float = 0.0
        self.high: float = 1.0

    def fit(self, data: np.ndarray) -> LightCurveNormalizer:
        """Estimate robust baseline and trough bounds from the dataset."""
        self.low = float(np.percentile(data, self.lower_percentile))
        self.high = float(np.percentile(data, self.upper_percentile))
        # Sanity check: prevent ZeroDivisionError in transform() if data is flat or degenerate
        if self.high <= self.low:
            self.high = self.low + 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Scale array from [low, high] to [-1, 1] with clipping."""
        scaled = 2.0 * (data - self.low) / (self.high - self.low) - 1.0
        return np.clip(scaled, -1.0, 1.0)

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        """Denormalize from [-1, 1] back to physical flux values."""
        return (scaled + 1.0) / 2.0 * (self.high - self.low) + self.low


class DreamGAN:
    """Manager class for GAN training, generation, and light-curve reconstruction.

    Parameters
    ----------
    params : dict, str, Path, or Parameters, optional
        Configuration dictionary, path to YAML configuration file, or Parameters instance.
    **kwargs : Any
        Overriding configuration options.
    """

    def __init__(
        self,
        params: Optional[Union[Dict[str, Any], str, Path, Parameters]] = None,
        **kwargs: Any,
    ):
        user_config: Dict[str, Any] = {}
        if isinstance(params, (str, Path)):
            path = Path(params)
            if not path.is_file():
                path = resolve_file_path(path)
            if not path.is_file():
                raise FileNotFoundError(f"Configuration file not found: {params}")
            with open(path, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}
                user_config = raw_cfg.get("dream", raw_cfg)
        elif isinstance(params, Parameters):
            raw_cfg = copy.deepcopy(params.params)
            user_config = raw_cfg.get("dream", raw_cfg)
        elif isinstance(params, dict):
            user_config = copy.deepcopy(params.get("dream", params))

        if kwargs:
            user_config = deep_merge(user_config, kwargs)

        self.config = {**DEFAULT_DREAM_CONFIG, **user_config}
        setup_logging(self.config.get("logging", {}).get("level", "INFO") if isinstance(self.config.get("logging"), dict) else "INFO")

        # Set seeds for reproducibility and device
        seed = int(self.config.get("random_state", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("DreamGAN using device: %s", self.device)

        # Networks
        self.nz = int(self.config.get("nz", 100))
        self.ngf = int(self.config.get("ngf", 32))
        self.ndf = int(self.config.get("ndf", 32))
        dropout = float(self.config.get("dropout", 0.25))

        self.netG = Generator(nz=self.nz, ngf=self.ngf, nc=1).to(self.device)
        self.netD = Discriminator(nc=1, ndf=self.ndf, dropout=dropout).to(self.device)

        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        self.normalizer = LightCurveNormalizer()
        self.real_curves_1d: Optional[np.ndarray] = None
        self.d_losses: list[float] = []
        self.g_losses: list[float] = []

    def load_data(self) -> TensorDataset:
        """Load, truncate to 1936 samples, normalize, and reshape to (N, 1, 44, 44)."""
        csv_path = Path(self.config["data_path"])
        if not csv_path.is_file():
            csv_path = resolve_file_path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        logger.info("Loading light curve data from: %s", csv_path)
        df = pd.read_csv(csv_path)

        if self.config.get("filter_transits", True) and "label" in df.columns:
            df = df[df["label"] == 1].copy()
            logger.info("Filtered for confirmed transit curves (label=1): N = %d", len(df))

        # Extract numeric columns (flux_0000..flux_0999 and flux_err_0000..flux_err_0999)
        cols = [c for c in df.columns if c.startswith("flux_")]
        if not cols:
            raise ValueError(f"No flux columns found in {csv_path}")

        raw_curves = df[cols].values.astype(np.float32)
        n_samples, n_points = raw_curves.shape
        target_len = int(self.config.get("target_length", 1936))
        side = int(self.config.get("side_length", 44))

        if n_points >= target_len:
            # Truncate each 2000-sample sequence to 1936 points
            curves_1936 = raw_curves[:, :target_len]
        else:
            # Resample / interpolate to target_len if fewer points provided
            x_old = np.linspace(0, 1, n_points)
            x_new = np.linspace(0, 1, target_len)
            curves_1936 = np.array([np.interp(x_new, x_old, row) for row in raw_curves], dtype=np.float32)

        self.real_curves_1d = curves_1936
        self.normalizer.fit(curves_1936)
        scaled_curves = self.normalizer.transform(curves_1936)

        # Reshape to 44x44 2D single-channel images
        images = scaled_curves.reshape(n_samples, 1, side, side).astype(np.float32)
        tensor_x = torch.tensor(images, dtype=torch.float32)
        return TensorDataset(tensor_x)

    def train(self) -> None:
        """Train DCGAN with label smoothing and Adam optimizers."""
        dataset = self.load_data()
        batch_size = int(self.config.get("batch_size", 32))
        epochs = int(self.config.get("epochs", 30))
        lr = float(self.config.get("lr", 0.0002))
        beta1 = float(self.config.get("beta1", 0.5))

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        criterion = nn.BCELoss()

        optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-4)
        optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))

        # Real label smoothing (0.9 instead of 1.0) to regularize small dataset
        real_label_val = 0.9
        fake_label_val = 0.0

        self.d_losses = []
        self.g_losses = []

        logger.info("Starting GAN training for %d epochs on %d samples...", epochs, len(dataset))
        progress_bar = tqdm(range(epochs), desc="GAN Training")

        for epoch in progress_bar:
            d_loss_epoch = 0.0
            g_loss_epoch = 0.0
            n_batches = 0

            for (real_batch,) in dataloader:
                b_size = real_batch.size(0)
                real_batch = real_batch.to(self.device)

                # Update Discriminator
                self.netD.zero_grad()
                label = torch.full((b_size,), real_label_val, dtype=torch.float, device=self.device)

                out_real = self.netD(real_batch).view(-1)
                errD_real = criterion(out_real, label)
                errD_real.backward()

                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                fake_batch = self.netG(noise)
                label.fill_(fake_label_val)

                out_fake = self.netD(fake_batch.detach()).view(-1)
                errD_fake = criterion(out_fake, label)
                errD_fake.backward()

                optimizerD.step()
                errD = errD_real + errD_fake

                # Update Generator
                self.netG.zero_grad()
                label.fill_(1.0)  # Generator maximizes log(D(G(z)))
                out_fake_g = self.netD(fake_batch).view(-1)
                errG = criterion(out_fake_g, label)
                errG.backward()
                optimizerG.step()

                d_loss_epoch += errD.item()
                g_loss_epoch += errG.item()
                n_batches += 1

            avg_d = d_loss_epoch / max(1, n_batches)
            avg_g = g_loss_epoch / max(1, n_batches)
            self.d_losses.append(avg_d)
            self.g_losses.append(avg_g)

            progress_bar.set_postfix(Loss_D=f"{avg_d:.4f}", Loss_G=f"{avg_g:.4f}")

        logger.info("GAN training finished: Final Loss_D=%.4f, Loss_G=%.4f", self.d_losses[-1], self.g_losses[-1])

    def generate(self, num_samples: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic 2D representations and reconstructed 1D light curves.

        Returns
        -------
        synthetic_2d : np.ndarray
            Generated 2D images in [-1, 1], shape (num_samples, 44, 44).
        reconstructed_1d : np.ndarray
            Reconstructed 1D light curves in physical flux, shape (num_samples, 1936).
        """
        self.netG.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.nz, 1, 1, device=self.device)
            fake_imgs = self.netG(noise).cpu().numpy()  # (num_samples, 1, 44, 44)

        synthetic_2d = fake_imgs[:, 0, :, :]
        flattened = synthetic_2d.reshape(num_samples, -1)
        reconstructed_1d = self.normalizer.inverse_transform(flattened)
        return synthetic_2d, reconstructed_1d

    def plot_results(
        self,
        output_path: Union[str, Path] = "dream_synthetic_transits.png",
        num_display: int = 6,
    ) -> Path:
        """Plot synthetic 2D heatmaps, 1D reconstructed light curves, and comparison."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        synth_2d, synth_1d = self.generate(num_samples=max(16, num_display))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

        # Panel 1: Synthetic 2D representation grid
        grid_imgs = torch.tensor(synth_2d[:16, None, :, :])
        grid = vutils.make_grid(grid_imgs, nrow=4, normalize=True, padding=2)
        grid_np = np.transpose(grid.numpy(), (1, 2, 0))
        axes[0, 0].imshow(grid_np)
        axes[0, 0].set_title(r"$\mathbf{Synthetic\ 2D\ Light\ Curves\ (44\times 44)}$", fontsize=11)
        axes[0, 0].axis("off")

        # Panel 2: Reconstructed 1D Synthetic Light Curves
        time_axis = np.arange(synth_1d.shape[1])
        for i in range(num_display):
            axes[0, 1].plot(time_axis, synth_1d[i], alpha=0.6, lw=1.2, label=f"Synthetic #{i+1}" if i < 3 else None)
        axes[0, 1].plot(time_axis, np.median(synth_1d, axis=0), color="black", lw=2.2, label="Synthetic Median")
        axes[0, 1].axhline(1.0, color="gray", ls="--", lw=0.9, alpha=0.7)
        axes[0, 1].set_title(r"$\mathbf{Reconstructed\ 1D\ Synthetic\ Light\ Curves\ (1936\ pts)}$", fontsize=11)
        axes[0, 1].set_xlabel("Time index (sample)", fontsize=10)
        axes[0, 1].set_ylabel("Relative flux", fontsize=10)
        axes[0, 1].grid(True, linestyle=":", alpha=0.4)
        axes[0, 1].legend(loc="lower left", fontsize=8)

        # Panel 3: Real 1D Transit Light Curves for Comparison
        if self.real_curves_1d is not None:
            n_real = min(num_display, len(self.real_curves_1d))
            for i in range(n_real):
                axes[1, 0].plot(time_axis, self.real_curves_1d[i], alpha=0.6, lw=1.2, label=f"Real #{i+1}" if i < 3 else None)
            axes[1, 0].plot(time_axis, np.median(self.real_curves_1d, axis=0), color="crimson", lw=2.2, label="Real Median")
            axes[1, 0].axhline(1.0, color="gray", ls="--", lw=0.9, alpha=0.7)
            axes[1, 0].set_title(r"$\mathbf{Real\ Transit\ Light\ Curves\ (TESS\ N=472)}$", fontsize=11)
            axes[1, 0].set_xlabel("Time index (sample)", fontsize=10)
            axes[1, 0].set_ylabel("Relative flux", fontsize=10)
            axes[1, 0].grid(True, linestyle=":", alpha=0.4)
            axes[1, 0].legend(loc="lower left", fontsize=8)

        # Panel 4: GAN Loss History
        if self.d_losses and self.g_losses:
            axes[1, 1].plot(self.d_losses, label="Discriminator Loss", color="royalblue", lw=1.8)
            axes[1, 1].plot(self.g_losses, label="Generator Loss", color="darkorange", lw=1.8)
            axes[1, 1].set_title(r"$\mathbf{GAN\ Training\ Loss\ Curves}$", fontsize=11)
            axes[1, 1].set_xlabel("Epoch", fontsize=10)
            axes[1, 1].set_ylabel("Binary Cross Entropy Loss", fontsize=10)
            axes[1, 1].grid(True, linestyle=":", alpha=0.4)
            axes[1, 1].legend(loc="upper right", fontsize=9)

        fig.tight_layout()
        fig.savefig(out_file, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved synthetic transit figure to: %s", out_file)
        return out_file

    def run(self, output_file: Optional[str] = None) -> Path:
        """Run complete pipeline: train GAN, generate light curves, and save plot."""
        self.train()
        dest = output_file or self.config.get("output_plot", "dream_synthetic_transits.png")
        return self.plot_results(dest)
