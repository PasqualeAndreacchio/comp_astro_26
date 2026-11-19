"""Data preprocessing utilities for exoplanet detection.

This module provides functions for balancing datasets through data augmentation,
which is crucial for training machine learning models on imbalanced exoplanet
detection datasets where positive samples (planets) are typically much rarer
than negative samples (non-planets).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Random state for reproducibility
RANDOM_STATE = 42


def create_balanced_dataset(
    X: np.ndarray,
    y: np.ndarray,
    samples_per_class: int = 350,
    random_state: Optional[int] = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a balanced dataset through data augmentation.
    
    This function balances an imbalanced dataset by either downsampling the majority
    class or augmenting the minority class. Data augmentation techniques include:
    - Adding Gaussian noise
    - Scaling the signal
    - Shifting the signal in time
    - Combining scaling with noise
    
    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features) containing the input data.
        Typically light curve flux values for exoplanet detection.
    y : numpy.ndarray
        Target labels of shape (n_samples,) with binary values (0 or 1).
        0 represents non-planet samples, 1 represents planet samples.
    samples_per_class : int, optional
        Number of samples to generate for each class in the balanced dataset.
        Default is 350.
    random_state : int, optional
        Seed for the random number generator. Default is 42.
    
    Returns
    -------
    X_balanced : numpy.ndarray
        Balanced feature matrix of shape (2 * samples_per_class, n_features).
    y_balanced : numpy.ndarray
        Balanced target labels of shape (2 * samples_per_class,).
    
    Notes
    -----
    The augmentation process randomly applies one of four techniques:
    - 25% probability: Add Gaussian noise (mean=0, std=0.01)
    - 25% probability: Scale by factor in range [0.97, 1.03]
    - 25% probability: Shift signal by -20 to +20 positions
    - 25% probability: Combined scaling and noise addition
    
    Examples
    --------
    >>> X_train = np.random.randn(100, 3000)  # 100 samples, 3000 time points
    >>> y_train = np.array([0]*90 + [1]*10)   # Imbalanced: 90 non-planets, 10 planets
    >>> X_bal, y_bal = create_balanced_dataset(X_train, y_train, samples_per_class=200)
    >>> X_bal.shape
    (400, 3000)
    >>> np.sum(y_bal == 0), np.sum(y_bal == 1)
    (200, 200)
    """
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")

    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be 2-D and y must be a matching 1-D array")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain only binary labels 0 and 1")

    rng = np.random.default_rng(random_state)

    # Separate samples by class
    X0 = X[y == 0]
    X1 = X[y == 1]
    if len(X0) == 0 or len(X1) == 0:
        raise ValueError("Both classes (0 and 1) must be present in the dataset.")
    logger.info("Balancing dataset (original Class 0: %d, Class 1: %d)", len(X0), len(X1))

    def augment_to_target(X_orig: np.ndarray, n_target: int) -> np.ndarray:
        """Augment dataset to reach target number of samples.
        
        If the original dataset has enough samples, randomly subsample.
        Otherwise, augment using various techniques until target is reached.
        
        Parameters
        ----------
        X_orig : numpy.ndarray
            Original samples to augment.
        n_target : int
            Target number of samples to generate.
        
        Returns
        -------
        numpy.ndarray
            Augmented dataset with exactly n_target samples.
        """
        if len(X_orig) >= n_target:
            idx = rng.choice(len(X_orig), n_target, replace=False)
            return X_orig[idx]

        X_result = [X_orig]
        current_count = len(X_orig)

        while current_count < n_target:
            n_needed = n_target - current_count
            sample_size = min(len(X_orig), n_needed)
            idx = rng.choice(len(X_orig), sample_size, replace=(sample_size > len(X_orig)))
            aug_type = rng.random()

            if aug_type < 0.25:
                # 1. Add Gaussian noise
                noise = rng.normal(0.0, 0.01, (len(idx), X_orig.shape[1]))
                X_aug = X_orig[idx] + noise
            elif aug_type < 0.50:
                # 2. Scale amplitude
                scale = 1.0 + rng.uniform(-0.03, 0.03, (len(idx), 1))
                X_aug = X_orig[idx] * scale
            elif aug_type < 0.75:
                # 3. Circular roll (transit phase shift)
                shifts = rng.integers(-20, 20, size=len(idx))
                X_aug = np.array([np.roll(X_orig[i], s) for i, s in zip(idx, shifts)])
            else:
                # 4. Scale + Gaussian noise
                scale = 1.0 + rng.uniform(-0.02, 0.02, (len(idx), 1))
                noise = rng.normal(0.0, 0.008, (len(idx), X_orig.shape[1]))
                X_aug = (X_orig[idx] * scale) + noise

            X_result.append(X_aug)
            current_count += len(X_aug)

        X_final = np.vstack(X_result)
        return X_final[:n_target]

    # Balance both classes
    X0_bal = augment_to_target(X0, samples_per_class)
    X1_bal = augment_to_target(X1, samples_per_class)

    # Combine and shuffle
    X_bal = np.vstack([X0_bal, X1_bal])
    y_bal = np.concatenate([np.zeros(samples_per_class), np.ones(samples_per_class)])

    shuffle_idx = rng.permutation(len(X_bal))
    logger.info("Balanced dataset created: %d samples per class (Total: %d)", samples_per_class, len(X_bal))
    return X_bal[shuffle_idx], y_bal[shuffle_idx]


def resolve_file_path(path_str: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """Resolves a file path, checking base directory, current working directory, and repo root."""
    path = Path(path_str)
    if path.is_file():
        return path.resolve()

    if base_dir and (base_dir / path).is_file():
        return (base_dir / path).resolve()

    # Check relative to repo root (where tess_data.csv usually resides)
    repo_root = Path(__file__).resolve().parents[3]
    if (repo_root / path).is_file():
        return (repo_root / path).resolve()

    # Check parent directory fallback
    parent_fallback = Path("..") / path
    if parent_fallback.is_file():
        return parent_fallback.resolve()

    return path


def load_data(
    csv_path: Union[str, Path] = "tess_data.csv",
    n_bins: int = 1000,
    balanced: bool = True,
    test_size: float = 0.2,
    samples_per_class: int = 350,
    random_state: Optional[int] = RANDOM_STATE,
    base_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Loads light curve dataset from CSV, performs stratified split and optional balancing.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to the CSV file containing TESS light curve data.
        Default is 'tess_data.csv'.
    n_bins : int, optional
        Number of time bins in the light curve. The CSV should contain
        columns named 'flux_0000' through 'flux_{n_bins-1:04d}'.
        Default is 1000.
    balanced : bool, optional
        Whether to balance the training set using data augmentation.
        Default is True.
    test_size : float, optional
        Proportion of the dataset allocated to the test set.
        Default is 0.2.
    samples_per_class : int, optional
        Number of samples to generate for each class in the balanced training set.
        Default is 350.
    random_state : int, optional
        Seed for deterministic data splitting and augmentation.
        Default is 42.
    base_dir : Path, optional
        Optional base directory used to resolve relative paths.

    Returns
    -------
    X_train : numpy.ndarray
        Training feature matrix of shape (n_train_samples, n_bins).
    X_test : numpy.ndarray
        Test feature matrix of shape (n_test_samples, n_bins).
    y_train : numpy.ndarray
        Training labels of shape (n_train_samples,).
    y_test : numpy.ndarray
        Test labels of shape (n_test_samples,).
    metadata_test : pandas.DataFrame
        Metadata for test samples containing columns:
        ['toi_name', 'tic', 'label', 'disp', 'period_d', 't0_bjd', 'dur_hr', 'sector']
    X_err_test : numpy.ndarray
        Flux uncertainties for test samples of shape (n_test_samples, n_bins).

    Raises
    ------
    FileNotFoundError
        If the specified CSV file does not exist.
    ValueError
        If required columns or classes are missing from the dataset.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

    resolved_csv = resolve_file_path(csv_path, base_dir=base_dir)
    if not resolved_csv.is_file():
        raise FileNotFoundError(f"Dataset CSV file not found: {resolved_csv}")

    logger.info("Loading dataset from: %s", resolved_csv)
    df = pd.read_csv(resolved_csv)
    logger.info("Dataset shape: %s", df.shape)

    flux_cols = [f"flux_{i:04d}" for i in range(n_bins)]
    flux_err_cols = [f"flux_err_{i:04d}" for i in range(n_bins)]
    required_cols = flux_cols + ["label"]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns (first 5 missing: {missing_cols[:5]})")

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

    n_class_0 = int((y == 0).sum())
    n_class_1 = int((y == 1).sum())
    if n_class_0 < 2 or n_class_1 < 2:
        raise ValueError("Insufficient samples per class for stratified splitting.")

    X_train, X_test, y_train, y_test, _, X_err_test, _, idx_test = train_test_split(
        X,
        y,
        X_err,
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Stratified split -> Train: %d, Test: %d (test_size=%.2f, random_state=%s)",
        len(X_train),
        len(X_test),
        test_size,
        random_state,
    )

    if balanced:
        X_train, y_train = create_balanced_dataset(
            X_train,
            y_train,
            samples_per_class=samples_per_class,
            random_state=random_state,
        )

    metadata_test = metadata.iloc[idx_test].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, metadata_test, X_err_test
