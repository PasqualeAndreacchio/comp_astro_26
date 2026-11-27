"""Exoplanet Transit Classification using Random Forest.

This module provides an end-to-end Machine Learning pipeline for binary
classification of exoplanetary transit light curves (Planet vs Non-Planet).
It executes a single-configuration training and evaluation run driven by a
modular YAML configuration file, with deterministic evaluation splits across runs,
scikit-learn Pipeline construction, optimal decision threshold tuning via
Youden's J statistic, and a clean, clutter-free evaluation report.
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import yaml

logger = logging.getLogger(__name__)

# Default configuration: fallback if no YAML is provided
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
        "standardize": False,
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": None,
        "random_state": 42,
        "n_jobs": -1,
        "oob_score": True,
    },
    "output": {
        "report_path": "random_forest/results/report_config_1.txt",
    },
    "logging": {
        "level": "INFO",
    },
}


def deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merges an override dictionary into a base configuration dictionary.
    Even if keys are not in the override dictionary, they will be present in the merged dictionary,
    allowing fallback to the default values for any missing keys.

    Args:
        base: The base/default configuration dictionary.
        override: The overriding dictionary (e.g., loaded from a user YAML file).

    Returns:
        A new dictionary representing the deep merge of base and override.
    """
    merged = copy.deepcopy(base)
    if not override:
        return merged

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads configuration from a YAML file with recursive fallback to DEFAULT_CONFIG.

    If `config_path` is None or omitted, the default configuration is returned directly.
    If a path is provided, the file is parsed and its parameters are deep-merged
    over `DEFAULT_CONFIG`, ensuring that any missing keys smoothly fall back to default values.

    Args:
        config_path: Optional path to the user's YAML configuration file.

    Returns:
        The effective merged configuration dictionary.

    Raises:
        FileNotFoundError: If the specified configuration path does not exist on disk.
        yaml.YAMLError: If the YAML configuration file contains syntax errors.
    """
    # Fallback to default if no path is provided
    if config_path is None:
        logger.info("No configuration file specified (-i / --input). Using default configuration.")
        return copy.deepcopy(DEFAULT_CONFIG)

    # Check if file exists
    path = Path(config_path)
    if not path.is_file():
        # Fallback check if invoked from within a subfolder or root
        fallback = Path("..") / config_path
        if fallback.is_file():
            path = fallback
        else:
            logger.error("Configuration file not found at: %s", path.resolve())
            raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    # Load YAML file and merge with default configuration
    try:
        with open(path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        logger.info("Successfully loaded configuration from: %s", path.resolve())
    except yaml.YAMLError as exc:
        logger.error("Failed to parse YAML configuration in '%s': %s", path, exc)
        raise
    except Exception as exc:
        logger.error("Failed to read configuration file '%s': %s", path, exc)
        raise

    effective_config = deep_merge(DEFAULT_CONFIG, user_config)
    return effective_config


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
        force=True,
    )


def create_balanced_dataset(
    X: np.ndarray,
    y: np.ndarray,
    samples_per_class: int = 400,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Balances a binary dataset using random sampling and data augmentation.

    If a class contains more samples than `samples_per_class`, it is randomly
    downsampled without replacement. If it contains fewer samples, synthetic light
    curves are generated using one of four photometric augmentation strategies:
    1. Gaussian noise addition.
    2. Uniform amplitude scaling.
    3. Temporal phase shift (circular roll).
    4. Combined amplitude scaling and Gaussian noise.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary target array of shape (n_samples,).
        samples_per_class: Desired number of samples for each class.
        random_state: Seed for pseudo-random number generator.

    Returns:
        A tuple containing:
            - Balanced feature matrix of shape (2 * samples_per_class, n_features).
            - Balanced target vector of shape (2 * samples_per_class,).
    """
    logger.info("=" * 70)
    logger.info("CREATING BALANCED DATASET")
    logger.info("=" * 70)

    rng = np.random.default_rng(random_state)

    X0 = X[y == 0]
    X1 = X[y == 1]
    logger.info("Original split - Class 0 (Non-Planet): %d, Class 1 (Planet): %d", len(X0), len(X1))

    def augment_to_target(X_orig: np.ndarray, n_target: int) -> np.ndarray:
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

    X0_bal = augment_to_target(X0, samples_per_class)
    X1_bal = augment_to_target(X1, samples_per_class)
    logger.info("Balanced split - Class 0: %d, Class 1: %d", len(X0_bal), len(X1_bal))

    X_bal = np.vstack([X0_bal, X1_bal])
    y_bal = np.concatenate([np.zeros(samples_per_class), np.ones(samples_per_class)])

    shuffle_idx = rng.permutation(len(X_bal))
    return X_bal[shuffle_idx], y_bal[shuffle_idx]


def load_data(
    csv_path: Union[str, Path] = "tess_data.csv",
    n_bins: int = 1000,
    balanced: bool = True,
    test_size: float = 0.2,
    samples_per_class: int = 350,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Loads light curve dataset from CSV, performs stratified split and optional balancing.

    Uses a fixed `random_state` so that each configuration run evaluates on the exact
    same train/test split.

    Args:
        csv_path: Path to the input dataset CSV.
        n_bins: Number of flux feature bins to extract (expects columns flux_0000 to flux_{n_bins-1}).
        balanced: Whether to balance the training split via augmentation/sampling.
        test_size: Proportion of the dataset allocated to the test set.
        samples_per_class: Number of samples per class when balancing the training set.
        random_state: Random state for deterministic data splitting.

    Returns:
        A tuple containing:
            - X_train: Training features array.
            - X_test: Test features array.
            - y_train: Training labels array.
            - y_test: Test labels array.
            - metadata_test: DataFrame containing metadata for test instances.
            - X_err_test: Measurement error values corresponding to X_test.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        ValueError: If required columns or classes are missing from the dataset.
    """
    logger.info("=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)

    # Check if file exists, with fallback to root directory
    path = Path(csv_path)   
    if not path.is_file():
        # Fallback check if executed from within random_forest/ subdirectory
        fallback = Path("..") / csv_path
        if fallback.is_file():
            path = fallback
        else:
            logger.error("Dataset file not found at: %s", path.resolve())
            raise FileNotFoundError(f"Dataset CSV file not found: {path.resolve()}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.error("Failed to read CSV file '%s': %s", path, exc)
        raise

    logger.info("Loaded CSV with shape: %s", df.shape)

    # Generate column names and check for missing columns
    flux_cols = [f"flux_{i:04d}" for i in range(n_bins)]
    flux_err_cols = [f"flux_err_{i:04d}" for i in range(n_bins)]
    required_cols = flux_cols + ["label"]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        err_msg = f"Dataset is missing required columns (first 5 missing: {missing_cols[:5]})"
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Separate flux, error, label columns and get metadata columns
    X = df[flux_cols].values
    X_err = df[flux_err_cols].values if set(flux_err_cols).issubset(df.columns) else np.zeros_like(X)
    y = df["label"].values

    metadata_cols = [
        col for col in ["toi_name", "tic", "label", "disp", "period_d", "t0_bjd", "dur_hr", "sector"]
        if col in df.columns
    ]
    metadata = df[metadata_cols] if metadata_cols else pd.DataFrame(index=df.index)

    n_class_0 = int((y == 0).sum())
    n_class_1 = int((y == 1).sum())
    logger.info("Original class distribution -> Class 0: %d, Class 1: %d", n_class_0, n_class_1)
    if n_class_0 > 0:
        logger.info("Imbalance ratio (Class 1 : Class 0): %.2f : 1", n_class_1 / n_class_0)

    if n_class_0 < 2 or n_class_1 < 2:
        raise ValueError("Insufficient samples per class for stratified splitting.")

    # Execute stratified split once with deterministic random_state
    # stratify=y ensures that the proportion of classes in the training and testing sets is the same as in the original dataset
    # We pass np.arange(len(y)) to get the indices of the test set, which we can use to identify the corresponding metadata and error values
    X_train, X_test, y_train, y_test, _, X_err_test, _, idx_test = train_test_split(
        X,
        y,
        X_err,
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info("Stratified split completed -> Train: %d, Test: %d (test_size=%.2f, random_state=%d)",
                len(X_train), len(X_test), test_size, random_state)

    # Balance the training set if requested
    if balanced:
        X_train, y_train = create_balanced_dataset(
            X_train,
            y_train,
            samples_per_class=samples_per_class,
            random_state=random_state,
        )

    metadata_test = metadata.iloc[idx_test].reset_index(drop=True)
    logger.info(
        "Final training split: %s (Class 0: %d, Class 1: %d)",
        X_train.shape,
        int((y_train == 0).sum()),
        int((y_train == 1).sum()),
    )
    logger.info("Final test split: %s (Class 0: %d, Class 1: %d)",
                X_test.shape, int((y_test == 0).sum()), int((y_test == 1).sum()))

    return X_train, X_test, y_train, y_test, metadata_test, X_err_test


def build_pipeline(
    standardize: bool = False,
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    max_features: Union[str, int, float] = "sqrt",
    bootstrap: bool = True,
    class_weight: Optional[Union[str, Dict[int, float]]] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    oob_score: bool = True,
) -> Pipeline:
    """Constructs a scikit-learn Pipeline encapsulating preprocessing and Random Forest.
    A Pipeline is used to encapsulate sequential transformations and the estimator
    into a single object, ensuring reproducibility and preventing data leakage
    between training and test splits.

    Args:
        standardize: Whether to include StandardScaler in the pipeline.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of the individual decision trees.
        min_samples_leaf: Minimum number of samples required at a leaf node.
        max_features: Number of features to consider when looking for the best split.
        bootstrap: Whether bootstrap samples are used when building trees.
        class_weight: Weights associated with classes ('balanced', 'balanced_subsample', or dict).
        random_state: Controls the randomness of the bootstrapping and tree splits.
        n_jobs: Number of parallel jobs to run (-1 uses all available processors).
        oob_score: Whether to use out-of-bag samples to estimate generalization score.

    Returns:
        A configured scikit-learn `Pipeline` instance.
    """
    steps = []
    if standardize:
        logger.info("Pipeline configuration: StandardScaler enabled.")
        steps.append(("scaler", StandardScaler()))
    else:
        logger.info("Pipeline configuration: StandardScaler disabled (raw features).")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        oob_score=oob_score,
    )
    steps.append(("classifier", rf))

    return Pipeline(steps=steps)


def train_and_evaluate_model(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, Any], str]:
    """Trains the model pipeline and computes validation metrics with threshold tuning.

    Calculates Out-of-Bag (OOB) score, ROC-AUC, optimal decision threshold using
    Youden's J statistic (J = TPR - FPR), class-specific and aggregated Precision,
    confusion matrix, and classification report.

    Args:
        pipeline: scikit-learn Pipeline instance containing the classifier.
        X_train: Training features.
        y_train: Training ground-truth labels.
        X_test: Testing features.
        y_test: Testing ground-truth labels.

    Returns:
        A tuple containing:
            - y_pred_best: Predictions at the optimal threshold.
            - proba: Predicted probabilities for the positive class (Planet).
            - best_thresh: Optimal probability decision threshold.
            - roc_data: Tuple of (fpr, tpr, thresholds) from ROC curve calculation.
            - metrics: Dictionary summarizing evaluation scores.
            - report_text: Formatted classification report string.
    """
    logger.info("=" * 70)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info("Fitting pipeline on training data (shape: %s)...", X_train.shape)

    pipeline.fit(X_train, y_train)

    # Compute OOB score if requested (only possible if bootstrap=True)
    rf_step = pipeline.named_steps.get("classifier")
    oob_score_val: Optional[float] = None
    if rf_step and getattr(rf_step, "oob_score", False) and hasattr(rf_step, "oob_score_"):
        oob_score_val = float(rf_step.oob_score_)
        logger.info("Out-of-Bag (OOB) Score: %.4f", oob_score_val)

    logger.info("=" * 70)
    logger.info("EVALUATION & THRESHOLD OPTIMIZATION")
    logger.info("=" * 70)

    proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, proba)

    # Youden's J statistic to identify optimal operating point on ROC curve
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(np.clip(thresholds[best_idx], 0.0, 1.0))

    y_pred_default = (proba >= 0.5).astype(int)
    y_pred_best = (proba >= best_thresh).astype(int)

    auc_score = float(roc_auc_score(y_test, proba))
    acc_default = float(accuracy_score(y_test, y_pred_default))
    acc_best = float(accuracy_score(y_test, y_pred_best))

    # Precision calculations (per-class, macro, weighted)
    prec_planet = float(precision_score(y_test, y_pred_best, pos_label=1, zero_division=0))
    prec_non_planet = float(precision_score(y_test, y_pred_best, pos_label=0, zero_division=0))
    prec_macro = float(precision_score(y_test, y_pred_best, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_test, y_pred_best, average="weighted", zero_division=0))

    # Recall and F1 metrics
    rec_planet = float(recall_score(y_test, y_pred_best, pos_label=1, zero_division=0))
    rec_non_planet = float(recall_score(y_test, y_pred_best, pos_label=0, zero_division=0))
    f1_planet = float(f1_score(y_test, y_pred_best, pos_label=1, zero_division=0))
    f1_non_planet = float(f1_score(y_test, y_pred_best, pos_label=0, zero_division=0))

    cm = confusion_matrix(y_test, y_pred_best)
    report_text = classification_report(
        y_test,
        y_pred_best,
        target_names=["Non-Planet", "Planet"],
        digits=4,
        zero_division=0,
    )

    logger.info("Optimal threshold: %.4f (Accuracy: %.2f%%, Planet Precision: %.2f%%, AUC: %.4f)",
                best_thresh, acc_best * 100, prec_planet * 100, auc_score)
    logger.info("Confusion Matrix:\n%s", cm)
    logger.info("Classification Report:\n%s", report_text)

    metrics: Dict[str, Any] = {
        "auc_roc": auc_score,
        "oob_score": oob_score_val,
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
    }

    return y_pred_best, proba, best_thresh, (fpr, tpr, thresholds), metrics, report_text


def save_report(
    report_path: Union[str, Path],
    model_cfg: Dict[str, Any],
    metrics: Dict[str, Any],
    classification_report_str: str,
) -> Path:
    """Saves model hyperparameters, confusion matrix, and evaluation metrics to a clean report.

    Writes strictly the hyperparameters, confusion matrix, precision metrics, and
    classification report without timestamps, full YAML dumps, or irrelevant metadata.

    Args:
        report_path: Destination path for the report file.
        model_cfg: Dictionary containing the model hyperparameters used in the run.
        metrics: Dictionary containing calculated evaluation metrics.
        classification_report_str: Formatted scikit-learn classification report string.

    Returns:
        Path pointing to the written report file.

    Raises:
        OSError: If writing to the file fails.
    """
    raw_path_str = str(report_path)
    # Prevent duplicate subdirectory prefix if executed from inside random_forest/
    if Path.cwd().name == "random_forest" and raw_path_str.startswith("random_forest/"):
        path = Path(raw_path_str[len("random_forest/") :])
    else:
        path = Path(report_path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        cm = metrics.get("confusion_matrix", np.array([[0, 0], [0, 0]]))
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        oob_val = metrics.get("oob_score")
        oob_display = f"{oob_val:.4f}" if oob_val is not None else "N/A"

        lines = [
            "=" * 80,
            "RANDOM FOREST EVALUATION REPORT",
            "=" * 80,
            "",
            "MODEL HYPERPARAMETERS:",
            "-" * 80,
            f"  n_estimators      : {model_cfg.get('n_estimators')}",
            f"  max_depth         : {model_cfg.get('max_depth')}",
            f"  min_samples_leaf  : {model_cfg.get('min_samples_leaf')}",
            f"  max_features      : {model_cfg.get('max_features')}",
            f"  bootstrap         : {model_cfg.get('bootstrap')}",
            f"  class_weight      : {model_cfg.get('class_weight')}",
            f"  random_state      : {model_cfg.get('random_state')}",
            f"  standardize       : {model_cfg.get('standardize')}",
            f"  oob_score         : {model_cfg.get('oob_score')}",
            f"  n_jobs            : {model_cfg.get('n_jobs')}",
            "",
            "EVALUATION METRICS & PRECISION:",
            "-" * 80,
            f"  Precision (Planet / Positive Class)     : {metrics.get('precision_planet', 0.0):.4f} ({metrics.get('precision_planet', 0.0) * 100:.2f}%)",
            f"  Precision (Non-Planet / Negative Class) : {metrics.get('precision_non_planet', 0.0):.4f} ({metrics.get('precision_non_planet', 0.0) * 100:.2f}%)",
            f"  Precision (Macro Average)               : {metrics.get('precision_macro', 0.0):.4f} ({metrics.get('precision_macro', 0.0) * 100:.2f}%)",
            f"  Precision (Weighted Average)            : {metrics.get('precision_weighted', 0.0):.4f} ({metrics.get('precision_weighted', 0.0) * 100:.2f}%)",
            f"  Accuracy (at Optimal Threshold)         : {metrics.get('accuracy_optimal', 0.0) * 100:.2f}%",
            f"  Accuracy (at Default Threshold 0.5000)  : {metrics.get('accuracy_default', 0.0) * 100:.2f}%",
            f"  Optimal Decision Threshold (Youden's J) : {metrics.get('optimal_threshold', 0.5):.4f} (Default: 0.5000)",
            f"  ROC-AUC Score                           : {metrics.get('auc_roc', 0.0):.4f}",
            f"  Out-of-Bag (OOB) Score                  : {oob_display}",
            f"  Recall (Planet)                         : {metrics.get('recall_planet', 0.0):.4f} ({metrics.get('recall_planet', 0.0) * 100:.2f}%)",
            f"  Recall (Non-Planet)                     : {metrics.get('recall_non_planet', 0.0):.4f} ({metrics.get('recall_non_planet', 0.0) * 100:.2f}%)",
            f"  F1-Score (Planet)                       : {metrics.get('f1_planet', 0.0):.4f}",
            f"  F1-Score (Non-Planet)                   : {metrics.get('f1_non_planet', 0.0):.4f}",
            "",
            "CONFUSION MATRIX (Evaluated at Optimal Threshold):",
            "-" * 80,
            "                       Predicted Non-Planet    Predicted Planet",
            f"  Actual Non-Planet :        {tn:<18}    {fp:<18}",
            f"  Actual Planet     :        {fn:<18}    {tp:<18}",
            "",
            "DETAILED CLASSIFICATION REPORT (Optimal Threshold):",
            "-" * 80,
            classification_report_str.rstrip(),
            "=" * 80,
        ]

        report_content = "\n".join(lines) + "\n"
        path.write_text(report_content, encoding="utf-8")
        logger.info("Evaluation report saved successfully to: %s", path.resolve())
        return path
    except Exception as exc:
        logger.error("Failed to write evaluation report to '%s': %s", path, exc)
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for the Random Forest pipeline.

    Args:
        argv: Sequence of argument strings to parse (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace containing the configuration file path.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate a single Random Forest configuration for exoplanet transit detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="config_path",
        type=str,
        default=None,
        help="Path to YAML configuration file (e.g., config_1.yaml). If omitted, default configuration is used.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main execution orchestrator for single-configuration Random Forest evaluation.

    Args:
        argv: Optional sequence of CLI arguments for programmatic invocation.

    Returns:
        Process exit code (0 for success, non-zero for error).
    """
    args = parse_args(argv)

    try:
        # Load configuration (merging user YAML over DEFAULT_CONFIG)
        config = load_config(args.config_path)

        # Configure logging according to effective configuration
        log_level = config.get("logging", {}).get("level", "INFO")
        setup_logging(log_level)

        logger.info("Initializing Exoplanet Transit Random Forest Pipeline...")

        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})
        output_cfg = config.get("output", {})

        logger.info("Configuration loaded: n_estimators=%s, max_depth=%s, random_state=%s",
                    model_cfg.get("n_estimators"), model_cfg.get("max_depth"), model_cfg.get("random_state"))

        # Load dataset with deterministic split
        X_train, X_test, y_train, y_test, metadata_test, _ = load_data(
            csv_path=data_cfg.get("csv_path", "tess_data.csv"),
            n_bins=data_cfg.get("n_bins", 1000),
            balanced=data_cfg.get("balanced", True),
            test_size=data_cfg.get("test_size", 0.2),
            samples_per_class=data_cfg.get("samples_per_class", 350),
            random_state=data_cfg.get("random_state", 42),
        )

        # Build Pipeline
        pipeline = build_pipeline(
            standardize=model_cfg.get("standardize", False),
            n_estimators=model_cfg.get("n_estimators", 500),
            max_depth=model_cfg.get("max_depth", None),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
            max_features=model_cfg.get("max_features", "sqrt"),
            bootstrap=model_cfg.get("bootstrap", True),
            class_weight=model_cfg.get("class_weight", None),
            random_state=model_cfg.get("random_state", 42),
            n_jobs=model_cfg.get("n_jobs", -1),
            oob_score=model_cfg.get("oob_score", True),
        )

        # Train and evaluate
        _, _, _, _, metrics, report_str = train_and_evaluate_model(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Save clean report
        report_path = output_cfg.get("report_path", "random_forest/results/report.txt")
        saved_path = save_report(
            report_path=report_path,
            model_cfg=model_cfg,
            metrics=metrics,
            classification_report_str=report_str,
        )

        logger.info("Random Forest execution completed successfully. Output: %s", saved_path)
        return 0

    except Exception as exc:
        logger.exception("Execution halted due to an unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
