"""Random Forest detection and classification module for exoplanet transits.

This module provides an end-to-end Machine Learning pipeline and detection class
for binary classification of exoplanetary transit light curves (Planet vs Non-Planet).
It supports configuration via YAML parameter files, deterministic data splits,
photometric data augmentation, scikit-learn Pipeline construction, optimal decision
threshold tuning via Youden's J statistic, and comprehensive evaluation statistics.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
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


# Centralized default configuration
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
        "report_path": "random_forest/results/report.txt",
    },
    "logging": {
        "level": "INFO",
    },
}


def deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merges an override dictionary into a base configuration dictionary.

    Args:
        base: The base/default configuration dictionary.
        override: The overriding dictionary.

    Returns:
        A new deeply merged dictionary.
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


def normalize_user_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes various YAML schemas into standard {data, model, output, logging} structure.

    Supports:
      1. Standard structure: {data: {...}, model: {...}, output: {...}}
      2. Nested under 'classifier' or 'rf' or 'random_forest':
         {classifier: {type: 'rf', data: {...}, model: {...}}}
         or {classifier: {n_estimators: 100, csv_path: '...'}}
      3. Flat parameters at root level: {csv_path: '...', n_estimators: 100, ...}
    """
    config: Dict[str, Any] = {"data": {}, "model": {}, "output": {}, "logging": {}}

    # Check for wrapper sections
    target_dict = raw_config
    for key in ("classifier", "rf", "random_forest", "detection"):
        if key in raw_config and isinstance(raw_config[key], dict):
            target_dict = raw_config[key]
            break

    # If target has structured keys, merge them
    for section in ("data", "model", "output", "logging"):
        if section in target_dict and isinstance(target_dict[section], dict):
            config[section].update(target_dict[section])
        elif section in raw_config and isinstance(raw_config[section], dict):
            config[section].update(raw_config[section])

    # Map possible flat / synonym keys
    data_keys = {
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
    model_keys = {
        "standardize": "standardize",
        "n_estimators": "n_estimators",
        "max_depth": "max_depth",
        "min_samples_leaf": "min_samples_leaf",
        "max_features": "max_features",
        "bootstrap": "bootstrap",
        "class_weight": "class_weight",
        "random_state": "random_state",
        "n_jobs": "n_jobs",
        "oob_score": "oob_score",
    }
    output_keys = {
        "report_path": "report_path",
        "output_path": "report_path",
        "output_file": "report_path",
        "report_file": "report_path",
    }

    # Extract flat keys from target_dict or raw_config
    for source in (raw_config, target_dict):
        for k, v in source.items():
            if k in data_keys and data_keys[k] not in config["data"]:
                config["data"][data_keys[k]] = v
            elif k in model_keys and model_keys[k] not in config["model"]:
                config["model"][model_keys[k]] = v
            elif k in output_keys and output_keys[k] not in config["output"]:
                config["output"][output_keys[k]] = v

    # Clean up empty sections so deep_merge will use defaults
    clean_config = {k: v for k, v in config.items() if v}
    return deep_merge(DEFAULT_CONFIG, clean_config)


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


def create_balanced_dataset(
    X: np.ndarray,
    y: np.ndarray,
    samples_per_class: int = 350,
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
    """
    rng = np.random.default_rng(random_state)

    X0 = X[y == 0]
    X1 = X[y == 1]

    if len(X0) == 0 or len(X1) == 0:
        raise ValueError("Both classes (0 and 1) must be present in the dataset.")

    logger.info("Balancing dataset (original Class 0: %d, Class 1: %d)", len(X0), len(X1))

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

    X_bal = np.vstack([X0_bal, X1_bal])
    y_bal = np.concatenate([np.zeros(samples_per_class), np.ones(samples_per_class)])

    shuffle_idx = rng.permutation(len(X_bal))
    logger.info("Balanced dataset created: %d samples per class (Total: %d)", samples_per_class, len(X_bal))
    return X_bal[shuffle_idx], y_bal[shuffle_idx]


class RandomForestDetection:
    """Random Forest detection and classification class for exoplanet transit light curves.

    Parameters
    ----------
    params : dict, str, Path, or Parameters, optional
        Configuration dictionary, path to YAML configuration file, or Parameters instance.
    **kwargs : Any
        Additional keyword arguments overriding configuration values.
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
                # Try finding relative to repo root
                resolved = resolve_file_path(self.config_path)
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

        # Merge kwargs into user_config if any
        if kwargs:
            user_config = deep_merge(user_config, kwargs)

        self.config = normalize_user_config(user_config)
        self.data_config = self.config["data"]
        self.model_config = self.config["model"]
        self.output_config = self.config["output"]

        # Configure logger
        log_level = self.config.get("logging", {}).get("level", "INFO")
        setup_logging(log_level)

        # Data split holders
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.metadata_test: Optional[pd.DataFrame] = None
        self.X_err_test: Optional[np.ndarray] = None

        # Evaluation results
        self.pipeline: Optional[Pipeline] = None
        self.optimal_threshold: float = 0.5
        self.metrics: Dict[str, Any] = {}
        self.report_text: str = ""
        self.roc_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def load_data(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        n_bins: Optional[int] = None,
        balanced: Optional[bool] = None,
        test_size: Optional[float] = None,
        samples_per_class: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
        """Loads dataset from CSV file, performs stratified split and optional balancing."""
        csv_p = csv_path or self.data_config.get("csv_path", "tess_data.csv")
        base_dir = self.config_path.parent if self.config_path else None
        resolved_csv = resolve_file_path(csv_p, base_dir=base_dir)

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
            test_size=test_size_val,
            random_state=rs_val,
            stratify=y,
        )

        logger.info(
            "Stratified split -> Train: %d, Test: %d (test_size=%.2f, random_state=%d)",
            len(X_train),
            len(X_test),
            test_size_val,
            rs_val,
        )

        if balanced_val:
            X_train, y_train = create_balanced_dataset(
                X_train,
                y_train,
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

    def build_pipeline(self) -> Pipeline:
        """Constructs scikit-learn Pipeline with preprocessing and Random Forest classifier."""
        steps = []
        if self.model_config.get("standardize", False):
            logger.info("Pipeline: StandardScaler enabled.")
            steps.append(("scaler", StandardScaler()))
        else:
            logger.info("Pipeline: StandardScaler disabled (raw flux).")

        rf = SklearnRandomForestClassifier(
            n_estimators=self.model_config.get("n_estimators", 500),
            max_depth=self.model_config.get("max_depth", None),
            min_samples_leaf=self.model_config.get("min_samples_leaf", 1),
            max_features=self.model_config.get("max_features", "sqrt"),
            bootstrap=self.model_config.get("bootstrap", True),
            class_weight=self.model_config.get("class_weight", None),
            random_state=self.model_config.get("random_state", 42),
            n_jobs=self.model_config.get("n_jobs", -1),
            oob_score=self.model_config.get("oob_score", True),
        )
        steps.append(("classifier", rf))
        self.pipeline = Pipeline(steps=steps)
        return self.pipeline

    def fit(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> RandomForestDetection:
        """Fits the Random Forest pipeline on provided or loaded training data."""
        if X is None or y is None:
            if self.X_train is None or self.y_train is None:
                self.load_data()
            X = self.X_train
            y = self.y_train

        if self.pipeline is None:
            self.build_pipeline()

        logger.info(
            "Fitting Random Forest pipeline (n_estimators=%s, max_depth=%s, random_state=%s)...",
            self.model_config.get("n_estimators"),
            self.model_config.get("max_depth"),
            self.model_config.get("random_state"),
        )
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for input features."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted. Call fit() before predict_proba().")
        return self.pipeline.predict_proba(X)

    def predict(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Predicts binary class labels using the optimal or specified threshold."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        proba = self.predict_proba(X)[:, 1]
        return (proba >= thresh).astype(int)

    def evaluate(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Evaluates the model on test data, tuning the decision threshold via Youden's J."""
        if X is None or y is None:
            if self.X_test is None or self.y_test is None:
                self.load_data()
            X = self.X_test
            y = self.y_test

        if self.pipeline is None:
            self.fit()

        rf_step = self.pipeline.named_steps.get("classifier")
        oob_val: Optional[float] = None
        if rf_step and getattr(rf_step, "oob_score", False) and hasattr(rf_step, "oob_score_"):
            oob_val = float(rf_step.oob_score_)
            logger.info("Out-of-Bag (OOB) generalization score: %.4f", oob_val)

        proba = self.pipeline.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, proba)

        # Youden's J statistic
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
            y,
            y_pred_best,
            target_names=["Non-Planet", "Planet"],
            digits=4,
            zero_division=0,
        )

        logger.info(
            "Evaluation complete -> Optimal threshold: %.4f | AUC: %.4f | Accuracy: %.2f%%",
            best_thresh,
            auc_score,
            acc_best * 100,
        )

        self.metrics = {
            "auc_roc": auc_score,
            "oob_score": oob_val,
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

        return self.metrics

    def display_statistics(self) -> None:
        """Displays formatted evaluation statistics to standard output using the logger."""
        if not self.metrics:
            self.evaluate()

        cm = self.metrics.get("confusion_matrix", np.array([[0, 0], [0, 0]]))
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        oob_val = self.metrics.get("oob_score")
        oob_display = f"{oob_val:.4f}" if oob_val is not None else "N/A"

        logger.info("=" * 80)
        logger.info("RANDOM FOREST EXOPLANET DETECTION STATISTICS")
        logger.info("=" * 80)
        logger.info("MODEL HYPERPARAMETERS:")
        logger.info("-" * 80)
        for k, v in self.model_config.items():
            logger.info("  %-18s: %s", k, v)
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
        logger.info("  Out-of-Bag (OOB) Score                  : %s", oob_display)
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
        """Saves evaluation report to the specified file path."""
        if not self.metrics:
            self.evaluate()

        dest_str = report_path or self.output_config.get(
            "report_path", "random_forest/results/report.txt"
        )
        dest_path = Path(dest_str)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        cm = self.metrics.get("confusion_matrix", np.array([[0, 0], [0, 0]]))
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        oob_val = self.metrics.get("oob_score")
        oob_display = f"{oob_val:.4f}" if oob_val is not None else "N/A"

        lines = [
            "=" * 80,
            "RANDOM FOREST EVALUATION REPORT",
            "=" * 80,
            "",
            "MODEL HYPERPARAMETERS:",
            "-" * 80,
        ]
        for k, v in self.model_config.items():
            lines.append(f"  {k:<18}: {v}")
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
            f"  Out-of-Bag (OOB) Score                  : {oob_display}",
            f"  Recall (Planet)                         : {self.metrics.get('recall_planet', 0.0):.4f} ({self.metrics.get('recall_planet', 0.0) * 100:.2f}%)",
            f"  Recall (Non-Planet)                     : {self.metrics.get('recall_non_planet', 0.0):.4f} ({self.metrics.get('recall_non_planet', 0.0) * 100:.2f}%)",
            f"  F1-Score (Planet)                       : {self.metrics.get('f1_planet', 0.0):.4f}",
            f"  F1-Score (Non-Planet)                   : {self.metrics.get('f1_non_planet', 0.0):.4f}",
            "",
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
        logger.info("Evaluation report saved successfully to: %s", dest_path)
        return dest_path

    def run(
        self,
        output_file: Optional[Union[str, Path]] = None,
        display: bool = True,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Runs complete detection pipeline: load, fit, evaluate, display, and save report."""
        self.load_data()
        self.build_pipeline()
        self.fit()
        self.evaluate()

        if display:
            self.display_statistics()

        if save:
            target_report = output_file or self.output_config.get("report_path")
            if target_report:
                self.save_report(target_report)

        return self.metrics


# Aliases for backwards compatibility and intuitive imports
RandomForestClassifier = RandomForestDetection
RandomForest = RandomForestDetection
