"""Abstract base class for exoplanet transit detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
import yaml

try:
    from daneel.parameters import Parameters
except ImportError:
    from ..parameters import Parameters

from ..utils import deep_merge, normalize_config, setup_logging
from .data_process import load_data, resolve_file_path

logger = logging.getLogger(__name__)


class BaseTransitDetector(ABC):
    """Abstract base detector class defining the lifecycle for transit classification models.

    Subclasses must implement:
      - :meth:`build_model`: Architecture / pipeline setup.
      - :meth:`fit`: Model training on provided or loaded data.
      - :meth:`predict_proba`: Probability estimation for positive class.

    Parameters
    ----------
    params : dict, str, Path, or Parameters, optional
        Configuration dictionary, path to YAML configuration file, or Parameters instance.
    **kwargs : Any
        Additional keyword arguments overriding configuration values.
    """

    model_name: str = "Transit Detector"
    DEFAULT_CONFIG: Dict[str, Any] = {}
    MODEL_KEYS: Dict[str, str] = {}
    WRAPPER_KEYS: Sequence[str] = ("classifier", "rf", "random_forest", "cnn", "detection")

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

        if kwargs:
            user_config = deep_merge(user_config, kwargs)

        self.config = self.normalize_config(user_config)
        self.data_config = self.config.get("data", {})
        self.model_config = self.config.get("model", {})
        self.output_config = self.config.get("output", {})

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
        self.optimal_threshold: float = 0.5
        self.metrics: Dict[str, Any] = {}
        self.report_text: str = ""
        self.roc_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def normalize_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes user configuration using class-level defaults and aliases."""
        return normalize_config(
            raw_config=user_config,
            default_config=self.DEFAULT_CONFIG,
            model_keys=self.MODEL_KEYS,
            wrapper_keys=self.WRAPPER_KEYS,
        )

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
        n_bins_val = n_bins if n_bins is not None else self.data_config.get("n_bins", 1000)
        balanced_val = balanced if balanced is not None else self.data_config.get("balanced", True)
        test_size_val = test_size if test_size is not None else self.data_config.get("test_size", 0.2)
        samples_val = (
            samples_per_class
            if samples_per_class is not None
            else self.data_config.get("samples_per_class", 350)
        )
        rs_val = random_state if random_state is not None else self.data_config.get("random_state", 42)

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.metadata_test,
            self.X_err_test,
        ) = load_data(
            csv_path=csv_p,
            n_bins=n_bins_val,
            balanced=balanced_val,
            test_size=test_size_val,
            samples_per_class=samples_val,
            random_state=rs_val,
            base_dir=base_dir,
        )
        return (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.metadata_test,
            self.X_err_test,
        )

    # Optional subclass hooks (Template Method pattern).
    # These let child classes inject model-specific metrics (e.g. RF's OOB score)
    # or extra summary sections (e.g. CNN's epoch curves) into the shared evaluate(),
    # display_statistics(), and save_report() lifecycle without copying code.

    def _is_fitted(self) -> bool:
        """Returns True if the underlying model is fitted."""
        return True

    def _ensure_fitted(self) -> None:
        """Ensures the model is fitted before prediction or evaluation."""
        if not self._is_fitted():
            self.fit()

    def hook_extra_metrics(self, metrics: Dict[str, Any]) -> None:
        """Called inside evaluate(). Subclasses can inject model-specific metrics (e.g. OOB score, loss curves)."""
        pass

    def hook_display_extra_metrics(self) -> None:
        """Called inside display_statistics(). Subclasses can print extra rows in the metrics table."""
        pass

    def hook_display_extra_sections(self) -> None:
        """Called inside display_statistics(). Subclasses can print standalone sections before the confusion matrix."""
        pass

    def hook_report_extra_metrics(self, lines: List[str]) -> None:
        """Called inside save_report(). Subclasses can append extra metric rows to report text."""
        pass

    def hook_report_extra_sections(self, lines: List[str]) -> None:
        """Called inside save_report(). Subclasses can append standalone sections to report text."""
        pass

    # Abstract methods
    @abstractmethod
    def build_model(self) -> Any:
        """Constructs or initializes the underlying model architecture or pipeline."""

    @abstractmethod
    def fit(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> BaseTransitDetector:
        """Fits or trains the model on training data."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts positive class probabilities for input features."""

    def predict(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Predicts binary class labels using the optimal or specified threshold."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        proba = self.predict_proba(X)
        if proba.ndim == 2:
            proba = proba[:, 1]
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

        self._ensure_fitted()

        proba = self.predict_proba(X)
        if proba.ndim == 2:
            proba = proba[:, 1]

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

        self.hook_extra_metrics(self.metrics)
        return self.metrics

    def display_statistics(self) -> None:
        """Displays formatted evaluation statistics to standard output using the logger."""
        if not self.metrics:
            self.evaluate()

        cm = self.metrics.get("confusion_matrix", np.array([[0, 0], [0, 0]]))
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        logger.info("=" * 80)
        logger.info("%s EXOPLANET DETECTION STATISTICS", self.model_name.upper())
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

        self.hook_display_extra_metrics()

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

        self.hook_display_extra_sections()

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
            "report_path", f"{self.model_name.lower().replace(' ', '_')}/results/report.txt"
        )
        dest_path = Path(dest_str)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        cm = self.metrics.get("confusion_matrix", np.array([[0, 0], [0, 0]]))
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        lines = [
            "=" * 80,
            f"{self.model_name.upper()} EVALUATION REPORT",
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
        ])

        self.hook_report_extra_metrics(lines)

        lines.extend([
            f"  Recall (Planet)                         : {self.metrics.get('recall_planet', 0.0):.4f} ({self.metrics.get('recall_planet', 0.0) * 100:.2f}%)",
            f"  Recall (Non-Planet)                     : {self.metrics.get('recall_non_planet', 0.0):.4f} ({self.metrics.get('recall_non_planet', 0.0) * 100:.2f}%)",
            f"  F1-Score (Planet)                       : {self.metrics.get('f1_planet', 0.0):.4f}",
            f"  F1-Score (Non-Planet)                   : {self.metrics.get('f1_non_planet', 0.0):.4f}",
            "",
        ])

        self.hook_report_extra_sections(lines)

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
        logger.info("%s evaluation report saved successfully to: %s", self.model_name, dest_path)
        return dest_path

    def run(
        self,
        output_file: Optional[Union[str, Path]] = None,
        display: bool = True,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Runs complete detection pipeline: load, build, fit, evaluate, display, and save report."""
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
