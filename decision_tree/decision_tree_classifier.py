"""Object-oriented Decision Tree detection pipeline for exoplanet transits (Task E).

Provides :class:`DecisionTreeDetection`, inheriting from
:class:`~daneel.detection.base.BaseTransitDetector` so that data loading,
stratified splitting, balanced augmentation, Youden's J threshold optimization,
evaluation metrics, logging, and report generation are completely unified with
the Random Forest (Task D) and CNN (Tasks F/H) detectors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

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
        "name": "Decision Tree",
        "standardize": False,
        "criterion": "gini",
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 10,
        "max_features": None,
        "class_weight": None,
        "ccp_alpha": 0.0,
        "random_state": 42,
    },
    "output": {
        "report_path": "decision_tree/results/report_decision_tree.txt",
    },
    "logging": {
        "level": "INFO",
    },
}

MODEL_KEYS: Dict[str, str] = {
    "name": "name",
    "standardize": "standardize",
    "criterion": "criterion",
    "max_depth": "max_depth",
    "min_samples_split": "min_samples_split",
    "min_samples_leaf": "min_samples_leaf",
    "max_features": "max_features",
    "class_weight": "class_weight",
    "ccp_alpha": "ccp_alpha",
    "random_state": "random_state",
}


class DecisionTreeDetection(BaseTransitDetector):
    """Decision Tree detection and classification class for exoplanet transit light curves.

    Parameters
    ----------
    params : dict, str, Path, or Parameters, optional
        Configuration dictionary, path to YAML configuration file, or Parameters instance.
    **kwargs : Any
        Additional keyword arguments overriding configuration values.
    """

    model_name: str = "Decision Tree"
    DEFAULT_CONFIG: Dict[str, Any] = DEFAULT_CONFIG
    MODEL_KEYS: Dict[str, str] = MODEL_KEYS
    WRAPPER_KEYS: Sequence[str] = ("classifier", "dt", "decision_tree", "detection")

    def __init__(
        self,
        params: Optional[Union[Dict[str, Any], str, Path, Parameters]] = None,
        **kwargs: Any,
    ):
        super().__init__(params, **kwargs)
        # Normalize string "None" for max_depth if loaded from YAML
        if self.model_config.get("max_depth") == "None":
            self.model_config["max_depth"] = None
        if self.model_config.get("max_features") == "None":
            self.model_config["max_features"] = None
        if self.model_config.get("class_weight") == "None":
            self.model_config["class_weight"] = None

        self.pipeline: Optional[Pipeline] = None
        self._fitted: bool = False

    def _is_fitted(self) -> bool:
        """Returns True if the pipeline has been constructed and fitted."""
        return self.pipeline is not None and self._fitted

    def build_model(self) -> Pipeline:
        """Constructs scikit-learn Pipeline with optional scaling and Decision Tree classifier."""
        steps: List[tuple[str, Any]] = []
        if self.model_config.get("standardize", False):
            logger.info("Pipeline: StandardScaler enabled.")
            steps.append(("scaler", StandardScaler()))
        else:
            logger.info("Pipeline: StandardScaler disabled (raw flux).")

        max_depth = self.model_config.get("max_depth", 10)
        if max_depth is not None:
            max_depth = int(max_depth)

        tree = SklearnDecisionTreeClassifier(
            criterion=str(self.model_config.get("criterion", "gini")),
            max_depth=max_depth,
            min_samples_split=int(self.model_config.get("min_samples_split", 2)),
            min_samples_leaf=int(self.model_config.get("min_samples_leaf", 10)),
            max_features=self.model_config.get("max_features", None),
            class_weight=self.model_config.get("class_weight", None),
            ccp_alpha=float(self.model_config.get("ccp_alpha", 0.0)),
            random_state=int(self.model_config.get("random_state", 42)),
        )
        steps.append(("classifier", tree))
        self.pipeline = Pipeline(steps=steps)
        self._fitted = False
        return self.pipeline

    # Alias for backwards compatibility
    build_pipeline = build_model

    def fit(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> DecisionTreeDetection:
        """Fits the Decision Tree pipeline on provided or loaded training data."""
        if X is None or y is None:
            if self.X_train is None or self.y_train is None:
                self.load_data()
            X = self.X_train
            y = self.y_train

        if self.pipeline is None:
            self.build_model()

        logger.info(
            "Fitting Decision Tree pipeline (criterion=%s, max_depth=%s, "
            "min_samples_leaf=%s, random_state=%s)...",
            self.model_config.get("criterion"),
            self.model_config.get("max_depth"),
            self.model_config.get("min_samples_leaf"),
            self.model_config.get("random_state"),
        )
        self.pipeline.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for input features."""
        if self.pipeline is None or not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() before predict_proba().")
        return self.pipeline.predict_proba(X)

    # Subclass hooks for Decision Tree structure & default-threshold summary
    def hook_extra_metrics(self, metrics: Dict[str, Any]) -> None:
        """Attaches fitted tree structure metrics and default-threshold (0.5000) breakdown."""
        tree_step = self.pipeline.named_steps.get("classifier") if self.pipeline else None
        tree_depth: Optional[int] = None
        n_leaves: Optional[int] = None
        if tree_step is not None and self._fitted:
            tree_depth = int(tree_step.get_depth())
            n_leaves = int(tree_step.get_n_leaves())
            logger.info(
                "Fitted Decision Tree structure -> depth=%d | leaves=%d",
                tree_depth,
                n_leaves,
            )

        metrics["config_name"] = str(self.model_config.get("name", self.model_name))
        metrics["tree_depth"] = tree_depth
        metrics["n_leaves"] = n_leaves
        metrics["n_train"] = int(len(self.X_train)) if self.X_train is not None else 0
        metrics["n_test"] = int(len(self.X_test)) if self.X_test is not None else 0

        if self.X_test is not None and self.y_test is not None and self._fitted:
            proba = self.predict_proba(self.X_test)
            if proba.ndim == 2:
                proba = proba[:, 1]
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
        """Displays fitted tree depth and leaf count in console statistics."""
        depth = self.metrics.get("tree_depth")
        leaves = self.metrics.get("n_leaves")
        logger.info(
            "  Fitted Tree Depth                       : %s",
            depth if depth is not None else "N/A",
        )
        logger.info(
            "  Number of Leaves                        : %s",
            leaves if leaves is not None else "N/A",
        )

    def hook_report_extra_metrics(self, lines: List[str]) -> None:
        """Appends fitted tree depth and leaf count to report metrics."""
        depth = self.metrics.get("tree_depth")
        leaves = self.metrics.get("n_leaves")
        lines.append(
            f"  Fitted Tree Depth                       : {depth if depth is not None else 'N/A'}"
        )
        lines.append(
            f"  Number of Leaves                        : {leaves if leaves is not None else 'N/A'}"
        )

    def hook_report_extra_sections(self, lines: List[str]) -> None:
        """Appends default-threshold (0.5000) summary to the report."""
        if "precision_planet_default" not in self.metrics:
            return
        cm_def = self.metrics.get("confusion_matrix_default", np.array([[0, 0], [0, 0]]))
        tn_d, fp_d, fn_d, tp_d = cm_def.ravel() if cm_def.size == 4 else (0, 0, 0, 0)
        lines.extend([
            "DEFAULT THRESHOLD SUMMARY (p >= 0.5000):",
            "-" * 80,
            f"  Precision (Planet / Positive Class)     : {self.metrics['precision_planet_default']:.4f} ({self.metrics['precision_planet_default'] * 100:.2f}%)",
            f"  Precision (Non-Planet / Negative Class) : {self.metrics['precision_non_planet_default']:.4f} ({self.metrics['precision_non_planet_default'] * 100:.2f}%)",
            f"  Precision (Macro Average)               : {self.metrics['precision_macro_default']:.4f} ({self.metrics['precision_macro_default'] * 100:.2f}%)",
            f"  Recall (Planet)                         : {self.metrics['recall_planet_default']:.4f} ({self.metrics['recall_planet_default'] * 100:.2f}%)",
            f"  Confusion Matrix (TN / FP / FN / TP)    : {tn_d} / {fp_d} / {fn_d} / {tp_d}",
            "",
        ])


# Aliases for backwards compatibility and intuitive imports
DecisionTreeTransitDetector = DecisionTreeDetection
DecisionTreeClassifier = DecisionTreeDetection
DecisionTree = DecisionTreeDetection
