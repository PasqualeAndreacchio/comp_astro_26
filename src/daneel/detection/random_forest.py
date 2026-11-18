"""Random Forest detection and classification module for exoplanet transits.

This module provides an end-to-end Machine Learning pipeline and detection class
for binary classification of exoplanetary transit light curves (Planet vs Non-Planet).
It supports configuration via YAML parameter files, deterministic data splits,
photometric data augmentation, scikit-learn Pipeline construction, optimal decision
threshold tuning via Youden's J statistic, and comprehensive evaluation statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from daneel.parameters import Parameters
except ImportError:
    from ..parameters import Parameters

from .base import BaseTransitDetector

logger = logging.getLogger(__name__)


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

MODEL_KEYS: Dict[str, str] = {
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


class RandomForestDetection(BaseTransitDetector):
    """Random Forest detection and classification class for exoplanet transit light curves.

    Parameters
    ----------
    params : dict, str, Path, or Parameters, optional
        Configuration dictionary, path to YAML configuration file, or Parameters instance.
    **kwargs : Any
        Additional keyword arguments overriding configuration values.
    """

    model_name: str = "Random Forest"
    DEFAULT_CONFIG: Dict[str, Any] = DEFAULT_CONFIG
    MODEL_KEYS: Dict[str, str] = MODEL_KEYS

    def __init__(
        self,
        params: Optional[Union[Dict[str, Any], str, Path, Parameters]] = None,
        **kwargs: Any,
    ):
        super().__init__(params, **kwargs)
        self.pipeline: Optional[Pipeline] = None

    def _is_fitted(self) -> bool:
        """Returns True if the pipeline has been constructed and fitted."""
        return self.pipeline is not None

    def build_model(self) -> Pipeline:
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

    # Alias for backwards compatibility
    build_pipeline = build_model

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

    # Subclass hooks for Out-of-Bag (OOB) score reporting
    def hook_extra_metrics(self, metrics: Dict[str, Any]) -> None:
        """Extracts and logs Out-of-Bag (OOB) generalization score if available."""
        rf_step = self.pipeline.named_steps.get("classifier") if self.pipeline else None
        oob_val: Optional[float] = None
        if rf_step and getattr(rf_step, "oob_score", False) and hasattr(rf_step, "oob_score_"):
            oob_val = float(rf_step.oob_score_)
            logger.info("Out-of-Bag (OOB) generalization score: %.4f", oob_val)
        metrics["oob_score"] = oob_val

    def hook_display_extra_metrics(self) -> None:
        """Displays Out-of-Bag (OOB) score in console statistics."""
        oob_val = self.metrics.get("oob_score")
        oob_display = f"{oob_val:.4f}" if oob_val is not None else "N/A"
        logger.info("  Out-of-Bag (OOB) Score                  : %s", oob_display)

    def hook_report_extra_metrics(self, lines: List[str]) -> None:
        """Appends Out-of-Bag (OOB) score to report lines."""
        oob_val = self.metrics.get("oob_score")
        oob_display = f"{oob_val:.4f}" if oob_val is not None else "N/A"
        lines.append(f"  Out-of-Bag (OOB) Score                  : {oob_display}")


# Aliases for backwards compatibility and intuitive imports
RandomForestClassifier = RandomForestDetection
RandomForest = RandomForestDetection
