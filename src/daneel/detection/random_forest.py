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
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

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
        # Early Stopping configuration
        "early_stopping": {
            "enabled": True,
            "step": 20,
            "patience": 5,
            "min_delta": 0.001,
            "monitor": "oob_score",
            "mode": "max",
            "restore_best": True,
        },
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
    "warm_start": "warm_start",
    "step": "step",
    "early_stopping": "early_stopping",
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

        es_cfg = self.model_config.get("early_stopping", {})
        warm_start_val = bool(
            es_cfg if isinstance(es_cfg, bool) else (es_cfg.get("enabled", True) if isinstance(es_cfg, dict) else False)
        ) or bool(self.model_config.get("warm_start", False))

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
            warm_start=warm_start_val,
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
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> RandomForestDetection:
        """Fits the Random Forest pipeline on provided or loaded training data.

        Supports YAML-configurable incremental Early Stopping using warm_start=True
        and live tqdm progress tracking of incremental estimator steps.

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

        if self.pipeline is None:
            self.build_pipeline()

        rf: SklearnRandomForestClassifier = self.pipeline.named_steps["classifier"]
        total_estimators = int(self.model_config.get("n_estimators", 500))

        es_cfg = self.model_config.get("early_stopping", {})
        if isinstance(es_cfg, bool):
            es_enabled = es_cfg
            es_cfg = {}
        elif isinstance(es_cfg, dict):
            es_enabled = es_cfg.get("enabled", True)
        else:
            es_enabled = False

        if es_enabled:
            step = int(es_cfg.get("step", 20))
            patience = int(es_cfg.get("patience", 5))
            min_delta = float(es_cfg.get("min_delta", 0.001))
            monitor = str(es_cfg.get("monitor", "oob_score"))
            mode = str(es_cfg.get("mode", "min" if "loss" in monitor else "max"))
            restore_best = bool(es_cfg.get("restore_best", True))

            # Ensure warm_start is active on the classifier
            rf.warm_start = True

            # If monitoring oob_score, ensure bootstrap and oob_score are enabled
            if "oob" in monitor:
                if getattr(rf, "bootstrap", True):
                    rf.oob_score = True
                else:
                    logger.warning(
                        "OOB score requested but bootstrap is False. Falling back to validation data."
                    )
                    monitor = "val_score"
                    mode = "max"

            # Generate milestone tree counts up to total_estimators
            if step <= 0 or step >= total_estimators:
                trees_steps = [total_estimators]
            else:
                trees_steps = list(range(step, total_estimators + 1, step))
                if trees_steps[-1] < total_estimators:
                    trees_steps.append(total_estimators)

            logger.info(
                "Fitting Random Forest with incremental Early Stopping | total_estimators=%d | "
                "step=%d | patience=%d | monitor=%s",
                total_estimators,
                step,
                patience,
                monitor,
            )

            best_score = float("inf") if mode == "min" else float("-inf")
            best_state = None
            best_trees = 0
            patience_counter = 0

            pbar = tqdm(trees_steps, desc="Random Forest Training", unit="step")
            for current_trees in pbar:
                rf.n_estimators = current_trees
                self.pipeline.fit(X, y)

                # Evaluate current score
                current_score: Optional[float] = None
                if "oob" in monitor and getattr(rf, "oob_score", False) and hasattr(rf, "oob_score_"):
                    current_score = float(rf.oob_score_)
                elif X_val is not None and y_val is not None:
                    if "loss" in monitor:
                        from sklearn.metrics import log_loss
                        proba = self.pipeline.predict_proba(X_val)
                        current_score = float(log_loss(y_val, proba))
                    elif "auc" in monitor:
                        from sklearn.metrics import roc_auc_score
                        proba = self.pipeline.predict_proba(X_val)
                        if proba.ndim == 2:
                            proba = proba[:, 1]
                        current_score = float(roc_auc_score(y_val, proba))
                    else:
                        current_score = float(self.pipeline.score(X_val, y_val))
                elif hasattr(rf, "oob_score_") and rf.oob_score_ is not None:
                    current_score = float(rf.oob_score_)

                if current_score is not None:
                    if mode == "min":
                        improved = current_score < (best_score - min_delta)
                    else:
                        improved = current_score > (best_score + min_delta)

                    if improved:
                        best_score = current_score
                        best_trees = current_trees
                        best_state = copy.deepcopy(self.pipeline)
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    postfix = {
                        "trees": current_trees,
                        monitor: f"{current_score:.4f}",
                        f"best_{monitor}": f"{best_score:.4f}",
                        "pat": f"{patience_counter}/{patience}",
                    }
                else:
                    postfix = {"trees": current_trees}

                pbar.set_postfix(postfix)

                if patience_counter >= patience:
                    logger.info(
                        "Early stopping triggered at %d trees: '%s' did not improve for %d consecutive steps "
                        "(best %.4f at %d trees).",
                        current_trees,
                        monitor,
                        patience,
                        best_score,
                        best_trees,
                    )
                    pbar.close()
                    break

            if restore_best and best_state is not None:
                self.pipeline = best_state
                logger.info(
                    "Restored best Random Forest pipeline checkpoint with %d trees (%s: %.4f).",
                    best_trees,
                    monitor,
                    best_score,
                )
        else:
            logger.info(
                "Fitting Random Forest pipeline (n_estimators=%s, max_depth=%s, random_state=%s)...",
                self.model_config.get("n_estimators"),
                self.model_config.get("max_depth"),
                self.model_config.get("random_state"),
            )
            with tqdm(total=1, desc="Random Forest Training", unit="fit") as pbar:
                self.pipeline.fit(X, y)
                pbar.update(1)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for input features."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted. Call fit() before predict_proba().")
        return self.pipeline.predict_proba(X)

    # Subclass hooks for Out-of-Bag (OOB) score and fitted tree count reporting
    def hook_extra_metrics(self, metrics: Dict[str, Any]) -> None:
        """Extracts and logs Out-of-Bag (OOB) generalization score and fitted trees if available."""
        rf_step = self.pipeline.named_steps.get("classifier") if self.pipeline else None
        oob_val: Optional[float] = None
        if rf_step and getattr(rf_step, "oob_score", False) and hasattr(rf_step, "oob_score_"):
            oob_val = float(rf_step.oob_score_)
            logger.info("Out-of-Bag (OOB) generalization score: %.4f", oob_val)
        metrics["oob_score"] = oob_val
        if rf_step and hasattr(rf_step, "estimators_"):
            metrics["trees_fitted"] = len(rf_step.estimators_)

    def hook_display_extra_metrics(self) -> None:
        """Displays Out-of-Bag (OOB) score and retained tree count in console statistics."""
        oob_val = self.metrics.get("oob_score")
        oob_display = f"{oob_val:.4f}" if oob_val is not None else "N/A"
        logger.info("  Out-of-Bag (OOB) Score                  : %s", oob_display)
        if "trees_fitted" in self.metrics:
            logger.info("  Fitted Trees Count                      : %d", self.metrics["trees_fitted"])

    def hook_report_extra_metrics(self, lines: List[str]) -> None:
        """Appends Out-of-Bag (OOB) score and retained tree count to report lines."""
        oob_val = self.metrics.get("oob_score")
        oob_display = f"{oob_val:.4f}" if oob_val is not None else "N/A"
        lines.append(f"  Out-of-Bag (OOB) Score                  : {oob_display}")
        if "trees_fitted" in self.metrics:
            lines.append(f"  Fitted Trees Count                      : {self.metrics['trees_fitted']}")


# Aliases for backwards compatibility and intuitive imports
RandomForestClassifier = RandomForestDetection
RandomForest = RandomForestDetection
