"""Object-oriented Decision Tree pipeline for TESS transit detection.

Mirrors the modular YAML-driven design used in Task D while reusing
``daneel.detection.data_process.load_data`` to guarantee the exact same
stratified 80/20 evaluation split and balanced training augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import yaml

from daneel.detection.data_process import RANDOM_STATE, load_data


@dataclass
class DataConfig:
    """Dataset and train/test split configuration."""

    csv_path: str = "tess_data.csv"
    n_bins: int = 1000
    samples_per_class: int = 350


@dataclass
class DecisionTreeModelConfig:
    """Hyperparameters for ``sklearn.tree.DecisionTreeClassifier``."""

    name: str = "Decision Tree"
    standardize: bool = False
    criterion: str = "gini"
    max_depth: int | None = 10
    min_samples_split: int = 2
    min_samples_leaf: int = 10
    max_features: str | int | float | None = None
    class_weight: str | dict[int, float] | None = None
    ccp_alpha: float = 0.0
    random_state: int = RANDOM_STATE


@dataclass
class DecisionTreeExperimentConfig:
    """Complete YAML-serializable configuration for a Decision Tree run."""

    data: DataConfig = field(default_factory=DataConfig)
    model: DecisionTreeModelConfig = field(default_factory=DecisionTreeModelConfig)
    report_path: str = "decision_tree/results/report_decision_tree.txt"

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> DecisionTreeExperimentConfig:
        """Load configuration from a YAML file with safe defaults."""
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}

        data_raw = raw.get("data", {})
        model_raw = raw.get("model", {})
        output_raw = raw.get("output", {})

        data_cfg = DataConfig(
            csv_path=str(data_raw.get("csv_path", "tess_data.csv")),
            n_bins=int(data_raw.get("n_bins", 1000)),
            samples_per_class=int(data_raw.get("samples_per_class", 350)),
        )
        max_depth_val = model_raw.get("max_depth", 10)
        if max_depth_val == "None":
            max_depth_val = None

        model_cfg = DecisionTreeModelConfig(
            name=str(model_raw.get("name", path.stem)),
            standardize=bool(model_raw.get("standardize", False)),
            criterion=str(model_raw.get("criterion", "gini")),
            max_depth=int(max_depth_val) if max_depth_val is not None else None,
            min_samples_split=int(model_raw.get("min_samples_split", 2)),
            min_samples_leaf=int(model_raw.get("min_samples_leaf", 10)),
            max_features=model_raw.get("max_features", None),
            class_weight=model_raw.get("class_weight", None),
            ccp_alpha=float(model_raw.get("ccp_alpha", 0.0)),
            random_state=int(model_raw.get("random_state", RANDOM_STATE)),
        )
        report_path = str(
            output_raw.get(
                "report_path", f"decision_tree/results/report_{path.stem}.txt"
            )
        )
        return cls(data=data_cfg, model=model_cfg, report_path=report_path)


@dataclass
class ThresholdMetrics:
    """Evaluation metrics at a given probability decision threshold."""

    threshold: float
    accuracy: float
    precision_planet: float
    precision_non_planet: float
    precision_macro: float
    precision_weighted: float
    recall_planet: float
    recall_non_planet: float
    f1_planet: float
    f1_non_planet: float
    confusion_matrix: np.ndarray
    classification_report: str


@dataclass
class EvaluationMetrics:
    """Full evaluation summary for a Decision Tree experiment."""

    config_name: str
    roc_auc: float
    default_threshold: ThresholdMetrics
    optimal_threshold: ThresholdMetrics
    n_train: int
    n_test: int
    tree_depth: int
    n_leaves: int


class DecisionTreeTransitDetector:
    """Encapsulates loading, training, evaluation, and reporting for Task E."""

    def __init__(self, config: DecisionTreeExperimentConfig) -> None:
        self.config = config
        self.pipeline: Pipeline | None = None
        self.metrics: EvaluationMetrics | None = None

    def load_dataset(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset via ``daneel.detection.data_process.load_data``."""
        csv_path = self._resolve_csv_path(self.config.data.csv_path)
        X_train, X_test, y_train, y_test, *_ = load_data(
            csv_path=str(csv_path),
            n_bins=self.config.data.n_bins,
            use_scaler=False,
            samples_per_class=self.config.data.samples_per_class,
        )
        return X_train, X_test, y_train, y_test

    @staticmethod
    def _resolve_csv_path(csv_path: str) -> Path:
        path = Path(csv_path)
        if path.is_file():
            return path
        fallback = Path("..") / csv_path
        if fallback.is_file():
            return fallback
        raise FileNotFoundError(f"Dataset CSV not found: {path.resolve()}")

    def build_pipeline(self) -> Pipeline:
        """Build scikit-learn Pipeline (optional scaler + ``DecisionTreeClassifier``)."""
        m = self.config.model
        steps: list[tuple[str, Any]] = []
        if m.standardize:
            steps.append(("scaler", StandardScaler()))

        tree = DecisionTreeClassifier(
            criterion=m.criterion,
            max_depth=m.max_depth,
            min_samples_split=m.min_samples_split,
            min_samples_leaf=m.min_samples_leaf,
            max_features=m.max_features,
            class_weight=m.class_weight,
            ccp_alpha=m.ccp_alpha,
            random_state=m.random_state,
        )
        steps.append(("classifier", tree))
        self.pipeline = Pipeline(steps)
        return self.pipeline

    @staticmethod
    def _compute_threshold_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float,
    ) -> ThresholdMetrics:
        return ThresholdMetrics(
            threshold=float(threshold),
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision_planet=float(
                precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            ),
            precision_non_planet=float(
                precision_score(y_true, y_pred, pos_label=0, zero_division=0)
            ),
            precision_macro=float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            precision_weighted=float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            recall_planet=float(
                recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            ),
            recall_non_planet=float(
                recall_score(y_true, y_pred, pos_label=0, zero_division=0)
            ),
            f1_planet=float(
                f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            ),
            f1_non_planet=float(
                f1_score(y_true, y_pred, pos_label=0, zero_division=0)
            ),
            confusion_matrix=confusion_matrix(y_true, y_pred),
            classification_report=classification_report(
                y_true,
                y_pred,
                target_names=["Non-Planet", "Planet"],
                digits=4,
                zero_division=0,
            ),
        )

    def fit_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> EvaluationMetrics:
        """Train the tree and compute metrics at default (0.5) and Youden-J."""
        pipeline = self.build_pipeline()
        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred_default = (proba >= 0.5).astype(int)
        roc_auc = float(roc_auc_score(y_test, proba))

        fpr, tpr, thresholds = roc_curve(y_test, proba)
        best_idx = int(np.argmax(tpr - fpr))
        best_thresh = float(np.clip(thresholds[best_idx], 0.0, 1.0))
        y_pred_opt = (proba >= best_thresh).astype(int)

        tree_step: DecisionTreeClassifier = pipeline.named_steps["classifier"]

        self.metrics = EvaluationMetrics(
            config_name=self.config.model.name,
            roc_auc=roc_auc,
            default_threshold=self._compute_threshold_metrics(
                y_test, y_pred_default, 0.5
            ),
            optimal_threshold=self._compute_threshold_metrics(
                y_test, y_pred_opt, best_thresh
            ),
            n_train=int(len(X_train)),
            n_test=int(len(X_test)),
            tree_depth=int(tree_step.get_depth()),
            n_leaves=int(tree_step.get_n_leaves()),
        )
        return self.metrics

    def format_report(self) -> str:
        """Render a clean evaluation report for a single configuration."""
        if self.metrics is None:
            raise RuntimeError("Model must be evaluated before formatting report.")

        m = self.config.model
        res = self.metrics
        d_m = res.default_threshold
        o_m = res.optimal_threshold

        def _cm_lines(cm: np.ndarray) -> list[str]:
            tn, fp, fn, tp = cm.ravel()
            return [
                "                       Predicted Non-Planet    Predicted Planet",
                f"  Actual Non-Planet :        {tn:<18d}    {fp:<18d}",
                f"  Actual Planet     :        {fn:<18d}    {tp:<18d}",
            ]

        lines = [
            "=" * 80,
            f"DECISION TREE EVALUATION REPORT - {res.config_name}",
            "=" * 80,
            "",
            "MODEL HYPERPARAMETERS & TREE STRUCTURE:",
            "-" * 80,
            f"  criterion         : {m.criterion}",
            f"  max_depth         : {m.max_depth}",
            f"  min_samples_split : {m.min_samples_split}",
            f"  min_samples_leaf  : {m.min_samples_leaf}",
            f"  max_features      : {m.max_features}",
            f"  ccp_alpha         : {m.ccp_alpha}",
            f"  standardize       : {m.standardize}",
            f"  random_state      : {m.random_state}",
            f"  fitted_depth      : {res.tree_depth}",
            f"  n_leaves          : {res.n_leaves}",
            f"  train / test size : {res.n_train} / {res.n_test}",
            f"  ROC-AUC Score     : {res.roc_auc:.4f}",
            "",
            "METRICS AT DEFAULT THRESHOLD (p >= 0.5000):",
            "-" * 80,
            f"  Precision (Planet)     : {d_m.precision_planet:.4f} ({d_m.precision_planet * 100:.2f}%)",
            f"  Precision (Non-Planet) : {d_m.precision_non_planet:.4f} ({d_m.precision_non_planet * 100:.2f}%)",
            f"  Precision (Macro Avg)  : {d_m.precision_macro:.4f} ({d_m.precision_macro * 100:.2f}%)",
            f"  Recall (Planet)        : {d_m.recall_planet:.4f} ({d_m.recall_planet * 100:.2f}%)",
            f"  Accuracy               : {d_m.accuracy * 100:.2f}%",
            "",
            "CONFUSION MATRIX (Default Threshold 0.5000):",
            *_cm_lines(d_m.confusion_matrix),
            "",
            "CLASSIFICATION REPORT (Default Threshold 0.5000):",
            d_m.classification_report.rstrip(),
            "",
            f"METRICS AT OPTIMAL THRESHOLD (Youden's J = {o_m.threshold:.4f}):",
            "-" * 80,
            f"  Precision (Planet)     : {o_m.precision_planet:.4f} ({o_m.precision_planet * 100:.2f}%)",
            f"  Precision (Non-Planet) : {o_m.precision_non_planet:.4f} ({o_m.precision_non_planet * 100:.2f}%)",
            f"  Precision (Macro Avg)  : {o_m.precision_macro:.4f} ({o_m.precision_macro * 100:.2f}%)",
            f"  Recall (Planet)        : {o_m.recall_planet:.4f} ({o_m.recall_planet * 100:.2f}%)",
            f"  Accuracy               : {o_m.accuracy * 100:.2f}%",
            "",
            "CONFUSION MATRIX (Optimal Youden's J Threshold):",
            *_cm_lines(o_m.confusion_matrix),
            "",
            "CLASSIFICATION REPORT (Optimal Youden's J Threshold):",
            o_m.classification_report.rstrip(),
            "=" * 80,
        ]
        return "\n".join(lines) + "\n"

    def save_report(self, output_path: str | Path | None = None) -> Path:
        """Write the individual experiment report to disk."""
        target = Path(output_path or self.config.report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.format_report(), encoding="utf-8")
        return target

    def run(self) -> EvaluationMetrics:
        """Execute loading, training, evaluation, and report export."""
        X_train, X_test, y_train, y_test = self.load_dataset()
        metrics = self.fit_and_evaluate(X_train, y_train, X_test, y_test)
        self.save_report()
        return metrics
