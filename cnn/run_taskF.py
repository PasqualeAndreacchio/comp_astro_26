"""CLI runner and consolidated Task F report builder for 1D CNN experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnn.cnn_classifier import (  # noqa: E402
    CNNExperimentConfig,
    CNNTransitDetector,
    EvaluationMetrics,
)

# Reference benchmarks on the identical 189-sample test split (Tasks D & E)
PRIOR_TASK_BENCHMARKS = [
    {
        "task": "Task D (RF)",
        "name": "RF Config 2 (300 trees, depth=20) [Best RF]",
        "prec_p_opt": 0.7881,
        "prec_np_opt": 0.9859,
        "prec_macro_opt": 0.8870,
        "rec_p_opt": 0.9894,
        "acc_05": 0.8466,
        "acc_opt": 0.8624,
        "auc": 0.8885,
    },
    {
        "task": "Task D (RF)",
        "name": "RF Config 3 (500 trees, depth=30)",
        "prec_p_opt": 0.7881,
        "prec_np_opt": 0.9859,
        "prec_macro_opt": 0.8870,
        "rec_p_opt": 0.9894,
        "acc_05": 0.8360,
        "acc_opt": 0.8624,
        "auc": 0.8926,
    },
    {
        "task": "Task E (DT)",
        "name": "DT Config 2 (depth=10, leaf=10) [Best DT]",
        "prec_p_opt": 0.7961,
        "prec_np_opt": 0.8605,
        "prec_macro_opt": 0.8283,
        "rec_p_opt": 0.8723,
        "acc_05": 0.7460,
        "acc_opt": 0.8254,
        "auc": 0.8450,
    },
]


def run_all_configs(
    params_dir: Path,
    consolidated_report: Path,
) -> list[tuple[CNNExperimentConfig, EvaluationMetrics]]:
    """Run every YAML configuration in ``params_dir`` on the same data split."""
    yaml_files = sorted(params_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML configurations found in {params_dir}")

    first_cfg = CNNExperimentConfig.from_yaml(yaml_files[0])
    base_detector = CNNTransitDetector(first_cfg)
    X_train, X_test, y_train, y_test = base_detector.load_dataset()

    results: list[tuple[CNNExperimentConfig, EvaluationMetrics]] = []
    for yaml_path in yaml_files:
        cfg = CNNExperimentConfig.from_yaml(yaml_path)
        detector = CNNTransitDetector(cfg)
        metrics = detector.fit_and_evaluate(X_train, y_train, X_test, y_test)
        saved_individual = detector.save_report()
        print(
            f"Completed {cfg.model.name} | "
            f"AUC={metrics.roc_auc:.4f} | "
            f"Prec(P)@0.5={metrics.default_threshold.precision_planet * 100:.2f}% | "
            f"Prec(P)@Opt={metrics.optimal_threshold.precision_planet * 100:.2f}% | "
            f"Saved: {saved_individual}"
        )
        results.append((cfg, metrics))

    report_text = build_consolidated_report(results)
    consolidated_report.write_text(report_text, encoding="utf-8")
    print(f"Saved consolidated Task F report: {consolidated_report}")
    return results


def build_consolidated_report(
    results: list[tuple[CNNExperimentConfig, EvaluationMetrics]],
) -> str:
    """Build the consolidated Task F report across all four CNN configurations."""
    best_cfg, best_res = max(results, key=lambda pair: pair[1].roc_auc)
    best_prec_cfg, best_prec_res = max(
        results, key=lambda pair: pair[1].optimal_threshold.precision_planet
    )
    default_cfg, default_res = results[0]

    lines = [
        "=" * 96,
        "ASSIGNMENT 2 - TASK F: 1D CNN EXOPLANET TRANSIT CLASSIFICATION REPORT",
        "=" * 96,
        "",
        "EXPERIMENTAL SETUP & SPLIT CONSISTENCY WITH TASKS D & E:",
        "-" * 96,
        "  Dataset            : tess_data.csv (944 samples, 1000 flux bins)",
        "  Train/Test Split   : Stratified 80/20 (random_state=42) -> 755 train / 189 test",
        "  Test Distribution  : 95 Non-Planet (class 0), 94 Planet (class 1)",
        "  Train Balancing    : daneel.detection.data_process.load_data",
        "                       (samples_per_class=350, random_state=42 -> 700 train samples)",
        "  Internal Val Split : Stratified 80/20 on balanced train (560 train / 140 val)",
        "                       used for EarlyStopping & ReduceLROnPlateau (X_test kept unseen)",
        "  Standardization    : Enabled (use_scaler=True, StandardScaler fit on training set)",
        "  Reference Notebook : transit_analysis_deep_learning.ipynb (implemented in PyTorch)",
        "",
    ]

    for cfg, res in results:
        t = cfg.training
        m = cfg.model
        d_m = res.default_threshold
        o_m = res.optimal_threshold
        tn_d, fp_d, fn_d, tp_d = d_m.confusion_matrix.ravel()
        tn_o, fp_o, fn_o, tp_o = o_m.confusion_matrix.ravel()
        loss_desc = (
            f"FocalLoss(gamma={t.focal_gamma}, alpha={t.focal_alpha})"
            if t.loss_type.lower() == "focal"
            else "BCEWithLogitsLoss"
        )

        lines.extend(
            [
                "=" * 96,
                f"{res.config_name}",
                "=" * 96,
                "NEURAL NETWORK ARCHITECTURE:",
                "-" * 96,
                *res.architecture_lines,
                "",
                "TRAINING HYPERPARAMETERS:",
                "-" * 96,
                f"  loss_function     : {loss_desc}",
                f"  optimizer         : Adam (lr={t.learning_rate}, dense_l2={m.l2_reg})",
                f"  batch_size        : {t.batch_size}",
                f"  epochs            : max={t.epochs}, stopped={res.epochs_run}, best_val_epoch={res.best_epoch}",
                f"  best_val_roc_auc  : {res.val_auc_best:.4f}",
                f"  test_roc_auc      : {res.roc_auc:.4f}",
                "",
                "CONFUSION MATRIX & PRECISION (Default Threshold p >= 0.5000):",
                "-" * 96,
                "                       Predicted Non-Planet    Predicted Planet",
                f"  Actual Non-Planet :        {tn_d:<18d}    {fp_d:<18d}",
                f"  Actual Planet     :        {fn_d:<18d}    {tp_d:<18d}",
                "",
                f"  Precision (Planet)     : {d_m.precision_planet:.4f} ({d_m.precision_planet * 100:.2f}%)",
                f"  Precision (Non-Planet) : {d_m.precision_non_planet:.4f} ({d_m.precision_non_planet * 100:.2f}%)",
                f"  Precision (Macro Avg)  : {d_m.precision_macro:.4f} ({d_m.precision_macro * 100:.2f}%)",
                f"  Recall (Planet)        : {d_m.recall_planet:.4f} ({d_m.recall_planet * 100:.2f}%)",
                f"  F1-Score (Planet)      : {d_m.f1_planet:.4f}",
                f"  Accuracy               : {d_m.accuracy * 100:.2f}%",
                "",
                f"CONFUSION MATRIX & PRECISION (Optimal Youden's J Threshold = {o_m.threshold:.4f}):",
                "-" * 96,
                "                       Predicted Non-Planet    Predicted Planet",
                f"  Actual Non-Planet :        {tn_o:<18d}    {fp_o:<18d}",
                f"  Actual Planet     :        {fn_o:<18d}    {tp_o:<18d}",
                "",
                f"  Precision (Planet)     : {o_m.precision_planet:.4f} ({o_m.precision_planet * 100:.2f}%)",
                f"  Precision (Non-Planet) : {o_m.precision_non_planet:.4f} ({o_m.precision_non_planet * 100:.2f}%)",
                f"  Precision (Macro Avg)  : {o_m.precision_macro:.4f} ({o_m.precision_macro * 100:.2f}%)",
                f"  Recall (Planet)        : {o_m.recall_planet:.4f} ({o_m.recall_planet * 100:.2f}%)",
                f"  F1-Score (Planet)      : {o_m.f1_planet:.4f}",
                f"  Accuracy               : {o_m.accuracy * 100:.2f}%",
                "",
            ]
        )

    lines.extend(
        [
            "=" * 96,
            "COMPARISON ACROSS CNN CONFIGURATIONS & TASKS D-E",
            "=" * 96,
            "",
            "1. Task F CNN Configurations Summary (same 189 test samples):",
            "-" * 96,
            "   Configuration                                             Params    Prec(P)@0.5  Acc@0.5  Prec(P)@Opt  Prec(NP)@Opt  MacroPrec@Opt  Rec(P)@Opt  Acc@Opt  ROC-AUC",
            "   " + "-" * 158,
        ]
    )

    for _, res in results:
        d_m = res.default_threshold
        o_m = res.optimal_threshold
        lines.append(
            f"   {res.config_name:<57s}"
            f" {res.trainable_params:>7,d}  "
            f" {d_m.precision_planet * 100:6.2f}%     "
            f" {d_m.accuracy * 100:6.2f}%  "
            f" {o_m.precision_planet * 100:6.2f}%      "
            f" {o_m.precision_non_planet * 100:6.2f}%       "
            f" {o_m.precision_macro * 100:6.2f}%        "
            f" {o_m.recall_planet * 100:6.2f}%     "
            f" {o_m.accuracy * 100:6.2f}%  "
            f" {res.roc_auc:6.4f}"
        )

    lines.extend(
        [
            "",
            "2. Classical ML Benchmarks from Tasks D & E (same 189 test samples):",
            "-" * 96,
            "   Task          Model                                         Acc@0.5  Prec(P)@Opt  Prec(NP)@Opt  MacroPrec@Opt  Rec(P)@Opt  Acc@Opt  ROC-AUC",
            "   " + "-" * 138,
        ]
    )
    for bm in PRIOR_TASK_BENCHMARKS:
        lines.append(
            f"   {bm['task']:<13s}"
            f" {bm['name']:<45s}"
            f" {bm['acc_05'] * 100:6.2f}%  "
            f" {bm['prec_p_opt'] * 100:6.2f}%      "
            f" {bm['prec_np_opt'] * 100:6.2f}%       "
            f" {bm['prec_macro_opt'] * 100:6.2f}%        "
            f" {bm['rec_p_opt'] * 100:6.2f}%     "
            f" {bm['acc_opt'] * 100:6.2f}%  "
            f" {bm['auc']:6.4f}"
        )

    best_rf = PRIOR_TASK_BENCHMARKS[0]
    best_dt = PRIOR_TASK_BENCHMARKS[2]
    auc_vs_default = best_res.roc_auc - default_res.roc_auc
    auc_vs_rf = best_res.roc_auc - best_rf["auc"]
    auc_vs_dt = best_res.roc_auc - best_dt["auc"]

    lines.extend(
        [
            "",
            "3. Summary & Key Findings:",
            "-" * 96,
            f"   - Default CNN ({default_cfg.model.name}): "
            f"ROC-AUC = {default_res.roc_auc:.4f}, "
            f"Prec(P)@0.5 = {default_res.default_threshold.precision_planet * 100:.2f}%, "
            f"Prec(P)@Opt = {default_res.optimal_threshold.precision_planet * 100:.2f}% "
            f"({default_res.trainable_params:,} trainable params).",
            f"   - Best CNN by ROC-AUC ({best_cfg.model.name}): "
            f"ROC-AUC = {best_res.roc_auc:.4f} ({auc_vs_default:+.4f} vs default), "
            f"Prec(P)@Opt = {best_res.optimal_threshold.precision_planet * 100:.2f}%, "
            f"Rec(P)@Opt = {best_res.optimal_threshold.recall_planet * 100:.2f}%, "
            f"Acc@Opt = {best_res.optimal_threshold.accuracy * 100:.2f}%.",
            f"   - Highest Planet Precision@Opt ({best_prec_cfg.model.name}): "
            f"Prec(P)@Opt = {best_prec_res.optimal_threshold.precision_planet * 100:.2f}% "
            f"(ROC-AUC = {best_prec_res.roc_auc:.4f}).",
            f"   - Cross-Model Comparison on identical test split: Best CNN vs Best RF delta ROC-AUC = {auc_vs_rf:+.4f}; "
            f"Best CNN vs Best Decision Tree delta ROC-AUC = {auc_vs_dt:+.4f}.",
            "=" * 96,
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task F 1D CNN exoplanet detection experiments."
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="config_path",
        type=str,
        default=None,
        help="Optional single YAML config path. Default runs all configs in cnn/parameters/.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_report",
        type=str,
        default="report_cnn_assignment2_taskF.txt",
        help="Path for consolidated Task F report when running all configs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.config_path:
        cfg = CNNExperimentConfig.from_yaml(args.config_path)
        detector = CNNTransitDetector(cfg)
        metrics = detector.run()
        print(
            f"Completed {cfg.model.name} | "
            f"AUC={metrics.roc_auc:.4f} | "
            f"Acc@0.5={metrics.default_threshold.accuracy * 100:.2f}% | "
            f"Prec(P)@0.5={metrics.default_threshold.precision_planet * 100:.2f}% | "
            f"Prec(P)@Opt={metrics.optimal_threshold.precision_planet * 100:.2f}%"
        )
        return 0

    run_all_configs(
        params_dir=REPO_ROOT / "cnn" / "parameters",
        consolidated_report=REPO_ROOT / args.output_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
