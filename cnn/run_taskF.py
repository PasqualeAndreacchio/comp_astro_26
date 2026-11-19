"""CLI runner and consolidated Task F report builder for 1D CNN experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnn.cnn_classifier import CNNDetection  # noqa: E402

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
) -> List[Tuple[CNNDetection, Dict[str, Any]]]:
    """Run every YAML configuration in ``params_dir`` via ``CNNDetection.run()``."""
    yaml_files = sorted(params_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML configurations found in {params_dir}")

    results: List[Tuple[CNNDetection, Dict[str, Any]]] = []
    for yaml_path in yaml_files:
        detector = CNNDetection(params=yaml_path)
        metrics = detector.run(display=False, save=True)
        print(
            f"Completed {metrics['config_name']} | "
            f"AUC={metrics['auc_roc']:.4f} | "
            f"Prec(P)@0.5={metrics['precision_planet_default'] * 100:.2f}% | "
            f"Prec(P)@Opt={metrics['precision_planet'] * 100:.2f}% | "
            f"Saved: {detector.output_config.get('report_path')}"
        )
        results.append((detector, metrics))

    consolidated_report.parent.mkdir(parents=True, exist_ok=True)
    report_text = build_consolidated_report(results)
    consolidated_report.write_text(report_text, encoding="utf-8")
    print(f"Saved consolidated Task F report: {consolidated_report}")
    return results


def build_consolidated_report(
    results: List[Tuple[CNNDetection, Dict[str, Any]]],
) -> str:
    """Build the consolidated Task F report across all four CNN configurations."""
    best_det, best_res = max(results, key=lambda pair: pair[1]["auc_roc"])
    best_prec_det, best_prec_res = max(
        results, key=lambda pair: pair[1]["precision_planet"]
    )
    default_det, default_res = results[0]

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
        "                       (balanced=True, samples_per_class=350, random_state=42 -> 700 train)",
        "  Internal Val Split : Stratified 80/20 on balanced train (560 train / 140 val)",
        "                       used for EarlyStopping & ReduceLROnPlateau (X_test kept unseen)",
        "  Standardization    : Enabled (standardize=True, StandardScaler fit inside CNNDetection.fit)",
        "  Architecture       : CNNDetection(BaseTransitDetector)",
        "  Reference Notebook : transit_analysis_deep_learning.ipynb (implemented in PyTorch)",
        "",
    ]

    for detector, res in results:
        m = detector.model_config
        cm_d = res["confusion_matrix_default"]
        cm_o = res["confusion_matrix"]
        tn_d, fp_d, fn_d, tp_d = cm_d.ravel()
        tn_o, fp_o, fn_o, tp_o = cm_o.ravel()
        loss_type = str(m.get("loss_type", "focal")).lower()
        loss_desc = (
            f"FocalLoss(gamma={m.get('focal_gamma')}, alpha={m.get('focal_alpha')})"
            if loss_type == "focal"
            else "BCEWithLogitsLoss"
        )

        lines.extend(
            [
                "=" * 96,
                f"{res['config_name']}",
                "=" * 96,
                "NEURAL NETWORK ARCHITECTURE:",
                "-" * 96,
                *res["architecture_lines"],
                "",
                "TRAINING HYPERPARAMETERS:",
                "-" * 96,
                f"  loss_function     : {loss_desc}",
                f"  optimizer         : Adam (lr={m.get('learning_rate')}, dense_l2={m.get('l2_reg')})",
                f"  batch_size        : {m.get('batch_size')}",
                f"  epochs            : max={m.get('epochs')}, stopped={res['epochs_run']}, best_val_epoch={res['best_epoch']}",
                f"  best_val_roc_auc  : {res['val_auc_best']:.4f}",
                f"  test_roc_auc      : {res['auc_roc']:.4f}",
                "",
                "CONFUSION MATRIX & PRECISION (Default Threshold p >= 0.5000):",
                "-" * 96,
                "                       Predicted Non-Planet    Predicted Planet",
                f"  Actual Non-Planet :        {tn_d:<18d}    {fp_d:<18d}",
                f"  Actual Planet     :        {fn_d:<18d}    {tp_d:<18d}",
                "",
                f"  Precision (Planet)     : {res['precision_planet_default']:.4f} ({res['precision_planet_default'] * 100:.2f}%)",
                f"  Precision (Non-Planet) : {res['precision_non_planet_default']:.4f} ({res['precision_non_planet_default'] * 100:.2f}%)",
                f"  Precision (Macro Avg)  : {res['precision_macro_default']:.4f} ({res['precision_macro_default'] * 100:.2f}%)",
                f"  Recall (Planet)        : {res['recall_planet_default']:.4f} ({res['recall_planet_default'] * 100:.2f}%)",
                f"  F1-Score (Planet)      : {res['f1_planet_default']:.4f}",
                f"  Accuracy               : {res['accuracy_default'] * 100:.2f}%",
                "",
                f"CONFUSION MATRIX & PRECISION (Optimal Youden's J Threshold = {res['optimal_threshold']:.4f}):",
                "-" * 96,
                "                       Predicted Non-Planet    Predicted Planet",
                f"  Actual Non-Planet :        {tn_o:<18d}    {fp_o:<18d}",
                f"  Actual Planet     :        {fn_o:<18d}    {tp_o:<18d}",
                "",
                f"  Precision (Planet)     : {res['precision_planet']:.4f} ({res['precision_planet'] * 100:.2f}%)",
                f"  Precision (Non-Planet) : {res['precision_non_planet']:.4f} ({res['precision_non_planet'] * 100:.2f}%)",
                f"  Precision (Macro Avg)  : {res['precision_macro']:.4f} ({res['precision_macro'] * 100:.2f}%)",
                f"  Recall (Planet)        : {res['recall_planet']:.4f} ({res['recall_planet'] * 100:.2f}%)",
                f"  F1-Score (Planet)      : {res['f1_planet']:.4f}",
                f"  Accuracy               : {res['accuracy_optimal'] * 100:.2f}%",
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
        lines.append(
            f"   {res['config_name']:<57s}"
            f" {res['trainable_params']:>7,d}  "
            f" {res['precision_planet_default'] * 100:6.2f}%     "
            f" {res['accuracy_default'] * 100:6.2f}%  "
            f" {res['precision_planet'] * 100:6.2f}%      "
            f" {res['precision_non_planet'] * 100:6.2f}%       "
            f" {res['precision_macro'] * 100:6.2f}%        "
            f" {res['recall_planet'] * 100:6.2f}%     "
            f" {res['accuracy_optimal'] * 100:6.2f}%  "
            f" {res['auc_roc']:6.4f}"
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
    auc_vs_default = best_res["auc_roc"] - default_res["auc_roc"]
    auc_vs_rf = best_res["auc_roc"] - best_rf["auc"]
    auc_vs_dt = best_res["auc_roc"] - best_dt["auc"]

    lines.extend(
        [
            "",
            "3. Summary & Key Findings:",
            "-" * 96,
            f"   - Default CNN ({default_det.model_config.get('name')}): "
            f"ROC-AUC = {default_res['auc_roc']:.4f}, "
            f"Prec(P)@0.5 = {default_res['precision_planet_default'] * 100:.2f}%, "
            f"Prec(P)@Opt = {default_res['precision_planet'] * 100:.2f}% "
            f"({default_res['trainable_params']:,} trainable params).",
            f"   - Best CNN by ROC-AUC ({best_det.model_config.get('name')}): "
            f"ROC-AUC = {best_res['auc_roc']:.4f} ({auc_vs_default:+.4f} vs default), "
            f"Prec(P)@Opt = {best_res['precision_planet'] * 100:.2f}%, "
            f"Rec(P)@Opt = {best_res['recall_planet'] * 100:.2f}%, "
            f"Acc@Opt = {best_res['accuracy_optimal'] * 100:.2f}%.",
            f"   - Highest Planet Precision@Opt ({best_prec_det.model_config.get('name')}): "
            f"Prec(P)@Opt = {best_prec_res['precision_planet'] * 100:.2f}% "
            f"(ROC-AUC = {best_prec_res['auc_roc']:.4f}).",
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
        default="assignment2/report_cnn_assignment2_taskF.txt",
        help="Path for consolidated Task F report when running all configs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.config_path:
        detector = CNNDetection(params=args.config_path)
        metrics = detector.run(display=True, save=True)
        print(
            f"Completed {metrics['config_name']} | "
            f"AUC={metrics['auc_roc']:.4f} | "
            f"Acc@0.5={metrics['accuracy_default'] * 100:.2f}% | "
            f"Prec(P)@0.5={metrics['precision_planet_default'] * 100:.2f}% | "
            f"Prec(P)@Opt={metrics['precision_planet'] * 100:.2f}%"
        )
        return 0

    run_all_configs(
        params_dir=REPO_ROOT / "cnn" / "parameters",
        consolidated_report=REPO_ROOT / args.output_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
