"""CLI runner and consolidated Task E report builder for Decision Tree."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decision_tree.decision_tree_classifier import DecisionTreeDetection  # noqa: E402

# Reference Random Forest results from Task D (Pasquale, commit 5e3563a)
TASK_D_RF_BENCHMARKS = [
    {
        "name": "RF Config 1 (100, depth=10)",
        "prec_p": 0.7931,
        "prec_np": 0.9726,
        "prec_macro": 0.8829,
        "rec_p": 0.9787,
        "acc_opt": 0.8624,
        "acc_05": 0.8360,
        "auc": 0.8837,
    },
    {
        "name": "RF Config 2 (300, depth=20) [Best]",
        "prec_p": 0.7881,
        "prec_np": 0.9859,
        "prec_macro": 0.8870,
        "rec_p": 0.9894,
        "acc_opt": 0.8624,
        "acc_05": 0.8466,
        "auc": 0.8885,
    },
    {
        "name": "RF Config 3 (500, depth=30)",
        "prec_p": 0.7881,
        "prec_np": 0.9859,
        "prec_macro": 0.8870,
        "rec_p": 0.9894,
        "acc_opt": 0.8624,
        "acc_05": 0.8360,
        "auc": 0.8926,
    },
    {
        "name": "RF Config 4 (500, depth=None)",
        "prec_p": 0.7881,
        "prec_np": 0.9859,
        "prec_macro": 0.8870,
        "rec_p": 0.9894,
        "acc_opt": 0.8624,
        "acc_05": 0.8360,
        "auc": 0.8926,
    },
]


def run_all_configs(
    params_dir: Path,
    consolidated_report: Path,
) -> List[Tuple[DecisionTreeDetection, Dict[str, Any]]]:
    """Run every YAML configuration in ``params_dir`` via ``DecisionTreeDetection.run()``."""
    yaml_files = sorted(params_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML configurations found in {params_dir}")

    results: List[Tuple[DecisionTreeDetection, Dict[str, Any]]] = []
    for yaml_path in yaml_files:
        detector = DecisionTreeDetection(params=yaml_path)
        metrics = detector.run(display=False, save=True)
        print(
            f"Completed {metrics['config_name']} | "
            f"AUC={metrics['auc_roc']:.4f} | "
            f"Acc@0.5={metrics['accuracy_default'] * 100:.2f}% | "
            f"Acc@Opt={metrics['accuracy_optimal'] * 100:.2f}% | "
            f"Saved: {detector.output_config.get('report_path')}"
        )
        results.append((detector, metrics))

    consolidated_report.parent.mkdir(parents=True, exist_ok=True)
    report_text = build_consolidated_report(results)
    consolidated_report.write_text(report_text, encoding="utf-8")
    print(f"Saved consolidated Task E report: {consolidated_report}")
    return results


def build_consolidated_report(
    results: List[Tuple[DecisionTreeDetection, Dict[str, Any]]],
) -> str:
    """Build the consolidated Task E report comparing Decision Tree vs Task D RF."""
    best_det, best_res = max(results, key=lambda pair: pair[1]["auc_roc"])
    rf_best = TASK_D_RF_BENCHMARKS[1]  # RF Config 2 (300 trees, depth=20)
    rf_matched_pair = next(
        (
            pair
            for pair in results
            if pair[0].model_config.get("max_depth") == 20
            and pair[0].model_config.get("min_samples_leaf") == 1
        ),
        results[0],
    )

    lines = [
        "=" * 92,
        "ASSIGNMENT 2 - TASK E: DECISION TREE EXOPLANET CLASSIFICATION REPORT",
        "=" * 92,
        "",
        "EXPERIMENTAL SETUP & SPLIT CONSISTENCY WITH TASK D:",
        "-" * 92,
        "  Dataset          : tess_data.csv (944 samples, 1000 flux bins)",
        "  Train/Test Split : Stratified 80/20 (random_state=42) -> 755 train / 189 test",
        "  Test Distribution: 95 Non-Planet (class 0), 94 Planet (class 1)",
        "  Train Balancing  : daneel.detection.data_process.load_data",
        "                     (balanced=True, samples_per_class=350, random_state=42 -> 700 train)",
        "  Standardization  : Disabled (standardize=False), as tree splits are invariant",
        "                     to monotonic feature scaling (consistent with Task D)",
        "  Architecture     : DecisionTreeDetection(BaseTransitDetector)",
        "",
    ]

    for detector, res in results:
        m = detector.model_config
        cm_d = res["confusion_matrix_default"]
        cm_o = res["confusion_matrix"]
        tn_d, fp_d, fn_d, tp_d = cm_d.ravel()
        tn_o, fp_o, fn_o, tp_o = cm_o.ravel()
        lines.extend(
            [
                "=" * 92,
                f"{res['config_name']}",
                "=" * 92,
                "MODEL HYPERPARAMETERS & TREE STRUCTURE:",
                "-" * 92,
                f"  criterion         : {m.get('criterion')}",
                f"  max_depth         : {m.get('max_depth')} (fitted depth: {res['tree_depth']})",
                f"  min_samples_leaf  : {m.get('min_samples_leaf')}",
                f"  min_samples_split : {m.get('min_samples_split')}",
                f"  n_leaves          : {res['n_leaves']}",
                f"  random_state      : {m.get('random_state')}",
                f"  ROC-AUC Score     : {res['auc_roc']:.4f}",
                "",
                "CONFUSION MATRIX & PRECISION (Default Threshold p >= 0.5000):",
                "-" * 92,
                "                       Predicted Non-Planet    Predicted Planet",
                f"  Actual Non-Planet :        {tn_d:<18d}    {fp_d:<18d}",
                f"  Actual Planet     :        {fn_d:<18d}    {tp_d:<18d}",
                "",
                f"  Precision (Planet)     : {res['precision_planet_default']:.4f} ({res['precision_planet_default'] * 100:.2f}%)",
                f"  Precision (Non-Planet) : {res['precision_non_planet_default']:.4f} ({res['precision_non_planet_default'] * 100:.2f}%)",
                f"  Precision (Macro Avg)  : {res['precision_macro_default']:.4f} ({res['precision_macro_default'] * 100:.2f}%)",
                f"  Recall (Planet)        : {res['recall_planet_default']:.4f} ({res['recall_planet_default'] * 100:.2f}%)",
                f"  Accuracy               : {res['accuracy_default'] * 100:.2f}%",
                "",
                f"CONFUSION MATRIX & PRECISION (Optimal Youden's J Threshold = {res['optimal_threshold']:.4f}):",
                "-" * 92,
                "                       Predicted Non-Planet    Predicted Planet",
                f"  Actual Non-Planet :        {tn_o:<18d}    {fp_o:<18d}",
                f"  Actual Planet     :        {fn_o:<18d}    {tp_o:<18d}",
                "",
                f"  Precision (Planet)     : {res['precision_planet']:.4f} ({res['precision_planet'] * 100:.2f}%)",
                f"  Precision (Non-Planet) : {res['precision_non_planet']:.4f} ({res['precision_non_planet'] * 100:.2f}%)",
                f"  Precision (Macro Avg)  : {res['precision_macro']:.4f} ({res['precision_macro'] * 100:.2f}%)",
                f"  Recall (Planet)        : {res['recall_planet']:.4f} ({res['recall_planet'] * 100:.2f}%)",
                f"  Accuracy               : {res['accuracy_optimal'] * 100:.2f}%",
                "",
            ]
        )

    lines.extend(
        [
            "=" * 92,
            "COMPARISON WITH TASK D (RANDOM FOREST)",
            "=" * 92,
            "",
            "1. Decision Tree Configurations Summary:",
            "-" * 92,
            "   Configuration                                        Prec(P)@0.5  Acc@0.5  Prec(P)@Opt  Prec(NP)@Opt  MacroPrec@Opt  Rec(P)@Opt  Acc@Opt  ROC-AUC",
            "   " + "-" * 143,
        ]
    )

    for _, res in results:
        lines.append(
            f"   {res['config_name']:<54s}"
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
            "2. Task D Random Forest Benchmarks (same 189 test samples):",
            "-" * 92,
            "   Configuration                                        Acc@0.5  Prec(P)@Opt  Prec(NP)@Opt  MacroPrec@Opt  Rec(P)@Opt  Acc@Opt  ROC-AUC",
            "   " + "-" * 131,
        ]
    )
    for rf in TASK_D_RF_BENCHMARKS:
        lines.append(
            f"   {rf['name']:<54s}"
            f" {rf['acc_05'] * 100:6.2f}%  "
            f" {rf['prec_p'] * 100:6.2f}%      "
            f" {rf['prec_np'] * 100:6.2f}%       "
            f" {rf['prec_macro'] * 100:6.2f}%        "
            f" {rf['rec_p'] * 100:6.2f}%     "
            f" {rf['acc_opt'] * 100:6.2f}%  "
            f" {rf['auc']:6.4f}"
        )

    matched_res = rf_matched_pair[1]
    auc_gain = rf_best["auc"] - matched_res["auc_roc"]
    acc05_gain = (rf_best["acc_05"] - matched_res["accuracy_default"]) * 100
    best_dt_auc_gap = best_res["auc_roc"] - rf_best["auc"]
    best_dt_acc05_gap = (best_res["accuracy_default"] - rf_best["acc_05"]) * 100

    lines.extend(
        [
            "",
            "3. Summary Comparison:",
            "-" * 92,
            f"   - Best Decision Tree by ROC-AUC: {best_det.model_config.get('name')} "
            f"(ROC-AUC = {best_res['auc_roc']:.4f}, "
            f"Prec(P)@Opt = {best_res['precision_planet'] * 100:.2f}%).",
            f"   - Best Task D Random Forest: {rf_best['name']} "
            f"(ROC-AUC = {rf_best['auc']:.4f}, "
            f"Prec(P)@Opt = {rf_best['prec_p'] * 100:.2f}%).",
            f"   - Direct Single-Tree vs Ensemble Comparison (both depth=20, min_samples_leaf=1):",
            f"     * 1 Decision Tree  -> Acc@0.5 = {matched_res['accuracy_default'] * 100:.2f}%, "
            f"ROC-AUC = {matched_res['auc_roc']:.4f}",
            f"     * 300-Tree RF      -> Acc@0.5 = {rf_best['acc_05'] * 100:.2f}%, "
            f"ROC-AUC = {rf_best['auc']:.4f}",
            f"     * Ensemble Gain    -> +{acc05_gain:.2f} pp Accuracy@0.5, +{auc_gain:.4f} ROC-AUC.",
            f"   - Best Regularized Tree vs Best RF: ROC-AUC delta = {best_dt_auc_gap:+.4f}, "
            f"Accuracy@0.5 delta = {best_dt_acc05_gap:+.2f} pp.",
            "=" * 92,
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task E Decision Tree exoplanet detection experiments."
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="config_path",
        type=str,
        default=None,
        help="Optional single YAML config path. Default runs all configs in decision_tree/parameters/.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_report",
        type=str,
        default="assignment2/report_decision_tree_assignment2_taskE.txt",
        help="Path for consolidated Task E report when running all configs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.config_path:
        detector = DecisionTreeDetection(params=args.config_path)
        metrics = detector.run(display=True, save=True)
        print(
            f"Completed {metrics['config_name']} | "
            f"AUC={metrics['auc_roc']:.4f} | "
            f"Acc@0.5={metrics['accuracy_default'] * 100:.2f}% | "
            f"Prec(P)@Opt={metrics['precision_planet'] * 100:.2f}%"
        )
        return 0

    run_all_configs(
        params_dir=REPO_ROOT / "decision_tree" / "parameters",
        consolidated_report=REPO_ROOT / args.output_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
