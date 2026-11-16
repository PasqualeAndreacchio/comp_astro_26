import copy
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

# Ensure 'src' package root is on sys.path
repo_root = Path(__file__).resolve().parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from daneel.parameters import Parameters
from daneel.detection.transit import build_batman_params, compute_light_curve


def run_task(
    yaml_file: str | Path = "examples/WASP43-B.yaml",
    output_filename: str = "assignment2_taskB.png",
):
    yaml_path = Path(yaml_file)
    if not yaml_path.is_file():
        # Fallback to alternative naming if needed
        fallback = repo_root / "examples" / "wasp43b_params.yaml"
        if fallback.is_file():
            yaml_path = fallback
        else:
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    # Load reference parameters
    loaded = Parameters(yaml_path).params
    params_ref_dict = copy.deepcopy(loaded.get("transit", loaded))

    rp_ref = float(params_ref_dict["rp"])
    rp_half = rp_ref / 2.0

    # Build parameters for reference planet (WASP-43 b)
    params_ref = build_batman_params(params_ref_dict)

    # Build parameters for hypothetical planet (half radius)
    params_half_dict = copy.deepcopy(params_ref_dict)
    params_half_dict["rp"] = rp_half
    params_half = build_batman_params(params_half_dict)

    # Define time window
    # Compute analytical transit duration to frame the plot nicely
    b = params_ref.a * np.cos(np.radians(params_ref.inc))
    arg = max(0.0, (1.0 + params_ref.rp) ** 2 - b ** 2)
    t_dur = (params_ref.per / np.pi) * (1.0 / params_ref.a) * np.sqrt(arg) / max(0.01, np.sin(np.radians(params_ref.inc)))
    t_half = max(0.035, 1.4 * (t_dur / 2.0))
    t = np.linspace(-t_half, t_half, 1000)

    # 5. Compute both synthetic light curves
    _, flux_ref = compute_light_curve(params_ref, t)
    _, flux_half = compute_light_curve(params_half, t)

    # Physical transit depths
    depth_ref = (1.0 - np.min(flux_ref)) * 100
    depth_half = (1.0 - np.min(flux_half)) * 100

    # Plot both curves
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)

    ax.plot(
        t,
        flux_ref,
        color="red",
        lw=2.2,
        label=f"Reference: WASP-43 b ($R_p/R_* = {rp_ref:.5f}$, depth = {depth_ref:.2f}%)",
    )
    ax.plot(
        t,
        flux_half,
        color="blue",
        lw=2.2,
        ls="--",
        label=f"Hypothetical: Half Radius ($R_p/R_* = {rp_half:.5f}$, depth = {depth_half:.2f}%)",
    )
    ax.axhline(1.0, color="gray", lw=0.9, ls=":", alpha=0.7, label="Out-of-transit baseline")

    ax.set_xlabel("Time from transit centre [days]", fontsize=11)
    ax.set_ylabel("Relative stellar flux", fontsize=11)
    ax.set_title("Transit Light Curve Comparison — WASP-43 b vs. Half-Radius Planet", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_ylim(min(flux_ref) - 0.003, 1.003)

    # Information box
    info_text = (
        r"$\mathbf{Model\ Parameters:}$" + "\n"
        f"Period $P$: {params_ref.per:.5f} d\n"
        f"Scaled semi-major axis $a/R_*$: {params_ref.a:.3f}\n"
        f"Inclination $i$: {params_ref.inc:.2f}° (impact $b = {b:.2f}$)\n"
        f"Limb darkening: {params_ref.limb_dark} {params_ref.u}\n"
        f"Depth ratio: $\\delta_1 / \\delta_2 \\approx {(depth_ref / depth_half):.2f}$ (geometric limit $\\approx 4$)"
    )
    ax.text(
        0.03,
        0.08,
        info_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)
    fig.tight_layout()

    # Save outputs
    fig.savefig(output_filename, dpi=180, bbox_inches="tight")
    print(f"Saved comparison light curve to: {output_filename}")

    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Assignment 2 Task B: Planet Radius Comparison")
    parser.add_argument(
        "-i",
        "--input",
        default="examples/WASP43-B.yaml",
        help="Input YAML configuration file (default: examples/WASP43-B.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="assignment2_taskB.png",
        help="Output PNG file path (default: assignment2_taskB.png)",
    )

    args = parser.parse_args()
    run_task(yaml_file=args.input, output_filename=args.output)
