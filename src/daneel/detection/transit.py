"""
Transit modelling module for exoplanet light curves.

Target: WASP-43 b
Host star: WASP-43 (K7V, Teff ~ 4400 K)
References:
    - Hellier et al. (2011), A&A, 535, L7 (Discovery paper)
    - Southworth, J. (2012), MNRAS, 426(2), 1291-1309 (Homogeneous studies of transiting extrasolar planets)
    - NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
"""

import importlib
from pathlib import Path
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import setuptools  # noqa: F401  # Provides distutils compatibility for batman on Python 3.12+

batman = importlib.import_module("batman")

# Default target parameters: WASP-43 b
PLANET_NAME = "WASP-43 b"
PLANET_SLUG = "WASP-43_b"

DEFAULT_PARAMS = {
    "t0": 0.0,              # Time of inferior conjunction [days]
    "per": 0.81347753,      # Orbital period [days]
    "rp": 0.15949,          # Planet-to-star radius ratio (Rp / Rstar)
    "a": 4.872,             # Scaled semi-major axis (a / Rstar)
    "inc": 82.33,           # Orbital inclination [degrees]
    "ecc": 0.0,             # Eccentricity (circular orbit)
    "w": 0.0,               # Longitude of periastron [degrees]
    "u": [0.36, 0.22],      # Quadratic limb darkening coefficients (optical/V band)
    "limb_dark": "quadratic",
}


def build_batman_params(custom_params=None):
    """
    Construct a batman.TransitParams object from a dictionary.

    Parameters
    ----------
    custom_params : dict, optional
        Dictionary with transit parameters. Missing keys fall back to DEFAULT_PARAMS.

    Returns
    -------
    params : batman.TransitParams
    """
    p_dict = dict(DEFAULT_PARAMS)
    if custom_params:
        p_dict.update(custom_params)

    params = batman.TransitParams()
    params.t0 = float(p_dict.get("t0", 0.0))
    params.per = float(p_dict.get("per", 0.81347753))
    params.rp = float(p_dict.get("rp", 0.15949))
    params.a = float(p_dict.get("a", 4.872))
    params.inc = float(p_dict.get("inc", 82.33))
    params.ecc = float(p_dict.get("ecc", 0.0))
    params.w = float(p_dict.get("w", 0.0))
    params.limb_dark = str(p_dict.get("limb_dark", "quadratic"))
    params.u = list(p_dict.get("u", [0.36, 0.22]))
    return params


def compute_light_curve(params=None, t=None):
    """
    Compute normalised synthetic flux using batman.

    Parameters
    ----------
    params : batman.TransitParams, optional
    t : ndarray, optional
        Observation times relative to transit centre [days].

    Returns
    -------
    t : ndarray
    flux : ndarray
    """
    if params is None:
        params = build_batman_params()
    if t is None:
        # 1000 points spanning [-0.10, +0.10] days (~ +/- 2.4 hours)
        t = np.linspace(-0.10, 0.10, 1000)

    model = batman.TransitModel(params, t)
    flux = model.light_curve(params)
    return t, flux


def plot_light_curve(output_file=f"{PLANET_SLUG}_assignment1_taskF.png", params=None, t=None, show=False):
    """
    Generate and save the static transit light curve.

    Parameters
    ----------
    output_file : str or Path
    params : batman.TransitParams, optional
    t : ndarray, optional
    show : bool, optional
    """
    if params is None:
        params = build_batman_params()
    t, flux = compute_light_curve(params, t)

    # Physical transit depth and impact parameter
    depth_geom = (params.rp ** 2) * 100  # percent
    actual_depth = (1.0 - np.min(flux)) * 100  # percent with limb darkening
    b = params.a * np.cos(np.radians(params.inc))  # impact parameter in R*

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=180)

    # Plot light curve
    ax.plot(t, flux, color="#1f77b4", lw=2.2, label=f"BATMAN model ({params.limb_dark} LD)")
    ax.axhline(1.0, color="gray", lw=1.0, ls="--", alpha=0.7, label="Out-of-transit baseline")

    # Ingress / egress approximate markers (where flux dips below 0.9999)
    in_transit = np.where(flux < 0.9999)[0]
    if len(in_transit) > 0:
        t_ingress = t[in_transit[0]]
        t_egress = t[in_transit[-1]]
        duration_hr = (t_egress - t_ingress) * 24.0
        ax.axvline(t_ingress, color="#d62728", lw=0.8, ls=":", label=f"Ingress / Egress (~{duration_hr:.2f} h)")
        ax.axvline(t_egress, color="#d62728", lw=0.8, ls=":")

    ax.set_xlabel("Time from transit centre [days]", fontsize=11)
    ax.set_ylabel("Relative stellar flux", fontsize=11)
    ax.set_title(f"Model Transit Light Curve — {PLANET_NAME}", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_ylim(min(flux) - 0.003, 1.003)

    # Information box with physical parameters and references
    info_text = (
        f"Planet: {PLANET_NAME}\n"
        f"Period $P$: {params.per:.4f} d\n"
        f"Radius ratio $R_p/R_*$: {params.rp:.5f}\n"
        f"Scaled semi-major axis $a/R_*$: {params.a:.2f}\n"
        f"Inclination $i$: {params.inc:.2f}° (impact $b={b:.2f}$)\n"
        f"Limb darkening $u$: {params.u}\n"
        f"Transit depth: {actual_depth:.2f}%\n"
        f"Source: Southworth (2012)"
    )
    ax.text(
        0.03,
        0.06,
        info_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Static light curve saved to: {output_file}")


def save_transit_animation(output_file=f"{PLANET_SLUG}_transit.gif", params=None, n_frames=60, fps=15):
    """
    Generate an animated GIF showing the 2D transit geometry (sky-plane)
    synchronized with the real-time formation of the light curve.

    Parameters
    ----------
    output_file : str or Path
    params : batman.TransitParams, optional
    n_frames : int, optional
    fps : int, optional
    """
    if params is None:
        params = build_batman_params()

    t, flux = compute_light_curve(params, np.linspace(-0.08, 0.08, 500))

    # Transit geometry
    # Impact parameter b = a * cos(i) in units of stellar radius R*
    b = params.a * np.cos(np.radians(params.inc))
    # Semi-major axis in stellar radii
    # Projection x(t) = a * sin(2*pi * t / per)
    t_frames = np.linspace(t[0], t[-1], n_frames)
    x_positions = params.a * np.sin(2.0 * np.pi * t_frames / params.per)
    y_position = b

    # Precalculate model flux at the animation sample times
    model_anim = batman.TransitModel(params, t_frames)
    flux_frames = model_anim.light_curve(params)

    fig, (ax_sky, ax_lc) = plt.subplots(2, 1, figsize=(7.5, 6.5), dpi=120, gridspec_kw={"height_ratios": [1.1, 1.0]})

    # --- Top panel: Star & Planet disk ---
    ax_sky.set_xlim(-1.6, 1.6)
    ax_sky.set_ylim(-1.3, 1.3)
    ax_sky.set_aspect("equal")
    ax_sky.set_facecolor("#0b0c10")
    ax_sky.set_title(f"Transit Geometry (Sky-plane view) — {PLANET_NAME}", fontsize=11, fontweight="bold")
    ax_sky.axis("off")

    # Star disk with radial limb darkening representation
    for r_star in np.linspace(1.0, 0.05, 30):
        intensity = 1.0 - 0.37 * (1.0 - np.sqrt(max(0.0, 1.0 - r_star**2)))
        ax_sky.add_patch(patches.Circle((0, 0), r_star, color="#fdb813", alpha=0.12 * intensity))
    ax_sky.add_patch(patches.Circle((0, 0), 1.0, color="#f39c12", fill=False, lw=1.5))

    # Transit chord path
    ax_sky.axhline(y_position, color="#ffffff", ls="--", lw=0.8, alpha=0.35)

    # Planet disk
    planet_patch = patches.Circle((x_positions[0], y_position), params.rp, color="#111111", ec="#555555", lw=1.0)
    ax_sky.add_patch(planet_patch)

    # --- Bottom panel: Light curve ---
    ax_lc.set_xlim(t[0], t[-1])
    ax_lc.set_ylim(min(flux) - 0.003, 1.003)
    ax_lc.set_xlabel("Time from transit centre [days]", fontsize=10)
    ax_lc.set_ylabel("Relative flux", fontsize=10)
    ax_lc.grid(True, linestyle="--", alpha=0.3)
    ax_lc.plot(t, flux, color="#bdc3c7", lw=1.2, ls=":", label="Full model curve")

    (curve_line,) = ax_lc.plot([], [], color="#1f77b4", lw=2.2, label="Observed flux")
    (current_pt,) = ax_lc.plot([], [], marker="o", color="#e74c3c", markersize=7)
    ax_lc.legend(loc="upper right", fontsize=8)

    fig.tight_layout()

    def update(frame):
        # Update planet position on star
        planet_patch.center = (x_positions[frame], y_position)
        # Update light curve trail
        curve_line.set_data(t_frames[: frame + 1], flux_frames[: frame + 1])
        current_pt.set_data([t_frames[frame]], [flux_frames[frame]])
        return planet_patch, curve_line, current_pt

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_file, writer=writer)
    plt.close(fig)
    print(f"Transit animation GIF saved to: {output_file}")


if __name__ == "__main__":
    print(f"Running transit modelling for {PLANET_NAME}...")
    # 1. Generate static light curve
    static_png = f"{PLANET_SLUG}_assignment1_taskF.png"
    plot_light_curve(output_file=static_png)

    # 2. Generate animation GIF
    anim_gif = f"{PLANET_SLUG}_transit.gif"
    save_transit_animation(output_file=anim_gif, n_frames=60, fps=15)

    print("\nExecution completed successfully.")
