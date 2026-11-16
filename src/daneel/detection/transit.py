"""
Transit modelling module for exoplanet light curves.

Target: HD 209458 b (Osiris)
Host star: HD 209458 (G0V, Teff ~ 6065 K)
References:
    - Charbonneau et al. (2000), ApJ, 529, L45 (First detection of transit)
    - Southworth, J. (2010), MNRAS, 408(3), 1689-1713 (Homogeneous studies of transiting extrasolar planets)
    - NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
"""

import argparse
import importlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import setuptools  # noqa: F401  # Provides distutils compatibility for batman on Python 3.12+

batman = importlib.import_module("batman")

# Default target parameters: HD 209458 b
PLANET_NAME = "HD 209458 b"
PLANET_SLUG = "HD_209458_b"

DEFAULT_PARAMS = {
    "t0": 0.0,              # Time of inferior conjunction [days]
    "per": 3.52474859,      # Orbital period [days]
    "rp": 0.12086,          # Planet-to-star radius ratio (Rp / Rstar)
    "a": 8.76,              # Scaled semi-major axis (a / Rstar)
    "inc": 86.71,           # Orbital inclination [degrees]
    "ecc": 0.0,             # Eccentricity (circular orbit)
    "w": 90.0,              # Longitude of periastron [degrees]
    "u": [0.37, 0.28],      # Quadratic limb darkening coefficients (optical/V band)
    "limb_dark": "quadratic",
}


def get_planet_slug(planet_name: str) -> str:
    """
    Sanitise a planet name for use in filenames.

    Example
    -------
    >>> get_planet_slug("HD 209458 b")
    'HD_209458_b'
    >>> get_planet_slug("WASP-12 b")
    'WASP_12_b'
    """
    return re.sub(r"[^\w]+", "_", planet_name).strip("_")


def fetch_planet_from_archive(planet_name: str, max_retries: int = 3):
    """
    Query the NASA Exoplanet Archive (TAP API) to retrieve orbital and physical parameters.

    Parameters
    ----------
    planet_name : str
        Name of the planet (e.g. 'HD 209458 b', 'WASP-12 b', 'TRAPPIST-1 b').
    max_retries : int, optional
        Number of retry attempts in case of transient network issues.

    Returns
    -------
    official_name : str
    params_dict : dict
        Parameters dictionary ready for `build_batman_params`.
    trandur_days : float
        Transit duration in days (used to size the observation window).
    source : str
        Provenance description string.
    """
    clean_name = planet_name.strip()
    query = (
        "SELECT pl_name, hostname, pl_orbper, pl_orbincl, pl_orbeccen, pl_orblper, "
        "pl_ratror, pl_ratdor, pl_rade, pl_radj, st_rad, pl_orbsmax, pl_trandur, st_teff "
        f"FROM pscomppars WHERE lower(pl_name) = lower('{clean_name}')"
    )
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=" + urllib.parse.quote(query) + "&format=json"

    data = None
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "comp_astro_daneel/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                break
        except Exception as e:
            last_err = e
            time.sleep(1.0)

    if data is None:
        raise RuntimeError(f"Could not connect to NASA Exoplanet Archive: {last_err}")
    if not data:
        raise ValueError(f"Planet '{planet_name}' not found in NASA Exoplanet Archive (pscomppars).")

    row = data[0]

    # 1. Orbital period [days]
    if row.get("pl_orbper") is None:
        raise ValueError(f"Orbital period (pl_orbper) missing for '{planet_name}'.")
    per = float(row["pl_orbper"])

    # 2. Rp / Rstar
    rp = row.get("pl_ratror")
    if rp is None and row.get("st_rad"):
        st_rad = float(row["st_rad"])
        if row.get("pl_radj") is not None:
            rp = (float(row["pl_radj"]) * 0.10276) / st_rad
        elif row.get("pl_rade") is not None:
            rp = (float(row["pl_rade"]) * 0.0091675) / st_rad
    if rp is None:
        raise ValueError(f"Radius ratio Rp/R* missing and cannot be derived for '{planet_name}'.")
    rp = float(rp)

    # 3. a / Rstar
    a = row.get("pl_ratdor")
    if a is None and row.get("st_rad") and row.get("pl_orbsmax"):
        a = (float(row["pl_orbsmax"]) * 215.032) / float(row["st_rad"])
    if a is None:
        a = 10.0
    a = float(a)

    # 4. Inclination [degrees]
    inc = float(row.get("pl_orbincl") or 90.0)

    # 5. Eccentricity
    ecc = float(row.get("pl_orbeccen") or 0.0)

    # 6. Longitude of periastron [degrees]
    w = float(row.get("pl_orblper") or 90.0)

    # 7. Transit duration in days
    trandur_days = None
    if row.get("pl_trandur") is not None:
        trandur_days = float(row["pl_trandur"]) / 24.0
    else:
        # Analytical approximation
        b = a * np.cos(np.radians(inc))
        if b < 1.0 + rp:
            arg = max(0.0, (1.0 + rp) ** 2 - b ** 2)
            trandur_days = (per / np.pi) * (1.0 / a) * np.sqrt(arg) / max(0.01, np.sin(np.radians(inc)))
        else:
            trandur_days = 0.1

    params_dict = {
        "t0": 0.0,
        "per": per,
        "rp": rp,
        "a": a,
        "inc": inc,
        "ecc": ecc,
        "w": w,
        "u": [0.37, 0.28],
        "limb_dark": "quadratic",
    }

    official_name = row.get("pl_name", clean_name)
    source = "NASA Exoplanet Archive (pscomppars)"
    return official_name, params_dict, trandur_days, source


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
    params.per = float(p_dict.get("per", 3.52474859))
    params.rp = float(p_dict.get("rp", 0.12086))
    params.a = float(p_dict.get("a", 8.76))
    params.inc = float(p_dict.get("inc", 86.71))
    params.ecc = float(p_dict.get("ecc", 0.0))
    params.w = float(p_dict.get("w", 90.0))
    params.limb_dark = str(p_dict.get("limb_dark", "quadratic"))
    params.u = list(p_dict.get("u", [0.37, 0.28]))
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


def plot_light_curve(
    output_file=None,
    params=None,
    t=None,
    planet_name=PLANET_NAME,
    source_ref="Southworth (2010)",
    show=False,
):
    """
    Generate and save the static transit light curve.

    Parameters
    ----------
    output_file : str or Path, optional
    params : batman.TransitParams, optional
    t : ndarray, optional
    planet_name : str, optional
    source_ref : str, optional
    show : bool, optional
    """
    if params is None:
        params = build_batman_params()
    if output_file is None:
        slug = get_planet_slug(planet_name)
        output_file = f"{slug}_light_curve.png"

    t, flux = compute_light_curve(params, t)

    # Physical transit depth and impact parameter
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
    ax.set_title(f"Model Transit Light Curve — {planet_name}", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_ylim(min(flux) - 0.003, 1.003)

    # Information box with physical parameters and references
    info_text = (
        f"Planet: {planet_name}\n"
        f"Period $P$: {params.per:.4f} d\n"
        f"Radius ratio $R_p/R_*$: {params.rp:.5f}\n"
        f"Scaled semi-major axis $a/R_*$: {params.a:.2f}\n"
        f"Inclination $i$: {params.inc:.2f}° (impact $b={b:.2f}$)\n"
        f"Limb darkening $u$: {params.u}\n"
        f"Transit depth: {actual_depth:.2f}%\n"
        f"Source: {source_ref}"
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


def save_transit_animation(
    output_file=None,
    params=None,
    t_span=None,
    planet_name=PLANET_NAME,
    n_frames=60,
    fps=15,
):
    """
    Generate an animated GIF showing the 2D transit geometry (sky-plane)
    synchronized with the real-time formation of the light curve.

    Parameters
    ----------
    output_file : str or Path, optional
    params : batman.TransitParams, optional
    t_span : float, optional
        Half-width of the time window [days]. If None, defaults to 0.08.
    planet_name : str, optional
    n_frames : int, optional
    fps : int, optional
    """
    if params is None:
        params = build_batman_params()
    if output_file is None:
        slug = get_planet_slug(planet_name)
        output_file = f"{slug}_transit.gif"

    half_window = t_span if t_span is not None else 0.08
    t, flux = compute_light_curve(params, np.linspace(-half_window, half_window, 500))

    # Transit geometry
    # Impact parameter b = a * cos(i) in units of stellar radius R*
    b = params.a * np.cos(np.radians(params.inc))
    t_frames = np.linspace(t[0], t[-1], n_frames)
    x_positions = params.a * np.sin(2.0 * np.pi * t_frames / params.per)
    y_position = b

    # Precalculate model flux at the animation sample times
    model_anim = batman.TransitModel(params, t_frames)
    flux_frames = model_anim.light_curve(params)

    fig, (ax_sky, ax_lc) = plt.subplots(2, 1, figsize=(7.5, 6.5), dpi=120, gridspec_kw={"height_ratios": [1.1, 1.0]})

    # --- Top panel: Star & Planet disk ---
    x_limit = max(1.6, float(np.max(np.abs(x_positions)) * 1.15))
    y_limit = max(1.3, float(abs(y_position) + params.rp + 0.3))
    ax_sky.set_xlim(-x_limit, x_limit)
    ax_sky.set_ylim(-y_limit, y_limit)
    ax_sky.set_aspect("equal")
    ax_sky.set_facecolor("#0b0c10")
    ax_sky.set_title(f"Transit Geometry (Sky-plane view) — {planet_name}", fontsize=11, fontweight="bold")
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


def run_for_planet(planet_name: str = PLANET_NAME, from_archive: bool = True, output_suffix: str = "light_curve"):
    """
    Orchestrate parameter fetching, static light curve plotting, and GIF animation.

    Parameters
    ----------
    planet_name : str, optional
    from_archive : bool, optional
    output_suffix : str, optional
    """
    slug = get_planet_slug(planet_name)
    source_ref = "Southworth (2010)"
    trandur_days = None

    if from_archive and planet_name.lower() != "default":
        print(f"Querying NASA Exoplanet Archive for '{planet_name}'...")
        try:
            official_name, params_dict, trandur_days, source_ref = fetch_planet_from_archive(planet_name)
            planet_name = official_name
            slug = get_planet_slug(planet_name)
            params = build_batman_params(params_dict)
            print(f"Retrieved parameters for {planet_name}:")
            print(f"  Period: {params.per:.5f} d | Rp/R*: {params.rp:.5f} | a/R*: {params.a:.2f} | inc: {params.inc:.2f}°")
        except Exception as e:
            if planet_name == PLANET_NAME:
                print(f"Archive query failed ({e}); falling back to local DEFAULT_PARAMS.")
                params = build_batman_params()
            else:
                raise
    else:
        params = build_batman_params()

    # Determine adaptive time grid
    if trandur_days is not None and trandur_days > 0:
        t_half = max(0.03, 1.4 * (trandur_days / 2.0))
    else:
        t_half = 0.10

    t_grid = np.linspace(-t_half, t_half, 1000)

    # 1. Static light curve
    static_png = f"{slug}_{output_suffix}.png"
    plot_light_curve(
        output_file=static_png,
        params=params,
        t=t_grid,
        planet_name=planet_name,
        source_ref=source_ref,
    )

    # 2. Animation GIF
    anim_gif = f"{slug}_transit.gif"
    save_transit_animation(
        output_file=anim_gif,
        params=params,
        t_span=t_half * 0.9,
        planet_name=planet_name,
        n_frames=60,
        fps=15,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exoplanet Transit Light Curve and Animation Generator")
    parser.add_argument(
        "planet",
        nargs="?",
        default=PLANET_NAME,
        help=f"Target planet name (default: '{PLANET_NAME}'). Example: 'WASP-12 b', 'TRAPPIST-1 b'",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force using local DEFAULT_PARAMS instead of querying the NASA Exoplanet Archive",
    )
    parser.add_argument(
        "--suffix",
        default="light_curve",
        help="Suffix for output PNG (default: 'light_curve')",
    )

    args = parser.parse_args()

    use_archive = not args.local
    print(f"Running transit modelling for target: {args.planet}...")
    run_for_planet(planet_name=args.planet, from_archive=use_archive, output_suffix=args.suffix)
    print("\nExecution completed successfully.")
