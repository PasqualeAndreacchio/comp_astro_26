import matplotlib.pyplot as plt
import numpy as np

from daneel.detection.transit import build_batman_params, compute_light_curve
from daneel.parameters import Parameters

pars_ref = dict(Parameters("examples/HD_209458_b.yaml").params["transit"])

pars_half = dict(pars_ref)
pars_half["rp"] = 0.5 * pars_ref["rp"]

pars_double = dict(pars_ref)
pars_double["rp"] = 2.0 * pars_ref["rp"]

t = np.linspace(-0.10, 0.10, 1000)
_, flux_half = compute_light_curve(build_batman_params(pars_half), t)
_, flux_ref = compute_light_curve(build_batman_params(pars_ref), t)
_, flux_double = compute_light_curve(build_batman_params(pars_double), t)

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=180)
ax.plot(t, flux_half, color="#d62728", lw=2.2, ls="--", label="Hypothetical (0.5 $R_p$)")
ax.plot(t, flux_ref, color="#1f77b4", lw=2.2, ls="-", label="HD 209458 b (1.0 $R_p$)")
ax.plot(t, flux_double, color="#2ca02c", lw=2.2, ls="-.", label="Hypothetical (2.0 $R_p$)")
ax.axhline(1.0, color="gray", lw=1.0, ls=":", alpha=0.6)
ax.set_xlabel("Time from transit centre [days]")
ax.set_ylabel("Relative stellar flux")
ax.set_title("Transit Light Curve Comparison — HD 209458 b (0.5, 1.0, 2.0 $R_p$)")
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig("assignment2_taskC.png", dpi=180, bbox_inches="tight")
plt.close(fig)