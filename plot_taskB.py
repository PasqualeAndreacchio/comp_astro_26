import numpy as np
import matplotlib.pyplot as plt
from daneel.parameters import Parameters
from daneel.detection.transit import build_batman_params, compute_light_curve

pars_ref = dict(Parameters("examples/HD_209458_b.yaml").params["transit"])
pars_half = dict(pars_ref)
pars_half["rp"] = 0.5 * pars_ref["rp"]

t = np.linspace(-0.09, 0.09, 1000)
_, flux_ref = compute_light_curve(build_batman_params(pars_ref), t)
_, flux_half = compute_light_curve(build_batman_params(pars_half), t)

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=180)
ax.plot(t, flux_ref, color="#1f77b4", lw=2.2, label="HD 209458 b (1.0 $R_p$)")
ax.plot(t, flux_half, color="#d62728", lw=2.2, ls="--", label="Hypothetical planet (0.5 $R_p$)")
ax.axhline(1.0, color="gray", lw=1.0, ls="--", alpha=0.6)
ax.set_xlabel("Time from transit centre [days]")
ax.set_ylabel("Relative stellar flux")
ax.set_title("Transit Light Curve Comparison — HD 209458 b (1.0 $R_p$ vs 0.5 $R_p$)")
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig("assignment2/assignment2_taskB.png", dpi=180, bbox_inches="tight")
plt.close(fig)