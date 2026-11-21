"""Fig. 3: states and global cost on the 5-agent network from the nominal x(t0).

Output: results/figures/{rl,am12,am14,gm21,yue22}_{states,cost}.eps
"""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
# embed text as Type 42 (TrueType), not Type 3, as required for the camera-ready PDF
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt

import adagm_do as m
import evaluate as ev

ROOT = Path("results")
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DT = 0.1

P = m.PROBLEMS["paper5"]()
x_star, f_star = P.solve_optimum()
T = json.load(open(ROOT / "evaluation" / "trajectories_nominal_x0.json"))

# RL: the seed with the median settling time (seed 1), re-simulated from the saved actor
run = ROOT / "paper5_fixed_s1" / "rl"
actor = m.load_saved_actor_for_evaluation(run / "actor_torch.pt")
x_rl, c_rl, _ = ev.simulate_env(ev.load_run_args(run), actor, P, P.x0)

# name: (states, cost, inset window [s])
cases = {
    "rl": (x_rl, c_rl, (10.0, 60.0)),
    "am12": (T["paper5|[12]"]["x"], T["paper5|[12]"]["cost"], (10.0, 60.0)),
    "am14": (T["paper5|[13]"]["x"], T["paper5|[13]"]["cost"], (40.0, 100.0)),
    "gm21": (T["paper5|[21]"]["x"], T["paper5|[21]"]["cost"], (40.0, 100.0)),
    "yue22": (T["paper5|[Yue22]"]["x"], T["paper5|[Yue22]"]["cost"], (40.0, 100.0)),
}

# Drawn at the exact size used in the paper (width=4.2cm, height=2.7cm), so LaTeX does not rescale
# and every size below is the printed size in pt.
CM = 1 / 2.54
FIGSIZE = (4.2 * CM, 2.7 * CM)
LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, INSET_TICK_SIZE = 8, 7, 6.5, 5


def fit_to_paper(fig, ax):
    ax.xaxis.label.set_size(LABEL_SIZE)
    ax.yaxis.label.set_size(LABEL_SIZE)
    ax.xaxis.labelpad = 1
    ax.yaxis.labelpad = 1
    for a in fig.axes:
        main = a is ax
        a.tick_params(axis="both", which="both", width=0.5, length=2.0 if main else 1.2, pad=1.5)
        for spine in a.spines.values():
            spine.set_linewidth(0.6 if main else 0.5)
        for line in a.get_lines():
            line.set_linewidth(line.get_linewidth() * (0.5 if main else 0.6))
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.tick_params(axis="x", which="major", pad=2.5)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    for axins in fig.axes:
        if axins is ax:
            continue
        # only the zoomed time window is labelled; y-values do not fit at this size
        axins.set_yticks([])
        t0, t1 = axins.get_xlim()
        axins.set_xticks([t0, t1])
        axins.set_xticklabels([f"{t0:g}", f"{t1:g}"])
        # above the inset: the space below it is crossed by the x* / f* line
        axins.tick_params(axis="x", labelsize=INSET_TICK_SIZE, pad=1.0, labeltop=True, labelbottom=False)
        axins.get_xticklabels()[0].set_ha("left")
        axins.get_xticklabels()[-1].set_ha("right")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(labels),
        frameon=True,
        prop={"weight": "bold", "size": LEGEND_SIZE},
        handlelength=1.4,
        handletextpad=0.3,
        columnspacing=0.6,
        borderpad=0.2,
        borderaxespad=0.2,
        framealpha=1.0,
    )


for name, (x, cost, zoom) in cases.items():
    x = np.asarray(x, dtype=float)
    cost = np.asarray(cost, dtype=float)
    t = np.arange(len(cost)) * DT

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=180)
    m._plot_state_panel(ax, t, x, x_star, zoom_start=zoom[0], zoom_end=zoom[1])
    if np.min(x) < 0.0:  # AM methods leave the positive orthant; do not clip it away
        lo, hi = ax.get_ylim()
        ax.set_ylim(np.min(x) - 0.12 * (hi - np.min(x)), hi)
    fit_to_paper(fig, ax)
    fig.set_layout_engine("constrained", w_pad=0.01, h_pad=0.01)
    fig.savefig(OUT / f"{name}_states.eps", format="eps")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=180)
    m._plot_cost_panel(ax, t, cost, f_star, color="#1f77b4", zoom_start=zoom[0], zoom_end=zoom[1])
    fit_to_paper(fig, ax)
    fig.set_layout_engine("constrained", w_pad=0.01, h_pad=0.01)
    fig.savefig(OUT / f"{name}_cost.eps", format="eps")
    plt.close(fig)


def outline_text(eps: Path) -> None:
    """Convert all text in the EPS to vector outlines, so the paper PDF gets no embedded (Type 3) fonts."""
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "fig.pdf"
        subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-dNoOutputFonts", "-dEPSCrop",
                        "-sDEVICE=pdfwrite", f"-sOutputFile={pdf}", str(eps)], check=True)
        subprocess.run(["pdftops", "-eps", str(pdf), str(eps)], check=True)


for eps in sorted(OUT.glob("*.eps")):
    outline_text(eps)

print("saved to", OUT)
