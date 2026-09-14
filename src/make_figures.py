"""Generate report figures from results/*.json.

Run:  python -m src.make_figures
Writes PDFs to results/figures/ -- upload those four files to Overleaf.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import RESULTS

FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 200})
W, H = 3.4, 2.5          # single IEEE column


def save(fig, name):
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


# ---------------------------------------------------------------- Fig 1: floor
def fig_floor():
    a = json.load(open(RESULTS / "audit.json"))
    init = [2, 3, 4, 5]
    rate = [1.000, 0.630, 0.344, 0.164]
    n = [32, 238, 503, 377]
    fig, ax = plt.subplots(figsize=(W, H))
    bars = ax.bar([str(i) for i in init], rate, color="#4a6fa5", edgecolor="black", lw=0.5)
    ax.axhline(a["F2_base_rate"], ls="--", c="crimson", lw=1,
               label=f"corpus base rate = {a['F2_base_rate']:.3f}")
    for b, v, m in zip(bars, rate, n):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"n={m}", ha="center", fontsize=6)
    ax.set_xlabel("Initial distress intensity (1-5 scale)")
    ax.set_ylabel("P(insufficient improvement)")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=6, loc="upper right")
    save(fig, "fig_floor")


# ----------------------------------------------------------- Fig 2: inversion
def fig_inversion():
    arms = ["none", "mean", "gru"]
    val, test, vsd, tsd = [], [], [], []
    for a in arms:
        fs = [f for f in glob.glob(str(RESULTS / f"cradle_{a}_seed*.json")) if "ep12" not in f]
        v = np.array([json.load(open(f))["val_micro_f1_at_best"] for f in fs]) * 100
        t = np.array([json.load(open(f))["micro_f1"] for f in fs]) * 100
        val.append(v.mean()); test.append(t.mean())
        vsd.append(v.std(ddof=1)); tsd.append(t.std(ddof=1))
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(W, H))
    ax.errorbar(x, val, yerr=vsd, marker="o", lw=1.4, capsize=3,
                c="#c0392b", label="synthetic validation")
    ax.errorbar(x, test, yerr=tsd, marker="s", lw=1.4, capsize=3,
                c="#2c5f8a", label="human test")
    ax.set_xticks(x); ax.set_xticklabels(["no history", "order-free", "order-sensitive"])
    ax.set_ylabel("Turn-level micro F1")
    ax.set_ylim(20, 70)
    ax.legend(fontsize=6, loc="lower left", framealpha=0.9)
    save(fig, "fig_inversion")


# -------------------------------------------------------------- Fig 3: epochs
def fig_epochs():
    seeds = [13, 71, 97]
    fig, ax = plt.subplots(figsize=(W, H))
    for a, c in [("none", "#2c5f8a"), ("gru", "#c0392b")]:
        v6, t6, v12, t12 = [], [], [], []
        for s in seeds:
            p6 = RESULTS / f"cradle_{a}_seed{s}.json"
            p12 = RESULTS / f"cradle_{a}_seed{s}_ep12.json"
            if not (os.path.exists(p6) and os.path.exists(p12)):
                continue
            d6, d12 = json.load(open(p6)), json.load(open(p12))
            v6.append(d6["val_micro_f1_at_best"] * 100); t6.append(d6["micro_f1"] * 100)
            v12.append(d12["val_micro_f1_at_best"] * 100); t12.append(d12["micro_f1"] * 100)
        ax.plot([6, 12], [np.mean(v6), np.mean(v12)], "o--", c=c, lw=1.4,
                label=f"{a}: synth. val")
        ax.plot([6, 12], [np.mean(t6), np.mean(t12)], "s-", c=c, lw=1.4, alpha=0.6,
                label=f"{a}: human test")
    ax.set_xlabel("Training epochs"); ax.set_ylabel("Turn-level micro F1")
    ax.set_xticks([6, 12]); ax.set_ylim(20, 70)
    ax.legend(fontsize=5.5, ncol=2, loc="center right")
    save(fig, "fig_epochs")


# ------------------------------------------------------------ Fig 4: transfer
def fig_transfer():
    conds = ["human400", "synth400", "synth400+cal", "synthfull"]
    lab = ["human\n400", "synthetic\n400", "synthetic 400\n+human cal.", "synthetic\n3058"]
    vals = {c: [] for c in conds}
    for f in glob.glob(str(RESULTS / "transfer_seed*.json")):
        d = json.load(open(f))
        for r in d["per_fold"]:
            vals[r["condition"]].append(r["micro_f1"] * 100)
    m = [np.mean(vals[c]) for c in conds]
    sd = [np.std(vals[c], ddof=1) for c in conds]
    fig, ax = plt.subplots(figsize=(W * 1.05, H))
    cols = ["#2c5f8a", "#c0392b", "#e08a54", "#7f8c8d"]
    b = ax.bar(range(4), m, yerr=sd, capsize=3, color=cols, edgecolor="black", lw=0.5)
    for i, (bb, v) in enumerate(zip(b, m)):
        ax.text(bb.get_x() + bb.get_width() / 2, v + sd[i] + 0.6, f"{v:.1f}",
                ha="center", fontsize=6.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=6)
    ax.set_ylabel("Turn-level micro F1 (human test)")
    ax.set_ylim(0, 45)
    ax.annotate("", xy=(0, 39.5), xytext=(1, 39.5),
                arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(0.5, 40.2, "+6.42, p<.001", ha="center", fontsize=6)
    save(fig, "fig_transfer")


if __name__ == "__main__":
    fig_floor(); fig_inversion(); fig_epochs(); fig_transfer()
    print("\nUpload results/figures/*.pdf to Overleaf.")