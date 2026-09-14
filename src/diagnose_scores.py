"""Stage 2.5 - go/no-go diagnostic on the transferred turn encoder.

The encoder is trained on Dreaddit (long retrospective Reddit prose) and applied
to ESConv (short synchronous chat turns). If that transfer fails, every
trajectory arm downstream is modelling noise and no amount of sequence
modelling fixes it. This script decides that before any of it gets built.

It answers four questions:
  Q1  Do the scores vary at all, or has the encoder collapsed to one value?
  Q2  Do they vary WITHIN a conversation? Between-conversation variance is
      useless to a trajectory model -- only within-conversation movement can
      carry a slope. Reported as ICC: the share of total variance that is
      between-conversation. High ICC means flat trajectories.
  Q3  Does any simple summary of the score sequence separate the target at all?
      If pooled AUROC is at chance for every summary, stop.
  Q4  Is there a turn-position artefact -- e.g. scores drifting purely with
      position regardless of content?

Run:  python -m src.diagnose_scores --seed 13
"""
import argparse
import json
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import ARTIFACTS, RESULTS


def load(seed):
    p = ARTIFACTS / f"scores_seed{seed}.pkl"
    if not p.exists():
        raise SystemExit(f"{p} not found. Run: python -m src.score_conversations --seed {seed}")
    with open(p, "rb") as f:
        return pickle.load(f)


def safe_auroc(y, s):
    s = np.asarray(s, dtype=float)
    if len(set(y)) < 2 or not np.isfinite(s).all() or np.allclose(s, s[0]):
        return float("nan")
    return roc_auc_score(y, s)


def summaries(seq):
    """Pointwise and order-sensitive summaries of one score sequence."""
    x = np.asarray(seq, dtype=float)
    n = len(x)
    t = np.arange(n)
    slope = np.polyfit(t, x, 1)[0] if n >= 2 and x.std() > 0 else 0.0
    half = max(1, n // 2)
    return {
        "last": x[-1],
        "first": x[0],
        "mean": x.mean(),
        "max": x.max(),
        "sd": x.std(),
        "slope": slope,
        "last_minus_first": x[-1] - x[0],
        "secondhalf_minus_firsthalf": x[half:].mean() - x[:half].mean(),
        "frac_above_half": float((x > 0.5).mean()),
    }


def main(seed):
    d = load(seed)
    scores, meta = d["scores"], d["meta"]
    y = np.array([m["label"] for m in meta])
    flat = np.concatenate(scores)

    print(f"\nconversations {len(scores)}   turns {len(flat)}   "
          f"positives {y.sum()} ({y.mean():.3f})")

    # ---------------------------------------------------------------- Q1
    print("\n[Q1] pooled turn-score distribution")
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    pct = np.percentile(flat, qs)
    print("     " + "  ".join(f"p{q}={v:.3f}" for q, v in zip(qs, pct)))
    print(f"     mean {flat.mean():.4f}   sd {flat.std():.4f}   "
          f"IQR {pct[6] - pct[4]:.4f}")
    dead = flat.std() < 0.02
    print(f"     -> {'COLLAPSED' if dead else 'varies'}"
          f"  (sd {'<' if dead else '>='} 0.02)")

    # ---------------------------------------------------------------- Q2
    print("\n[Q2] within- vs between-conversation variance")
    conv_means = np.array([s.mean() for s in scores])
    within = np.array([s.std() for s in scores])
    grand = flat.mean()
    ns = np.array([len(s) for s in scores])
    ss_between = float((ns * (conv_means - grand) ** 2).sum())
    ss_within = float(sum(((s - s.mean()) ** 2).sum() for s in scores))
    icc = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else float("nan")
    print(f"     mean within-conversation sd : {within.mean():.4f}")
    print(f"     between-conversation sd     : {conv_means.std():.4f}")
    print(f"     ICC (between / total)       : {icc:.4f}")
    if icc > 0.75:
        print("     -> FLAT trajectories: variance is almost all between "
              "conversations. A slope has nothing to ride on.")
    elif within.mean() < 0.05:
        print("     -> within-conversation movement is very small.")
    else:
        print("     -> usable within-conversation movement.")

    # ---------------------------------------------------------------- Q3
    print("\n[Q3] does any summary separate the target? (pooled AUROC)")
    feats = {}
    for s in scores:
        for k, v in summaries(s).items():
            feats.setdefault(k, []).append(v)
    order_free = {"last", "first", "mean", "max", "sd", "frac_above_half"}
    rows = []
    for k, v in feats.items():
        a = safe_auroc(y, v)
        rows.append((k, a, "pointwise" if k in order_free else "ORDER-SENSITIVE"))
    rows.sort(key=lambda r: -(r[1] if np.isfinite(r[1]) else 0))
    print(f"     {'summary':28s} {'AUROC':>7}   kind")
    for k, a, kind in rows:
        mark = " *" if np.isfinite(a) and abs(a - 0.5) >= 0.05 else "  "
        print(f"    {mark} {k:28s} {a:7.4f}   {kind}")
    best = max((r[1] for r in rows if np.isfinite(r[1])), default=float("nan"))
    best_ord = max((r[1] for r in rows if r[2] != "pointwise" and np.isfinite(r[1])),
                   default=float("nan"))
    best_pt = max((r[1] for r in rows if r[2] == "pointwise" and np.isfinite(r[1])),
                  default=float("nan"))

    # ---------------------------------------------------------------- Q4
    print("\n[Q4] score by normalised turn position (artefact check)")
    bins = np.zeros(10)
    cnt = np.zeros(10)
    for s in scores:
        n = len(s)
        for j, v in enumerate(s):
            b = min(9, int(10 * j / n))
            bins[b] += v
            cnt[b] += 1
    prof = bins / np.maximum(cnt, 1)
    print("     " + "  ".join(f"{v:.3f}" for v in prof))
    print(f"     drift first->last decile: {prof[-1] - prof[0]:+.4f}")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 62)
    if dead or (np.isfinite(icc) and icc > 0.85):
        v = ("STOP. The encoder does not transfer. Fine-tune it on ESConv turns "
             "or use a domain-pretrained encoder before building anything.")
    elif not np.isfinite(best) or best < 0.55:
        v = ("STOP. No summary beats chance. The scores carry no signal about "
             "the target; the trajectory layer cannot rescue this.")
    elif np.isfinite(best_ord) and np.isfinite(best_pt) and best_ord > best_pt + 0.02:
        v = (f"GO, and the premise holds: an order-sensitive summary "
             f"({best_ord:.3f}) already beats the best pointwise one "
             f"({best_pt:.3f}) before any model is fitted.")
    else:
        v = (f"GO, with a caveat: there is signal (best AUROC {best:.3f}) but "
             f"order-sensitive summaries do not yet beat pointwise ones "
             f"({best_ord:.3f} vs {best_pt:.3f}). H1 is live but unproven -- "
             f"which is exactly what the ablation is for.")
    print(v)
    print("=" * 62)

    out = {"seed": seed, "n_conversations": len(scores), "n_turns": int(len(flat)),
           "pooled_mean": float(flat.mean()), "pooled_sd": float(flat.std()),
           "mean_within_sd": float(within.mean()),
           "between_sd": float(conv_means.std()), "icc": float(icc),
           "auroc": {k: (float(a) if np.isfinite(a) else None) for k, a, _ in rows},
           "position_profile": [float(x) for x in prof], "verdict": v}
    (RESULTS / f"diagnostic_seed{seed}.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved results/diagnostic_seed{seed}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=13)
    main(p.parse_args().seed)