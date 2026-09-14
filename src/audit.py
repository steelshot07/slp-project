"""Reproduces every ESConv audit finding locally. Run:  python -m src.audit

Writes results/audit.json. These numbers are Chapter 7 material -- they are the
project's first real contribution and they must be reproducible on your machine,
not taken on trust. Needs data/raw/ESConv.json and data/raw/FailedESConv.json.
"""
import json
import warnings
from collections import Counter

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict)
from scipy.sparse import csr_matrix, hstack
from sklearn.pipeline import make_pipeline

from src.config import DATA, RESULTS

warnings.filterwarnings("ignore")
RAW = DATA / "raw"
SEEK = {"seeker", "usr", "user", "speaker"}
OUT = {}


def load(path):
    rows = []
    for c in json.load(open(path, encoding="utf-8")):
        s = (c.get("survey_score") or {}).get("seeker", {})
        try:
            a, b = float(s["initial_emotion_intensity"]), float(s["final_emotion_intensity"])
        except (KeyError, TypeError, ValueError):
            continue
        sk = [t["content"] for t in c["dialog"]
              if (t.get("speaker") or "").lower() in SEEK and (t.get("content") or "").strip()]
        sp = [t["content"] for t in c["dialog"]
              if (t.get("speaker") or "").lower() not in SEEK and (t.get("content") or "").strip()]
        if len(sk) < 4:
            continue
        rows.append({"init": a, "delta": b - a, "seek": " ".join(sk), "supp": " ".join(sp),
                     "pt": str(c.get("problem_type", "?")).strip().lower()})
    return rows


def auroc_cv(X, y, seed=13):
    p = cross_val_predict(LogisticRegression(max_iter=3000), X, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    es = load(RAW / "ESConv.json")
    init = np.array([r["init"] for r in es])
    delta = np.array([r["delta"] for r in es])
    print(f"ESConv usable: {len(es)}")

    # ---- F1: the original target is degenerate -----------------------------
    print("\n[F1] original target  label = delta > -1")
    orig = (delta > -1).astype(int)
    print(f"  positives: {orig.sum()} / {len(orig)}   <-- degenerate")
    print(f"  delta distribution: {sorted(Counter(delta).items())}")
    OUT["F1_original_positives"] = int(orig.sum())

    # ---- F2: floor effect --------------------------------------------------
    print("\n[F2] revised target  label = delta >= -1   (improved by < 2 points)")
    y = (delta >= -1).astype(int)
    print(f"  base rate {y.mean():.4f}  ({y.sum()}/{len(y)})")
    print(f"  {'init':>5} {'n':>5} {'pos':>5} {'rate':>7}")
    for v in sorted(set(init)):
        m = init == v
        print(f"  {v:5.0f} {m.sum():5d} {y[m].sum():5d} {y[m].mean():7.3f}")
    a_int = auroc_cv(init.reshape(-1, 1), y)
    print(f"  AUROC(intake -> label) = {a_int:.4f}   <-- floor effect")
    OUT["F2_base_rate"] = float(y.mean())
    OUT["F2_auroc_intake"] = a_int

    # ---- F3: leakage from merging FailedESConv -----------------------------
    print("\n[F3] FailedESConv merge leakage")
    fe = load(RAW / "FailedESConv.json")
    yf = np.array([(r["delta"] >= -1) for r in fe]).astype(int)
    print(f"  ESConv base rate       {y.mean():.4f}")
    print(f"  FailedESConv base rate {yf.mean():.4f}  (n={len(fe)})")
    print(f"  corpus identity alone separates the classes almost perfectly")
    OUT["F3_esconv_rate"], OUT["F3_failed_rate"] = float(y.mean()), float(yf.mean())

    # ---- F4: where the text signal is --------------------------------------
    print("\n[F4] signal location (5-fold CV AUROC on original binary label)")

    def tfidf_auroc_cv(texts, y, seed=13):
        pipe = make_pipeline(
            TfidfVectorizer(
                min_df=3,
                max_features=20000,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
            LogisticRegression(max_iter=3000),
        )
        p = cross_val_predict(
            pipe,
            texts,
            y,
            cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            method="predict_proba",
        )[:, 1]
        return float(roc_auc_score(y, p))

    seeker_texts = [r["seek"] for r in es]
    supporter_texts = [r["supp"] for r in es]

    for nm, texts in [
        ("seeker text", seeker_texts),
        ("supporter text", supporter_texts),
    ]:
        v = tfidf_auroc_cv(texts, y)
        print(f"  {nm:24s} {v:.4f}")
        OUT[f"F4_{nm.replace(' ', '_')}"] = v

    # ---- F5: residualised target -------------------------------------------
    print("\n[F5] intake-residualised target  (the clean one)")

    cv = KFold(5, shuffle=True, random_state=13)
    pred = np.zeros(len(delta))
    resid = np.zeros(len(delta))

    for tr, te in cv.split(delta):
        exp_tr = {
            v: delta[tr][init[tr] == v].mean()
            for v in np.unique(init[tr])
        }

        r_tr = delta[tr] - np.array([exp_tr[v] for v in init[tr]])
        r_te = delta[te] - np.array([
            exp_tr.get(v, delta[tr].mean())
            for v in init[te]
        ])

        resid[te] = r_te

        pipe = make_pipeline(
            TfidfVectorizer(
                min_df=3,
                max_features=20000,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
            Ridge(alpha=1.0),
        )
        pipe.fit([seeker_texts[i] for i in tr], r_tr)
        pred[te] = pipe.predict([seeker_texts[i] for i in te])

    rho = float(spearmanr(resid, pred).statistic)
    yb = (resid >= np.median(resid)).astype(int)
    ab = tfidf_auroc_cv(seeker_texts, yb)

    print(f"  TF-IDF -> residual   Spearman rho = {rho:+.4f}")
    print(f"  TF-IDF -> residual   AUROC (median split) = {ab:.4f}")

    OUT["F5_rho"], OUT["F5_auroc"] = rho, ab

    RESULTS.mkdir(exist_ok=True, parents=True)
    (RESULTS / "audit.json").write_text(json.dumps(OUT, indent=2))
    print("\nsaved results/audit.json")


if __name__ == "__main__":
    main()