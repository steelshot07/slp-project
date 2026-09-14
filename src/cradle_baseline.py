"""CRADLE turn-level floor baselines. CPU only, a few minutes.

Establishes what a trivial system scores BEFORE any GPU time, so the neural
model has a real bar to clear. Follows the paper's protocol: predict labels for
each USER turn, with dialogue history available as context; listener turns are
context only, never prediction targets.

Published turn-level micro F1 on this benchmark (Byun et al.):
    Qwen3-32B 43.75 | gpt-oss-120b 47.20 | their FT-32B 51.31 | Claude 56.85

Run:  python -m src.cradle_baseline
"""
import json
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import DATA, RESULTS

CR = DATA / "cradle"
CTX = 4  # preceding turns fed as context


def load(name):
    return [json.loads(l) for l in open(CR / name, encoding="utf-8")]


def to_examples(dialogues):
    """One example per USER turn; context = preceding CTX turns."""
    X, Y, meta = [], [], []
    for d in dialogues:
        turns = d["turns"]
        for i, t in enumerate(turns):
            if t["speaker"] != "user":
                continue
            hist = turns[max(0, i - CTX):i]
            ctx = " ".join(f"<{h['speaker']}> {h['text']}" for h in hist)
            X.append(f"{ctx} <CURRENT> {t['text']}")
            Y.append(t["labels"])
            meta.append({"conv": d["conv_id"], "idx": i,
                         "user_idx": sum(1 for x in turns[:i + 1] if x["speaker"] == "user")})
    return X, Y, meta


def prf(gold, pred):
    """Multi-label micro/macro P, R, F1 over the label matrix."""
    tp = (gold & pred).sum(0)
    fp = ((1 - gold) & pred).sum(0)
    fn = (gold & (1 - pred)).sum(0)
    mi_p = tp.sum() / max(tp.sum() + fp.sum(), 1)
    mi_r = tp.sum() / max(tp.sum() + fn.sum(), 1)
    mi_f = 2 * mi_p * mi_r / max(mi_p + mi_r, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.divide(tp, tp + fp, out=np.zeros(len(tp), float), where=(tp + fp) > 0)
        r = np.divide(tp, tp + fn, out=np.zeros(len(tp), float), where=(tp + fn) > 0)
        f = np.divide(2 * p * r, p + r, out=np.zeros(len(tp), float), where=(p + r) > 0)
    em = (gold == pred).all(1).mean()
    return dict(micro_p=mi_p, micro_r=mi_r, micro_f1=mi_f,
                macro_f1=f.mean(), exact_match=em), f


def main():
    tr = load("cradle_train.jsonl")          # synthetic
    te = load("cradle_test.jsonl")           # human-annotated benchmark
    Xtr, Ytr, _ = to_examples(tr)
    Xte, Yte, Mte = to_examples(te)
    print(f"train user-turns {len(Xtr)}   test user-turns {len(Xte)}")
    print(f"test turns carrying >=1 label: {sum(bool(y) for y in Yte)} "
          f"({sum(bool(y) for y in Yte)/len(Yte):.3%})")

    mlb = MultiLabelBinarizer()
    mlb.fit(Ytr + Yte)
    Gte = mlb.transform(Yte).astype(int)
    print(f"label space: {len(mlb.classes_)}")

    res = {}

    # ---- baseline 0: predict nothing -------------------------------------
    m, _ = prf(Gte, np.zeros_like(Gte))
    res["always_none"] = m
    print(f"\n[always-none]  microF1 {m['micro_f1']:.4f}  EM {m['exact_match']:.4f}")
    print("  (EM is high because most turns carry no label -- this is why the "
          "paper's EM column looks flattering and micro F1 is the real metric)")

    # ---- baseline 1: TF-IDF on the current turn only ----------------------
    # ---- baseline 2: TF-IDF with CTX turns of history ---------------------
    for tag, getter in [("tfidf_current_only", lambda s: s.split("<CURRENT>")[-1]),
                        ("tfidf_with_context", lambda s: s)]:
        V = TfidfVectorizer(min_df=3, max_features=60000, ngram_range=(1, 2),
                            sublinear_tf=True)
        A = V.fit_transform([getter(x) for x in Xtr])
        B = V.transform([getter(x) for x in Xte])
        clf = OneVsRestClassifier(
            LogisticRegression(max_iter=1500, class_weight="balanced", C=1.0), n_jobs=-1)
        clf.fit(A, mlb.transform(Ytr).astype(int))
        P = clf.predict(B)
        m, per = prf(Gte, P)
        res[tag] = m
        print(f"\n[{tag}]  microF1 {m['micro_f1']:.4f}  macroF1 {m['macro_f1']:.4f}  "
              f"P {m['micro_p']:.4f}  R {m['micro_r']:.4f}  EM {m['exact_match']:.4f}")
        if tag == "tfidf_with_context":
            order = np.argsort(-Gte.sum(0))
            print(f"    {'label':30s} {'support':>8} {'F1':>7}")
            for j in order:
                print(f"    {mlb.classes_[j]:30s} {Gte[:, j].sum():8d} {per[j]:7.4f}")
            al = [j for j, c in enumerate(mlb.classes_) if c.startswith("alert")]
            g2, p2 = Gte[:, al], P[:, al]
            ma, _ = prf(g2, p2)
            print(f"\n    ALERT-only microF1 {ma['micro_f1']:.4f}  "
                  f"(the paper's weakest category across all models)")
            res["alert_only_microf1"] = ma["micro_f1"]

    print("\n" + "=" * 64)
    print(f"{'floor (tfidf+context) turn-level microF1':48s} "
          f"{res['tfidf_with_context']['micro_f1']*100:6.2f}")
    for k, v in [("Qwen3-32B", 43.75), ("gpt-oss-120b", 47.20),
                 ("their FT Qwen3-32B", 51.31), ("Claude-4.5-Sonnet", 56.85)]:
        print(f"{'  paper: ' + k:48s} {v:6.2f}")
    print("=" * 64)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cradle_baseline.json").write_text(
        json.dumps({k: (v if not isinstance(v, dict) else
                        {kk: float(vv) for kk, vv in v.items()})
                    for k, v in res.items()}, indent=2))
    print("saved results/cradle_baseline.json")


if __name__ == "__main__":
    main()