"""
Stage 1 - build the two corpora.

Dreaddit  [16] -> turn-encoder supervision (binary stress on short spans)
ESConv    [17] -> multi-turn support conversations + seeker pre/post distress intensity

Run:  python -m src.data_prep
Outputs: data/dreaddit_{train,val,test}.jsonl, data/esconv.jsonl
"""
import json
import os
import re
import sys
import zipfile
import urllib.request

import numpy as np
import pandas as pd

from src.config import (DATA, CHUNK_TRAIN, CHUNK_MIN_WORDS, CHUNK_MAX_SENTS,
                        MIN_SEEKER_TURNS, IMPROVEMENT_MARGIN, SEEDS)

DREADDIT_URLS = {
    # Mirrors of Turcan & McKeown (2019). If both fail, download manually (see README)
    # and drop dreaddit-train.csv / dreaddit-test.csv into data/raw/.
    "dreaddit-train.csv": "https://raw.githubusercontent.com/gillian850413/Insight_Stress_Analysis/master/data/dreaddit-train.csv",
    "dreaddit-test.csv": "https://raw.githubusercontent.com/gillian850413/Insight_Stress_Analysis/master/data/dreaddit-test.csv",
}

RAW = DATA / "raw"
RAW.mkdir(exist_ok=True, parents=True)


# --------------------------------------------------------------------- helpers
def _fetch(name, url):
    dest = RAW / name
    if dest.exists():
        print(f"  [cached] {name}")
        return dest
    print(f"  [fetch ] {name}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _sent_split(text):
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p for p in parts if p.strip()]


def _chunk(text, max_sents=CHUNK_MAX_SENTS, min_words=CHUNK_MIN_WORDS):
    """Split a long post into short spans so the encoder sees turn-length input."""
    sents = _sent_split(text)
    out, buf = [], []
    for s in sents:
        buf.append(s)
        if len(buf) >= max_sents:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return [c for c in out if len(c.split()) >= min_words] or [str(text)]


# ------------------------------------------------------------------- dreaddit
def build_dreaddit():
    print("Dreaddit:")
    try:
        for n, u in DREADDIT_URLS.items():
            _fetch(n, u)
    except Exception as e:                                    # noqa: BLE001
        print(f"  ! automatic download failed ({e}).")
        print("  ! Place dreaddit-train.csv and dreaddit-test.csv in data/raw/ and rerun.")
        if not (RAW / "dreaddit-train.csv").exists():
            sys.exit(1)

    tr = pd.read_csv(RAW / "dreaddit-train.csv")
    te = pd.read_csv(RAW / "dreaddit-test.csv")

    text_col = "text" if "text" in tr.columns else tr.columns[tr.columns.str.contains("text")][0]
    tr = tr[[text_col, "label", "subreddit"]].rename(columns={text_col: "text"})
    te = te[[text_col, "label", "subreddit"]].rename(columns={text_col: "text"})

    # carve a calibration/validation split out of train, post-disjoint
    rng = np.random.default_rng(SEEDS[0])
    idx = rng.permutation(len(tr))
    n_val = int(0.15 * len(tr))
    val = tr.iloc[idx[:n_val]].copy()
    trn = tr.iloc[idx[n_val:]].copy()

    def emit(df, path, chunk):
        rows = []
        for _, r in df.iterrows():
            spans = _chunk(r["text"]) if chunk else [r["text"]]
            for sp in spans:
                rows.append({"text": sp, "label": int(r["label"]), "subreddit": r["subreddit"]})
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  {path.name}: {len(rows)} spans from {len(df)} posts")

    emit(trn, DATA / "dreaddit_train.jsonl", CHUNK_TRAIN)
    emit(val, DATA / "dreaddit_val.jsonl", CHUNK_TRAIN)
    emit(te,  DATA / "dreaddit_test.jsonl", False)   # test on full posts, unchunked


# --------------------------------------------------------------------- esconv
def _load_esconv_raw():
    """Try HuggingFace first, then a direct download of the released JSON."""
    local = RAW / "ESConv.json"
    if local.exists():
        with open(local, encoding="utf-8") as f:
            return json.load(f)
    try:
        from datasets import load_dataset
        ds = load_dataset("thu-coai/esconv")
        rows = []
        for split in ds:
            for r in ds[split]:
                rows.append(json.loads(r["text"]) if "text" in r else r)
        return rows
    except Exception as e:                                    # noqa: BLE001
        print(f"  ! HF load failed ({e}).")
    try:
        url = "https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/ESConv.json"
        _fetch("ESConv.json", url)
        with open(local, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:                                    # noqa: BLE001
        print(f"  ! direct download failed ({e}).")
        print("  ! Download ESConv.json manually (see README) into data/raw/ and rerun.")
        sys.exit(1)


def _as_int(x, default=None):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


SEEKER_ALIASES = {"seeker", "usr", "user", "speaker"}


def _is_seeker(raw):
    """ESConv uses seeker/supporter; FailedESConv uses speaker/listener.
    Without this, every FailedESConv conversation silently yields zero turns."""
    return (raw or "").strip().lower() in SEEKER_ALIASES


def build_failed_esconv():
    """FailedESConv -> EXTERNAL OOD set only. NEVER merged into training.

    Its release rule selects ON the outcome ("negative emotion intensity does
    not decrease"), so merging it with ESConv makes corpus identity a
    near-perfect predictor of the label: base rate 0.885 vs 0.363. Any gain a
    trajectory model shows on the merged corpus is provenance detection.
    """
    local = RAW / "FailedESConv.json"
    if not local.exists():
        print("FailedESConv: not in data/raw -- skipping OOD set.")
        return
    print("FailedESConv (external OOD only):")
    raw = json.load(open(local, encoding="utf-8"))
    kept = []
    for i, conv in enumerate(raw):
        survey = (conv.get("survey_score") or {}).get("seeker", {})
        init = _as_int(survey.get("initial_emotion_intensity"))
        fin = _as_int(survey.get("final_emotion_intensity"))
        if init is None or fin is None:
            continue
        st = [(t.get("content") or "").strip() for t in conv.get("dialog", [])
              if _is_seeker(t.get("speaker")) and (t.get("content") or "").strip()]
        if len(st) < MIN_SEEKER_TURNS:
            continue
        delta = fin - init
        kept.append({"conv_id": f"failed_{i:05d}",
                     "problem_type": str(conv.get("problem_type", "unknown")).strip().lower(),
                     "emotion_type": conv.get("emotion_type", "unknown"),
                     "situation": conv.get("situation", ""),
                     "seeker_turns": st, "n_seeker_turns": len(st),
                     "initial_intensity": init, "final_intensity": fin,
                     "delta": delta,
                     "label": int(delta > -IMPROVEMENT_MARGIN)})
    with open(DATA / "failed_esconv_ood.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    pos = sum(r["label"] for r in kept)
    print(f"  kept {len(kept)} conversations, {pos} positive "
          f"({pos / max(len(kept), 1):.1%}) -- NOT for training")


def build_esconv():
    print("ESConv:")
    raw = _load_esconv_raw()
    kept, dropped = [], {"no_survey": 0, "too_short": 0}

    for i, conv in enumerate(raw):
        if isinstance(conv, str):
            conv = json.loads(conv)
        survey = (conv.get("survey_score") or {}).get("seeker", {})
        init = _as_int(survey.get("initial_emotion_intensity"))
        fin = _as_int(survey.get("final_emotion_intensity"))
        if init is None or fin is None:
            dropped["no_survey"] += 1
            continue

        turns = []
        for t in conv.get("dialog", []):
            speaker = t.get("speaker", "")
            content = (t.get("content") or "").strip()
            if not content:
                continue
            turns.append({"speaker": speaker, "text": content})

        seeker_turns = [t["text"] for t in turns if _is_seeker(t["speaker"])]
        if len(seeker_turns) < MIN_SEEKER_TURNS:
            dropped["too_short"] += 1
            continue

        delta = fin - init
        kept.append({
            "conv_id": f"esconv_{i:05d}",
            "problem_type": str(conv.get("problem_type", "unknown")).strip().lower(),
            "emotion_type": conv.get("emotion_type", "unknown"),
            "experience_type": conv.get("experience_type", "unknown"),
            "situation": conv.get("situation", ""),
            "seeker_turns": seeker_turns,
            "n_seeker_turns": len(seeker_turns),
            "n_all_turns": len(turns),
            "initial_intensity": init,
            "final_intensity": fin,
            "delta": delta,
            # label 1 = NON-IMPROVING: distress did not fall by at least the margin
            "label": int(delta > -IMPROVEMENT_MARGIN),
        })

    with open(DATA / "esconv.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    pos = sum(r["label"] for r in kept)
    print(f"  kept {len(kept)} conversations "
          f"(dropped: {dropped['no_survey']} no survey, {dropped['too_short']} < {MIN_SEEKER_TURNS} seeker turns)")
    print(f"  label balance: {pos} non-improving / {len(kept) - pos} improving "
          f"({pos / max(len(kept), 1):.1%} positive)")
    print(f"  median seeker turns: {np.median([r['n_seeker_turns'] for r in kept]):.0f}")
    types = pd.Series([r["problem_type"] for r in kept]).value_counts()
    print("  problem types:\n" + types.to_string())


if __name__ == "__main__":
    build_dreaddit()
    build_esconv()
    build_failed_esconv()
    print("\nStage 1 complete.")
