"""Parse CRADLE into dialogues and VERIFY against the paper's published counts.

The release ships three columns only (turn_id, text, labels). Dialogue
boundaries, speaker, and reveal timing are all implicit. This script infers
them and then checks the result against Byun et al. Table 2, so a wrong
inference fails loudly instead of silently corrupting everything downstream.

Expected for test.csv (the 600 clinician-annotated dialogues):
    600 dialogues | 8,975 turns | 4,527 user turns | 713 labels
    226 with Alert | 417 with Confirm | 203 with both | 160 with neither

Run:  python -m src.prep_cradle
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from src.config import DATA

RAW = DATA / "raw" / "CRADLE"
csv.field_size_limit(10_000_000)

PAPER = {"dialogues": 600, "turns": 8975, "user_turns": 4527, "labels": 713,
         "with_alert": 226, "with_confirm": 417, "with_both": 203, "with_neither": 160}


def read_rows(path):
    if path.suffix.lower() == ".json":
        rows = []
        txt = path.read_text(encoding="utf-8").strip()
        try:
            d = json.loads(txt)
            rows = d if isinstance(d, list) else [d]
        except json.JSONDecodeError:
            for line in txt.splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def split_labels(raw):
    s = (raw or "").strip()
    if not s or s.lower() in {"none", "nan", "[]"}:
        return []
    if s.startswith("["):
        try:
            v = json.loads(s.replace("'", '"'))
            return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass
    return [p.strip() for p in re.split(r"[;,|]", s) if p.strip()]


def parse(path):
    rows = read_rows(path)
    dialogues, cur = [], []
    for r in rows:
        tid = int(str(r["turn_id"]).strip())
        if tid == 0 and cur:
            dialogues.append(cur)
            cur = []
        text = (r.get("text") or "").strip()
        m = re.match(r"^\s*(User|Listener|Seeker|Supporter|Client|Therapist)\s*:\s*(.*)$",
                     text, flags=re.S | re.I)
        speaker = (m.group(1).lower() if m else "unknown")
        body = (m.group(2).strip() if m else text)
        cur.append({"turn_id": tid, "speaker": speaker, "text": body,
                    "labels": split_labels(r.get("labels"))})
    if cur:
        dialogues.append(cur)
    return dialogues


def summarise(dialogues, name, check=False):
    turns = sum(len(d) for d in dialogues)
    user = sum(1 for d in dialogues for t in d if t["speaker"] == "user")
    vocab = Counter(l for d in dialogues for t in d for l in t["labels"])
    n_labels = sum(vocab.values())

    def has(d, kind):
        return any(l.lower().startswith(kind) for t in d for l in t["labels"])

    a = sum(has(d, "alert") for d in dialogues)
    c = sum(not has(d, "alert") and False for d in dialogues)  # placeholder
    c = sum(any(l.lower().startswith("confirm") or
                (not l.lower().startswith("alert")) for t in d for l in t["labels"])
            for d in dialogues)
    both = sum(has(d, "alert") and
               any((l.lower().startswith("confirm") or not l.lower().startswith("alert"))
                   for t in d for l in t["labels"]) for d in dialogues)
    neither = sum(not any(t["labels"] for t in d) for d in dialogues)

    got = {"dialogues": len(dialogues), "turns": turns, "user_turns": user,
           "labels": n_labels, "with_alert": a, "with_confirm": c,
           "with_both": both, "with_neither": neither}

    print(f"\n=== {name} ===")
    tl = sorted(len(d) for d in dialogues)
    print(f"  turns/dialogue  min {tl[0]}  med {tl[len(tl)//2]}  max {tl[-1]}")
    print(f"  speakers: {Counter(t['speaker'] for d in dialogues for t in d).most_common()}")
    print(f"\n  label vocabulary ({len(vocab)} distinct):")
    for k, v in vocab.most_common():
        print(f"    {k:44s} {v}")

    if check:
        print(f"\n  {'field':14s} {'parsed':>8} {'paper':>8}  ok")
        ok_all = True
        for k, exp in PAPER.items():
            ok = got[k] == exp
            ok_all &= ok
            print(f"  {k:14s} {got[k]:>8} {exp:>8}  {'YES' if ok else '*** NO ***'}")
        print("\n  PARSE VERIFIED against the paper." if ok_all else
              "\n  PARSE MISMATCH -- dialogue boundary or label rule is wrong. "
              "Do not build on this until it matches.")
    return got


def reveal_timing(d):
    """Paper bins by the turn at which the crisis becomes explicit."""
    for i, t in enumerate(d):
        if any(not l.lower().startswith("alert") for l in t["labels"]):
            u = sum(1 for x in d[:i + 1] if x["speaker"] == "user")
            return "early" if u <= 3 else ("mid" if u <= 6 else "late")
    return "none"


def write(dialogues, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, d in enumerate(dialogues):
            f.write(json.dumps({
                "conv_id": f"{path.stem}-{i:05d}",
                "n_turns": len(d),
                "n_user_turns": sum(1 for t in d if t["speaker"] == "user"),
                "reveal": reveal_timing(d),
                "turns": d}) + "\n")
    print(f"  -> {len(dialogues)} dialogues to {path.name}")


def main():
    if not RAW.exists():
        sys.exit(f"{RAW} not found")
    out = DATA / "cradle"
    out.mkdir(parents=True, exist_ok=True)
    for fn, name, chk in [("test.csv", "test.csv (HUMAN-ANNOTATED BENCHMARK)", True),
                          ("validation.json", "validation.json (synthetic dev)", False),
                          ("train.csv", "train.csv (synthetic train)", False)]:
        p = RAW / fn
        if not p.exists():
            print(f"skip {fn} (missing)")
            continue
        d = parse(p)
        summarise(d, name, check=chk)
        print(f"  reveal timing: {Counter(reveal_timing(x) for x in d).most_common()}")
        write(d, out / f"cradle_{p.stem}.jsonl")


if __name__ == "__main__":
    main()