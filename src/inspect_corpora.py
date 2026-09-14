"""Inspect newly added corpora so the loaders can be written against reality.

Run:  python -m src.inspect_corpora
Paste the whole output back.
"""
import json
from collections import Counter
from pathlib import Path

from src.config import DATA

RAW = DATA / "raw"
SKIP = {"ESConv.json", "FailedESConv.json", "dreaddit-train.csv", "dreaddit-test.csv"}


def peek_json(p):
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        # maybe JSONL
        rows = []
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                try:
                    rows.append(json.loads(line))
                except Exception:
                    return print("    (not JSON or JSONL)")
        d = rows
        print("    format: JSONL")
    if isinstance(d, dict):
        print(f"    top-level dict, keys: {list(d.keys())[:15]}")
        for k in list(d.keys())[:2]:
            v = d[k]
            print(f"    ['{k}'] -> {type(v).__name__}"
                  f"{' len ' + str(len(v)) if hasattr(v, '__len__') else ''}")
            if isinstance(v, list) and v and isinstance(v[0], dict):
                print(f"      item keys: {list(v[0].keys())}")
                print(f"      sample: {json.dumps(v[0])[:500]}")
        return
    if isinstance(d, list):
        print(f"    list of {len(d)}")
        if d and isinstance(d[0], dict):
            print(f"    item keys: {list(d[0].keys())}")
            print(f"    sample item:\n      {json.dumps(d[0], indent=1)[:1200]}")
            # look for a dialogue-ish field
            for k, v in d[0].items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    print(f"    nested list '{k}' -> turn keys: {list(v[0].keys())}")
                    print(f"      first 2 turns: {json.dumps(v[:2])[:700]}")
                    spk = Counter()
                    for conv in d[:200]:
                        for t in conv.get(k, []) or []:
                            for kk in ("speaker", "role", "from", "who"):
                                if kk in t:
                                    spk[f"{kk}={t[kk]}"] += 1
                    if spk:
                        print(f"      speaker values (first 200 convs): {spk.most_common(8)}")


def peek_csv(p):
    import csv
    with open(p, encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        try:
            hdr = next(r)
        except StopIteration:
            return print("    (empty)")
        print(f"    columns ({len(hdr)}): {hdr}")
        rows = [next(r, None) for _ in range(2)]
        for row in rows:
            if row:
                print(f"    row: {dict(zip(hdr, row))}")
        n = sum(1 for _ in r) + 2
        print(f"    ~{n} data rows")


def main():
    print(f"scanning {RAW}\n")
    files = sorted(q for q in RAW.rglob("*") if q.is_file() and q.name not in SKIP)
    if not files:
        print("nothing new found under data/raw/")
        return
    for p in files:
        sz = p.stat().st_size
        print(f"--- {p.relative_to(RAW)}  ({sz/1024:.0f} KB) ---")
        s = p.suffix.lower()
        if s in (".json", ".jsonl"):
            peek_json(p)
        elif s in (".csv", ".tsv"):
            peek_csv(p)
        elif s in (".md", ".txt"):
            print("    " + "\n    ".join(
                open(p, encoding="utf-8", errors="replace").read()[:800].splitlines()[:20]))
        else:
            print("    (binary or unhandled)")
        print()


if __name__ == "__main__":
    main()