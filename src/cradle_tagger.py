"""CRADLE turn-level tagger: hierarchical encoder + trajectory state.

THE EXPERIMENT. Three arms, identical except for how prior turns reach the
classifier. This is the H1 test that ESConv could not support, now on a corpus
with clinician turn-level labels:

    --arm none   turn embedding only              (no history)
    --arm mean   turn + MEAN of prior turn embs   (history, ORDER-FREE)
    --arm gru    turn + GRU state over prior embs (history, ORDER-SENSITIVE)

The mean-vs-gru comparison is the one that matters: both see exactly the same
history, only the ordering differs. That isolates trajectory from mere context.

Reference points on the 600 clinician-annotated dialogues (turn-level micro F1):
    TF-IDF floor 22.09 | Qwen3-32B 43.75 | gpt-oss-120b 47.20
    their FT-32B 51.31 | Claude-4.5-Sonnet 56.85

Motivated by the baseline: concatenating history into the input HURT
(22.90 -> 22.09), so history gets its own pathway rather than a longer window.

HEADLINE METRIC: micro F1 under a single GLOBAL threshold tuned on validation.
Per-label thresholds are reported alongside as an upper bound only. Fixed in
advance: validation is synthetic and test is human-annotated, so sixteen
independently tuned thresholds are sixteen chances to overfit a distribution
the test set does not share.

Run:  python -m src.cradle_tagger --arm gru --seed 13
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.config import DATA, DEVICE, RESULTS

CR = DATA / "cradle"
ENCODER = "distilroberta-base"
MAX_LEN = 128
LR_ENC, LR_HEAD = 2e-5, 1e-3
EPOCHS, DIALOGS_PER_BATCH = 6, 4
HID = 256

MIN_SUPP = 10      # labels with fewer val positives keep 0.5; too few to tune on
THR_FLOOR = 0.20   # never tune below this -- guards against degenerate thresholds
THR_GRID = np.arange(THR_FLOOR, 0.96, 0.05)

LABELS = ["alert_ongoing", "alert_past",
          "confirm_CA_ongoing", "confirm_CA_past", "confirm_DV_ongoing",
          "confirm_DV_past", "confirm_RA_ongoing", "confirm_RA_past",
          "confirm_SH_ongoing", "confirm_SH_past", "confirm_SHA_ongoing",
          "confirm_SHA_past", "confirm_SI_active_ongoing", "confirm_SI_active_past",
          "confirm_SI_passive_ongoing", "confirm_SI_passive_past"]
L2I = {l: i for i, l in enumerate(LABELS)}
ALERT_IDX = [L2I[l] for l in LABELS if l.startswith("alert")]


# ------------------------------------------------------------------ data
class Dialogues(Dataset):
    """One item = one dialogue = an ordered list of user turns."""

    def __init__(self, path):
        self.d = []
        for line in open(path, encoding="utf-8"):
            g = json.loads(line)
            turns, texts, ys = g["turns"], [], []
            for i, t in enumerate(turns):
                if t["speaker"] != "user":
                    continue
                prev = ""
                for h in reversed(turns[:i]):
                    if h["speaker"] == "listener":
                        prev = h["text"]
                        break
                texts.append((prev, t["text"]))
                y = np.zeros(len(LABELS), dtype=np.float32)
                for l in t["labels"]:
                    if l in L2I:
                        y[L2I[l]] = 1.0
                ys.append(y)
            if texts:
                self.d.append({"texts": texts, "y": np.stack(ys), "id": g["conv_id"]})

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        return self.d[i]


def collate(batch):
    return batch


# ----------------------------------------------------------------- model
class Tagger(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.enc = AutoModel.from_pretrained(ENCODER)
        h = self.enc.config.hidden_size
        if arm == "gru":
            self.gru = nn.GRU(h, HID, batch_first=True)
            ctx = HID
        elif arm == "mean":
            self.proj = nn.Linear(h, HID)
            ctx = HID
        else:
            ctx = 0
        self.head = nn.Sequential(nn.Dropout(0.1),
                                  nn.Linear(h + ctx, 256), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(256, len(LABELS)))

    def embed(self, tok):
        out = self.enc(**tok).last_hidden_state
        mask = tok["attention_mask"].unsqueeze(-1).float()
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, tok, n_turns):
        """tok holds all turns of all dialogues, flattened; n_turns splits them.

        The context vector for turn t is built from turns 1..t-1 only -- never
        turn t itself, never the future. Alert labels sit BEFORE explicit
        disclosure, so forward leakage would make them trivially predictable.
        """
        e = self.embed(tok)
        outs, off = [], 0
        for n in n_turns:
            seq = e[off:off + n]
            off += n
            if self.arm == "none":
                outs.append(self.head(seq))
                continue
            if self.arm == "gru":
                h, _ = self.gru(seq.unsqueeze(0))
                h = h.squeeze(0)
                prior = torch.cat([torch.zeros(1, HID, device=seq.device), h[:-1]], 0)
            else:
                cs = torch.cumsum(seq, 0)
                idx = torch.arange(1, n + 1, device=seq.device).unsqueeze(1).float()
                m = cs / idx
                prior = torch.cat([torch.zeros(1, seq.size(1), device=seq.device),
                                   m[:-1]], 0)
                prior = self.proj(prior)
            outs.append(self.head(torch.cat([seq, prior], -1)))
        return torch.cat(outs, 0)


# ------------------------------------------------------------------ eval
def prf(gold, pred):
    """Multi-label micro/macro P, R, F1 over the label matrix."""
    tp = (gold & pred).sum(0)
    fp = ((1 - gold) & pred).sum(0)
    fn = (gold & (1 - pred)).sum(0)
    mp = tp.sum() / max(tp.sum() + fp.sum(), 1)
    mr = tp.sum() / max(tp.sum() + fn.sum(), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.divide(tp, tp + fp, out=np.zeros(len(tp)), where=(tp + fp) > 0)
        r = np.divide(tp, tp + fn, out=np.zeros(len(tp)), where=(tp + fn) > 0)
        f = np.divide(2 * p * r, p + r, out=np.zeros(len(tp)), where=(p + r) > 0)
    return dict(micro_p=float(mp), micro_r=float(mr),
                micro_f1=float(2 * mp * mr / max(mp + mr, 1e-9)),
                macro_f1=float(f.mean())), f


def encode(tok_fn, batch, device):
    pairs = [t for d in batch for t in d["texts"]]
    tok = tok_fn([p for _, p in pairs], [c for c, _ in pairs],
                 truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in tok.items()}, [len(d["texts"]) for d in batch]


@torch.no_grad()
def evaluate_probs(model, tokz, loader):
    model.eval()
    G, P = [], []
    for b in loader:
        tok, n = encode(tokz, b, DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            logits = model(tok, n)
        P.append(torch.sigmoid(logits.float()).cpu().numpy())
        G.append(np.concatenate([d["y"] for d in b], 0).astype(int))
    return np.concatenate(G), np.concatenate(P)


def evaluate(model, tokz, loader, thr=0.5):
    gold, probs = evaluate_probs(model, tokz, loader)
    return gold, (probs > thr).astype(int)


def tune_global(gv, pv):
    """One threshold for every label. The headline setting."""
    best_t, best_f = 0.5, -1.0
    for t in THR_GRID:
        m, _ = prf(gv, (pv > t).astype(int))
        if m["micro_f1"] > best_f:
            best_f, best_t = m["micro_f1"], float(t)
    return best_t, best_f


def tune_per_label(gv, pv):
    """Per-label thresholds. Secondary, reported as an upper bound.

    Labels with < MIN_SUPP validation positives keep 0.5. Without that guard a
    zero-support label scores F1 = 0 at every threshold and silently takes the
    first grid value, then fires on nearly every test turn.
    """
    thr = np.full(len(LABELS), 0.5)
    for j in range(len(LABELS)):
        if gv[:, j].sum() < MIN_SUPP:
            continue
        best_t, best_f = 0.5, -1.0
        for t in THR_GRID:
            pj = (pv[:, j] > t).astype(int)
            tp = int((gv[:, j] & pj).sum())
            fp = int(((1 - gv[:, j]) & pj).sum())
            fn = int((gv[:, j] & (1 - pj)).sum())
            f = 2 * tp / max(2 * tp + fp + fn, 1)
            if f > best_f:
                best_f, best_t = f, float(t)
        thr[j] = best_t
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["none", "mean", "gru"], default="gru")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    tokz = AutoTokenizer.from_pretrained(ENCODER)
    tr = Dialogues(CR / "cradle_train.jsonl")
    va = Dialogues(CR / "cradle_validation.jsonl")
    te = Dialogues(CR / "cradle_test.jsonl")
    print(f"arm={a.arm} seed={a.seed} | dialogues tr/va/te {len(tr)}/{len(va)}/{len(te)}")

    def mk(ds, sh):
        return DataLoader(ds, batch_size=DIALOGS_PER_BATCH, shuffle=sh, collate_fn=collate)

    dl_tr, dl_va, dl_te = mk(tr, True), mk(va, False), mk(te, False)

    Y = np.concatenate([d["y"] for d in tr.d], 0)
    pw = torch.tensor(((len(Y) - Y.sum(0)) / np.maximum(Y.sum(0), 1)).clip(1, 50),
                      dtype=torch.float32, device=DEVICE)
    print(f"pos_weight range {pw.min():.1f}-{pw.max():.1f}")

    model = Tagger(a.arm).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"parameters: {n_par:.1f}M")

    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": LR_ENC},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("enc.")],
         "lr": LR_HEAD}], weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

    val_curve = []
    t0, best, best_state, best_ep = time.time(), -1.0, None, -1
    for ep in range(a.epochs):
        model.train()
        for i, b in enumerate(dl_tr):
            tok, n = encode(tokz, b, DEVICE)
            y = torch.tensor(np.concatenate([d["y"] for d in b], 0), device=DEVICE)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                loss = lossf(model(tok, n), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if i % 150 == 0:
                print(f"  ep{ep} step{i}/{len(dl_tr)} loss {loss.item():.4f}")
        g, p = evaluate(model, tokz, dl_va)
        m, _ = prf(g, p)
        print(f"  epoch {ep}: val microF1 {m['micro_f1']:.4f} macroF1 {m['macro_f1']:.4f}")
        val_curve.append({"epoch": ep, "val_micro_f1": m["micro_f1"], "val_macro_f1": m["macro_f1"]})
        if m["micro_f1"] > best:
            best, best_ep = m["micro_f1"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"  restored epoch {best_ep} (val microF1 {best:.4f})")
    if best_ep == a.epochs - 1:
        print("  NOTE: best epoch is the last one -- may still be undertrained.")

    # ---- thresholds, tuned on VALIDATION only ------------------------------
    gv, pv = evaluate_probs(model, tokz, dl_va)
    gt, gt_val_f1 = tune_global(gv, pv)
    thr = tune_per_label(gv, pv)
    print(f"\nglobal threshold {gt:.2f} (val microF1 {gt_val_f1:.4f})")
    print("per-label thresholds:",
          {LABELS[j]: round(float(thr[j]), 2) for j in range(len(LABELS))})

    # ---- test ---------------------------------------------------------------
    g, probs = evaluate_probs(model, tokz, dl_te)

    p_glob = (probs > gt).astype(int)
    m_glob, per_glob = prf(g, p_glob)
    ma_glob, _ = prf(g[:, ALERT_IDX], p_glob[:, ALERT_IDX])

    p_pl = (probs > thr).astype(int)
    m_pl, per_pl = prf(g, p_pl)
    ma_pl, _ = prf(g[:, ALERT_IDX], p_pl[:, ALERT_IDX])

    print(f"\n=== TEST (600 clinician-annotated dialogues) | arm={a.arm} seed={a.seed} ===")
    print(f"  HEADLINE   global thr {gt:.2f}")
    print(f"    micro F1 {m_glob['micro_f1']*100:.2f}   macro F1 {m_glob['macro_f1']*100:.2f}"
          f"   P {m_glob['micro_p']*100:.2f}   R {m_glob['micro_r']*100:.2f}"
          f"   ALERT {ma_glob['micro_f1']*100:.2f}")
    print(f"  secondary  per-label thr (upper bound)")
    print(f"    micro F1 {m_pl['micro_f1']*100:.2f}   macro F1 {m_pl['macro_f1']*100:.2f}"
          f"   P {m_pl['micro_p']*100:.2f}   R {m_pl['micro_r']*100:.2f}"
          f"   ALERT {ma_pl['micro_f1']*100:.2f}")

    print(f"\n  {'label':30s} {'support':>8} {'F1 glob':>9} {'F1 perL':>9}")
    for j in np.argsort(-g.sum(0)):
        print(f"  {LABELS[j]:30s} {g[:, j].sum():8d} "
              f"{per_glob[j]:9.4f} {per_pl[j]:9.4f}")

    print(f"\n  reference: tfidf 22.09 | Qwen3-32B 43.75 | gpt-oss-120b 47.20 | "
          f"FT-32B 51.31 | Claude 56.85")
    print(f"  params {n_par:.1f}M   train+eval minutes {(time.time()-t0)/60:.1f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {
        "arm": a.arm, "seed": a.seed, "epochs": a.epochs, "val_curve": val_curve, "best_epoch": best_ep,
        "params_m": n_par, "encoder": ENCODER,
        "val_micro_f1_at_best": float(best),
        "global_threshold": gt, "global_val_micro_f1": float(gt_val_f1),
        "per_label_thresholds": thr.tolist(),
        "micro_f1": m_glob["micro_f1"], "macro_f1": m_glob["macro_f1"],
        "micro_p": m_glob["micro_p"], "micro_r": m_glob["micro_r"],
        "alert_micro_f1": ma_glob["micro_f1"],
        "perlabel_micro_f1": m_pl["micro_f1"], "perlabel_macro_f1": m_pl["macro_f1"],
        "perlabel_micro_p": m_pl["micro_p"], "perlabel_micro_r": m_pl["micro_r"],
        "perlabel_alert_micro_f1": ma_pl["micro_f1"],
        "per_label_f1_global": {LABELS[j]: float(per_glob[j]) for j in range(len(LABELS))},
        "per_label_f1_perlabel": {LABELS[j]: float(per_pl[j]) for j in range(len(LABELS))},
        "test_support": {LABELS[j]: int(g[:, j].sum()) for j in range(len(LABELS))},
        "minutes": (time.time() - t0) / 60,
    }
    tag = f"cradle_{a.arm}_seed{a.seed}_ep{a.epochs}"
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"  saved results/cradle_{a.arm}_seed{a.seed}_ep{a.epochs}.json")


if __name__ == "__main__":
    main()