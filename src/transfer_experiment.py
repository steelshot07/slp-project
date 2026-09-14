"""Does the synthetic-to-human gap come from the TRAINING SOURCE?

The val-test gap is currently an inference. This measures it. Everything is held
fixed -- architecture, training size, validation size, test set, epochs, seed --
and ONLY the source of the training dialogues changes.

5-fold CV over the 600 clinician-annotated dialogues. Per fold:
    test   = 120 human dialogues (identical across all conditions)
    val    =  80 dialogues (threshold tuning + epoch selection)
    train  = 400 dialogues

Conditions:
    human400   train 400 human,     val 80 human        (in-domain)
    synth400   train 400 synthetic, val 80 synthetic    (matched size, transfer)
    synth400+c train 400 synthetic, val 80 human        (same weights as
                                                         synth400, recalibrated
                                                         on human -- isolates
                                                         representation from
                                                         calibration)
    synthfull  train 3058 synthetic, val synthetic dev  (the paper's setting;
                                                         trained once, scored
                                                         on every fold's test)

Reading it:
  human400 >> synth400 at equal size  -> the gap is the training SOURCE, and
      the artefact claim is measured rather than argued.
  synth400+c ~= synth400              -> it is the representation, not
      miscalibration.
  synthfull <= human400 despite 7.6x more data -> synthetic volume does not
      substitute for human data.

Run:  python -m src.transfer_experiment --seed 13
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from src.cradle_tagger import (ALERT_IDX, DIALOGS_PER_BATCH, Dialogues, LABELS,
                               LR_ENC, LR_HEAD, Tagger, collate, encode,
                               evaluate_probs, prf, tune_global)
from src.config import DATA, DEVICE, RESULTS
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

CR = DATA / "cradle"
ENCODER = "distilroberta-base"
N_FOLDS = 5
N_VAL = 80
EPOCHS_SMALL = 12     # 400 dialogues -> cheap epochs
EPOCHS_FULL = 12


class Subset:
    """Dialogues-compatible view over a list of parsed dialogue records."""

    def __init__(self, records):
        self.d = records

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        return self.d[i]


def mk(ds, shuffle):
    return DataLoader(ds, batch_size=DIALOGS_PER_BATCH, shuffle=shuffle,
                      collate_fn=collate)


def train_model(train_ds, val_ds, tokz, seed, epochs, tag):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dl_tr, dl_va = mk(train_ds, True), mk(val_ds, False)

    Y = np.concatenate([d["y"] for d in train_ds.d], 0)
    pw = torch.tensor(((len(Y) - Y.sum(0)) / np.maximum(Y.sum(0), 1)).clip(1, 50),
                      dtype=torch.float32, device=DEVICE)

    model = Tagger("gru").to(DEVICE)
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": LR_ENC},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("enc.")],
         "lr": LR_HEAD}], weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

    best, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        for b in dl_tr:
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
        gv, pv = evaluate_probs(model, tokz, dl_va)
        m, _ = prf(gv, (pv > 0.5).astype(int))
        if m["micro_f1"] > best:
            best, best_ep = m["micro_f1"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"    [{tag}] best_ep {best_ep}  val@0.5 {best*100:.2f}")
    return model


def score(model, tokz, val_ds, test_ds):
    """Tune one global threshold on val_ds, apply to test_ds."""
    gv, pv = evaluate_probs(model, tokz, mk(val_ds, False))
    thr, _ = tune_global(gv, pv)
    g, p = evaluate_probs(model, tokz, mk(test_ds, False))
    pred = (p > thr).astype(int)
    m, _ = prf(g, pred)
    ma, _ = prf(g[:, ALERT_IDX], pred[:, ALERT_IDX])
    return {"micro_f1": m["micro_f1"], "macro_f1": m["macro_f1"],
            "micro_p": m["micro_p"], "micro_r": m["micro_r"],
            "alert_micro_f1": ma["micro_f1"], "threshold": thr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    tokz = AutoTokenizer.from_pretrained(ENCODER)

    human = Dialogues(CR / "cradle_test.jsonl").d
    synth = Dialogues(CR / "cradle_train.jsonl").d
    sdev = Dialogues(CR / "cradle_validation.jsonl")
    print(f"human {len(human)}  synthetic {len(synth)}  synth-dev {len(sdev)}")

    hidx = rng.permutation(len(human))
    folds = np.array_split(hidx, N_FOLDS)

    # synth-full trained ONCE, scored on every fold's test split
    print("\n[synthfull] training on all synthetic dialogues ...")
    full_model = train_model(Subset(synth), sdev, tokz, a.seed, EPOCHS_FULL, "synthfull")

    rows = []
    for k, te_idx in enumerate(folds):
        rest = np.setdiff1d(hidx, te_idx)
        rest = rng.permutation(rest)
        va_idx, tr_idx = rest[:N_VAL], rest[N_VAL:]
        te = Subset([human[i] for i in te_idx])
        hva = Subset([human[i] for i in va_idx])
        htr = Subset([human[i] for i in tr_idx])
        n_tr = len(htr)

        sperm = rng.permutation(len(synth))
        str_ = Subset([synth[i] for i in sperm[:n_tr]])
        sva = Subset([synth[i] for i in sperm[n_tr:n_tr + N_VAL]])

        print(f"\n--- fold {k}: train {n_tr}  val {N_VAL}  test {len(te)} ---")

        hm = train_model(htr, hva, tokz, a.seed + k, EPOCHS_SMALL, "human400")
        r_h = score(hm, tokz, hva, te)
        del hm
        torch.cuda.empty_cache()

        sm = train_model(str_, sva, tokz, a.seed + k, EPOCHS_SMALL, "synth400")
        r_s = score(sm, tokz, sva, te)
        r_sc = score(sm, tokz, hva, te)      # same weights, human calibration
        del sm
        torch.cuda.empty_cache()

        r_f = score(full_model, tokz, sdev, te)

        for name, r in [("human400", r_h), ("synth400", r_s), ("synth400+cal", r_sc), ("synthfull", r_f)]:
            rows.append({"fold": k, "condition": name, "n_train": len(synth) if name == "synthfull" else n_tr, **r})
            print(f"    {name:13s} microF1 {r['micro_f1']*100:6.2f}  "
                  f"macroF1 {r['macro_f1']*100:6.2f}  ALERT {r['alert_micro_f1']*100:6.2f}")

    print("\n" + "=" * 66)
    print(f"{'condition':14s} {'micro F1':>16s} {'macro F1':>15s} {'ALERT':>15s}")
    summary = {}
    for name in ["human400", "synth400", "synth400+cal", "synthfull"]:
        v = np.array([r["micro_f1"] for r in rows if r["condition"] == name]) * 100
        ma = np.array([r["macro_f1"] for r in rows if r["condition"] == name]) * 100
        al = np.array([r["alert_micro_f1"] for r in rows if r["condition"] == name]) * 100
        summary[name] = {"micro_mean": v.mean(), "micro_sd": v.std(ddof=1),
                         "macro_mean": ma.mean(), "alert_mean": al.mean()}
        print(f"{name:14s} {v.mean():9.2f} ± {v.std(ddof=1):4.2f} "
              f"{ma.mean():9.2f} ± {ma.std(ddof=1):4.2f} "
              f"{al.mean():9.2f} ± {al.std(ddof=1):4.2f}")
    print("=" * 66)
    d = summary["human400"]["micro_mean"] - summary["synth400"]["micro_mean"]
    print(f"human400 - synth400 (matched size) = {d:+.2f} micro F1")
    print(f"synth400 - synth400+cal            = "
          f"{summary['synth400']['micro_mean'] - summary['synth400+cal']['micro_mean']:+.2f}"
          f"   (small => representation, not calibration)")
    print(f"human400 - synthfull               = "
          f"{summary['human400']['micro_mean'] - summary['synthfull']['micro_mean']:+.2f}"
          f"   (synthfull uses {len(synth)} dialogues vs 400)")
    print(f"\nminutes {(time.time()-t0)/60:.1f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"transfer_seed{a.seed}.json").write_text(json.dumps(
        {"seed": a.seed, "n_folds": N_FOLDS, "n_val": N_VAL,
         "epochs_small": EPOCHS_SMALL, "epochs_full": EPOCHS_FULL,
         "n_synth": len(synth), "n_human": len(human),
         "per_fold": rows, "summary": summary}, indent=2))
    print(f"saved results/transfer_seed{a.seed}.json")


if __name__ == "__main__":
    main()