"""
Stage 2 - fine-tune the turn-level distress encoder on Dreaddit, then calibrate it.

The encoder is deliberately small (distilroberta-base, ~82M). Chrifi Alaoui et al. [1]
established that competitive post-level performance does not require a large
domain-pretrained encoder, so encoder capacity is not the variable this project studies.

Run:  python -m src.train_turn_encoder --seed 13
Outputs: artifacts/encoder_seed{S}/ (weights + temperature.json), results/encoder_seed{S}.json
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src.config import (ARTIFACTS, BATCH_SIZE, DATA, DEVICE, ENCODER_NAME, EPOCHS,
                        LR, MAX_LEN, RESULTS, WARMUP_RATIO, WEIGHT_DECAY)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


class SpanDataset(Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, max_length=MAX_LEN,
                       padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()} | {"labels": torch.tensor(r["label"])}


@torch.no_grad()
def collect_logits(model, loader):
    model.eval()
    logits, labels = [], []
    for batch in loader:
        y = batch.pop("labels")
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch).logits.float().cpu()
        logits.append(out)
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits, labels):
    """Single-parameter temperature scaling (Guo et al.) on the held-out split.
    Calibration is required by gap 3.3: reviewers are shown confidence values."""
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / torch.exp(logT), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(logT).item())


def main(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        ENCODER_NAME, num_labels=2).to(DEVICE)

    tr = read_jsonl(DATA / "dreaddit_train.jsonl")
    va = read_jsonl(DATA / "dreaddit_val.jsonl")
    te = read_jsonl(DATA / "dreaddit_test.jsonl")

    dl_tr = DataLoader(SpanDataset(tr, tok), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dl_va = DataLoader(SpanDataset(va, tok), batch_size=BATCH_SIZE)
    dl_te = DataLoader(SpanDataset(te, tok), batch_size=BATCH_SIZE)

    steps = len(dl_tr) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO * steps), steps)

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        running = 0.0
        for i, batch in enumerate(dl_tr):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += loss.item()
            if i % 50 == 0:
                print(f"  ep{ep} step{i}/{len(dl_tr)} loss {running / (i + 1):.4f}", flush=True)
        vl, vy = collect_logits(model, dl_va)
        vf1 = f1_score(vy, vl.argmax(1))
        print(f"  epoch {ep}: val F1 {vf1:.4f}")

    # calibrate on val, evaluate on test
    vl, vy = collect_logits(model, dl_va)
    T = fit_temperature(vl, vy)
    tl, ty = collect_logits(model, dl_te)
    probs = torch.softmax(tl / T, dim=1)[:, 1].numpy()

    metrics = {
        "seed": seed,
        "temperature": T,
        "test_f1": float(f1_score(ty, (probs > 0.5).astype(int))),
        "test_acc": float(accuracy_score(ty, (probs > 0.5).astype(int))),
        "test_auroc": float(roc_auc_score(ty, probs)),
        "train_minutes": (time.time() - t0) / 60,
        "encoder": ENCODER_NAME,
    }
    print(json.dumps(metrics, indent=2))

    out = ARTIFACTS / f"encoder_seed{seed}"
    out.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "temperature.json").write_text(json.dumps({"T": T}))
    (RESULTS / f"encoder_seed{seed}.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=13)
    main(p.parse_args().seed)
