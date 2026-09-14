"""
Stage 3 - run the calibrated turn encoder over every seeker turn in ESConv.

This is the transfer step, and it is where the domain shift of gap 3.5 actually bites:
the encoder was supervised on Reddit narratives and is now applied to short synchronous
turns. Nothing here is fine-tuned on ESConv text, so the score sequence is an
out-of-domain reading by construction. That is reported, not hidden.

Run:  python -m src.score_conversations --seed 13
Outputs: artifacts/scores_seed{S}.npz  (per-conversation score sequence + pooled embeddings)
"""
import argparse
import json
import pickle

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import ARTIFACTS, DATA, DEVICE, MAX_LEN


@torch.no_grad()
def score_batch(model, tok, texts, T):
    enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True)
    probs = torch.softmax(out.logits.float() / T, dim=1)[:, 1].cpu().numpy()
    # mean-pooled last hidden state, masked
    h = out.hidden_states[-1].float()
    m = enc["attention_mask"].unsqueeze(-1).float()
    pooled = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
    return probs, pooled


def main(seed):
    src = ARTIFACTS / f"encoder_seed{seed}"
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src).to(DEVICE).eval()
    T = json.loads((src / "temperature.json").read_text())["T"]

    convs = [json.loads(l) for l in open(DATA / "esconv.jsonl")]
    all_scores, all_emb, meta = [], [], []

    for i, c in enumerate(convs):
        turns = c["seeker_turns"]
        probs, pooled = [], []
        for j in range(0, len(turns), 32):
            p, e = score_batch(model, tok, turns[j:j + 32], T)
            probs.append(p)
            pooled.append(e)
        all_scores.append(np.concatenate(probs))
        all_emb.append(np.concatenate(pooled))
        meta.append({k: c[k] for k in
                     ("conv_id", "problem_type", "label", "delta",
                      "initial_intensity", "final_intensity", "n_seeker_turns")})
        if i % 100 == 0:
            print(f"  scored {i}/{len(convs)}", flush=True)

    with open(ARTIFACTS / f"scores_seed{seed}.pkl", "wb") as f:
        pickle.dump({"scores": all_scores, "embeddings": all_emb, "meta": meta}, f)

    flat = np.concatenate(all_scores)
    print(f"\nturn score distribution: mean {flat.mean():.3f}  sd {flat.std():.3f}  "
          f"p10 {np.percentile(flat, 10):.3f}  p90 {np.percentile(flat, 90):.3f}")
    print(f"saved artifacts/scores_seed{seed}.pkl")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=13)
    main(p.parse_args().seed)
