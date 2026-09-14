"""
Stage 5 - the architectural claim of gap 3.4, made measurable.

The literature computes SHAP / LIME / integrated gradients offline on a static corpus and
never reports a latency budget. This module puts attribution on a background worker behind
a queue, so the synchronous path a counsellor waits on contains only the classifier
forward pass. It then measures both paths separately, which is the number the reviewed
papers do not report.

  synchronous  : tokenize + one forward pass -> score, flag decision
  asynchronous : occlusion attribution, queued ONLY for turns above FLAG_THRESHOLD

Run:  python -m src.attribution_service --seed 13 --n 200
Outputs: results/latency_seed{S}.json
"""
import argparse
import json
import queue
import threading
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import ARTIFACTS, DATA, DEVICE, FLAG_THRESHOLD, MAX_LEN, RESULTS


class TriageService:
    def __init__(self, seed):
        src = ARTIFACTS / f"encoder_seed{seed}"
        self.tok = AutoTokenizer.from_pretrained(src)
        self.model = AutoModelForSequenceClassification.from_pretrained(src).to(DEVICE).eval()
        self.T = json.loads((src / "temperature.json").read_text())["T"]
        self.q = queue.Queue()
        self.explanations = {}
        self.async_latencies = []
        self._stop = threading.Event()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    # --------------------------------------------------------- synchronous path
    @torch.no_grad()
    def score_turn(self, text):
        """Everything on the counsellor's critical path lives in this method."""
        enc = self.tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = self.model(**enc).logits.float()
        return float(torch.softmax(logits / self.T, dim=1)[0, 1])

    def handle_turn(self, turn_id, text):
        t0 = time.perf_counter()
        score = self.score_turn(text)
        flagged = score >= FLAG_THRESHOLD
        if flagged:
            self.q.put((turn_id, text, score))     # non-blocking handoff
        sync_ms = (time.perf_counter() - t0) * 1000
        return score, flagged, sync_ms

    # -------------------------------------------------------- asynchronous path
    @torch.no_grad()
    def _occlusion(self, text, base_score, max_tokens=64):
        """Leave-one-token-out attribution. Chosen over integrated gradients because it
        needs forward passes only, and over SHAP because the number of passes is bounded
        and known in advance (one per token), which is what makes the queue size
        predictable."""
        ids = self.tok(text, truncation=True, max_length=MAX_LEN)["input_ids"]
        toks = self.tok.convert_ids_to_tokens(ids)
        n = min(len(ids), max_tokens)
        variants = []
        for i in range(1, n - 1):                  # skip <s> and </s>
            v = ids.copy()
            v[i] = self.tok.mask_token_id or self.tok.unk_token_id
            variants.append(v)
        if not variants:
            return []
        batch = torch.tensor(variants).to(DEVICE)
        attrs = []
        for j in range(0, len(batch), 32):
            chunk = batch[j:j + 32]
            logits = self.model(input_ids=chunk,
                                attention_mask=torch.ones_like(chunk)).logits.float()
            p = torch.softmax(logits / self.T, dim=1)[:, 1].cpu().numpy()
            attrs.append(base_score - p)
        attrs = np.concatenate(attrs)
        pairs = list(zip(toks[1:n - 1], attrs.tolist()))
        return sorted(pairs, key=lambda x: -abs(x[1]))[:10]

    def _loop(self):
        while not self._stop.is_set():
            try:
                turn_id, text, score = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            self.explanations[turn_id] = self._occlusion(text, score)
            self.async_latencies.append((time.perf_counter() - t0) * 1000)
            self.q.task_done()

    def shutdown(self):
        self.q.join()
        self._stop.set()
        self.worker.join(timeout=2)


def main(seed, n_turns):
    convs = [json.loads(l) for l in open(DATA / "esconv.jsonl")]
    turns = [t for c in convs for t in c["seeker_turns"]][:n_turns]

    svc = TriageService(seed)
    for t in turns[:20]:                            # warm-up, excluded from timings
        svc.score_turn(t)

    sync, flags = [], 0
    t_start = time.perf_counter()
    for i, t in enumerate(turns):
        _, f, ms = svc.handle_turn(i, t)
        sync.append(ms)
        flags += int(f)
    wall = time.perf_counter() - t_start
    svc.shutdown()

    sync = np.array(sync)
    asyn = np.array(svc.async_latencies) if svc.async_latencies else np.array([np.nan])
    out = {
        "device": DEVICE,
        "n_turns": len(turns),
        "flag_rate": flags / len(turns),
        "sync_ms": {"p50": float(np.percentile(sync, 50)),
                    "p95": float(np.percentile(sync, 95)),
                    "p99": float(np.percentile(sync, 99)),
                    "mean": float(sync.mean())},
        "async_attribution_ms": {"p50": float(np.nanpercentile(asyn, 50)),
                                 "p95": float(np.nanpercentile(asyn, 95)),
                                 "mean": float(np.nanmean(asyn))},
        "attribution_over_classifier_ratio": float(np.nanmean(asyn) / sync.mean()),
        "wall_seconds_for_stream": wall,
    }
    print(json.dumps(out, indent=2))
    (RESULTS / f"latency_seed{seed}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    main(a.seed, a.n)
