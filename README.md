# Order Without Benefit

Code and results for *Order Without Benefit: Auditing Trajectory Modelling and
Synthetic Benchmarks in Multi-Turn Crisis Detection*.

**What the project found.** Order-sensitive modelling of conversational history
does not outperform an order-free control on clinician-annotated crisis
dialogues. The apparent benefit reported elsewhere is traceable to
LLM-generated benchmark corpora, which are measurably more predictable than
human-annotated ones. A widely used self-report outcome target is separately
shown to be degenerate under its standard threshold.

All numbers below are what a correct run reproduces. If yours differ
materially, something is wrong — say so rather than adjusting constants.

---

## Install

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1      Linux/macOS:  source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

CUDA is strongly preferred. Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Developed on an RTX 4050 Laptop (6 GB VRAM), 32 GB RAM, Windows 11,
Python 3.11. Everything fits 6 GB with fp16. On CPU the CRADLE stages are
impractical; run stages 1–3 only.

---

## Data

Nothing is downloaded automatically. Place these in `data/raw/`:

| File | Source |
| --- | --- |
| `dreaddit-train.csv`, `dreaddit-test.csv` | Kaggle mirror `turcan/dreaddit`, or the LOUHI 2019 release |
| `ESConv.json`, `FailedESConv.json` | `thu-coai/Emotional-Support-Conversation` on GitHub |
| `CRADLE/train.csv`, `CRADLE/validation.json`, `CRADLE/test.csv` | CRADLE benchmark release |

`data/`, `artifacts/`, and `.venv/` are gitignored. `results/` is committed.

---

## Pipeline

```bash
bash run_all.sh          # everything, ~8.5 h on a 6 GB GPU
```

Or stage by stage. Expected output is given for each, because these are the
checks that catch a broken run.

### 1. Corpora

```bash
python -m src.data_prep      # Dreaddit + ESConv
python -m src.prep_cradle    # CRADLE
```

`data_prep` should report **ESConv: 1150 kept, 417 non-improving (36.3%)** and
**FailedESConv: 139 kept, 123 positive**. The two ESConv files are written
separately on purpose — see *Two traps* below.

`prep_cradle` reconstructs dialogue boundaries and speaker roles, which the
release does not ship, then checks the result against eight published corpus
statistics. It must print **PARSE VERIFIED** (600 dialogues, 8,975 turns, 4,527
user turns, 713 labels). If it does not, stop; every downstream number would be
built on a wrong segmentation.

### 2. ESConv audit (~2 min, CPU)

```bash
python -m src.audit
```

Reproduces the five audit findings into `results/audit.json`: intake
AUROC 0.7078, base rate 0.3626, FailedESConv base rate 0.8849, seeker text
0.5850, residualised Spearman rho +0.2352.

### 3. ESConv trajectory null (~4 min per seed)

```bash
for S in 13 29 47 71 97; do
  python -m src.train_turn_encoder --seed $S     # Dreaddit encoder, ~3 min
  python -m src.score_conversations --seed $S    # score ESConv seeker turns
  python -m src.diagnose_scores --seed $S        # go/no-go diagnostic
done
```

The encoder should reach ~0.82 F1 on Dreaddit test. `diagnose_scores` prints a
**STOP** verdict: turn scores carry no signal about the ESConv target
(best order-sensitive summary 0.5206 AUROC vs best pointwise 0.5241). The
measured ICC of 0.111 confirms trajectories are not flat, so the null is real
rather than an artefact of degenerate score sequences.

### 4. CRADLE floor and ablation

```bash
python -m src.cradle_baseline                          # TF-IDF floor: 22.09 micro F1
for S in 13 29 47 71 97; do
  for A in gru none mean; do
    python -m src.cradle_tagger --arm $A --seed $S     # ~10 min each
  done
done
for S in 13 71 97; do                                  # 12-epoch robustness check
  for A in gru none mean; do
    python -m src.cradle_tagger --arm $A --seed $S --epochs 12   # ~20 min each
  done
done
```

Expected at 6 epochs, 5 seeds (human test micro F1): none 31.37 ± 1.28,
mean 33.02 ± 3.21, gru 32.08 ± 2.04. Order-sensitive does not beat order-free.

### 5. Matched-size transfer (~50 min per seed)

```bash
for S in 13 29 47; do python -m src.transfer_experiment --seed $S; done
```

The headline result. Pooled over 15 folds: human400 34.13 ± 3.56 vs
synth400 27.71 ± 4.76, a 6.42-point gap at identical training size
(paired t = 4.98, p = 0.0002).

### 6. Figures

```bash
python -m src.make_figures    # results/figures/*.pdf
```

---

## Two traps this code guards against

**ESConv's standard target has zero positives.** The seeker change distribution
is `{-4: 107, -3: 193, -2: 433, -1: 417}` and never reaches zero, so
`label = delta > -1` yields 0 positives from 1,150 dialogues. `config.py` sets
`IMPROVEMENT_MARGIN = 2.0`, giving *insufficient improvement* — 417 positives,
base rate 0.363. Do not set it back to 1.0.

**Merging ESConv with FailedESConv leaks the label.** FailedESConv is selected
on the outcome, so base rates are 0.363 and 0.885 and corpus identity nearly
separates the classes. Any trajectory gain measured on the merged corpus is
provenance detection. `data_prep` writes the two files separately and never
merges them; FailedESConv is out-of-distribution evaluation only. The two
releases also use different speaker labels (`seeker`/`supporter` vs
`speaker`/`listener`); without normalisation every FailedESConv dialogue yields
zero seeker turns, silently.

---

## Files

```
src/config.py                 hyperparameters, paths, split definitions
src/data_prep.py              Dreaddit chunking + ESConv/FailedESConv construction
src/prep_cradle.py            CRADLE parse with published-statistic verification
src/audit.py                  ESConv audit: floor effect, leakage, residual target
src/train_turn_encoder.py     Dreaddit encoder + temperature scaling
src/score_conversations.py    score every ESConv seeker turn
src/diagnose_scores.py        ICC, positional drift, go/no-go on the transfer
src/cradle_baseline.py        TF-IDF one-vs-rest floor
src/cradle_tagger.py          hierarchical tagger, three history arms
src/transfer_experiment.py    matched-size provenance study (5-fold CV)
src/make_figures.py           report figures
src/attribution_service.py    async attribution worker (built, NOT evaluated)
src/inspect_corpora.py        one-off schema inspector for new corpora
```

`attribution_service.py` is described in the report and drawn in the
architecture figure but was never measured under load; the report lists that as
future work.

---

## Reporting discipline

`src/config.py` is the single source of truth for every threshold. Three knobs
are the ones most easily tuned into a convenient answer: `IMPROVEMENT_MARGIN`,
`MIN_SEEKER_TURNS`, and `FLAG_THRESHOLD`. Fix them before the first full run.

Decision thresholds are tuned on validation only. The headline metric uses a
single global threshold rather than 16 per-label thresholds, and that choice was
fixed before any test evaluation — validation is generated and test is human, so
per-label tuning offers 16 chances to overfit a distribution the test set does
not share. Per-label results are reported alongside as an upper bound.

Anything tuned after seeing test results must be reported as such.