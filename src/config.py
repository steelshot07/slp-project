"""Central configuration. Edit here, not in the individual scripts."""
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
for _d in (DATA, ARTIFACTS, RESULTS):
    _d.mkdir(exist_ok=True, parents=True)

SEEDS = [13, 29, 47, 71, 97]          # five seeded runs, per the review's methodology standard
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- turn encoder
# distilroberta-base is the default: 82M params, fits a 6GB consumer GPU.
# Swap to "mental/mental-roberta-base" to reproduce the domain-pretrained comparison.
ENCODER_NAME = "distilroberta-base"
MAX_LEN = 128            # ESConv turns are short; long Dreaddit posts get chunked instead
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# Chunking of Dreaddit posts into short spans, to close the length/register gap
# between long retrospective Reddit narratives and short synchronous helpline turns.
CHUNK_TRAIN = True
CHUNK_MIN_WORDS = 8
CHUNK_MAX_SENTS = 3

# ---------------------------------------------------------------- trajectory
# Escalation target from ESConv seeker survey: delta = final - initial intensity.
# label = 1  ("non-improving") when delta > -IMPROVEMENT_MARGIN
IMPROVEMENT_MARGIN = 2.0   # label=1 when delta > -2 (improved by <2 points)
# WAS 1.0, which yields ZERO positives. ESConv is positively selected: its
# seeker delta distribution is {-4:107, -3:193, -2:433, -1:417} and never
# reaches 0 -- the non-improving conversations were removed during collection
# and released separately as FailedESConv. At 2.0 the target is "insufficient
# improvement": 417 positives of 1150, base rate 0.363.
MIN_SEEKER_TURNS = 4       # conversations shorter than this cannot show a trajectory

GRU_HIDDEN = 64
GRU_EPOCHS = 60
GRU_LR = 1e-3
GRU_DROPOUT = 0.4
USE_EMBEDDINGS = True      # model D2: scores + pooled turn embeddings (PCA-reduced)
EMB_PCA_DIM = 16

# ---------------------------------------------------------------- evaluation
ALERT_BUDGETS = [0.10, 0.20, 0.30]   # fraction of conversations a reviewer can open
ECE_BINS = 15
N_BOOTSTRAP = 1000
N_FOLDS = 5
N_REPEATS = 5              # repeated stratified CV: 5 x 5 = 25 conversation-level splits

OOD_GROUPS = ["ongoing depression", "job crisis", "breakup with partner",
              "problems with friends", "academic pressure"]
# Leave-one-group-out: five folds. The ~86 conversations in minor types
# (sleep problems, alcohol abuse, etc.) stay in training in every fold.

FLAG_THRESHOLD = 0.6       # turn score above which asynchronous attribution is queued
