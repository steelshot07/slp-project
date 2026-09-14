#!/usr/bin/env bash
set -e
python -m src.data_prep                 # Dreaddit + ESConv
python -m src.prep_cradle               # CRADLE, self-verifying parse
python -m src.audit                     # ESConv audit -> results/audit.json
python -m src.cradle_baseline           # TF-IDF floor

for S in 13 29 47 71 97; do             # ESConv trajectory null
  python -m src.train_turn_encoder --seed $S
  python -m src.score_conversations --seed $S
  python -m src.diagnose_scores --seed $S
done

for S in 13 29 47 71 97; do             # CRADLE 3-arm ablation
  for A in gru none mean; do
    python -m src.cradle_tagger --arm $A --seed $S
  done
done
for S in 13 71 97; do                   # 12-epoch robustness check
  for A in gru none mean; do
    python -m src.cradle_tagger --arm $A --seed $S --epochs 12
  done
done

for S in 13 29 47; do                   # matched-size transfer
  python -m src.transfer_experiment --seed $S
done
python -m src.make_figures