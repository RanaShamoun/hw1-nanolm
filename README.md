# HW 1 - Language models from scratch (Part 1)

Course code: https://github.com/lkk688/DeepDataMiningLearning (llm/nanolm). Not copied here; this repo holds only my scripts, configs and results.

Machine: HP laptop, CPU only (no NVIDIA GPU), Python 3.13, PyTorch 2.14 (CPU).

## Task 1 - metrics by hand
- scripts/task1_metrics.py recomputes ppl, bits/token and bpb from metrics.json
- results/task1/task1_table.txt is its output for five runs
- results/task1/<run>/ holds config, metrics, meta and sample for each run
- Runs: t1_trained_char (150 steps, warmup 15), t1_untrained_char_lr0 and t1_untrained_bpe512_lr0 (train.lr=0.0, 1 step), plus two extra rows showing the effect of one optimizer step (default warmup; warmup=1)

## Extra credit - GRU from scratch
- scripts/gru_scratch.py: gate-by-gate GRU, compared against nn.GRU
- results/extra/gru_check.txt: output (all checks pass, max diff about 1e-14)

## Task 2 / 3 / 4
To be added.

## Reduced budget (laptop)
configs/01_char_models_laptop.yaml is the course sweep 01_char_models.yaml with max_steps 300 and warmup_steps 15 instead of 2000 and 100.