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

## Task 2 - RNN vs GRU vs LSTM vs Transformer
- `configs/01_char_models_laptop.yaml` is the course sweep `01_char_models.yaml` with 300 steps (15 warmup) instead of 2000 (100 warmup). Everything else is unchanged: 6 models, seeds 1337 and 42, 8192 tokens per step, 2.46M tokens per run.
- `results/task2/` holds each run's config, metrics and logs, plus `summary.csv` and `summary.md` (written by `nanolm.analyze`) and `analyze_output.txt` (the results table with parameter counts and the seed noise floor).
- `scripts/task2_curves.py` makes `figures/task2_curves.png`: lines are training loss in bits per byte (logged every 50 steps, mean of 2 seeds); hollow squares are validation bpb at steps 250 and 300.
- `figures/task2_curves.png` is the figure to use. The tool's own `curves.png` is not included, because it only covers steps 250 to 300, the only steps where this sweep logged validation.

## Task 3 / 4
To be added.


## Reduced budget (laptop)
configs/01_char_models_laptop.yaml is the course sweep 01_char_models.yaml with max_steps 300 and warmup_steps 15 instead of 2000 and 100.