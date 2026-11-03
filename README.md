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

- `configs/03_full_lstm_laptop.yaml` and `results/task2_full_lstm/`: LSTM 2x256 vs 2x512 for the full 2000 steps (seed 1337). `scripts/task2_full_lstm.py` makes `figures/task2_full_lstm.png`. The 2x512 reaches its best validation bpb (2.1306) at step 1200 and ends at 2.1791 while its training bpb keeps falling; the 2x256 ends at 2.1392.

## Task 3 - context length
- `configs/02_context_laptop.yaml`: LSTM 2x256 and modern Transformer at seq_len 64/128/256/512, 300 steps (15 warmup), seed 1337. Tokens per step is held at 8192 (batch_size x seq_len), so every run sees 2.46M tokens. Logging every 10 steps, evaluation every 25.
- `results/task3/`: one folder per run (config, metrics, logs, sample).
- `scripts/task3_context.py` makes `figures/task3_context.png` and `results/task3_context.csv` from the finished runs (it reads `metrics.json`).
- Seeds: 1337 at all four lengths; 42 at T = 64, 256 and 512 (`configs/02_context_laptop_seed42.yaml`). The seed-42 T = 256 runs are copies of the Task 2 seed-42 runs. `results/task3/sweep.yaml` comes from the seed-42 sweep; the seed-1337 settings are in `configs/02_context_laptop.yaml`.
- Validation bpb at T = 64 / 128 / 256 / 512 (mean of seeds; T = 128 is seed 1337 only): LSTM 2.8655 / 2.8554 / 2.8538 / 2.8585; modern Transformer 2.4885 / 2.4859 / 2.5018 / 2.5743. Seed noise floor from Task 2 is 0.009 bpb.

## Task 4 - attention analysis
- `scripts/task4_heads.py` imports `attention_maps` from `nanolm.viz` and scores every head of a trained `gpt_modern` run: average attention weight on the current token (`cur`), previous token (`prev`), two back (`prev2`), first token (`first`), and the first character of the current word (`wordstart`, counted only where the word start is at least 3 characters back). Each score is printed next to a chance baseline (mean of 1/(t+1) over the same positions).
- Three texts: Shakespeare, modern prose, and repetitive text (all under 256 characters).
- `results/task4_heads.csv` (seed 1337 model) and `results/task4_heads_seed42.csv` (seed 42 model).
- `figures/task4_*.png`: attention maps from `python -m nanolm.viz attention`.
- `scripts/task4_heads.py` also reports the weight on colon and newline characters (`marker`) and each head's mean look-back distance (`dist`). `scripts/shakes_dialogue.txt` is the 184-character two-speaker text used for those two measures: `python scripts/task4_heads.py <run dir> <out.csv> @scripts/shakes_dialogue.txt` (run from the llm folder).
- `results/task4_markers_seed1337.csv` and `results/task4_markers_seed42.csv`: all measures on the dialogue text. Marker heads: seed 1337 head 3.2 (0.407, chance 0.058) and seed 42 head 2.2 (0.461). The same two heads have the longest mean look-back (34.2 and 31.5 characters; uniform attention would give 46).


## Reduced budget (laptop)
configs/01_char_models_laptop.yaml is the course sweep 01_char_models.yaml with max_steps 300 and warmup_steps 15 instead of 2000 and 100.