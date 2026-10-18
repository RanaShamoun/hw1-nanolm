import csv, glob, json, os, re, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main(sweep_dir, out_png, out_csv):
    final = defaultdict(dict)                      # final[model][T] -> list of bpb over seeds
    for d in sorted(glob.glob(os.path.join(sweep_dir, "*_T*_s*"))):
        m = re.match(r"(.+)_T(\d+)_s(\d+)$", os.path.basename(d))
        f = os.path.join(d, "eval_log.jsonl")
        if not m or not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        if rows:
            final[m.group(1)].setdefault(int(m.group(2)), []).append(rows[-1]["bpb"])
    if not final:
        sys.exit(f"No finished runs matching *_T*_s* with eval_log.jsonl in {sweep_dir}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    table = []
    for i, (model, d) in enumerate(sorted(final.items())):
        Ts = sorted(d)
        means = [sum(d[T]) / len(d[T]) for T in Ts]
        ax.plot(Ts, means, "-o", color=f"C{i}", label=model)
        for T in Ts:
            ax.plot([T] * len(d[T]), d[T], ".", color=f"C{i}", alpha=0.5)
        for T, mu in zip(Ts, means):
            table.append((model, T, len(d[T]), mu))
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256, 512]); ax.set_xticklabels(["64", "128", "256", "512"])
    ax.set_xlabel("context length (train.seq_len = eval window)")
    ax.set_ylabel("validation bits per byte (lower is better)")
    ax.set_title("Task 3: bpb vs context length (tokens/step fixed at 8192)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=150)

    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "seq_len", "n_seeds", "val_bpb_mean"]); w.writerows(table)
    print(f"{'model':14s} {'T':>4s} {'seeds':>5s} {'bpb':>8s} {'gain vs previous T':>20s}")
    prev = {}
    for model, T, n, mu in table:
        gain = f"{prev[model] - mu:+.4f}" if model in prev else ""
        print(f"{model:14s} {T:4d} {n:5d} {mu:8.4f} {gain:>20s}")
        prev[model] = mu
    print("wrote", out_png, "and", out_csv)

if __name__ == "__main__":
    main(*sys.argv[1:4])