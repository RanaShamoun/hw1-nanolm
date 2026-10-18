import json, math, os, re, sys, glob
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def main(sweep_dir, out_png, tokens_per_step=8192):
    runs = defaultdict(lambda: {"train": [], "eval": []})
    for d in sorted(glob.glob(os.path.join(sweep_dir, "*_s*"))):
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        name = m.group(1)
        t, e = os.path.join(d, "train_log.jsonl"), os.path.join(d, "eval_log.jsonl")
        if os.path.exists(t):
            runs[name]["train"].append({r["step"]: r["loss"] / math.log(2) for r in read(t)})   # char models: bytes/token = 1, so bpb = nats / ln 2
        if os.path.exists(e):
            runs[name]["eval"].append({r["step"]: r["bpb"] for r in read(e)})
    if not runs:
        sys.exit(f"No run folders matching *_s* found in {sweep_dir}")
   
    def mean_curve(curves):
        steps = sorted(set.intersection(*[set(c) for c in curves]))
        return steps, [sum(c[s] for c in curves) / len(curves) for s in steps]

    order = sorted(runs, key=lambda n: mean_curve(runs[n]["eval"])[1][-1] if runs[n]["eval"] else 99)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(order):
        col = f"C{i}"
        if runs[name]["train"]:
            s, y = mean_curve(runs[name]["train"])
            ax.plot([x * tokens_per_step / 1e6 for x in s], y, "-o", ms=4, color=col, label=f"{name} (train)")
        if runs[name]["eval"]:
            s, y = mean_curve(runs[name]["eval"])
            ax.plot([x * tokens_per_step / 1e6 for x in s], y, "s", ms=8, mfc="none", mew=1.8, color=col)
    ax.set_xlabel("tokens seen (M)")
    ax.set_ylabel("bits per byte (lower is better)")
    ax.set_title("Task 2: lines = training loss (mean of seeds), hollow squares = validation bpb")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("wrote", out_png, "| models:", ", ".join(order))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])