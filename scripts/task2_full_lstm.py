import glob, json, math, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def main(sweep_dir, out_png, tokens_per_step=8192):
    runs = {}
    for d in sorted(glob.glob(os.path.join(sweep_dir, "*_s*"))):
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "eval_log.jsonl")):
            continue
        tr = {r["step"]: r["loss"] / math.log(2) for r in read(os.path.join(d, "train_log.jsonl"))}
        ev = {r["step"]: r["bpb"] for r in read(os.path.join(d, "eval_log.jsonl"))}
        runs[m.group(1)] = (tr, ev)
    if not runs:
        sys.exit(f"No runs found in {sweep_dir}")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, (name, (tr, ev)) in enumerate(sorted(runs.items())):
        xs = sorted(tr); ax.plot([x * tokens_per_step / 1e6 for x in xs], [tr[x] for x in xs], "--", color=f"C{i}", alpha=0.7, label=f"{name} train")
        xs = sorted(ev); ax.plot([x * tokens_per_step / 1e6 for x in xs], [ev[x] for x in xs], "-o", ms=4, color=f"C{i}", label=f"{name} validation")
    ax.set_xlabel("tokens seen (M)"); ax.set_ylabel("bits per byte (lower is better)")
    ax.set_title("LSTM 2x256 vs 2x512, 2000 steps: training vs validation bpb")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out_png, dpi=150)
    for name, (tr, ev) in sorted(runs.items()):
        print(f"\n{name}: step, validation bpb, training bpb, gap (val - train)")
        for s in sorted(ev):
            t = tr.get(s)
            print(f"  {s:5d}  {ev[s]:7.4f}  " + (f"{t:7.4f}  {ev[s]-t:+7.4f}" if t is not None else "   (no train point)"))
        best = min(ev, key=ev.get)
        print(f"  best validation {ev[best]:.4f} at step {best}; final {ev[max(ev)]:.4f} (final - best = {ev[max(ev)]-ev[best]:+.4f})")
    print("\nwrote", out_png)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])