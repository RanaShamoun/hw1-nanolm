"""Score every attention head of a nanolm GPT.
Usage (run from the llm folder):
  python task4_heads.py RUN_DIR OUT.csv "text 1" "text 2" @textfile.txt
An argument that starts with @ is read from that UTF-8 text file (so texts can contain newlines).
Columns (average attention weight on the position named):
  cur, prev, prev2   current character, previous, two back
  first              first character of the text
  wordstart          first character of the current word (only positions >= 3 characters into a word, first word skipped)
  marker             any ':' or newline character at or before the current position (speaker-label marker)
  dist               mean look-back distance in characters, sum_s A[t,s] * (t - s), averaged over positions
"""
import csv, os, sys
import numpy as np

WS = " \n\t"
MARK = ":\n"

def word_starts(text):
    starts, s = [], 0
    for i, ch in enumerate(text):
        if ch in WS:
            starts.append(None)
        else:
            if i == 0 or text[i - 1] in WS:
                s = i
            starts.append(s)
    return starts

def chance(positions):
    return float(np.mean([1.0 / (t + 1) for t in positions])) if positions else float("nan")

def positions_for(text):
    T = len(text)
    ws = word_starts(text)
    pos = {
        "cur":  list(range(1, T)),
        "prev": list(range(1, T)),
        "prev2": list(range(2, T)),
        "first": list(range(1, T)),
        "wordstart": [t for t in range(T) if ws[t] is not None and ws[t] > 0 and t - ws[t] >= 3],
        "marker": list(range(1, T)),
        "dist": list(range(1, T)),
    }
    return pos, ws

def baselines(text, pos):
    marks = np.cumsum([1 if c in MARK else 0 for c in text])
    base = {k: chance(pos[k]) for k in ["cur", "prev", "prev2", "first", "wordstart"]}
    base["marker"] = float(np.mean([marks[t] / (t + 1) for t in pos["marker"]]))
    base["dist"] = float(np.mean([t / 2.0 for t in pos["dist"]]))
    return base

def score_head(A, text, pos=None, ws=None):
    if pos is None:
        pos, ws = positions_for(text)
    marker_idx = [i for i, c in enumerate(text) if c in MARK]
    def avg(ps, f):
        return float(np.mean([f(t) for t in ps])) if ps else float("nan")
    return {
        "cur":  avg(pos["cur"],  lambda t: A[t, t]),
        "prev": avg(pos["prev"], lambda t: A[t, t - 1]),
        "prev2": avg(pos["prev2"], lambda t: A[t, t - 2]),
        "first": avg(pos["first"], lambda t: A[t, 0]),
        "wordstart": avg(pos["wordstart"], lambda t: A[t, ws[t]]),
        "marker": avg(pos["marker"], lambda t: float(sum(A[t, i] for i in marker_idx if i <= t))),
        "dist": avg(pos["dist"], lambda t: float(sum(A[t, s] * (t - s) for s in range(t + 1)))),
    }

KEYS = ["cur", "prev", "prev2", "first", "wordstart", "marker", "dist"]

def load_text(arg):
    if arg.startswith("@"):
        return open(arg[1:], encoding="utf-8").read()
    return arg

def main(run_dir, out_csv, texts):
    sys.path.insert(0, os.getcwd())
    import torch
    from nanolm.viz import load_run, attention_maps
    cfg, model, tok, dinfo, _ = load_run(run_dir)
    model.eval()
    rows, per_text = [], []
    for ti, arg in enumerate(texts, 1):
        text = load_text(arg)
        ids = tok.encode(text)
        if len(ids) != len(text):
            print(f"WARNING text {ti}: {len(ids)} tokens for {len(text)} characters; scores assume 1 token per character")
        with torch.no_grad():
            maps = attention_maps(model, torch.tensor([ids]))
        pos, ws = positions_for(text)
        base = baselines(text, pos)
        print(f"\n=== text {ti} ({len(text)} chars): {text[:60]!r}")
        print(f"{'head':6s}" + "".join(f"{k:>11s}" for k in KEYS))
        print(f"{'chance':6s}" + "".join(f"{base[k]:11.3f}" for k in KEYS))
        table = {}
        for l, layer in enumerate(maps):
            for h in range(layer.shape[0]):
                s = score_head(layer[h], text, pos, ws)
                table[(l, h)] = s
                print(f"{l}.{h:<4d}" + "".join(f"{s[k]:11.3f}" for k in KEYS))
                rows.append([ti, l, h] + [round(s[k], 4) for k in KEYS] + [round(base[k], 4) for k in KEYS])
        per_text.append(table)
    heads = sorted(per_text[0])
    print("\n=== mean over the texts")
    print(f"{'head':6s}" + "".join(f"{k:>11s}" for k in KEYS))
    means = {hd: {k: float(np.nanmean([t[hd][k] for t in per_text])) for k in KEYS} for hd in heads}
    for hd in heads:
        print(f"{hd[0]}.{hd[1]:<4d}" + "".join(f"{means[hd][k]:11.3f}" for k in KEYS))
    for k in KEYS:
        best = max(heads, key=lambda hd: means[hd][k])
        print(f"best {k:10s}: head {best[0]}.{best[1]}  mean score {means[best][k]:.3f}")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "layer", "head"] + KEYS + [f"chance_{k}" for k in KEYS])
        w.writerows(rows)
    print("\nwrote", out_csv)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])