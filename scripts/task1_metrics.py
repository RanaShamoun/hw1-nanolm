import json, math, os, sys

def vocab_for(name, m):
    for k in ("vocab_size", "vocab"):
        if k in m: return m[k]
    return 512 if "bpe512" in name else 65      # fallback: our two tokenizers

print(f"{'run':34s} {'V':>4s} {'lnV':>6s} {'loss':>7s} {'-lnV':>7s} {'ppl':>8s} {'bits/tok':>8s} {'B/tok':>6s} {'bpb':>7s} | {'logged ppl':>10s} {'logged bpb':>10s}")
for run in sys.argv[1:]:
    name = os.path.basename(run.rstrip("\\/"))
    m = json.load(open(os.path.join(run, "metrics.json")))
    L, N, B = m["val_loss"], m["val_tokens"], m["val_bytes"]
    V = vocab_for(name, m)
    bits = L / math.log(2)
    bpb = L * N / (B * math.log(2))
    print(f"{name[20:]:34s} {V:4d} {math.log(V):6.3f} {L:7.4f} {L-math.log(V):+7.3f} {math.exp(L):8.2f} {bits:8.4f} {B/N:6.3f} {bpb:7.4f} | {m.get('val_ppl')!s:>10s} {m.get('val_bpb')!s:>10s}")