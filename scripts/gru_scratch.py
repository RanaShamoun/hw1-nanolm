import torch, torch.nn as nn

torch.manual_seed(0)

class GRUScratch(nn.Module):
    """GRU written gate by gate, using PyTorch's convention:
       r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)          reset gate
       z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)          update gate
       n = tanh(W_in x + b_in + r * (W_hn h + b_hn))       candidate state
       h' = (1 - z) * n + z * h
    """
    def __init__(self, d_in, d_h):
        super().__init__()
        self.d_h = d_h
        self.ir, self.hr = nn.Linear(d_in, d_h), nn.Linear(d_h, d_h)
        self.iz, self.hz = nn.Linear(d_in, d_h), nn.Linear(d_h, d_h)
        self.in_, self.hn = nn.Linear(d_in, d_h), nn.Linear(d_h, d_h)

    def forward(self, x, h=None):                      # x: [B, T, d_in]
        B, T, _ = x.shape
        if h is None:
            h = x.new_zeros(B, self.d_h)
        outs = []
        for t in range(T):
            xt = x[:, t]
            r = torch.sigmoid(self.ir(xt) + self.hr(h))
            z = torch.sigmoid(self.iz(xt) + self.hz(h))
            n = torch.tanh(self.in_(xt) + r * self.hn(h))   # r multiplies the hidden projection INCLUDING its bias
            h = (1 - z) * n + z * h
            outs.append(h)
        return torch.stack(outs, 1), h                 # outputs [B,T,H], last hidden [B,H]

def load_from_torch(scratch, gru):
    """nn.GRU stacks the three gates in order (r, z, n) inside weight_ih_l0 / weight_hh_l0 and both biases."""
    H = scratch.d_h
    pairs = [(scratch.ir, scratch.hr), (scratch.iz, scratch.hz), (scratch.in_, scratch.hn)]
    with torch.no_grad():
        for k, (li, lh) in enumerate(pairs):
            s = slice(k * H, (k + 1) * H)
            li.weight.copy_(gru.weight_ih_l0[s]); li.bias.copy_(gru.bias_ih_l0[s])
            lh.weight.copy_(gru.weight_hh_l0[s]); lh.bias.copy_(gru.bias_hh_l0[s])

def check(name, a, b, tol=1e-10):
    err = (a - b).abs().max().item()
    print(f"{name:34s} max|diff| = {err:.3e}   {'PASS' if err < tol else 'FAIL'}")
    return err < tol

if __name__ == "__main__":
    d_in, H, B, T = 16, 32, 4, 50
    ref = nn.GRU(d_in, H, batch_first=True).double()
    mine = GRUScratch(d_in, H).double()
    load_from_torch(mine, ref)

    x = torch.randn(B, T, d_in, dtype=torch.double)
    h0 = torch.randn(B, H, dtype=torch.double)

    ok = True
    out_r, h_r = ref(x)                                # zero initial state
    out_m, h_m = mine(x)
    ok &= check("outputs (zero init state)", out_m, out_r)
    ok &= check("final hidden (zero init state)", h_m, h_r[0])

    out_r, h_r = ref(x, h0.unsqueeze(0))               # non-zero initial state
    out_m, h_m = mine(x, h0)
    ok &= check("outputs (random init state)", out_m, out_r)

    # gradients: same loss, compare weight gradients
    ref.zero_grad(); mine.zero_grad()
    ref(x)[0].pow(2).sum().backward()
    mine(x)[0].pow(2).sum().backward()
    g_mine_ih = torch.cat([mine.ir.weight.grad, mine.iz.weight.grad, mine.in_.weight.grad])
    g_mine_hh = torch.cat([mine.hr.weight.grad, mine.hz.weight.grad, mine.hn.weight.grad])
    ok &= check("grad weight_ih", g_mine_ih, ref.weight_ih_l0.grad, tol=1e-8)
    ok &= check("grad weight_hh", g_mine_hh, ref.weight_hh_l0.grad, tol=1e-8)

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")