"""Batched training == M independent reference trainings (exact, float64, CPU).

Builds M reference ICNN pairs (f, g) exactly as cellot/models/cellot.py does for the paper config, copies their
weights into BatchedICNN, then runs the paper's training step (10 g steps with the g penalty, 1 f step, clamp of
f) for a few outer iterations on the same batches, with one Adam per reference network and one Adam on the
stacked parameters. Checks forward, transport and all parameters after training.
    python -m lc.test_equivalence
"""
import torch

from lc.batched_icnn import BatchedICNN, loss_f, loss_g
from lc.reference_icnn import ICNN

torch.set_default_dtype(torch.float64)
M, D, B, INNER, OUTER = 3, 13, 64, 10, 4
ADAM = dict(lr=1e-4, betas=(0.5, 0.9), weight_decay=0)
init = lambda w: torch.nn.init.uniform_(w, b=0.1)
torch.manual_seed(0)
ref_f = [ICNN(D, [64] * 4, fnorm_penalty=0, kernel_init_fxn=init) for _ in range(M)]
ref_g = [ICNN(D, [64] * 4, fnorm_penalty=1, kernel_init_fxn=init) for _ in range(M)]
bf, bg = BatchedICNN(M, D, fnorm_penalty=0), BatchedICNN(M, D, fnorm_penalty=1)
for m in range(M):
    bf.load_single(m, ref_f[m]); bg.load_single(m, ref_g[m])
x = torch.randn(M, B, D, requires_grad=True)
assert torch.allclose(bf(x), torch.stack([ref_f[m](x[m]) for m in range(M)]), atol=1e-12)
Tb = bg.transport(x)
Tr = torch.stack([ref_g[m].transport(x[m]) for m in range(M)])
assert torch.allclose(Tb, Tr, atol=1e-12)
print("[OK   ] forward and transport identical")

ro_f = [torch.optim.Adam(n.parameters(), **ADAM) for n in ref_f]
ro_g = [torch.optim.Adam(n.parameters(), **ADAM) for n in ref_g]
bo_f, bo_g = torch.optim.Adam(bf.parameters(), **ADAM), torch.optim.Adam(bg.parameters(), **ADAM)
for it in range(OUTER):
    tgt = torch.randn(M, B, D) + 0.5
    for _ in range(INNER):
        src = torch.randn(M, B, D).requires_grad_(True)
        for m in range(M):   # reference: cellot/train/train.py:225-235
            ro_g[m].zero_grad()
            s = src[m].detach().requires_grad_(True)
            T = ref_g[m].transport(s)
            gl = (ref_f[m](T) - (s * T).sum(-1, keepdim=True)).mean() + ref_g[m].penalize_w()
            gl.backward(); ro_g[m].step()
        bo_g.zero_grad()
        (loss_g(bf, bg, src) + bg.penalize_w()).sum().backward(); bo_g.step()
    src = torch.randn(M, B, D).requires_grad_(True)
    for m in range(M):       # reference: train.py:237-244
        ro_f[m].zero_grad()
        s = src[m].detach().requires_grad_(True)
        fl = (-ref_f[m](ref_g[m].transport(s)) + ref_f[m](tgt[m])).mean()
        fl.backward(); ro_f[m].step(); ref_f[m].clamp_w()
    bo_f.zero_grad()
    loss_f(bf, bg, src, tgt).sum().backward(); bo_f.step(); bf.clamp_w()

err = 0.0
for m in range(M):
    for l in range(5):
        err = max(err, (bf.A[l][m] - ref_f[m].A[l].weight).abs().max().item(),
                  (bg.A[l][m] - ref_g[m].A[l].weight).abs().max().item(),
                  (bg.Ab[l][m] - ref_g[m].A[l].bias).abs().max().item())
    for l in range(4):
        err = max(err, (bf.W[l][m] - ref_f[m].W[l].weight).abs().max().item(),
                  (bg.W[l][m] - ref_g[m].W[l].weight).abs().max().item())
print(f"[CHECK] max |param batched - reference| after {OUTER} outer iterations: {err:.2e}")
assert err < 1e-10, err
print("[OK   ] batched training == independent reference training")
