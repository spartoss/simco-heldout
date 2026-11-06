"""M conditional CellOT ICNNs at once: lc/batched_icnn.BatchedICNN plus a donor context c (CondOT-style, simplified).

The potential stays convex in the cells x for every context c, so its gradient is still an OT map; c only changes
WHICH convex potential (= which map) is used. The context enters each layer in two ways, both linear in c:
- a bias shift      b_l(c) = Ab_l + Bc_l c                      (moves where each unit switches on)
- a positive gate   s_l(c) = 2 sigmoid(Gc_l c) on the W path    (rescales how strongly each unit feeds the next)
    z_1     = leaky(A_0 x + b_0(c))^2
    z_{l+1} = leaky(s_l(c) * (W_l z_l) + A_l x + b_l(c))
    f(x; c) = s_L(c) * (W_L z_L) + A_L x + b_L(c)
Convexity in x: W_l >= 0 times a positive gate keeps a non-negative combination of convex functions; A_l x + b_l(c)
is affine in x; leaky ReLU (and its square) is convex and non-decreasing. For CondOT's full PICNN (context MLP
modulating every path) see Bunne et al. 2022; here the context path is linear on purpose (15 training donors).
Bc and Gc start at zero, so at init (or with c = 0) the network IS the plain BatchedICNN, weight for weight.
Other context architectures (arch, 2026-10-13; every context weight starts at zero, so all start as the plain ICNN):
- gate   (default) the above.
- bias   bias shifts b_l(c) only, no gates.
- film   bias shifts + a multiplicative modulation of the affine skip paths: (1 + tanh(Fc_l c)) * (A_l x); affine in
         x, so convexity is kept; the W paths are not gated.
- trans  f(x; c) = f(x) + <Dc c, x>: the map is grad f(x) + Dc c, a donor-level translation and nothing else.
- affine trans + 1/2 sum_j s_j(c) (x_j - center_j)^2 with s_j(c) = 0.5 tanh(Sc c): the map gets a donor-level
         translation AND a donor-level change of slope per marker in (-0.5, 0.5) around the map's training Unstim
         mean (buffer `center`, set by the trainer). A negative s can make the potential non-convex (not enforced).
Shapes: x (M, B, d), c (M, B, cdim) or (M, 1, cdim) -> f (M, B, 1).
"""
import torch
from torch import nn

from lc.batched_icnn import LEAK, BatchedICNN


class CondBatchedICNN(BatchedICNN):
    def __init__(self, M, input_dim, cdim, hidden_units=(64, 64, 64, 64), fnorm_penalty=0.0, init_b=0.1, arch="gate"):
        super().__init__(M, input_dim, hidden_units, fnorm_penalty, init_b)
        units = list(hidden_units) + [1]
        self.cdim, self.arch = cdim, arch
        z = lambda *sh: nn.Parameter(torch.zeros(*sh))
        if arch in ("gate", "bias", "film"):
            self.Bc = nn.ParameterList([z(M, o, cdim) for o in units])
        if arch == "gate":
            self.Gc = nn.ParameterList([z(M, o, cdim) for o in units[1:]])
        if arch == "film":
            self.Fc = nn.ParameterList([z(M, o, cdim) for o in units])
        if arch in ("trans", "affine"):
            self.Dc = z(M, input_dim, cdim)
        if arch == "affine":
            self.Sc = z(M, input_dim, cdim)
            self.register_buffer("center", torch.zeros(M, input_dim))

    def ctx_params(self):
        return [p for n in ("Bc", "Gc", "Fc") if hasattr(self, n) for p in getattr(self, n)] + \
               [getattr(self, n) for n in ("Dc", "Sc") if hasattr(self, n)]

    def forward(self, x, c=None):
        if c is None or self.cdim == 0: return super().forward(x)
        if self.arch in ("trans", "affine"):
            out = super().forward(x) + (x * self._lin(c, self.Dc)).sum(-1, keepdim=True)
            if self.arch == "affine":
                s = 0.5 * torch.tanh(self._lin(c, self.Sc))
                out = out + 0.5 * (s * (x - self.center[:, None, :]) ** 2).sum(-1, keepdim=True)
            return out
        act = nn.functional.leaky_relu
        cb = lambda l: self._lin(c, self.Bc[l])                      # (M, B or 1, o)
        gate = (lambda l: 2 * torch.sigmoid(self._lin(c, self.Gc[l]))) if self.arch == "gate" else (lambda l: 1.0)
        film = (lambda l: 1 + torch.tanh(self._lin(c, self.Fc[l]))) if self.arch == "film" else (lambda l: 1.0)
        z = act(film(0) * self._lin(x, self.A[0]) + self.Ab[0][:, None, :] + cb(0), LEAK)
        z = z * z
        L = len(self.W)
        for l in range(L - 1):
            z = act(gate(l) * self._lin(z, self.W[l]) + film(l + 1) * self._lin(x, self.A[l + 1]) + self.Ab[l + 1][:, None, :] + cb(l + 1), LEAK)
        return gate(L - 1) * self._lin(z, self.W[-1]) + film(L) * self._lin(x, self.A[-1]) + self.Ab[-1][:, None, :] + cb(L)

    def ctx_l2(self):
        """(M,) sum of squared context weights: the --ctx-wd penalty; 0 when cdim = 0."""
        if self.cdim == 0: return 0.0
        return sum((p ** 2).flatten(1).sum(1) for p in self.ctx_params())

    def transport(self, x, c=None):
        (g,) = torch.autograd.grad(self.forward(x, c).sum(), x, create_graph=True)
        return g


def sub(net, idx):
    """View restricted to maps idx (evaluation)."""
    v = CondBatchedICNN.__new__(CondBatchedICNN); nn.Module.__init__(v)
    v.fnorm_penalty, v.cdim, v.arch = net.fnorm_penalty, net.cdim, getattr(net, "arch", "gate")
    for name in ("W", "A", "Ab", "Bc", "Gc", "Fc"):
        if hasattr(net, name): setattr(v, name, [p[idx] for p in getattr(net, name)])
    for name in ("Dc", "Sc", "center"):
        if hasattr(net, name): setattr(v, name, getattr(net, name)[idx])
    return v


def loss_g(f, g, x, c):
    T = g.transport(x, c)
    return (f(T, c).squeeze(-1) - (x * T).sum(-1)).mean(1)


def loss_f(f, g, x, y, c):
    T = g.transport(x, c)
    return (-f(T, c).squeeze(-1) + f(y, c).squeeze(-1)).mean(1)
