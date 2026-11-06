"""M independent CellOT ICNNs trained at once, with stacked weights.

Same architecture as cellot/networks/icnns.py (copied as lc/reference_icnn.py). The paper's configuration is:
hidden [64, 64, 64, 64], LeakyReLU(0.2), squared first layer, non-negative W on f by clamping, a Frobenius penalty on
negative W for g (fnorm_penalty 1), and U(0, 0.1) init of every weight with zero biases.

The maps share no parameter, and the training loss is the SUM over maps of each map's own loss. So every
gradient, and every Adam update (elementwise, no weight decay), is exactly what training the maps one by one would
give; lc/test_equivalence.py checks this against the reference ICNN.
Shapes: x (M, B, d) -> f(x) (M, B, 1); weights (M, out, in) like nn.Linear.weight.
"""
import torch
from torch import nn

LEAK = 0.2


class BatchedICNN(nn.Module):
    def __init__(self, M, input_dim, hidden_units=(64, 64, 64, 64), fnorm_penalty=0.0, init_b=0.1):
        super().__init__()
        units = list(hidden_units) + [1]
        self.fnorm_penalty = fnorm_penalty
        self.W = nn.ParameterList([nn.Parameter(torch.empty(M, o, i)) for i, o in zip(units[:-1], units[1:])])
        self.A = nn.ParameterList([nn.Parameter(torch.empty(M, o, input_dim)) for o in units])
        self.Ab = nn.ParameterList([nn.Parameter(torch.zeros(M, o)) for o in units])
        for p in list(self.W) + list(self.A):
            nn.init.uniform_(p, 0.0, init_b)

    @staticmethod
    def _lin(x, w, b=None):
        y = torch.bmm(x, w.transpose(1, 2))
        return y if b is None else y + b[:, None, :]

    def forward(self, x):
        act = nn.functional.leaky_relu
        z = act(self._lin(x, self.A[0], self.Ab[0]), LEAK)
        z = z * z
        for W, A, b in zip(self.W[:-1], self.A[1:-1], self.Ab[1:-1]):
            z = act(self._lin(z, W) + self._lin(x, A, b), LEAK)
        return self._lin(z, self.W[-1]) + self._lin(x, self.A[-1], self.Ab[-1])

    def transport(self, x):
        (g,) = torch.autograd.grad(self.forward(x).sum(), x, create_graph=True)
        return g

    @torch.no_grad()
    def clamp_w(self):
        for w in self.W:
            w.clamp_(min=0)

    def penalize_w(self):
        """(M,) per-map penalty: fnorm_penalty * sum over layers of ||relu(-W_l)||_F."""
        return self.fnorm_penalty * sum(torch.relu(-w).flatten(1).norm(dim=1) for w in self.W)

    # --- copying to / from the reference single-map ICNN (tests, checkpoints) ---
    @torch.no_grad()
    def load_single(self, m, ref):
        for l, lay in enumerate(ref.W): self.W[l][m].copy_(lay.weight)
        for l, lay in enumerate(ref.A):
            self.A[l][m].copy_(lay.weight); self.Ab[l][m].copy_(lay.bias)


def loss_g(f, g, source):
    """Per-map mean of f(T(x)) - <x, T(x)>, shape (M,)."""
    T = g.transport(source)
    return (f(T).squeeze(-1) - (source * T).sum(-1)).mean(1)


def loss_f(f, g, source, target):
    """Per-map mean of -f(T(x)) + f(y), shape (M,)."""
    T = g.transport(source)
    return (-f(T).squeeze(-1) + f(target).squeeze(-1)).mean(1)
