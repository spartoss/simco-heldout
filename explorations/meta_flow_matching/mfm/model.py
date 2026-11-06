"""Meta Flow Matching (Atanackovic et al., ICLR 2025; github.com/lazaratan/meta-flow-matching, MIT licence),
re-implemented in plain torch after src/models/components/{gnn,mlp}.py and src/models/trellis_module.TrellisMFM.

Same pieces as GlobalGNN / TrellisMFM:
- population encoder: GCN layers on a kNN graph of the source population's cells (input = cell features + cell
  condition), ReLU between layers, mean-pooled over cells, L2-normalised -> embedding z.
  GCNConv on a kNN graph (torch_geometric knn_graph, k neighbours, self loops added, symmetric degree norm) has every
  in-degree = k + 1, so a layer is out_i = (H_i + sum_{j in kNN(i)} H_j) / (k + 1) + b with H = X W: written as such.
- velocity field: SkipMLP(SELU, skip connections) on [z, sinusoidal t embedding (128), random Fourier features of y
  (B ~ N(0, 0.01^2), 128 -> 256), y, condition] -> dy/dt in the flow space.
- loss (base = source, independent coupling): y_t = (1 - t) x0 + t x1, u = x1 - x0, ||b||^2 - 2 b.u.
Changes: the encoder can embed several populations and concatenate their embeddings (n_pop > 1: the donor's own
celltype + its dominant celltypes, "mfm_blood"); emb_dim = 0 gives the FM baseline (no population information).
"""
import math

import torch
from torch import nn


class SkipMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.act = nn.SELU()

    def forward(self, x):
        out = self.act(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.act(layer(out) + out)
        return self.output_layer(out)


class KnnGCN(nn.Module):
    def __init__(self, d_in, hidden, n_layers, knn_k):
        super().__init__()
        self.k = knn_k
        self.lins = nn.ModuleList([nn.Linear(d_in if i == 0 else hidden, hidden) for i in range(n_layers)])

    def forward(self, x, feats):
        """x (n, d) for the graph, feats (n, d_in) node features -> (hidden,) normalised mean embedding."""
        n = len(x)
        k = min(self.k, n - 1)
        if k > 0:
            nb = torch.cdist(x, x).topk(k + 1, largest=False).indices[:, 1:]       # drop self
        z = feats
        for i, lin in enumerate(self.lins):
            h = z @ lin.weight.t()
            agg = h + (h[nb].sum(1) if k > 0 else 0)
            z = agg / (k + 1) + lin.bias
            if i < len(self.lins) - 1: z = torch.relu(z)
        z = z.mean(0)
        return z / z.norm()


class MFM(nn.Module):
    def __init__(self, D, d_embed_in, n_cond, n_cell_cond, n_pop=1, use_embedding=True, num_hidden=512,
                 num_layers_decoder=7, num_hidden_gnn=128, num_layers_gnn=2, knn_k=100, t_dim=128, n_fourier=128,
                 fourier_scale=0.01):
        super().__init__()
        self.D, self.t_dim, self.n_pop, self.use_embedding = D, t_dim, n_pop, use_embedding
        self.gnn = KnnGCN(d_embed_in + n_cell_cond, num_hidden_gnn, num_layers_gnn, knn_k) if use_embedding else None
        emb = num_hidden_gnn * n_pop if use_embedding else 0
        self.decoder = SkipMLP(emb + t_dim + 2 * n_fourier + D + n_cond, D, num_hidden, num_layers_decoder)
        self.register_buffer("B", torch.randn(D, n_fourier) * fourier_scale)

    def embed(self, pops):
        """pops: list of (x_graph (n, d), cell_cond (d_cc,)) -> (n_pop * hidden,) concatenated embeddings."""
        out = []
        for x, cc in pops:
            feats = torch.cat([x, cc.expand(len(x), -1)], 1)
            out.append(self.gnn(x, feats))
        return torch.cat(out)

    def t_embed(self, t, max_positions=10000):
        half = self.t_dim // 2
        f = torch.exp(torch.arange(half, device=t.device) * -(math.log(max_positions) / (half - 1)))
        e = t[:, None] * f[None]
        return torch.cat([e.sin(), e.cos()], 1)

    def velocity(self, z, t, y, cond):
        """z (emb,) or None, t (n,), y (n, D), cond (n, n_cond) -> (n, D)."""
        yb = 2 * math.pi * y @ self.B
        parts = ([z.expand(len(y), -1)] if z is not None else []) + [self.t_embed(t), yb.cos(), yb.sin(), y, cond]
        return self.decoder(torch.cat(parts, 1))

    def loss(self, z, x0, x1, cond):
        t = torch.rand(len(x0), device=x0.device)
        y = (1 - t[:, None]) * x0 + t[:, None] * x1
        b = self.velocity(z, t, y, cond)
        return ((b ** 2).sum(1) - 2 * (b * (x1 - x0)).sum(1)).mean()

    @torch.no_grad()
    def transport(self, z, x0, cond, steps=100):
        """Integrate dy/dt from t = 0 (x0) to 1 with the midpoint rule."""
        y, h = x0.clone(), 1.0 / steps
        for s in range(steps):
            t = torch.full((len(y),), s * h, device=y.device)
            k1 = self.velocity(z, t, y, cond)
            y = y + h * self.velocity(z, t + h / 2, y + h / 2 * k1, cond)
        return y
