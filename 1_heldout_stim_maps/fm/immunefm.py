"""ImmuneFM (Ding, Bhattacharya, Butte, bioRxiv 2025.07.09.664020), vendored from github.com/buttelab/ImmuneFM.

`Transformer` is code/model.py as published (weights load with strict=True from ckpt/immuneFM_pretrain_100.pth);
only `encode` is added: the 128-d hidden states after the final LayerNorm, before the tied projection onto the 12
token embeddings that `forward` returns. The repo has no licence: internal research use only (see README).

Vocabulary: 203 markers in fm/ckpt/biomarkers_align.txt order + 1 trailing column always 0 (pretrain.py appends it).
Tokens: 0 = zero / marker absent, 1..10 = value bins, 11 = MASK (also PAD). See fm/tokenize.py.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

CKPT = Path(__file__).resolve().parent / "ckpt"
N_BINS = 10
N_TOKENS = N_BINS + 2          # 12
MASK = N_TOKENS - 1            # 11
SEQ_LEN = 203 + 1


class Marker2VecPositionalEmbedding(nn.Module):
    def __init__(self, marker_dim=62):
        super().__init__()
        self.emb = nn.Embedding(marker_dim, 128)

    def forward(self, x, pos=None):
        t = torch.arange(x.shape[1], device=x.device) if pos is None else pos
        return self.emb(t)


class Transformer(nn.Module):
    def __init__(self, num_tokens, max_seq_len, dim, depth, heads, ff_mult=4, norm_first=False, emb_dropout=0.05):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.depth = depth
        layers = []
        for i in range(depth):
            layers.append(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * ff_mult,
                                                     batch_first=True, norm_first=norm_first))
        self.transformer_encoder = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(dim)
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.marker2emb = Marker2VecPositionalEmbedding(marker_dim=max_seq_len)
        self.dropout = nn.Dropout(emb_dropout)

    def hidden(self, x, padding_mask=None, pos=None):
        """pos: vocabulary columns of x's positions (default all SEQ_LEN, as published); a subset (e.g. the panel's
        columns only) skips the always-zero columns, which fm/finetune.py does."""
        x = self.token_emb(x)
        x = x + self.marker2emb(x, pos)
        x = self.dropout(x)
        for mod in self.transformer_encoder:
            x = mod(x, src_key_padding_mask=padding_mask)
        return self.norm(x)                                    # (B, SEQ_LEN, dim)

    def forward(self, x, padding_mask=None, pos=None):
        return self.hidden(x, padding_mask, pos) @ self.token_emb.weight.t()   # (B, L, num_tokens) logits

    @torch.no_grad()
    def encode(self, tokens, cols, batch=4096, device="cpu", pos=None):
        """Cell embeddings: mean of the hidden states over positions `cols`. tokens (n, L) int64; pos as in hidden."""
        out = []
        p = None if pos is None else torch.as_tensor(pos, device=device)
        for s in range(0, len(tokens), batch):
            h = self.hidden(torch.as_tensor(tokens[s:s + batch], device=device), pos=p)
            out.append(h[:, cols].mean(1).float().cpu())
        return torch.cat(out).numpy()


def markers():
    return [m.strip().strip("'") for m in (CKPT / "biomarkers_align.txt").read_text().split(",")]


def load(pretrained=True, seed=0, num_tokens=N_TOKENS):
    """eval-mode model; pretrained=False gives the same architecture at random init (the ablation). num_tokens !=
    N_TOKENS (fm/finetune.py's tokenisation) loads everything but token_emb, which stays at random init."""
    torch.manual_seed(seed)
    m = Transformer(num_tokens=num_tokens, max_seq_len=SEQ_LEN, dim=128, depth=6, heads=8)
    if pretrained:
        sd = torch.load(CKPT / "immuneFM_pretrain_100.pth", map_location="cpu", weights_only=False)["model_state_dict"]
        if num_tokens != N_TOKENS:
            sd = {k: v for k, v in sd.items() if k != "token_emb.weight"}
            missing, unexpected = m.load_state_dict(sd, strict=False)
            assert missing == ["token_emb.weight"] and not unexpected, (missing, unexpected)
        else:
            m.load_state_dict(sd, strict=True)
    return m.eval()


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


if __name__ == "__main__":
    m = load()
    print(f"ImmuneFM: {n_params(m)} parameters, {len(markers())} markers")
    t = torch.from_numpy(np.random.default_rng(0).integers(0, 11, (3, SEQ_LEN)))
    h1, h2 = m.hidden(t), m.hidden(t)
    print("eval deterministic:", torch.equal(h1, h2), "logits", tuple(m(t).shape))
