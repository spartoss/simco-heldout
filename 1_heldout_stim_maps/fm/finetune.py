"""Phase 4: continue ImmuneFM's masked-marker pre-training on the atlas, with tokens that can see a stim response.

Why: with the published tokenisation (per-cell MinMax, 10 bins) 63 % of the live stim responses leave the median
token unmoved (fm/sanity.py); per-marker quantile tokens (fm/tokenize.q_tokens, 64 bins) leave 2.5 %.
Changes from pre-training: tokens = per-marker quantile bins of the training donors (token_emb warm-started from the
published value token of the same rank, everything else from the checkpoint; --init random = all random, the ablation
that isolates pre-training);
the sequence is the 38 panel columns only (positions = their vocabulary columns, fm/tokenize.panel_pos); 25 % of the
positions masked (zeros included: in this panel a zero is a measurement), 90 % of them replaced by MASK.
Data: DMSO cells of all conditions (Unstim and the 7 stims) of the repeat's 15 training donors only (design.json
order[:15]), the first 80 % of each cached block (lc's train side); celltypes sampled uniformly, then cells.
Eval every --eval-every steps on a fixed masked sample: val side of the training donors, and the 4 test donors
(never trained on): accuracy, +-1 accuracy, CE, and the per-marker marginal baseline.
Output --out: ckpt.pt (model_state_dict, num_tokens, panel, pos, edges, init, repeat), log.csv.
    python3 -m fm.finetune --repeat 0 --init pretrained --out WORK/fm/finetune/repeat_0/pretrained
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fm.immunefm import CKPT, MASK, N_BINS, load
from fm.tokenize import Q_BINS, Q_MASK, panel_pos, q_edges, q_tokens
from lc.paths import CELLTYPES, VAL_FRAC, WORK
from lc.train import durable_save

CACHE = WORK / "fm" / "cache_full"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def read_blocks(donors, part, cap=None):
    """{celltype: (n, 38) float32} of the given donors' blocks; part 'train' = first 80 %, 'val' = rest, 'all'."""
    out, panel = {}, None
    for c in CELLTYPES:
        z = np.load(CACHE / f"{c}.npz"); panel = list(z["markers"]); xs = []
        for k in z.files:
            if k == "markers" or k.split("|")[1] not in donors: continue
            a = z[k]; n_tr = int(round(len(a) * (1 - VAL_FRAC)))
            a = a[:n_tr] if part == "train" else a[n_tr:] if part == "val" else a
            xs.append(a[:cap])
        if xs: out[c] = np.concatenate(xs)
    return out, panel


def masked(t, prob, gen):
    """t (B, L) long -> (input, target with -100 off the mask)."""
    m = torch.rand(t.shape, device=t.device, generator=gen) < prob
    rep = m & (torch.rand(t.shape, device=t.device, generator=gen) < 0.9)
    return t.masked_fill(rep, Q_MASK), t.masked_fill(~m, -100)


@torch.no_grad()
def evaluate(model, t_in, t_tgt, pos, batch=8192):
    model.eval(); acc = near = ce = n = 0.0
    for s in range(0, len(t_in), batch):
        lg = model(t_in[s:s + batch], pos=pos)[..., :Q_BINS + 1]        # tokens 0..Q_BINS (never MASK)
        y = t_tgt[s:s + batch]; mm = y >= 0
        lg, y = lg[mm], y[mm]; p = lg.argmax(-1)
        acc += (p == y).sum().item(); near += ((p - y).abs() <= 1).sum().item()
        ce += torch.nn.functional.cross_entropy(lg, y, reduction="sum").item(); n += len(y)
    model.train()
    return acc / n, near / n, ce / n


def marginal(t_train, t_tgt):
    """Per-marker token frequencies of the training cells as the predictor of the masked targets."""
    acc = near = ce = n = 0.0
    for j in range(t_train.shape[1]):
        p = np.bincount(t_train[:, j], minlength=Q_BINS + 1)[:Q_BINS + 1] + 1.0; p /= p.sum()
        y = t_tgt[:, j]; y = y[y >= 0]
        acc += (y == p.argmax()).sum(); near += (np.abs(y - p.argmax()) <= 1).sum(); ce += -np.log(p[y]).sum(); n += len(y)
    return acc / n, near / n, ce / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--init", choices=["pretrained", "random"], default="pretrained")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--mask", type=float, default=0.25)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--n-eval", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dz = json.load(open(WORK / "design.json"))[a.repeat]; train_d, test_d = dz["order"][:15], dz["test"]

    tr, panel = read_blocks(set(train_d), "train")
    allx = np.concatenate(list(tr.values()))
    edges = q_edges(allx[rng.permutation(len(allx))[:2_000_000]])
    pos = torch.as_tensor(panel_pos(panel), device=DEV)
    T = {c: torch.as_tensor(q_tokens(x, edges), device=DEV) for c, x in tr.items()}
    cts = sorted(T); sizes = torch.tensor([len(T[c]) for c in cts])
    print(f"repeat {a.repeat} init {a.init}: {sum(sizes).item()} training cells, {len(cts)} celltypes, "
          f"panel {len(panel)}, {time.time() - t0:.0f}s", flush=True)

    gen = torch.Generator(device=DEV).manual_seed(a.seed + 1)
    def fixed_eval(xd):
        x = np.concatenate(list(xd.values())); x = x[rng.permutation(len(x))[:a.n_eval]]
        t = torch.as_tensor(q_tokens(x, edges), device=DEV).long()
        return masked(t, a.mask, gen)
    ev = {"val": fixed_eval(read_blocks(set(train_d), "val")[0]), "test": fixed_eval(read_blocks(set(test_d), "all")[0])}
    t_marg = np.concatenate([t.cpu().numpy()[:200_000] for t in T.values()])
    rows = [dict(step=0, split=s, arm="marginal", **dict(zip(["acc", "acc_pm1", "ce"], marginal(t_marg, e[1].cpu().numpy()))))
            for s, e in ev.items()]

    model = load(a.init == "pretrained", seed=a.seed, num_tokens=Q_BINS + 2)
    if a.init == "pretrained":     # warm start: each quantile bin takes the published value token of the same rank
        old = torch.load(CKPT / "immuneFM_pretrain_100.pth", map_location="cpu", weights_only=False)["model_state_dict"]["token_emb.weight"]
        src = [0] + [1 + (q - 1) * N_BINS // Q_BINS for q in range(1, Q_BINS + 1)] + [MASK]
        with torch.no_grad():
            model.token_emb.weight.copy_(old[src] + 0.05 * torch.randn(len(src), old.shape[1]))
    model = model.to(DEV).train()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) *
                                              (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps))))
    per_ct = a.batch // len(cts)
    for step in range(1, a.steps + 1):
        idx = [torch.randint(0, int(sizes[i]), (per_ct,), device=DEV, generator=gen) for i in range(len(cts))]
        t = torch.cat([T[c][ix] for c, ix in zip(cts, idx)]).long()
        x_in, y = masked(t, a.mask, gen)
        loss = torch.nn.functional.cross_entropy(model(x_in, pos=pos).transpose(1, 2), y, ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            for s, (xi, yt) in ev.items():
                rows.append(dict(step=step, split=s, arm=a.init, **dict(zip(["acc", "acc_pm1", "ce"], evaluate(model, xi, yt, pos)))))
            last = {r["split"]: r for r in rows[-2:]}
            print(f"step {step} loss {loss.item():.4f} | val acc {last['val']['acc']:.4f} ce {last['val']['ce']:.4f} | "
                  f"test acc {last['test']['acc']:.4f} ce {last['test']['ce']:.4f} | {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(out / "log.csv", index=False)
            ck = {"model_state_dict": model.state_dict(), "num_tokens": Q_BINS + 2, "panel": panel,
                  "pos": pos.cpu().numpy(), "edges": edges, "init": a.init, "repeat": a.repeat, "step": step,
                  "train_donors": train_d, "test_donors": test_d}
            def w(fh, ck=ck):
                buf = io.BytesIO(); torch.save(ck, buf); fh.write(buf.getvalue())
            durable_save(out / "ckpt.pt", w)
    print(f"[DONE ] {out} {time.time() - t0:.0f}s", flush=True)


def load_ft(path, device="cpu"):
    """(eval model, ckpt dict) of a fine-tuned checkpoint; encode with pos=ck['pos'] on q_tokens(x, ck['edges'])."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = load(False, num_tokens=ck["num_tokens"]); m.load_state_dict(ck["model_state_dict"], strict=True)
    return m.eval().to(device), ck


if __name__ == "__main__":
    main()
