"""ESM2 protein embeddings of the 13 functional markers (for a marker-similarity feature instead of a categorical marker).

Each marker -> its human UniProt protein(s) (MARKER_PROTEINS; the pERK and pSTAT5 antibodies see two proteins each:
their embeddings are averaged). Sequences fetched from rest.uniprot.org once (meta/marker_proteins.fasta), embedded
with ESM2 (facebook/esm2_t33_650M_UR50D, MIT licence; mean over residues of the last layer, no BOS/EOS; sequences
longer than 1022 residues are embedded in 1022-residue windows and averaged).
Writes WORK/marker_embed/esm2_650M.npz: markers, emb (13, 1280), cos (13, 13).
    python -m lc.marker_embed     (needs py-transformers/4.39.1_py312 and network for the first run)
"""
import os
import urllib.request

import numpy as np

from lc.paths import MARKERS, REPO, WORK

MARKER_PROTEINS = {"CD25": ["P01589"], "HLADR": ["P01903"], "IkB": ["P25963"], "pCREB": ["P16220"],
                   "pERK": ["P27361", "P28482"], "pMK2": ["P49137"], "pNFkB": ["Q04206"], "pS6": ["P62753"],
                   "pSTAT1": ["P42224"], "pSTAT3": ["P40763"], "pSTAT5": ["P42229", "P51692"], "pSTAT6": ["P42226"],
                   "pp38": ["Q16539"]}
MODEL = "facebook/esm2_t33_650M_UR50D"
FASTA = REPO / "meta" / "marker_proteins.fasta"
OUT = WORK / "marker_embed"


def sequences():
    if not FASTA.exists():
        txt = "".join(urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", timeout=30).read().decode()
                      for accs in MARKER_PROTEINS.values() for acc in accs)
        FASTA.write_text(txt)
    seqs, acc = {}, None
    for line in FASTA.read_text().splitlines():
        if line.startswith(">"): acc = line.split("|")[1]; seqs[acc] = ""
        else: seqs[acc] += line.strip()
    return seqs


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    seqs = sequences()
    tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModel.from_pretrained(MODEL).eval()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    emb = {}
    with torch.no_grad():
        for acc, s in seqs.items():
            parts = [s[i:i + 1022] for i in range(0, len(s), 1022)]
            vs = []
            for p in parts:
                h = model(**tok(p, return_tensors="pt")).last_hidden_state[0, 1:-1]
                vs.append((h.mean(0).numpy(), len(p)))
            emb[acc] = sum(v * n for v, n in vs) / sum(n for _, n in vs)
            print(f"[EMB  ] {acc} {len(s)} aa", flush=True)
    E = np.stack([np.mean([emb[a] for a in MARKER_PROTEINS[m]], 0) for m in MARKERS])
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    np.savez(OUT / "esm2_650M.npz", markers=np.array(MARKERS), emb=E, cos=En @ En.T)
    print(f"[DONE ] -> {OUT / 'esm2_650M.npz'}")


if __name__ == "__main__":
    main()
