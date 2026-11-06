"""Designs for the donor experiments (custom specs read by lc/train.py and lc/baselines.py via --spec).

- hvpv/r<i>.json   HV vs PV. Per repeat: 2 HV + 3 PV test donors; from the remaining 6 HV / 8 PV, training sets of equal
                   size k in {2, 6} made of HV only, PV only, or half-half. -> does training on one group hurt the other?
- pairwise.json    every donor alone (k = 1) as a training set, every donor as a test donor (19 x 19; the diagonal is
                   dropped at scoring). -> what makes a good training donor, and a hard test donor.
- small/r<i>.json  the 8 smallest celltypes (median 180-920 training cells per donor): the design.json repeats' test
                   donors and orders, k in {1, 2, 4, 7, 10, 15} x cells per donor in {25, 50, 100, 200, 400, all}.
                   -> at equal total cells, is it better to add donors or cells?
    python -m lc.design_donors
"""
import json

import numpy as np

from lc.paths import DONORS, WORK

SMALL = ["pDCs", "mDCs", "CD56hiCD16negNK", "CD8Tcm_CCR2pos", "ncMCs", "intMCs", "MDSCs", "CD4Teff"]
HV = [d for d in DONORS if d.startswith("HV")]; PV = [d for d in DONORS if d.startswith("PV")]


def main():
    out = WORK / "designs"; (out / "hvpv").mkdir(parents=True, exist_ok=True); (out / "small").mkdir(exist_ok=True)
    for r in range(6):
        rng = np.random.default_rng(2000 + r)
        hv, pv = list(rng.permutation(HV)), list(rng.permutation(PV))
        tests, hv_tr, pv_tr = hv[:2] + pv[:3], hv[2:], pv[3:]
        sets = []
        for k in (2, 6):
            sets += [{"name": f"HV{k}", "donors": hv_tr[:k]}, {"name": f"PV{k}", "donors": pv_tr[:k]},
                     {"name": f"MIX{k}", "donors": hv_tr[:k // 2] + pv_tr[:k // 2]}]
        json.dump({"tests": tests, "train_sets": sets, "caps": [0]}, open(out / f"hvpv/r{r}.json", "w"), indent=1)
        print("hvpv", r, "test", tests, "| HV6", hv_tr, "| PV6", pv_tr[:6])
    json.dump({"tests": DONORS, "train_sets": [{"name": d, "donors": [d]} for d in DONORS], "caps": [0]},
              open(out / "pairwise.json", "w"), indent=1)
    design = json.load(open(WORK / "design.json"))
    for dz in design:
        sets = [{"name": f"k{k}", "donors": dz["order"][:k]} for k in (1, 2, 4, 7, 10, 15)]
        json.dump({"tests": dz["test"], "train_sets": sets, "caps": [25, 50, 100, 200, 400, 0], "celltypes": SMALL},
                  open(out / f"small/r{dz['repeat']}.json", "w"), indent=1)
    print("written to", out)


if __name__ == "__main__":
    main()
