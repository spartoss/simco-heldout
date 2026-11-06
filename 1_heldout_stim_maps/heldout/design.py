"""Specs for lc/train.py --spec and the closed-form fits.
- drugs/r<i>.json : held-out-donor drug maps. Test donors and training order from the learning-curve design repeat i;
  trained on all 15 non-test donors. One map per (stim or Unstim, celltype of the 168-pair grid, drug): source = the
  drug's plate control under that stim, target = the drug under that stim (batch-corrected drug cache), as the paper.
- stims_all.json  : Unstim -> stim maps for all 196 (stim, celltype) pairs, trained on all 19 donors (OOL features).
    python -m heldout.design
"""
import json

from heldout.common import ALL_PAIRS, CACHE_DRUG, DONORS, DRUGS, WORK
from lc.paths import PAIRS, WORK as LCW


def main():
    out = WORK / "designs"; (out / "drugs").mkdir(parents=True, exist_ok=True)
    plate = json.load(open(CACHE_DRUG / "plate_controls.json"))
    conds = PAIRS + [("Unstim", c) for c in sorted({c for _, c in PAIRS})]
    for dz in json.load(open(LCW / "design.json")):
        maps = [{"stim": s, "celltype": c, "drug": d, "src": f"{s}|{plate[d]}", "tgt": f"{s}|{d}"}
                for s, c in conds for d in DRUGS]
        json.dump({"tests": dz["test"], "train_sets": [{"name": "k15", "donors": dz["order"]}], "maps": maps},
                  open(out / f"drugs/r{dz['repeat']}.json", "w"), indent=1)
    json.dump({"tests": [], "train_sets": [{"name": "all19", "donors": DONORS}], "pairs": ALL_PAIRS},
              open(out / "stims_all.json", "w"), indent=1)
    print(f"{len(maps)} drug maps per repeat; {len(ALL_PAIRS)} stim pairs -> {out}")


if __name__ == "__main__":
    main()
