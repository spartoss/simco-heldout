import numpy as np, pandas as pd
from lc.paths import PAIRS, WORK
D_=WORK/"dmso"; P=["DMSO1","DMSO2","DMSO3"]; QS=(10,25,50,75,90)
rows=[]
for ct in sorted({c for _,c in PAIRS}):
    Z=[np.load(D_/f"cache_{p}/{ct}.npz") for p in P]
    conds=["Unstim"]+sorted({s for s,c in PAIRS if c==ct})
    Q={}
    for cond in conds:
        for d in sorted({k.split("|")[1] for k in Z[0].files if k.startswith(cond+"|")}):
            k=f"{cond}|{d}"
            if all(k in z.files and len(z[k])>=50 for z in Z):
                Q[(cond,d)]=np.stack([np.percentile(z[k],QS,axis=0) for z in Z])  # (3,5,13)
    for cond in conds:
        ds=[d for (c,d) in Q if c==cond and (("Unstim",d) in Q)]
        if len(ds)<5: continue
        for kind in ["level","response"]:
            A=np.stack([Q[(cond,d)]-(Q[("Unstim",d)] if kind=="response" else 0) for d in ds])  # (n,3,5,13)
            if kind=="response" and cond=="Unstim": continue
            dev=A-A.mean(1,keepdims=True)              # within-donor plate deviations
            off=dev.mean(0)                            # shared by all donors (well effect)
            res=dev-off                                # donor x well
            dm=A.mean(1)                               # donor's 3-plate mean
            rows.append({"celltype":ct,"condition":cond,"kind":kind,"n":len(ds),
                "shared_well_rms":np.sqrt((off**2).mean()),"donor_x_well_rms":np.sqrt((res**2).mean()*3/2),
                "between_donor_sd":dm.std(0,ddof=1).mean(),
                "response_size":np.abs((A.mean((0,1)))).mean() if kind=="response" else np.nan})
R=pd.DataFrame(rows)
pd.set_option("display.width",200)
print(R.groupby(["kind","condition"])[["shared_well_rms","donor_x_well_rms","between_donor_sd","response_size"]].median().round(3))
R.to_csv(D_/"classify/plate_magnitude.csv",index=False)
