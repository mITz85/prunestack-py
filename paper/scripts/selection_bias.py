import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import sys
from pathlib import Path
# data root (layout <root>/<id>/results/): first CLI argument, default ../data (relative to this script)
B=sys.argv[1] if len(sys.argv)>1 else str(Path(__file__).resolve().parent.parent/'data')
CLS=[1,12,49,50,52,53,87,106]; REG=[25,38,39,51,66,73,74,82,83,85,90,95,96]; IDS=CLS+REG
TASK={i:('cls' if i in CLS else 'reg') for i in IDS}
rng=np.random.default_rng(1985)
def load(i):
    h=pd.read_csv(f'{B}/{i}/results/heldout_performance.csv')
    p=h[h.strategy.isin(['pruned_vote','pruned_stacked'])]
    return p.pivot_table(index='run',columns=['strategy','alpha'],values='score')
R={i:load(i) for i in IDS}
def summ(d,label):
    d=np.array(d); return dict(test=label,W=(d>1e-3).sum(),T=(abs(d)<=1e-3).sum(),L=(d<-1e-3).sum(),mean=d.mean(),med=np.median(d),p=wilcoxon(d).pvalue if np.any(d!=0) else 1)
rows=[]
for comb in ['pruned_stacked','pruned_vote']:
    # (a) on-target: alpha* = argmax of mean curve over all 30 runs, gain evaluated on same runs
    on=[]; split=[]; perm=[]
    for i in IDS:
        M=R[i][comb]; mean=M.mean(); a=mean.idxmax(); on.append(mean[a]-mean[1.0])
        # (b) split runs: select on odd runs, evaluate on even runs (and vice versa), average
        g=[]
        for sel,ev in [(M.index%2==1,M.index%2==0),(M.index%2==0,M.index%2==1)]:
            a=M[sel].mean().idxmax(); g.append(M[ev].mean()[a]-M[ev].mean()[1.0])
        split.append(np.mean(g))
        # (c) null: permute alpha labels within each run, then on-target argmax gain -> expected gain from pure noise
        pg=[]
        for _ in range(200):
            P=M.values.copy()
            for r in range(P.shape[0]): P[r]=P[r][rng.permutation(P.shape[1])]
            m=P.mean(0); pg.append(m.max()-m[list(M.columns).index(1.0)])
        perm.append(np.mean(pg))
    rows+= [summ(on,f'{comb}: on-target alpha* gain vs a=1'),
            summ(split,f'{comb}: split-run (select odd/eval even) gain vs a=1'),
            summ(perm,f'{comb}: expected on-target gain under alpha-permutation null')]
df=pd.DataFrame(rows); pd.set_option('display.width',200); print(df.round(4).to_string(index=False))
