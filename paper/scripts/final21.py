import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import sys
from pathlib import Path
# data root (layout <root>/<id>/results/): first CLI argument, default ../data (relative to this script)
B=sys.argv[1] if len(sys.argv)>1 else str(Path(__file__).resolve().parent.parent/'data')
CLS=[1,12,49,50,52,53,87,106]; REG=[25,38,39,51,66,73,74,82,83,85,90,95,96]
IDS=CLS+REG; TASK={i:('cls' if i in CLS else 'reg') for i in IDS}
curves={}; refs={}; runs={}
for i in IDS:
    h=pd.read_csv(f'{B}/{i}/results/heldout_performance.csv')
    refs[i]=h[h.strategy.isin(['full_vote','full_stacked','best_single'])].set_index('strategy').score.to_dict()
    p=h[h.strategy.isin(['pruned_vote','pruned_stacked'])]
    curves[i]=p.groupby(['strategy','alpha']).score.mean().unstack(0)
    runs[i]=p.pivot_table(index=['run','alpha'],columns='strategy',values='score')
T=0.001
def cmp(diffs,label):
    d=np.array(diffs); w=(d>T).sum(); l=(d<-T).sum(); t=len(d)-w-l
    p=wilcoxon(d).pvalue if np.any(d!=0) else 1.0
    dz=d.mean()/d.std(ddof=1) if d.std(ddof=1)>0 else 0
    return dict(cmp=label,W=w,T=t,L=l,med=np.median(d),mean=d.mean(),p=p,dz=dz)
def holm(df):
    o=np.argsort(df.p.values); m=len(o); adj=np.empty(m); run=0
    for k,j in enumerate(o):
        run=max(run,min(1,(m-k)*df.p.values[j])); adj[j]=run
    df['p_holm']=adj; return df
out=[]
# RQ1
rq1=[]
for comb in ['pruned_stacked','pruned_vote']:
  for a in [1.0,0.8]:
    for ref in ['full_vote','full_stacked','best_single']:
      rq1.append(cmp([curves[i].loc[a,comb]-refs[i][ref] for i in IDS],f'{comb}@{a} vs {ref}'))
rq1=holm(pd.DataFrame(rq1)); out.append(('RQ1 (svih 21)',rq1))
for tk,ids in [('cls',CLS),('reg',REG)]:
  r=[cmp([curves[i].loc[0.8,'pruned_stacked']-refs[i][ref] for i in ids],f'{tk}: ps@0.8 vs {ref}') for ref in ['full_vote','full_stacked','best_single']]
  out.append((f'RQ1 {tk}',pd.DataFrame(r)))
# RQ2 fixed alpha
rq2=[]
for comb in ['pruned_stacked','pruned_vote']:
  for a in [0.0,0.3,0.5,0.7,0.8,0.9]:
    rq2.append(cmp([curves[i].loc[a,comb]-curves[i].loc[1.0,comb] for i in IDS],f'{comb}: a={a} vs a=1'))
out.append(('RQ2 fiksni alpha vs alpha=1',pd.DataFrame(rq2)))
# LODO
def lodo(comb,scope):
  res=[];al=[]
  for i in IDS:
    others=[j for j in IDS if j!=i and (scope=='all' or TASK[j]==TASK[i])]
    Z=np.mean([(curves[j][comb]-curves[j][comb].mean())/curves[j][comb].std() for j in others],axis=0)
    a=curves[i].index[int(np.argmax(Z))]; al.append(a)
    res.append(curves[i].loc[a,comb]-curves[i].loc[1.0,comb])
  r=cmp(res,f'LODO {comb} scope={scope}'); r['alphas']=','.join(f'{x:g}' for x in al); return r
rq2b=[lodo(c,s) for c in ['pruned_stacked','pruned_vote'] for s in ['all','task']]
out.append(('RQ2 LODO alpha vs alpha=1',pd.DataFrame(rq2b)))
# stacking vs vote at same alpha
rq2c=[cmp([curves[i].loc[a,'pruned_stacked']-curves[i].loc[a,'pruned_vote'] for i in IDS],f'stack vs vote @a={a}') for a in [0.0,0.5,0.8,1.0]]
out.append(('Stacking vs vote, isti alpha',pd.DataFrame(rq2c)))
# run-level consistency ps@0.8 vs ps@1
cons=[]
for i in IDS:
  r=runs[i]['pruned_stacked'].unstack(1)
  d=r[0.8]-r[1.0]; dv=runs[i]['pruned_vote'].unstack(1); dv=dv[0.8]-dv[1.0]
  cons.append(dict(ds=i,task=TASK[i],stack_runs_win=(d>0).mean(),stack_mean=d.mean(),stack_sd=d.std(),vote_runs_win=(dv>0).mean(),vote_mean=dv.mean(),
    curve_range_stack=curves[i].pruned_stacked.max()-curves[i].pruned_stacked.min(),ref_gap_fs_fv=refs[i]['full_stacked']-refs[i]['full_vote']))
out.append(('Konzistentnost po runovima: a=0.8 vs a=1',pd.DataFrame(cons)))
pd.set_option('display.width',250); pd.set_option('display.max_columns',20)
for n,d in out: print('\n=== '+n); print(d.round(4).to_string(index=False))
