"""Claims frozen from the 118-bus pilot and evaluated unchanged on any new case.
C1 prediction error: Vg error rises approaching a generator's Qmax transition (slope on log distance < 0).
C2 harm: on the active side only, a constraint's recovered violation rises with depth (slope on log depth > 0), holding load.
C3 inversion: ridge has worse-or-similar MAE than the best MLP yet lower Q violation than every MLP.
C4 structure: Q violation is explained by branch-weighted differential Vg error beyond Vg MAE."""
import os
import pickle, json, warnings, sys
warnings.filterwarnings("ignore")
import numpy as np, statsmodels.api as sm
from common import *
name=sys.argv[1]; ntrain=int(sys.argv[2])
case=load_case(os.path.join(os.environ.get("PGLIB_DIR","pglib-opf"),f"pglib_opf_case{name}.m"))
res=[r for r in pickle.load(open(f"data_{name}.pkl","rb")) if r and r["ok"]]
rand=[r for r in res if r["kind"]=="rand"]; path=[r for r in res if r["kind"]=="path"]
train,test=rand[:ntrain],rand[ntrain:]; allS=test+path; n=len(test)
tag=f"_{sys.argv[3]}" if len(sys.argv)>3 else ""
ev=pickle.load(open(f"eval_{name}{tag}.pkl","rb")); mn=list(ev)
g=case["gen"];bus=case["bus"];br=case["branch"];nb,ng,nl=bus.shape[0],g.shape[0],br.shape[0]
ids={int(b):i for i,b in enumerate(bus[:,0])}; gb=np.array([ids[int(x)] for x in g[:,0]])
typ=np.array(["Pmax"]*ng+["Pmin"]*ng+["Qmax"]*ng+["Qmin"]*ng+["Vmax"]*nb+["Vmin"]*nb+["Sf"]*nl+["St"]*nl)
pr=np.maximum(g[:,PMAX]-g[:,PMIN],1.0);qr=np.maximum(g[:,QMAX]-g[:,QMIN],1.0)
slack_g=np.where(bus[gb,1]==3)[0]
clamped=np.zeros(typ.size,bool);clamped[:2*ng]=True;clamped[slack_g]=False;clamped[ng+slack_g]=False
clamped[4*ng+gb]=True;clamped[4*ng+nb+gb]=True
TAU=float(os.environ.get("ACOPF_TAU","1e-4"))  # Active-set tolerance; set to test sensitivity.
A=lambda S:np.array([(np.abs(r["mu"])>TAU).astype(np.int8) for r in S])
Atr=A(train);Aall=A(allS);f=Atr.mean(0);sw=(f>0)&(f<1)&~clamped
Yt=np.array([np.r_[r["pg"],r["vg"]] for r in allS]); load=np.array([r["pd"].sum() for r in allS])/bus[:,PD].sum()
V={}
for m in mn:
    a=np.full((len(allS),typ.size),np.nan)
    for i,x in enumerate(ev[m]["rec"]):
        if x is not None:a[i]=-x[0]
    V[m]=a
fam={}
for m in mn: fam.setdefault(m.split("_s")[0],[]).append(m)
pid=np.array([r["pid"] for r in path]);tt=np.array([r["t"] for r in path]);Ap=Aall[n:];NP=max(tt)+1 if len(tt) else 0
out={"case":name,"n_test":n,"n_path_pts":len(path),"n_switching":int(sw.sum()),"switch_by_type":{t:int((sw&(typ==t)).sum()) for t in np.unique(typ)},
     "novel_frac":float(np.mean([Aall[i][(f>0)&(f<1)].tobytes() not in set(a[(f>0)&(f<1)].tobytes() for a in Atr) for i in range(n)]))}
qidx=np.where(np.isin(typ,["Qmax","Qmin"]))[0]; qphys=np.r_[qr,qr]
out["models"]={}
for m in mn:
    y=ev[m]["yh"];v=V[m][:n]
    out["models"][m]=dict(mae_pg_pct=float(100*(np.abs(y[:n,:ng]-Yt[:n,:ng])/pr).mean()),mae_vg_pu=float(np.abs(y[:n,ng:]-Yt[:n,ng:]).mean()),
        q_viol_mvar=float(np.nanmean(np.nansum(v[:,qidx]*qphys,axis=1))),nc=int(np.isnan(v).all(1).sum()))
mlps=[m for m in mn if m!="ridge"]
out["C3"]=dict(ridge_mae_vg=out["models"]["ridge"]["mae_vg_pu"],best_mlp_mae_vg=min(out["models"][m]["mae_vg_pu"] for m in mlps),
  ridge_q=out["models"]["ridge"]["q_viol_mvar"],min_mlp_q=min(out["models"][m]["q_viol_mvar"] for m in mlps),max_mlp_q=max(out["models"][m]["q_viol_mvar"] for m in mlps))
out["C3"]["holds"]=bool(out["C3"]["ridge_q"]<out["C3"]["min_mlp_q"] and out["C3"]["ridge_mae_vg"]>=out["C3"]["best_mlp_mae_vg"])
# C1
out["C1"]={}
for lab,mods in fam.items():
    Eg=np.mean([np.abs(ev[m]["yh"][n:,ng:]-Yt[n:,ng:]) for m in mods],0);L=[]
    for k in np.unique(pid):
        I=np.where(pid==k)[0];I=I[np.argsort(tt[I])]
        if len(I)!=NP: continue
        for gi in range(ng):
            if not sw[2*ng+gi]: continue
            a=Ap[I,2*ng+gi];ch=np.where(np.diff(a)!=0)[0]
            if len(ch)!=1: continue
            b=ch[0]
            for j in range(NP): L.append((k*10000+gi,min(abs(j-b),abs(j-b-1)),Eg[I[j],gi],load[n+I[j]]))
    L=np.array(L)
    if len(L)==0: out["C1"][lab]=None; continue
    y=np.log10(L[:,2]+1e-6);w=y.copy();X=np.log1p(L[:,1]);Lo=L[:,3].copy()
    for u in np.unique(L[:,0]):
        s=L[:,0]==u
        for a_ in (w,X,Lo): a_[s]-=a_[s].mean()
    r=sm.OLS(w,np.column_stack([X,Lo])).fit(cov_type="cluster",cov_kwds={"groups":L[:,0]})
    out["C1"][lab]=dict(episodes=int(np.unique(L[:,0]).size),slope=float(r.params[0]),se=float(r.bse[0]),p=float(r.pvalues[0]),near=float(L[L[:,1]<=1,2].mean()),far=float(L[L[:,1]>=8,2].mean()) if (L[:,1]>=8).any() else None)
# C2
out["C2"]={}
for lab,mods in fam.items():
    Vb=np.nanmean([V[m][n:] for m in mods],0);R=[]
    for k in np.unique(pid):
        I=np.where(pid==k)[0];I=I[np.argsort(tt[I])]
        if len(I)!=NP: continue
        for c in np.where(sw)[0]:
            a=Ap[I,c];ch=np.where(np.diff(a)!=0)[0]
            if len(ch)!=1: continue
            b=ch[0]
            for j in range(NP):
                if a[j]==1:
                    d=(j-b-1) if a[-1]==1 else (b-j)
                    R.append((k*100000+c,d,Vb[I[j],c],load[n+I[j]],np.abs(allS[n+I[j]]["mu"][c])))
    R=np.array(R)
    if len(R)==0: out["C2"][lab]=None; continue
    ok=np.isfinite(R[:,2]);R=R[ok]
    w=np.log10(R[:,2]+1e-4);X=np.log1p(R[:,1]);Lo=R[:,3].copy();Mu=np.log10(R[:,4]+1e-6)
    for u in np.unique(R[:,0]):
        s=R[:,0]==u
        for a_ in (w,X,Lo,Mu): a_[s]-=a_[s].mean()
    r=sm.OLS(w,np.column_stack([X,Lo])).fit(cov_type="cluster",cov_kwds={"groups":R[:,0]})
    r2=sm.OLS(w,np.column_stack([Mu,Lo])).fit(cov_type="cluster",cov_kwds={"groups":R[:,0]})
    out["C2"][lab]=dict(episodes=int(np.unique(R[:,0]).size),slope_depth=float(r.params[0]),se=float(r.bse[0]),p=float(r.pvalues[0]),load_b=float(r.params[1]),load_p=float(r.pvalues[1]),
        mu_slope=float(r2.params[0]),mu_se=float(r2.bse[0]),mu_p=float(r2.pvalues[0]),near=float(R[R[:,1]<=2,2].mean()),deep=float(R[R[:,1]>=13,2].mean()) if (R[:,1]>=13).any() else None)
# C4: differential Vg error along generator-to-generator electrical links
Yb=np.zeros((nb,nb))
for b_ in br:
    if b_[10]>0:
        i,j=ids[int(b_[0])],ids[int(b_[1])];w_=1/np.hypot(b_[2],b_[3]);Yb[i,j]+=w_;Yb[j,i]+=w_
pairs=[]
L_=np.diag(Yb.sum(1))-Yb
Zg=np.linalg.pinv(L_)[np.ix_(gb,gb)]
Reff=np.diag(Zg)[:,None]+np.diag(Zg)[None,:]-2*Zg
iu=np.triu_indices(ng,1);wgt=1/np.maximum(Reff[iu],1e-4)
out["C4"]={};rows=[]
for k,m in enumerate(mn):
    e=ev[m]["yh"][:n,ng:]-Yt[:n,ng:]
    mae=np.abs(e).mean(1)
    diff=(np.abs(e[:,iu[0]]-e[:,iu[1]])*wgt).sum(1)/wgt.sum()
    q=np.nansum(V[m][:n][:,qidx]*qphys,axis=1)
    out["C4"][m]=dict(diff_vg=float(diff.mean()),mean_signed_common=float(np.abs(e.mean(1)).mean()),within=None)
    rows.append((np.full(n,k),mae,diff,q))
K=np.concatenate([r[0] for r in rows]);MAE=np.concatenate([r[1] for r in rows]);DF=np.concatenate([r[2] for r in rows]);Q=np.concatenate([r[3] for r in rows])
# Within-model association across samples, standardized
ww=lambda a:np.concatenate([(a[K==k]-a[K==k].mean())/a[K==k].std() for k in np.unique(K)])
rr=sm.OLS(ww(np.log10(Q+1e-3)),np.column_stack([ww(np.log10(MAE)),ww(np.log10(DF))])).fit(cov_type="cluster",cov_kwds={"groups":np.tile(np.arange(n),len(mn))})
out["C4_reg"]=dict(mae=dict(b=float(rr.params[0]),p=float(rr.pvalues[0])),diff=dict(b=float(rr.params[1]),p=float(rr.pvalues[1])),R2=float(rr.rsquared))
a=[out["models"][m]["mae_vg_pu"] for m in mn];d=[out["C4"][m]["diff_vg"] for m in mn];q=[out["models"][m]["q_viol_mvar"] for m in mn]
from scipy.stats import spearmanr
out["C4_between"]=dict(rho_mae_q=float(spearmanr(a,q)[0]),rho_diff_q=float(spearmanr(d,q)[0]),n_models=len(mn))
json.dump(out,open(f"claims_{name}{tag}"+(f"_tau{TAU:g}" if "ACOPF_TAU" in os.environ else "")+".json","w"),indent=1);print(json.dumps(out))
