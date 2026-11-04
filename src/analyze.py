"""Pilot analysis: stratified failure, path event study, zero-slack constraint comparison."""
import os
import sys, pickle, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, statsmodels.api as sm
from common import *
name=sys.argv[1]; ntrain=int(sys.argv[2])
case=load_case(os.path.join(os.environ.get("PGLIB_DIR","pglib-opf"),f"pglib_opf_case{name}.m"))
res=[r for r in pickle.load(open(f"data_{name}.pkl","rb")) if r and r["ok"]]
rand=[r for r in res if r["kind"]=="rand"]; path=[r for r in res if r["kind"]=="path"]
train,test=rand[:ntrain],rand[ntrain:]; allS=test+path
ev=pickle.load(open(f"eval_{name}.pkl","rb"))
g=case["gen"]; nb,ng,nl=case["bus"].shape[0],g.shape[0],case["branch"].shape[0]
gb=(g[:,0]-1).astype(int)
# Constraint order Pmax,Pmin,Qmax,Qmin,Vmax,Vmin,Sf,St
typ=np.array(["Pmax"]*ng+["Pmin"]*ng+["Qmax"]*ng+["Qmin"]*ng+["Vmax"]*nb+["Vmin"]*nb+["Sf"]*nl+["St"]*nl)
slack_g=np.where(case["bus"][gb,1]==3)[0]
clamped=np.zeros(typ.size,bool)  # enforced by recovery: non-slack Pg bounds; V at generator buses
clamped[:2*ng]=True; clamped[slack_g]=False; clamped[ng+slack_g]=False
clamped[4*ng+gb]=True; clamped[4*ng+nb+gb]=True
out={"case":name}
def actset(S,tol):
    return np.array([(np.abs(r["mu"])>tol).astype(np.int8) for r in S])
Atr=actset(train,1e-4); Aall=actset(allS,1e-4); n=len(test)
f=Atr.mean(0); sw=(f>0)&(f<1)
out["n_train"]=len(train); out["n_test"]=len(test); out["n_path_pts"]=len(path)
out["n_constraints"]=int(typ.size); out["n_switching"]=int(sw.sum())
out["switch_by_type"]={t:int((sw&(typ==t)).sum()) for t in np.unique(typ)}
key=lambda a: a[sw].tobytes()
trs=set(key(a) for a in Atr); out["distinct_AS_train"]=len(trs)
out["novel_frac_test"]=float(np.mean([key(a) not in trs for a in Aall[:n]]))
uniqtr=np.unique(Atr[:,sw],axis=0)
ham=np.array([np.abs(uniqtr-a[sw]).sum(1).min() for a in Aall])
# tolerance sensitivity of novelty
sens={}
for tol in [1e-5,1e-4,1e-3,1e-2]:
    A1=actset(train,tol);A2=actset(test,tol);ff=A1.mean(0);ss=(ff>0)&(ff<1)
    S=set(a[ss].tobytes() for a in A1); sens[str(tol)]=dict(n_switch=int(ss.sum()),distinct=len(S),novel=float(np.mean([a[ss].tobytes() not in S for a in A2])))
out["tol_sensitivity"]=sens
load=np.array([r["pd"].sum() for r in allS])/case["bus"][:,PD].sum()
nact=Aall.sum(1)
Yt=np.array([np.r_[r["pg"],r["vg"]] for r in allS]); fopt=np.array([r["f"] for r in allS])
pr=np.maximum(g[:,PMAX]-g[:,PMIN],1.0)
mn=list(ev.keys()); M=len(mn)
pgerr=np.zeros((M,len(allS))); viol=np.full((M,len(allS)),np.nan); gap=np.full((M,len(allS)),np.nan); nc=np.zeros((M,len(allS)),bool)
free=~clamped
vc=[]
for k,m in enumerate(mn):
    yh=ev[m]["yh"]; rec=ev[m]["rec"]
    pgerr[k]=np.abs(yh[:,:ng]-Yt[:,:ng]).__truediv__(pr).mean(1)
    V=np.full((len(allS),typ.size),np.nan)
    for i,x in enumerate(rec):
        if x is None: nc[k,i]=True; continue
        V[i]=-x[0]; gap[k,i]=(x[1]-fopt[i])/fopt[i]
    viol[k]=np.nanmax(V[:,free],axis=1); vc.append(V)
vc=np.array(vc)
te=np.arange(n); pa=np.arange(n,len(allS))
# A. per model nominal vs stratified
novel=ham[:n]>0
rows=[]
for k,m in enumerate(mn):
    rows.append(dict(model=m,mae_pg=float(pgerr[k,te].mean()),viol_mean=float(np.nanmean(viol[k,te])),viol_p95=float(np.nanpercentile(viol[k,te],95)),
       gap_mean=float(np.nanmean(np.abs(gap[k,te]))),nc=int(nc[k,te].sum()),
       viol_seen=float(np.nanmean(viol[k,te][~novel])) if (~novel).any() else None,viol_novel=float(np.nanmean(viol[k,te][novel])) if novel.any() else None,
       mae_seen=float(pgerr[k,te][~novel].mean()) if (~novel).any() else None,mae_novel=float(pgerr[k,te][novel].mean()) if novel.any() else None))
out["models"]=rows
# Pick best NN family for regressions: average over MLP256 seeds
idx=[k for k,m in enumerate(mn) if m.startswith("mlp256")]
vbar=np.nanmean(viol[idx],0); ebar=pgerr[idx].mean(0)
# B. regression on random test set: novelty vs load, tightness
def ols(y,Xd,names):
    X=sm.add_constant(np.column_stack(Xd)); ok=np.isfinite(y)
    r=sm.OLS(y[ok],X[ok]).fit(cov_type="HC3")
    return {nm:dict(b=float(r.params[i+1]),se=float(r.bse[i+1]),p=float(r.pvalues[i+1])) for i,nm in enumerate(names)}|{"R2":float(r.rsquared),"n":int(ok.sum())}
z=lambda a:(a-a.mean())/a.std()
y=np.log10(vbar[te]+1e-4); out["reg_viol_test"]=ols(y,[z(ham[te].astype(float)),z(load[te]),z(nact[te].astype(float))],["hamming","load","n_active"])
out["reg_err_test"]=ols(np.log10(ebar[te]),[z(ham[te].astype(float)),z(load[te]),z(nact[te].astype(float))],["hamming","load","n_active"])
out["ham_bins"]={}
for lo,hi,lb in [(0,0,"0"),(1,1,"1"),(2,3,"2-3"),(4,99,"4+")]:
    s=(ham[te]>=lo)&(ham[te]<=hi)
    if s.sum(): out["ham_bins"][lb]=dict(n=int(s.sum()),load=float(load[te][s].mean()),viol=float(np.nanmean(vbar[te][s])),err=float(ebar[te][s].mean()))
# C. path event study
pid=np.array([r["pid"] for r in path]); tt=np.array([r["t"] for r in path]); Ap=Aall[pa]; Ys=Yt[pa]
ev_rows=[];trans_per=[]
dist=np.full(len(path),np.nan); kink=np.full(len(path),np.nan)
scale=np.r_[pr,np.ones(ng)*0.1]
for k in np.unique(pid):
    I=np.where(pid==k)[0]; I=I[np.argsort(tt[I])]
    if len(I)<5: continue
    consecutive=np.diff(tt[I])==1
    ch=(np.abs(np.diff(Ap[I][:,sw],axis=0)).sum(1)>0)&consecutive
    trans_per.append(int(ch.sum()))
    tp=np.where(ch)[0]+0.5
    for j,i in enumerate(I):
        if len(tp): dist[i]=np.abs(j-tp).min()
    Yn=Ys[I,:ng]/pr
    for j in range(1,len(I)-1):
        if consecutive[j-1] and consecutive[j]: kink[I[j]]=np.linalg.norm(Yn[j+1]-2*Yn[j]+Yn[j-1])
out["path_transitions_mean"]=float(np.mean(trans_per)); out["path_transitions"]=trans_per
vp,ep=vbar[pa],ebar[pa]
# path fixed effects: demean within path
def within(a):
    a=a.astype(float).copy()
    for k in np.unique(pid):
        s=pid==k; a[s]-=np.nanmean(a[s])
    return a
lv,le=np.log10(vp+1e-4),np.log10(ep)
ok=np.isfinite(dist)&np.isfinite(kink)&np.isfinite(lv)
loadp=load[pa]; nap=nact[pa].astype(float)
def fe(y):
    X=np.column_stack([within(np.where(np.isfinite(kink),kink,np.nanmean(kink))),within(loadp),within(nap)])[ok]
    Y=within(y)[ok]; r=sm.OLS(Y,X/X.std(0)).fit(cov_type="cluster",cov_kwds={"groups":pid[ok]})
    return {nm:dict(b=float(r.params[i]),se=float(r.bse[i]),p=float(r.pvalues[i])) for i,nm in enumerate(["kink","load","n_active"])}|{"n":int(ok.sum())}
out["fe_viol"]=fe(lv); out["fe_err"]=fe(le)
out["dist_bins"]={}
wv,we=within(lv),within(le)
for lo,hi,lb in [(0,0.5,"adjacent"),(0.5,1.5,"1 step"),(1.5,3.5,"2-3 steps"),(3.5,99,"4+ steps")]:
    s=(dist>=lo)&(dist<hi)&np.isfinite(wv)
    if s.sum(): out["dist_bins"][lb]=dict(n=int(s.sum()),within_log_viol=float(np.mean(wv[s])),within_log_err=float(np.mean(we[s])),raw_viol=float(np.nanmean(vp[s])),raw_err=float(np.mean(ep[s])))
# D. zero-slack comparison over test set: all MLP256 seeds averaged
Vbar=np.nanmean(vc[idx][:,te],0); At=Aall[te]
bins=[("rare (<10%)",0,.10),("mid (10-90%)",.10,.90),("common (>90%)",.90,1.01)]
elig=free
out["zero_slack"]={}
for lb,lo,hi in bins:
    cm=elig&(f>=lo)&(f<hi)
    mask=(At==1)&cm[None,:]
    vals=Vbar[mask]
    if vals.size:
        ci=np.where(mask)
        out["zero_slack"][lb]=dict(n_events=int(vals.size),n_constraints=int(np.unique(ci[1]).size),mean_v=float(np.nanmean(vals)),p95=float(np.nanpercentile(vals,95)),viol_rate=float(np.nanmean(vals>1e-3)))
# by type, same rare vs common
out["zero_slack_type"]={}
for t in np.unique(typ):
    for lb,lo,hi in bins:
        cm=elig&(typ==t)&(f>=lo)&(f<hi); mask=(At==1)&cm[None,:]
        if mask.sum()>=20: out["zero_slack_type"][f"{t} {lb}"]=dict(n=int(mask.sum()),c=int(np.unique(np.where(mask)[1]).size),mean_v=float(np.nanmean(Vbar[mask])))
# cluster-robust bootstrap over constraints for RAG
rng=np.random.default_rng(0); ce={}
for lb,lo,hi in [("rare",0,.10),("common",.90,1.01)]:
    cm=np.where(elig&(f>=lo)&(f<hi))[0]; ce[lb]=[(i,Vbar[At[:,i]==1,i]) for i in cm if (At[:,i]==1).any()]
if ce["rare"] and ce["common"]:
    boots=[]
    for b in range(2000):
        pick=lambda L:[L[j][1] for j in rng.integers(0,len(L),len(L))]
        boots.append(np.nanmean(np.concatenate(pick(ce["rare"])))-np.nanmean(np.concatenate(pick(ce["common"]))))
    out["RAG"]=dict(point=float(out["zero_slack"]["rare (<10%)"]["mean_v"]-out["zero_slack"]["common (>90%)"]["mean_v"]),ci=[float(np.nanpercentile(boots,2.5)),float(np.nanpercentile(boots,97.5))],n_rare=len(ce["rare"]),n_common=len(ce["common"]))
# never-active constraints in training that activated at test: novel binding
nev=(f==0)&elig; mask=(At==1)&nev[None,:]
out["novel_binding"]=dict(events=int(mask.sum()),constraints=int(np.unique(np.where(mask)[1]).size),mean_v=float(np.nanmean(Vbar[mask])) if mask.sum() else None)
# which constraint types drive max violation
out["argmax_type"]={}
W=np.where(np.isnan(Vbar),-1,Vbar).copy(); W[:,~free]=-1
am=W.argmax(1); u,c=np.unique(typ[am],return_counts=True); out["argmax_type"]={a:int(b) for a,b in zip(u,c)}
json.dump(out,open(f"results_{name}.json","w"),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
