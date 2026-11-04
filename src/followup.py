"""Exploratory follow-up: units, model comparison on Vg, within-constraint event study at zero slack."""
import os
import pickle, json, warnings, sys
warnings.filterwarnings("ignore")
import numpy as np, statsmodels.api as sm
from scipy.stats import spearmanr
from common import *
name="118_ieee"; ntrain=1800
case=load_case(os.path.join(os.environ.get("PGLIB_DIR","pglib-opf"),f"pglib_opf_case{name}.m"))
res=[r for r in pickle.load(open(f"data_{name}.pkl","rb")) if r and r["ok"]]
rand=[r for r in res if r["kind"]=="rand"]; path=[r for r in res if r["kind"]=="path"]
train,test=rand[:ntrain],rand[ntrain:]; allS=test+path; n=len(test)
ev=pickle.load(open(f"eval_{name}.pkl","rb")); mn=list(ev)
g=case["gen"];nb,ng,nl=case["bus"].shape[0],g.shape[0],case["branch"].shape[0]; gb=(g[:,0]-1).astype(int)
typ=np.array(["Pmax"]*ng+["Pmin"]*ng+["Qmax"]*ng+["Qmin"]*ng+["Vmax"]*nb+["Vmin"]*nb+["Sf"]*nl+["St"]*nl)
pr=np.maximum(g[:,PMAX]-g[:,PMIN],1.0);qr=np.maximum(g[:,QMAX]-g[:,QMIN],1.0)
rate=case["branch"][:,RATE_A].copy()
vr=np.maximum(case["bus"][:,VMAX]-case["bus"][:,VMIN],1e-3)
phys=np.r_[pr,pr,qr,qr,vr,vr,rate,rate]   # range -> MW, MVAr, pu V, MVA
slack_g=np.where(case["bus"][gb,1]==3)[0]
clamped=np.zeros(typ.size,bool);clamped[:2*ng]=True;clamped[slack_g]=False;clamped[ng+slack_g]=False
clamped[4*ng+gb]=True;clamped[4*ng+nb+gb]=True
A=lambda S:np.array([(np.abs(r["mu"])>1e-4).astype(np.int8) for r in S])
Atr=A(train);Aall=A(allS);f=Atr.mean(0)
Yt=np.array([np.r_[r["pg"],r["vg"]] for r in allS])
V={}
for m in mn:
    a=np.full((len(allS),typ.size),np.nan)
    for i,x in enumerate(ev[m]["rec"]):
        if x is not None:a[i]=-x[0]
    V[m]=a
out={}
out["qrange_quantiles_mvar"]=[float(x) for x in np.percentile(qr,[0,10,25,50,75,90,100])]
out["models"]={}
qidx=np.where(np.isin(typ,["Qmax","Qmin"]))[0]
for m in mn:
    y=ev[m]["yh"];v=V[m][:n]
    qm=(v[:,qidx]*phys[qidx])
    sq=np.nansum(qm,axis=1)
    thermal=np.where(np.isin(typ,["Sf","St"]))[0]
    tv=np.nanmax(v[:,thermal],axis=1)
    pq=np.where(np.isin(typ,["Vmax","Vmin"])&~clamped)[0]
    out["models"][m]=dict(mae_pg_pct=float(100*(np.abs(y[:n,:ng]-Yt[:n,:ng])/pr).mean()),
      mae_vg_pu=float(np.abs(y[:n,ng:]-Yt[:n,ng:]).mean()),
      total_q_viol_mvar=float(np.nanmean(sq)),
      max_q_viol_mvar=float(np.nanmean(np.nanmax(qm,axis=1))),
      thermal_viol_rate=float(np.mean(tv>1e-3)),thermal_viol_mean=float(np.nanmean(tv)),
      pq_v_viol_rate=float(np.mean(np.nanmax(v[:,pq],axis=1)>1e-3)),
      slack_p_viol_mw=float(np.nanmean((v[:,slack_g].sum(1)+v[:,ng+slack_g].sum(1))*pr[slack_g].mean())),
      cost_gap_pct=None)
ms=[m for m in mn]
out["rank"]=dict(spearman_maepg_qviol=float(spearmanr([out["models"][m]["mae_pg_pct"] for m in ms],[out["models"][m]["total_q_viol_mvar"] for m in ms])[0]),
                 spearman_maevg_qviol=float(spearmanr([out["models"][m]["mae_vg_pu"] for m in ms],[out["models"][m]["total_q_viol_mvar"] for m in ms])[0]))
# Constraint level: Q violation in MVAr vs switching frequency, at zero slack on test set
mlp=[m for m in mn if m.startswith("mlp256")]
def rows(mods):
    Vb=np.nanmean([V[m][:n] for m in mods],0); At=Aall[:n]; L=[]
    for i in qidx:
        if clamped[i] or f[i]==0: continue
        s=At[:,i]==1
        if s.sum()<10: continue
        L.append((i,f[i],Vb[s,i].mean()*phys[i],qr[(i%ng)],Vb[s,i].mean()))
    return np.array(L)
for lab,mods in [("mlp256",mlp),("ridge",["ridge"])]:
    R=rows(mods); ent=R[:,1]*(1-R[:,1])
    r1=spearmanr(ent,R[:,2]); r2=spearmanr(ent,R[:,3]); r3=spearmanr(R[:,3],R[:,2])
    # partial: regress log Mvar on log range and entropy
    X=sm.add_constant(np.column_stack([ent/ent.std(),np.log10(R[:,3])]));reg=sm.OLS(np.log10(R[:,2]+1e-3),X).fit(cov_type="HC3")
    bin_=lambda lo,hi:R[(R[:,1]>=lo)&(R[:,1]<hi),2]
    out[f"q_constraints_{lab}"]=dict(n=len(R),rho_entropy_mvar=[float(r1[0]),float(r1[1])],rho_entropy_qrange=[float(r2[0]),float(r2[1])],rho_qrange_mvar=[float(r3[0]),float(r3[1])],
      partial_entropy=dict(b=float(reg.params[1]),se=float(reg.bse[1]),p=float(reg.pvalues[1])),partial_logrange=dict(b=float(reg.params[2]),p=float(reg.pvalues[2])),
      mean_mvar_rare=float(bin_(0,.1).mean()) if len(bin_(0,.1)) else None,mean_mvar_mid=float(bin_(.1,.9).mean()),mean_mvar_common=float(bin_(.9,1.01).mean()),n_rare=len(bin_(0,.1)),n_mid=len(bin_(.1,.9)),n_common=len(bin_(.9,1.01)))
# Within-constraint event study on paths. Signed event time: >0 active side, <0 inactive.
pid=np.array([r["pid"] for r in path]);tt=np.array([r["t"] for r in path]);Ap=Aall[n:]
sw=(f>0)&(f<1)&~clamped
def event(mods,target="viol"):
    Vb=np.nanmean([V[m][n:] for m in mods],0); rows=[]
    for k in np.unique(pid):
        I=np.where(pid==k)[0];I=I[np.argsort(tt[I])]
        if len(I)!=41: continue
        for c in np.where(sw)[0]:
            a=Ap[I,c]
            ch=np.where(np.diff(a)!=0)[0]
            if len(ch)!=1: continue  # single clean crossing only
            b=ch[0]
            for j in range(41):
                d=(j-b-1) if a[-1]==1 else (b-j)   # 0.. on active side, negative on inactive
                rows.append((k,c,d,Vb[I[j],c],a[j]))
    return np.array(rows)
out["event"]={}
for lab,mods in [("mlp256",mlp),("mlp64",[m for m in mn if m.startswith("mlp64")]),("ridge",["ridge"])]:
    E=event(mods); act=E[E[:,4]==1]
    # within path-constraint demean on active side only
    keys=act[:,0]*10000+act[:,1]; y=np.log10(act[:,3]+1e-4)
    w=y.copy()
    for u in np.unique(keys): w[keys==u]-=w[keys==u].mean()
    d=act[:,2]
    near=d<=2; 
    bins={}
    for lo,hi,l in [(0,2,"0-2 steps inside"),(3,6,"3-6"),(7,12,"7-12"),(13,99,"13+")]:
        s=(d>=lo)&(d<=hi)
        if s.sum(): bins[l]=dict(n=int(s.sum()),within_log=float(w[s].mean()),raw_mean=float(act[:,3][s].mean()))
    # cluster by path-constraint episode
    X=(d<=2).astype(float);X=X-X.mean()
    r=sm.OLS(w,X).fit(cov_type="cluster",cov_kwds={"groups":keys})
    # also continuous decay
    X2=np.log1p(d); 
    for u in np.unique(keys): X2[keys==u]-=X2[keys==u].mean()
    r2=sm.OLS(w,X2).fit(cov_type="cluster",cov_kwds={"groups":keys})
    # inactive side profile: slack-driven
    ina=E[E[:,4]==0]
    out["event"][lab]=dict(episodes=int(np.unique(keys).size),constraints=int(np.unique(act[:,1]).size),types={t:int(np.sum(typ[np.unique(act[:,1]).astype(int)]==t)) for t in np.unique(typ[np.unique(act[:,1]).astype(int)])},
      bins=bins,near_effect=dict(b=float(r.params[0]),se=float(r.bse[0]),p=float(r.pvalues[0]),ratio=float(10**r.params[0])),
      log_dist_slope=dict(b=float(r2.params[0]),se=float(r2.bse[0]),p=float(r2.pvalues[0])),
      inactive_side={l:float(ina[(ina[:,2]<=-lo)&(ina[:,2]>=-hi),3].mean()) for lo,hi,l in [(1,2,"1-2"),(3,6,"3-6"),(7,99,"7+")]})
# Vg error around Q/V transitions on paths, by generator with own Q/V switching
Yp=Yt[n:]; 
for lab,mods in [("mlp256",mlp),("ridge",["ridge"])]:
    Eg=np.mean([np.abs(ev[m]["yh"][n:,ng:]-Yp[:,ng:]) for m in mods],0)
    L=[]
    for k in np.unique(pid):
        I=np.where(pid==k)[0];I=I[np.argsort(tt[I])]
        if len(I)!=41: continue
        for gi in range(ng):
            c=2*ng+gi
            if not sw[c]: continue
            a=Ap[I,c];ch=np.where(np.diff(a)!=0)[0]
            if len(ch)!=1: continue
            b=ch[0]
            for j in range(41):
                L.append((k*1000+gi,min(abs(j-b),abs(j-b-1)),Eg[I[j],gi]))
    L=np.array(L);y=np.log10(L[:,2]+1e-6);w=y.copy();X=np.log1p(L[:,1]);
    for u in np.unique(L[:,0]):
        s=L[:,0]==u;w[s]-=w[s].mean();X[s]-=X[s].mean()
    r=sm.OLS(w,X).fit(cov_type="cluster",cov_kwds={"groups":L[:,0]})
    out[f"vg_err_vs_qmax_transition_{lab}"]=dict(episodes=int(np.unique(L[:,0]).size),slope_log_dist=float(r.params[0]),se=float(r.bse[0]),p=float(r.pvalues[0]),
        near=float(L[L[:,1]<=1,2].mean()),far=float(L[L[:,1]>=8,2].mean()))
out["vg_kink"]=None
json.dump(out,open("followup_118.json","w"),indent=1)
print(json.dumps(out))
