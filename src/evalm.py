"""Train surrogates, recover via AC PF, save per-sample and per-constraint results."""
import os, sys, warnings, pickle, time
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
warnings.filterwarnings("ignore")
import numpy as np, multiprocessing as mp
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from common import *
name=sys.argv[1]; ntrain=int(sys.argv[2])
case=load_case(os.path.join(os.environ.get("PGLIB_DIR","pglib-opf"),f"pglib_opf_case{name}.m"))
res=[r for r in pickle.load(open(f"data_{name}.pkl","rb")) if r and r["ok"]]
rand=[r for r in res if r["kind"]=="rand"]; path=[r for r in res if r["kind"]=="path"]
train,test=rand[:ntrain],rand[ntrain:]
g=case["gen"]; ldb=np.where((case["bus"][:,PD]!=0)|(case["bus"][:,QD]!=0))[0]
X=lambda S: np.array([np.r_[r["pd"][ldb],r["qd"][ldb]] for r in S])
Y=lambda S: np.array([np.r_[r["pg"],r["vg"]] for r in S])
ng=g.shape[0]; lb=np.r_[g[:,PMIN],case["bus"][(g[:,0]-1).astype(int),VMIN]]; ub=np.r_[g[:,PMAX],case["bus"][(g[:,0]-1).astype(int),VMAX]]
gc=case["gencost"]
def cost(pg): return float(np.sum(gc[:,4]*pg**2+gc[:,5]*pg+gc[:,6]))
def recover(a):
    pd,qd,yh=a
    c={k:(v.copy() if hasattr(v,"copy") else v) for k,v in case.items()}
    c["bus"][:,PD]=pd; c["bus"][:,QD]=qd
    c["gen"][:,PG]=yh[:ng]; c["gen"][:,VG]=yh[ng:]
    c["bus"][(g[:,0]-1).astype(int),VM]=yh[ng:]
    try:
        r,ok=runpf(c,PF_OPT)
    except Exception:
        return None
    if not r["success"]: return None
    s=slacks(r,c["baseMVA"]); return (np.minimum(s,0).astype(np.float32),cost(r["gen"][:,PG]))
if __name__=="__main__":
    Xs=StandardScaler().fit(X(train)); Ys=StandardScaler().fit(Y(train))
    Xtr,Ytr=Xs.transform(X(train)),Ys.transform(Y(train))
    models={"ridge":Ridge(alpha=1.0)}
    for sd in range(3):
        models[f"mlp64_s{sd}"]=MLPRegressor(hidden_layer_sizes=(64,),max_iter=600,early_stopping=True,random_state=sd,n_iter_no_change=30)
        models[f"mlp256_s{sd}"]=MLPRegressor(hidden_layer_sizes=(256,256),max_iter=600,early_stopping=True,random_state=sd,n_iter_no_change=30)
    out={}
    allS=test+path
    Xa=Xs.transform(X(allS)); Ya=Y(allS)
    pool=mp.Pool(2)
    for mn,m in models.items():
        t0=time.time(); m.fit(Xtr,Ytr)
        yh=np.clip(Ys.inverse_transform(m.predict(Xa)),lb,ub)
        rec=pool.map(recover,[(r["pd"],r["qd"],y) for r,y in zip(allS,yh)],chunksize=16)
        out[mn]=dict(yh=yh.astype(np.float32),rec=rec)
        print(mn,round(time.time()-t0),sum(x is None for x in rec),flush=True)
        pickle.dump(out,open(f"eval_{name}.pkl","wb"))
    print("done",flush=True)
