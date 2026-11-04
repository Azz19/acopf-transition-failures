"""Generate solved AC-OPF samples. Random: global b~U(.85,1.15) x local U(.9,1.1). Paths: straight-line load interpolation."""
import os, sys, warnings, pickle, time
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
warnings.filterwarnings("ignore")
import numpy as np, multiprocessing as mp
from common import *
name, n_rand, n_paths, n_pts, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
case = load_case(os.path.join(os.environ.get("PGLIB_DIR","pglib-opf"),f"pglib_opf_case{name}.m"))
pd0, qd0 = case["bus"][:,PD].copy(), case["bus"][:,QD].copy()
def draw(rng):
    b = rng.uniform(.85,1.15)
    return pd0*b*rng.uniform(.9,1.1,pd0.size), qd0*b*rng.uniform(.9,1.1,pd0.size)
def job(a):
    kind, pid, t, pd, qd = a
    try:
        r = solve(case, pd, qd)
    except Exception:
        return None
    if not r["success"]: return dict(kind=kind,pid=pid,t=t,ok=False)
    act, mu = active_vector(r)
    return dict(kind=kind,pid=pid,t=t,ok=True,pd=pd,qd=qd,pg=r["gen"][:,PG],qg=r["gen"][:,QG],vg=r["gen"][:,VG],
                vm=r["bus"][:,VM],f=r["f"],act=act,mu=mu,slk=slacks(r,case["baseMVA"]),lam=r["bus"][:,LAM_P])
if __name__=="__main__":
    rng=np.random.default_rng(seed); jobs=[]
    for i in range(n_rand):
        p,q=draw(rng); jobs.append(("rand",-1,i,p,q))
    for k in range(n_paths):
        pa,qa=draw(rng); pb,qb=draw(rng)
        for j,t in enumerate(np.linspace(0,1,n_pts)):
            jobs.append(("path",k,j,pa+t*(pb-pa),qa+t*(qb-qa)))
    t0=time.time(); res=[]
    with mp.Pool(2) as pool:
        for i,x in enumerate(pool.imap(job,jobs,chunksize=8)):
            res.append(x)
            if i%200==0:
                print(i,len(jobs),round(time.time()-t0),flush=True)
                pickle.dump(res,open(f"data_{name}.tmp","wb"))
    pickle.dump(res,open(f"data_{name}.pkl","wb"))
    print("done",len(res),sum(1 for r in res if r and r["ok"]),round(time.time()-t0),flush=True)
