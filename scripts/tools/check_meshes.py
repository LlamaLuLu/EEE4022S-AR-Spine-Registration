"""Check decimated vertebra meshes against full-res ones.

Usage (from repo root):
  python scripts/tools/check_meshes.py \
    --overlay phantom/meshes/v3_overlay \
    --full <folder with full-res phantom_l1..5.stl> \
    --entry annotations/pedicle_entry_targets.json --scale 1000

NOTE: expects files named phantom_l1..5.stl in BOTH folders. Your full-res files are
named full_phantom_l*.stl, so the script is adjusted below via --full-prefix.
"""
import argparse, json, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

def read_stl(p):
    b = Path(p).read_bytes()
    if b[:5] == b"solid" and b"facet" in b[:400]:
        v = np.array([[float(x) for x in l.split()[1:4]] for l in b.decode().splitlines() if l.strip().startswith("vertex")])
        return v.reshape(-1, 3, 3)
    n = np.frombuffer(b, "<u4", 1, 80)[0]
    d = np.frombuffer(b, np.dtype([("n","<f4",3),("v","<f4",(3,3)),("a","<u2")]), n, 84)
    return d["v"].astype(float)

def ptd(p, T):
    a, b, c = T[:,0], T[:,1], T[:,2]
    ab, ac, ap = b-a, c-a, p-a
    d1=(ab*ap).sum(1); d2=(ac*ap).sum(1)
    bp=p-b; d3=(ab*bp).sum(1); d4=(ac*bp).sum(1)
    cp=p-c; d5=(ab*cp).sum(1); d6=(ac*cp).sum(1)
    vc=d1*d4-d3*d2; vb=d5*d2-d1*d6; va=d3*d6-d5*d4
    den=va+vb+vc; den=np.where(np.abs(den)<1e-18,1e-18,den)
    v=vb/den; w=vc/den
    q=a+ab*v[:,None]+ac*w[:,None]
    # clamp via barycentric projection fallback: sample-free robust approach
    u=1-v-w
    inside=(u>=0)&(v>=0)&(w>=0)
    best=np.linalg.norm(p-q,axis=1)
    best=np.where(inside,best,np.inf)
    def seg(x,y):
        e=y-x; t=np.clip(((p-x)*e).sum(1)/np.maximum((e*e).sum(1),1e-18),0,1)
        return np.linalg.norm(p-(x+e*t[:,None]),axis=1)
    return np.minimum(best,np.minimum(seg(a,b),np.minimum(seg(b,c),seg(c,a))))

def surf_samples(T, n=300000, rng=np.random.default_rng(0)):
    A = 0.5*np.linalg.norm(np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]),axis=1)
    idx = rng.choice(len(T), n, p=A/A.sum())
    r1, r2 = rng.random(n), rng.random(n); s=np.sqrt(r1)
    t=T[idx]
    return np.vstack([T.reshape(-1,3),(1-s)[:,None]*t[:,0]+(s*(1-r2))[:,None]*t[:,1]+(s*r2)[:,None]*t[:,2]])

ap=argparse.ArgumentParser()
ap.add_argument("--overlay");ap.add_argument("--full");ap.add_argument("--entry")
ap.add_argument("--full-prefix",default="phantom_l")
ap.add_argument("--scale",type=float,default=1.0)
ap.add_argument("--trans",type=float,nargs=3,default=[-9.1,177.7,583.2])
a=ap.parse_args()
ov={i:read_stl(f"{a.overlay}/phantom_l{i}.stl")*a.scale for i in range(1,6)}
fu={i:read_stl(f"{a.full}/{a.full_prefix}{i}.stl")*a.scale for i in range(1,6)}
for i in ov: print(f"L{i}: overlay {len(ov[i]):>6} tris, full {len(fu[i]):>6} tris ({len(ov[i])/len(fu[i]):.3f}); bounds {ov[i].reshape(-1,3).min(0).round(1)} {ov[i].reshape(-1,3).max(0).round(1)}")
d=json.load(open(a.entry))
pts={c["label"]:np.array(c["position"])+a.trans for c in d["markups"][0]["controlPoints"]}
print("\nEntry point distance to own-level surface (mm); bar <= 1.0")
print(f"{'label':8}{'full':>8}{'overlay':>9}{'delta':>8}{'nearest overlay level':>24}")
dv=[];wrong=0
for lab,p in pts.items():
    L=int(lab[1]); 
    df=ptd(p,fu[L]).min(); do=ptd(p,ov[L]).min()
    allo={i:ptd(p,ov[i]).min() for i in ov}; near=min(allo,key=allo.get)
    wrong+= near!=L; dv.append(do)
    print(f"{lab:8}{df:8.2f}{do:9.2f}{do-df:8.2f}{'L%d'%near:>24}{'  <-- WRONG' if near!=L else ''}{'  FAIL' if do>1 else ''}")
print(f"max {max(dv):.2f} mean {np.mean(dv):.2f}; wrong-level {wrong}")
print("\nDeviation of overlay surface from full surface (mm); bar p95 <= 0.5")
for i in ov:
    kf=cKDTree(surf_samples(fu[i])); ko=cKDTree(surf_samples(ov[i]))
    d1=kf.query(surf_samples(ov[i],100000))[0]; d2=ko.query(surf_samples(fu[i],100000))[0]
    print(f"L{i}: ov->full mean {d1.mean():.3f} p95 {np.percentile(d1,95):.3f} max {d1.max():.3f} | full->ov p95 {np.percentile(d2,95):.3f} max {d2.max():.3f}")