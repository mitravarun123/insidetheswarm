"""Conditionally independent Bernoulli sensors on undirected connected graphs."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

def seed_all(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def graph(n, kind, rng, connectivity=.15):
    a=np.zeros((n,n),np.float32)
    if kind=='star':
        a[0,1:]=1; a[1:,0]=1
    else:
        for i in range(n): a[i,(i+1)%n]=a[(i+1)%n,i]=1
        if kind=='small_world':
            # Watts-Strogatz-style rewiring of a degree-four ring; retain the
            # nearest-neighbor backbone to guarantee connectivity.
            for i in range(n):
                j=(i+2)%n
                candidates=[v for v in range(n) if v!=i and a[i,v]==0]
                if rng.random()<connectivity and candidates: j=int(rng.choice(candidates))
                a[i,j]=a[j,i]=1
        if kind=='random':
            for i in range(n):
                for j in range(i+1,n):
                    if rng.random()<connectivity: a[i,j]=a[j,i]=1
    return a

def sample(size,n=10,topology='random',reliability=.7,heterogeneity=.1,connectivity=.15,seed=0,device='cpu'):
    rng=np.random.default_rng(seed)
    y=rng.integers(0,2,size)
    p=np.clip(reliability+rng.uniform(-heterogeneity,heterogeneity,(size,n)),.501,.99)
    s=np.where(rng.random((size,n))<p,y[:,None],1-y[:,None])
    evidence=(2*s-1)*np.log(p/(1-p))
    # Features contain only observed signal and known sensor reliability, never y.
    x=np.stack([2*s-1,p],-1)
    a=np.stack([graph(n,topology,rng,connectivity) for _ in range(size)])
    return {k:torch.as_tensor(v,dtype=torch.float32,device=device) for k,v in dict(x=x,a=a,y=y,evidence=evidence).items()}

def subset(b,sl): return {k:v[sl] for k,v in b.items()}

def reachable(a,rounds):
    r=torch.eye(a.shape[-1],device=a.device).expand_as(a).clone()
    for _ in range(rounds): r=((r+torch.bmm(a,r))>0).float()
    return r

def oracle(b,rounds=None):
    """Uniform prior: posterior log odds is sum of signal log likelihood ratios."""
    if rounds is None: return b['evidence'].sum(-1)
    return torch.bmm(reachable(b['a'],rounds),b['evidence'].unsqueeze(-1)).squeeze(-1)
