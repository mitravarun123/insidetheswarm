"""Episode-wise causal sender effects; graph controls and attention are correlates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from environments.task import sample
from models.networks import SwarmNet

@torch.no_grad()
def run(root='.'):
    root=Path(root); cfg=json.loads((root/'results/manifest.json').read_text())['config']; rows=[]; summaries=[]
    for seed in range(cfg['seeds']):
        b=sample(cfg['test'],seed=390000+seed)
        a=b['a']; centrality=torch.linalg.eigh(a).eigenvectors[:,:,-1].abs()
        for kind in ('gnn','gat'):
            model=SwarmNet(kind,cfg['hidden']); model.load_state_dict(torch.load(root/'checkpoints'/f'{kind}_{seed}.pt',weights_only=False,map_location='cpu')['model']); model.eval()
            z,_,attention=model(b['x'],a,4,cache=True); p=z.sigmoid().mean(-1)
            effects=[]
            for node in range(10):
                q=model(b['x'],a,4,{'nodes':[node]}).sigmoid().mean(-1); effects.append((q-p).abs())
            effects=torch.stack(effects,-1)
            for ep in range(len(p)):
                for node in range(10):
                    rows.append(dict(seed=seed,model=kind,scenario=ep,node=node,effect=float(effects[ep,node]),degree=float(a[ep,:,node].sum()),centrality=float(centrality[ep,node]),reliability=float(b['x'][ep,node,1]),attention=float(attention[-1][ep,:,node].sum())))
            # Hold out entire episodes; evaluate incremental predictive value of attention
            # after degree, centrality and reliability. Not a causal test of attention.
            X=torch.stack([a.sum(1),centrality,b['x'][...,1],attention[-1].sum(1)],-1).numpy(); Y=effects.numpy(); cut=len(Y)//2
            for label,cols in [('graph_reliability',3),('plus_attention',4)]:
                reg=LinearRegression().fit(X[:cut,:,:cols].reshape(-1,cols),Y[:cut].reshape(-1))
                residual=Y[cut:]-reg.predict(X[cut:,:,:cols].reshape(-1,cols)).reshape(Y[cut:].shape)
                summaries.append(dict(seed=seed,model=kind,control=label,heldout_r2=reg.score(X[cut:,:,:cols].reshape(-1,cols),Y[cut:].reshape(-1)),mean_episode_residual_sd=float(residual.std(-1).mean())))
    pd.DataFrame(rows).to_csv(root/'results/episode_agent_effects.csv',index=False)
    pd.DataFrame(summaries).to_csv(root/'results/agent_control_regression.csv',index=False)
if __name__=='__main__': run()
