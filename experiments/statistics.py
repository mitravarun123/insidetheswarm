"""Seed-level uncertainty; no agent-level pseudo-replication."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import LinearRegression

def summarize(root):
    root=Path(root); out=root/'results'; d=pd.read_csv(out/'evaluation.csv')
    keys=['model','n','topology','reliability','rounds']; rows=[]
    for key,g in d.groupby(keys):
        v=g.accuracy.to_numpy(); count=len(v); sd=v.std(ddof=1) if count>1 else np.nan
        half=t.ppf(.975,count-1)*sd/np.sqrt(count) if count>1 else np.nan
        rows.append(dict(zip(keys,key),mean=v.mean(),std=sd,n_seeds=count,ci_low=v.mean()-half,ci_high=v.mean()+half))
    pd.DataFrame(rows).to_csv(out/'accuracy_summary.csv',index=False)
    metric_rows=[]
    for key,g in d.groupby(keys):
        for metric in ('accuracy','individual_accuracy','consensus','brier','ece','message_cost'):
            v=g[metric].to_numpy(); count=len(v); sd=v.std(ddof=1) if count>1 else np.nan
            half=t.ppf(.975,count-1)*sd/np.sqrt(count) if count>1 else np.nan
            metric_rows.append(dict(zip(keys,key),metric=metric,mean=v.mean(),std=sd,n_seeds=count,ci_low=v.mean()-half,ci_high=v.mean()+half))
    pd.DataFrame(metric_rows).to_csv(out/'metric_summary.csv',index=False)
    p=pd.read_csv(out/'paired_scenarios.csv').pivot(index=['seed','scenario'],columns='model',values='correct')
    contrasts=[]
    for reference in ('mlp','average','shuffled','randomized'):
        delta=(p.gnn-p[reference]).groupby('seed').mean(); count=len(delta); sd=delta.std(); half=t.ppf(.975,count-1)*sd/np.sqrt(count) if count>1 else np.nan
        contrasts.append(dict(comparison='gnn - '+reference,mean=delta.mean(),std=sd,ci_low=delta.mean()-half,ci_high=delta.mean()+half,n_seeds=count))
    pd.DataFrame(contrasts).to_csv(out/'paired_contrasts.csv',index=False)
    # Descriptive controlled residual variance, separately within each seed.
    # Centrality: exact degree and eigenvector score of each episode's graph
    # are averaged by node index, rather than treated as independent replicates.
    from environments.task import sample
    import torch
    residual=[]
    inf=pd.read_csv(out/'influence.csv').query("intervention=='remove'")
    for seed,g in inf.groupby('seed'):
        import json
        count=json.loads((out/'manifest.json').read_text())['config']['test']
        b=sample(count,seed=350000+int(seed)); _,vectors=torch.linalg.eigh(b['a'])
        centrality=vectors[:,:,-1].abs().mean(0).numpy()
        g=g.sort_values('node'); X=np.c_[g.degree,g.reliability,centrality]
        fit=LinearRegression().fit(X,g.effect); res=g.effect-fit.predict(X)
        residual.append(dict(seed=seed,residual_sd=float(res.std()),explained_r2=fit.score(X,g.effect),n_node_positions=len(g)))
    pd.DataFrame(residual).to_csv(out/'influence_controlled.csv',index=False)
