import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from environments.task import oracle, sample
from experiments.metrics import metrics, per_scenario

@torch.no_grad()
def mechanisms(model,b,seed,rounds=4):
    clean,states,attention=model(b['x'],b['a'],rounds,cache=True)
    base=metrics(clean,b['y']); n=b['x'].shape[1]; h=model.hidden
    influence=[]
    for node in range(n):
        for intervention in ('remove','corrupt','replace'):
            if intervention=='remove': z=model(b['x'],b['a'],rounds,{'nodes':[node]})
            else:
                z=model(b['x'],b['a'],rounds,{'replace_nodes':[node],'replacement':'batch_shuffle' if intervention=='replace' else 'sign_flip'})
            p=clean.sigmoid().mean(-1); q=z.sigmoid().mean(-1)
            influence.append(dict(seed=seed,node=node,intervention=intervention,effect=float((q-p).abs().mean()),accuracy_drop=base['accuracy']-metrics(z,b['y'])['accuracy'],degree=float(b['a'][:,:,node].sum(-1).mean()),reliability=float(b['x'][:,node,1].mean()),attention=float(attention[-1][:,:,node].sum(-1).mean()) if attention else 0))
    # Rank on an independent discovery sample; evaluation sample never selects targets.
    discovery=sample(128,seed=400000+seed,device=b['x'].device)
    dz,ds,_=model(discovery['x'],discovery['a'],rounds,cache=True)
    ranking=[]
    for c in range(h):
        cz=model(discovery['x'],discovery['a'],rounds,{'channels':[c]})
        ranking.append(float((cz.sigmoid()-dz.sigmoid()).abs().mean()))
    k=max(1,h//8); top=np.argsort(ranking)[-k:].tolist()
    rng=np.random.default_rng(seed+100)
    random=rng.choice(h,k,replace=False).tolist()
    magnitude=model.message(ds[-2]).abs().mean((0,1)).cpu().numpy()
    # Greedy nearest-magnitude matching excluding targeted channels.
    pool=set(range(h))-set(top); matched=[]
    for c in top:
        m=min(pool,key=lambda j:abs(magnitude[j]-magnitude[c])); matched.append(m); pool.remove(m)
    # Independent discovery ranking for node and edge interventions.
    node_scores=[]
    for node in range(n):
        q=model(discovery['x'],discovery['a'],rounds,{'nodes':[node]})
        node_scores.append(float((q.sigmoid()-dz.sigmoid()).abs().mean()))
    node_order=np.argsort(node_scores)[::-1].tolist()
    ablations=[]
    for label,channels in [('targeted',top),('random',random),('magnitude_matched',matched)]:
        for kind in ('channels','hidden_channels'):
            z=model(b['x'],b['a'],rounds,{kind:channels})
            ablations.append(dict(seed=seed,control=label,kind=kind,**metrics(z,b['y']),accuracy_drop=base['accuracy']-metrics(z,b['y'])['accuracy']))
    # Layer-specific effects distinguish recurrence steps sharing weights.
    layers=[]
    for t in range(rounds):
        for label,channels in [('targeted',top),('random',random),('magnitude_matched',matched)]:
            z=model(b['x'],b['a'],rounds,{'channels':channels,'only_round':t})
            layers.append(dict(seed=seed,round=t,control=label,effect=float((z.sigmoid()-clean.sigmoid()).abs().mean()),**metrics(z,b['y'])))
    circuits=[dict(seed=seed,channel=c,effect=ranking[c],magnitude=float(magnitude[c])) for c in range(h)]
    # Edge ranking and matched random ablation on a fixed ring graph (paired signals).
    a=b['a'].clone(); a.zero_()
    for i in range(n): a[:,i,(i+1)%n]=a[:,(i+1)%n,i]=1
    ring=model(b['x'],a,rounds); edge_rows=[]
    da=a[:1].expand(len(discovery['y']),-1,-1); dr=model(discovery['x'],da,rounds)
    edge_scores=[]
    for i in range(n):
        q=model(discovery['x'],da,rounds,{'edges':[(i,(i+1)%n)]})
        edge_scores.append(float((q.sigmoid()-dr.sigmoid()).abs().mean()))
    for label,indices in [('targeted',np.argsort(edge_scores)[-2:]),('random',rng.choice(n,2,replace=False))]:
        q=model(b['x'],a,rounds,{'edges':[(int(i),int((i+1)%n)) for i in indices]})
        ablations.append(dict(seed=seed,control=label,kind='edges',**metrics(q,b['y']),accuracy_drop=metrics(ring,b['y'])['accuracy']-metrics(q,b['y'])['accuracy']))
    for i in range(n):
        z=model(b['x'],a,rounds,{'edges':[(i,(i+1)%n)]})
        edge_rows.append(dict(seed=seed,receiver=i,sender=(i+1)%n,effect=float((z.sigmoid()-ring.sigmoid()).abs().mean())))
    # Clean/corrupt pairs differ at a single sensor only.
    x=b['x'].clone(); x[:,0,0]*=-1
    bad,bs,_=model(x,b['a'],rounds,cache=True)
    patch=[]
    clean_margin=(2*b['y']-1)*clean.mean(-1); bad_margin=(2*b['y']-1)*bad.mean(-1)
    for t in range(rounds):
        for label,value,nodes in [('clean_target',states[t],[0]),('corrupt_identity',bs[t],[0]),('shuffled_clean',states[t].roll(1,0),[0]),('random_node',states[t],[1])]:
            z=model(x,b['a'],rounds,{'patch_round':t,'patch_nodes':nodes,'patch_value':value})
            margin=(2*b['y']-1)*z.mean(-1)
            damaged=(clean.sigmoid().mean(-1)>=.5)==b['y'].bool()
            damaged &= ((bad.sigmoid().mean(-1)>=.5)!=b['y'].bool())
            restored=((z.sigmoid().mean(-1)>=.5)==b['y'].bool())
            patch.append(dict(seed=seed,round=t,control=label,logit_recovery=float((margin-bad_margin).mean()),available_logit_gap=float((clean_margin-bad_margin).mean()),decision_recovery=float(restored[damaged].float().mean()) if damaged.any() else np.nan,damaged_count=int(damaged.sum()),accuracy=metrics(z,b['y'])['accuracy']))
    from interpretability.probes import probe_states
    probes=probe_states(states,b,clean,seed)
    return dict(node_ranking=[dict(seed=seed,node=node,discovery_effect=node_scores[node],rank=rank) for rank,node in enumerate(node_order)],layers=layers,influence=influence,ablations=ablations,circuits=circuits,edges=edge_rows,patching=patch,probes=probes),states
