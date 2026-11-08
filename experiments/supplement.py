"""Post-run circuit transfer, graph geometry, attribution and paired patch diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from environments.task import sample,reachable,seed_all
from models.networks import SwarmNet
from experiments.run import infer
from experiments.metrics import metrics

def run(root='.'):
    root=Path(root); manifest=json.loads((root/'results/manifest.json').read_text()); cfg=manifest['config']; torch.set_num_threads(2)
    transfer=[]; geometry=[]; patch=[]; attribution=[]; robust=[]; probes=[]
    for seed in range(cfg['seeds']):
        state=torch.load(root/'checkpoints'/f'gnn_{seed}.pt',map_location='cpu',weights_only=False)
        model=SwarmNet('gnn',cfg['hidden']); model.load_state_dict(state['model']); model.eval()
        circuit=pd.read_csv(root/'results/circuits.csv').query('seed==@seed'); k=max(1,cfg['hidden']//8)
        top=circuit.nlargest(k,'effect').channel.tolist(); rng=np.random.default_rng(seed); rand=rng.choice(cfg['hidden'],k,replace=False).tolist()
        for n in (5,10,20,40):
            for topology in ('ring','random','star','small_world'):
                b=sample(cfg['test'],n,topology,.75,seed=380000+seed)
                baseline=metrics(infer(model,b,4),b['y'])
                for label,ch in [('targeted',top),('random',rand)]:
                    q=infer(model,b,4,{'channels':ch}); m=metrics(q,b['y'])
                    transfer.append(dict(seed=seed,n=n,topology=topology,control=label,accuracy_drop=baseline['accuracy']-m['accuracy'],**m))
                a=b['a'][:min(64,len(b['y']))]; diameter=torch.zeros(len(a)); diameter[:]=float('nan')
                for r in range(n):
                    cover=reachable(a,r).mean((1,2)); complete=cover==1
                    diameter[complete&diameter.isnan()]=r
                    if r in (0,1,2,4,8): geometry.append(dict(seed=seed,n=n,topology=topology,rounds=r,mean_radius_coverage=float(cover.mean()),graphs=len(a)))
                geometry.append(dict(seed=seed,n=n,topology=topology,rounds=-1,mean_diameter=float(diameter.mean()),graphs=len(a)))
        b=sample(cfg['test'],seed=350000+seed)
        from interpretability.probes import probe_states
        cached=torch.load(root/'results'/f'activations_{seed}.pt',weights_only=False)
        with torch.no_grad(): clean_probe=model.readout(cached['states'][-1]).squeeze(-1)
        probes.extend(probe_states(cached['states'],cached['batch'],clean_probe,seed))
        with torch.enable_grad():
            z,states,_=model(b['x'],b['a'],4,cache=True)
            grads=torch.autograd.grad(((2*b['y']-1)[:,None]*z).mean(),states)
            for t,(activation,grad) in enumerate(zip(states,grads)):
                scores=(activation*grad).abs().mean((0,1)).detach().numpy()
                attribution.extend(dict(seed=seed,round=t,channel=c,gradient_times_activation=float(v)) for c,v in enumerate(scores))
        with torch.no_grad():
            clean,cs,_=model(b['x'],b['a'],4,cache=True); x=b['x'].clone(); x[:,0,0]*=-1
            corrupt,bs,_=model(x,b['a'],4,cache=True)
            for t in range(4):
                z,ps,_=model(x,b['a'],4,{'patch_round':t,'patch_nodes':[0],'patch_value':cs[t]},cache=True)
                immediate=bs[t].clone(); immediate[:,0]=cs[t][:,0]
                patch.append(dict(seed=seed,round=t,immediate_logit_change=float((model.readout(immediate)-model.readout(bs[t])).mean()),downstream_logit_change=float((z-corrupt).mean()),other_agent_probability_change=float((z[:,1:].sigmoid()-corrupt[:,1:].sigmoid()).abs().mean())))
            base=metrics(clean,b['y'])['accuracy']
            node=pd.read_csv(root/'results/node_ranking.csv').query('seed==@seed').sort_values('rank').node.tolist()
            for fraction in (0,.1,.2,.4):
                count=int(10*fraction); randnodes=rng.permutation(10)[:count].tolist()
                # Remove sampled undirected existing edges independently in each episode.
                a=b['a'].clone(); removed=0
                for episode in range(len(a)):
                    edges=np.argwhere(np.triu(a[episode].numpy(),1)>0); selected=rng.permutation(len(edges))[:int(len(edges)*fraction)]
                    for i,j in edges[selected]: a[episode,i,j]=a[episode,j,i]=0; removed+=2
                edge_batch={**b,'a':a}
                cases=[('random_nodes',b,{'nodes':randnodes}),('targeted_nodes',b,{'nodes':node[:count]}),('random_edges',edge_batch,{}),('message_noise',b,{'noise':fraction})]
                for label,batch,iv in cases:
                    seed_all(seed); q=infer(model,batch,4,iv); m=metrics(q,b['y'])
                    robust.append(dict(seed=seed,fraction=fraction,kind=label,degradation=base-m['accuracy'],realized_edge_removal=removed/float(b['a'].sum()) if label=='random_edges' else np.nan,**m))
    for name,rows in [('probes',probes),('circuit_transfer',transfer),('graph_geometry',geometry),('patching_diagnostics',patch),('gradient_attribution',attribution),('robustness',robust)]: pd.DataFrame(rows).to_csv(root/'results'/f'{name}.csv',index=False)
    d=pd.read_csv(root/'results/evaluation.csv'); baseline=d.query("n==10 and topology=='random'")[['seed','model','reliability','rounds','accuracy']].rename(columns={'accuracy':'reference_accuracy'})
    merged=d.merge(baseline,on=['seed','model','reliability','rounds']); merged['generalization_gap']=merged.reference_accuracy-merged.accuracy; merged.to_csv(root/'results/generalization_gaps.csv',index=False)
    from interpretability.agent_effects import run as agent_effects
    agent_effects(root)
    from visualization.figures import generate
    generate(root)
    print('Supplemental experiments and refreshed figures completed.')
if __name__=='__main__': run()
