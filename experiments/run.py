"""Run from repository root: python -m experiments.run --mode standard."""
import argparse,json,time,platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from environments.task import sample,subset,seed_all
from training.engine import train
from baselines.methods import predict
from experiments.metrics import metrics
from interpretability.analysis import mechanisms

CONFIGS={
 'quick':dict(seeds=1,hidden=32,epochs=5,steps=5,batch=64,validation=128,test=128,patience=4),
 'standard':dict(seeds=3,hidden=64,epochs=35,steps=20,batch=128,validation=512,test=512,patience=7),
 'full':dict(seeds=5,hidden=64,epochs=70,steps=30,batch=128,validation=1024,test=1024,patience=10)}

def infer(model,b,rounds,iv=None,chunk=128):
    with torch.no_grad():
        return torch.cat([model(b['x'][i:i+chunk],b['a'][i:i+chunk],rounds,iv) for i in range(0,len(b['y']),chunk)])

def run(mode='standard',device='cpu',root='.'):
    root=Path(root); cfg=CONFIGS[mode].copy()
    for name in ('results','checkpoints'): (root/name).mkdir(exist_ok=True)
    torch.set_num_threads(2); start=time.time(); rows=[]; collected={}; robustness=[]; scenario_rows=[]
    for seed in range(cfg['seeds']):
        print(f'Training seed {seed} ({mode}, {device})',flush=True)
        models={kind:train(kind,seed,cfg,root,device) for kind in ('mlp','gnn','gat')}
        for n in (5,10,20,40):
            for topology in ('ring','random','star','small_world'):
                for reliability in (.6,.75,.9):
                    b=sample(cfg['test'],n,topology,reliability,seed=300000+seed,device=device)
                    for rounds in (0,1,2,4,8):
                        for name in ('mlp','average','gnn','gat','oracle','local_oracle','shuffled','randomized'):
                            if name in ('average','oracle','local_oracle'): z=predict(b,name,rounds)
                            else:
                                model=models.get(name,models['gnn']); iv={'shuffle':True} if name=='shuffled' else {'randomize':True} if name=='randomized' else None
                                seed_all(900000+seed); z=infer(model,b,rounds,iv)
                            if n==10 and topology=='random' and reliability==.75 and rounds==4:
                                for scenario,(prob,y) in enumerate(zip(z.sigmoid().mean(-1).cpu().tolist(),b['y'].cpu().tolist())):
                                    scenario_rows.append(dict(seed=seed,scenario=scenario,model=name,probability=prob,y=y,correct=int((prob>=.5)==bool(y))))
                            rows.append(dict(seed=seed,n=n,topology=topology,reliability=reliability,rounds=rounds,model=name,**metrics(z,b['y'],b['a'],0 if name in ('mlp','oracle') else rounds)))
        b=sample(cfg['test'],seed=350000+seed,device=device)
        analysis,states=mechanisms(models['gnn'],b,seed)
        torch.save(dict(states=[s.cpu() for s in states],batch={k:v.cpu() for k,v in b.items()}),root/'results'/f'activations_{seed}.pt')
        for key,value in analysis.items(): collected.setdefault(key,[]).extend(value)
        model=models['gnn']; base=metrics(infer(model,b,4),b['y'])['accuracy']
        target=pd.DataFrame(analysis['node_ranking']).sort_values('rank').node.tolist()
        rng=np.random.default_rng(seed)
        for fraction in (0,.1,.2,.4):
            count=int(10*fraction); random=rng.permutation(10)[:count].tolist()
            for label,iv in [('random_nodes',{'nodes':random}),('targeted_nodes',{'nodes':target[:count]}),('random_edges',{'edges':[(int(i),int(j)) for i,j in rng.permutation(np.argwhere(b['a'][0].cpu().numpy()>0))[:int(fraction*b['a'][0].sum())]]}),('message_noise',{'noise':fraction})]:
                seed_all(seed); z=infer(model,b,4,iv)
                robustness.append(dict(seed=seed,fraction=fraction,kind=label,degradation=base-metrics(z,b['y'])['accuracy'],**metrics(z,b['y'])))
        # Sensitivity beyond the primary grid: heterogeneity and connectivity.
        sensitivity=collected.setdefault('sensitivity',[])
        for heterogeneity in (0,.1,.25):
            for connectivity in (0,.15,.4):
                sb=sample(cfg['test'],heterogeneity=heterogeneity,connectivity=connectivity,seed=370000+seed,device=device)
                sensitivity.append(dict(seed=seed,heterogeneity=heterogeneity,connectivity=connectivity,**metrics(infer(model,sb,4),sb['y'])))
        pd.DataFrame(scenario_rows).to_csv(root/'results/paired_scenarios.csv',index=False)
        pd.DataFrame(rows).to_csv(root/'results/evaluation.csv',index=False)
        for key,value in collected.items(): pd.DataFrame(value).to_csv(root/'results'/f'{key}.csv',index=False)
        pd.DataFrame(robustness).to_csv(root/'results/robustness.csv',index=False)
    manifest=dict(mode=mode,config=cfg,seconds=time.time()-start,torch=torch.__version__,python=platform.python_version(),device=device,training_distribution='n=10, connected random ring-backbone graph, reliability=.7 +/- .1, rounds=4',test_seeds=[300000+i for i in range(cfg['seeds'])],warning='Training seeds are replicates. QUICK is a smoke run. CIs with one seed are unavailable.')
    (root/'results/manifest.json').write_text(json.dumps(manifest,indent=2))
    from experiments.statistics import summarize
    summarize(root)
    from visualization.figures import generate
    generate(root)
    summary=pd.DataFrame(rows).query("n == 10 and topology == 'random' and reliability == 0.75 and rounds == 4").groupby('model').accuracy.agg(['mean','std'])
    (root/'results/summary.txt').write_text(summary.to_string()); print(summary); return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=CONFIGS,default='standard'); parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args(); run(args.mode,args.device)
