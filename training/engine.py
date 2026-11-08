import json
from pathlib import Path
import torch
from environments.task import sample,seed_all
from models.networks import SwarmNet

def atomic_save(state,path):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(state,temporary); temporary.replace(path)

def train(kind,seed,cfg,root,device):
    seed_all(seed); model=SwarmNet(kind,cfg['hidden']).to(device)
    path=Path(root)/'checkpoints'/f'{kind}_{seed}.pt'
    latest=path.with_name(path.stem+'_latest.pt'); done=path.with_suffix('.complete')
    if path.exists() and done.exists():
        saved=torch.load(path,map_location=device,weights_only=False)
        if saved['config']==cfg:
            model.load_state_dict(saved['model']); return model.eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    val=sample(cfg['validation'],seed=200000+seed,device=device)
    best=float('inf'); stale=0; logs=[]; first=0
    if latest.exists():
        saved=torch.load(latest,map_location=device,weights_only=False)
        if saved['config']==cfg:
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            first=saved['epoch']+1; best=saved['best']; stale=saved['stale']; logs=saved['logs']
            if stale>=cfg['patience']: first=cfg['epochs']
    for epoch in range(first,cfg['epochs']):
        model.train(); losses=[]
        for step in range(cfg['steps']):
            b=sample(cfg['batch'],seed=100000+seed*10000+epoch*cfg['steps']+step,device=device)
            optimizer.zero_grad(); z=model(b['x'],b['a'],4)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(z,b['y'][:,None].expand_as(z))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            z=model(val['x'],val['a'],4)
            vl=torch.nn.functional.binary_cross_entropy_with_logits(z,val['y'][:,None].expand_as(z)).item()
        logs.append(dict(epoch=epoch,train=sum(losses)/len(losses),validation=vl))
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=cfg,seed=seed,epoch=epoch)
        if vl<best-1e-5:
            best=vl; stale=0; atomic_save(state,path)
        else: stale+=1
        atomic_save(dict(**state,best=best,stale=stale,logs=logs),latest)
        (Path(root)/'results'/f'train_{kind}_{seed}.json').write_text(json.dumps(logs,indent=2))
        if stale>=cfg['patience']: break
    done.write_text('Training completed; best validation checkpoint selected.\n')
    model.load_state_dict(torch.load(path,map_location=device,weights_only=False)['model']); return model.eval()
