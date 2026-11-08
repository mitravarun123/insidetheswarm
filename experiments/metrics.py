import numpy as np
import torch

def metrics(z,y,a=None,rounds=0):
    p=z.sigmoid(); collective=p.mean(-1); pred=collective>=.5
    individual=(p>=.5)==y[:,None].bool()
    consensus=((p>=.5)==pred[:,None]).float().mean(-1)
    ece=0.
    for lo in np.linspace(0,.9,10):
        mask=(collective>=lo)&(collective<lo+.1+1e-7)
        if mask.any(): ece+=float(mask.float().mean()*abs(collective[mask].mean()-y[mask].mean()))
    return dict(accuracy=float((pred==y.bool()).float().mean()),individual_accuracy=float(individual.float().mean()),consensus=float(consensus.mean()),brier=float(((collective-y)**2).mean()),ece=ece,message_cost=float(a.sum((-1,-2)).mean()*rounds) if a is not None else 0.)

def per_scenario(z,y):
    p=z.sigmoid().mean(-1)
    return ((p>=.5)==y.bool()).float(),(p-y).square()
