import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from environments.task import oracle

def probe_states(states,b,clean,seed):
    rng=np.random.default_rng(500000+seed); rows=[]; n=b['x'].shape[1]
    collective=(clean.sigmoid().mean(-1)>=.5)
    agreement=((clean>=0)==collective[:,None]).float().mean(-1)
    for t,state in enumerate(states):
        posterior=oracle(b,t).sigmoid().clamp(1e-7,1-1e-7)
        entropy=-(posterior*posterior.log()+(1-posterior)*(1-posterior).log())
        targets={'private':b['evidence'],'accumulated':oracle(b,t),'uncertainty':entropy,'consensus':agreement[:,None].expand(-1,n),'predicted_decision':collective.float()[:,None].expand(-1,n)}
        cut=len(state)//2; h=state.shape[-1]; X=state.detach().cpu().numpy()
        for name,target in targets.items():
            Y=target.cpu().numpy()
            for control in ('actual','permuted'):
                train_y=Y[:cut] if control=='actual' else Y[:cut][rng.permutation(cut)]
                probe=Ridge(alpha=10).fit(X[:cut].reshape(-1,h),train_y.reshape(-1))
                score=r2_score(Y[cut:].reshape(-1),probe.predict(X[cut:].reshape(-1,h)))
                rows.append(dict(seed=seed,round=t,target=name,control=control,r2=score))
    return rows
