import torch
from environments.task import oracle

def predict(b,kind,rounds):
    if kind=='oracle': return oracle(b).unsqueeze(-1).expand_as(b['evidence'])
    if kind=='local_oracle': return oracle(b,rounds)
    h=b['evidence'].clone()
    a=b['a']+torch.eye(h.shape[1],device=h.device)
    w=a/a.sum(-1,keepdim=True)
    for _ in range(rounds): h=torch.bmm(w,h.unsqueeze(-1)).squeeze(-1)
    return h
