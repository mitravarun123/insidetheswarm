import torch
from torch import nn

class SwarmNet(nn.Module):
    def __init__(self,kind='gnn',hidden=64):
        super().__init__(); self.kind=kind; self.hidden=hidden
        self.encoder=nn.Sequential(nn.Linear(2,hidden),nn.Tanh())
        self.message=nn.Linear(hidden,hidden,bias=False)
        self.update=nn.GRUCell(hidden,hidden)
        self.readout=nn.Linear(hidden,1)
        self.query=nn.Linear(hidden,hidden,bias=False); self.key=nn.Linear(hidden,hidden,bias=False)

    def forward(self,x,a,rounds=4,intervention=None,cache=False):
        """a[receiver,sender]. Interventions operate before sending each round.
        Patch dictionary: round, nodes, value (batch,node,hidden); channels/edges
        remove outgoing channels/edges; nodes removes all outgoing messages.
        """
        iv=intervention or {}; h=self.encoder(x); states=[h]; attentions=[]
        for t in range(rounds if self.kind!='mlp' else 0):
            if iv.get('patch_round')==t:
                mask=torch.zeros_like(h); mask[:,iv['patch_nodes'],:]=1
                h=h*(1-mask)+iv['patch_value']*mask
            m=self.message(h)
            if 'replace_nodes' in iv:
                mask=torch.zeros_like(m); mask[:,iv['replace_nodes'],:]=1
                replacement=m.roll(1,0) if iv.get('replacement')=='batch_shuffle' else -m
                m=m*(1-mask)+replacement*mask
            if iv.get('only_round') is not None and iv['only_round']!=t:
                active=False
            else: active=True
            if 'nodes' in iv:
                mask=torch.ones_like(m); mask[:,iv['nodes'],:]=0; m=m*mask
            if 'channels' in iv and active:
                mask=torch.ones_like(m); mask[:,:,iv['channels']]=0; m=m*mask
            if 'hidden_channels' in iv and active:
                mask=torch.ones_like(h); mask[:,:,iv['hidden_channels']]=0; h=h*mask
            if iv.get('shuffle'): m=m[:,torch.arange(m.shape[1]-1,-1,-1,device=m.device)]
            if iv.get('randomize'): m=torch.randn_like(m)*m.detach().std()
            if iv.get('noise',0): m=m+torch.randn_like(m)*iv['noise']
            mask=a.clone()
            if 'edges' in iv:
                for receiver,sender in iv['edges']: mask[:,receiver,sender]=0
            if self.kind=='gat':
                scores=torch.bmm(self.query(h),self.key(h).transpose(1,2))/(self.hidden**.5)
                w=scores.masked_fill(mask==0,-1e9).softmax(-1)*mask
                w=w/w.sum(-1,keepdim=True).clamp_min(1e-9)
            else: w=mask/mask.sum(-1,keepdim=True).clamp_min(1)
            attentions.append(w)
            agg=torch.bmm(w,m); h=self.update(agg.reshape(-1,self.hidden),h.reshape(-1,self.hidden)).reshape_as(h)
            states.append(h)
        logits=self.readout(h).squeeze(-1)
        return (logits,states,attentions) if cache else logits
