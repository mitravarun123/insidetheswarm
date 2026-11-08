from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t

NAMES=['system_architecture','agent_communication_graph','consensus_emergence','causal_agent_influence','communication_circuit_map','neural_representation_heatmap','targeted_ablation','activation_patching_recovery','failure_robustness','swarm_size_generalization']

def generate(root):
    root=Path(root); out=root/'results/figures'; out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.figsize':(8,5)})
    mode=__import__('json').loads((root/'results/manifest.json').read_text())['mode']
    def save(i,fig,data):
        fig.suptitle(('QUICK smoke run — ' if mode=='quick' else '')+NAMES[i-1].replace('_',' '),fontsize=13)
        fig.tight_layout(); fig.savefig(out/f'F{i:02d}_{NAMES[i-1]}.png',dpi=350); plt.close(fig)
        data.to_csv(out/f'F{i:02d}_data.csv',index=False)
    def curves(ax,df,x,y,group):
        # Reduce scenarios/conditions within seed before seed-level uncertainty.
        for label,g in df.groupby(group):
            d=g.groupby(['seed',x])[y].mean().reset_index()
            s=d.groupby(x)[y].agg(['mean','std','count']); ci=t.ppf(.975,(s['count']-1).clip(lower=1))*s['std']/np.sqrt(s['count'])
            ax.plot(s.index,s['mean'],marker='o',label=str(label))
            if s['count'].min()>1: ax.fill_between(s.index,s['mean']-ci,s['mean']+ci,alpha=.15)
        ax.set(xlabel={'n':'agent count','round':'communication round','rounds':'communication rounds'}.get(x,x.replace('_',' ')),ylabel={'accuracy':'collective accuracy','individual_accuracy':'individual accuracy'}.get(y,y.replace('_',' ')))
        if len(df[x].unique())<=10 and pd.api.types.is_numeric_dtype(df[x]): ax.set_xticks(sorted(df[x].unique()))
        ax.legend(fontsize=8)
    fig,ax=plt.subplots(); ax.axis('off')
    labels=['Hidden world y\n(training label only)','Private sensors\nsᵢ ~ Bernoulli likelihood','Shared encoder\nindependent hᵢ','Local masked recurrent\nmessages × R','Per-agent logits\nexternal evaluation']
    for j,label in enumerate(labels):
        ax.text(.5,.92-j*.2,label,ha='center',va='center',bbox=dict(boxstyle='round',facecolor='#dceafa'))
        if j<4: ax.annotate('',xy=(.5,.78-j*.2),xytext=(.5,.86-j*.2),arrowprops=dict(arrowstyle='->'))
    save(1,fig,pd.DataFrame({'stage':labels,'type':'schematic'}))
    cache=__import__('torch').load(root/'results/activations_0.pt',weights_only=False); a=cache['batch']['a'][0].numpy(); n=len(a)
    theta=np.linspace(0,2*np.pi,n,endpoint=False); pos=np.c_[np.cos(theta),np.sin(theta)]
    fig,ax=plt.subplots()
    edges=[]
    for i,j in zip(*np.where(np.triu(a)>0)):
        ax.plot(pos[[i,j],0],pos[[i,j],1],color='gray',alpha=.5); edges.append(dict(source=i,target=j))
    points=ax.scatter(*pos.T,s=180,c=cache['batch']['x'][0,:,1],cmap='viridis'); fig.colorbar(points,ax=ax,label='sensor reliability');
    for i in range(n): ax.text(*pos[i],str(i),ha='center',va='center')
    ax.set_aspect('equal'); ax.axis('off'); save(2,fig,pd.DataFrame(edges))
    df=pd.read_csv(root/'results/evaluation.csv'); d=df.query("n==10 and topology=='random' and reliability==0.75 and model in ['mlp','average','gnn','gat']")
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,metric in zip(axes.flat,['accuracy','individual_accuracy','consensus','brier']): curves(ax,d,'rounds',metric,'model')
    save(3,fig,d)
    d=pd.read_csv(root/'results/influence.csv').query("intervention=='remove'"); fig,ax=plt.subplots(); curves(ax,d,'node','effect','intervention'); save(4,fig,d)
    d=pd.read_csv(root/'results/circuits.csv'); edge=pd.read_csv(root/'results/edges.csv'); fig,axes=plt.subplots(1,2,figsize=(11,4)); curves(axes[0],d.assign(model='gnn'),'channel','effect','model')
    matrix=np.zeros((n,n))
    for (receiver,sender),g in edge.groupby(['receiver','sender']): matrix[int(receiver),int(sender)]=g.effect.mean()
    im=axes[1].imshow(matrix,cmap='magma'); axes[1].set(xlabel='sender',ylabel='receiver',title='Directed ring-edge intervention effect'); fig.colorbar(im,ax=axes[1],label='mean |probability change|')
    save(5,fig,pd.concat([d.assign(component='channel'),edge.assign(component='edge')],ignore_index=True))
    d=pd.read_csv(root/'results/probes.csv'); fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(name,g) in zip(axes,d.groupby('control')):
        table=g.pivot_table(index='target',columns='round',values='r2'); im=ax.imshow(table,vmin=-.2,vmax=1,cmap='coolwarm'); ax.set_yticks(range(len(table)),table.index); ax.set_xticks(range(len(table.columns)),table.columns); ax.set(xlabel='communication round',title=name); fig.colorbar(im,ax=ax,label='held-out R²')
    save(6,fig,d)
    d=pd.read_csv(root/'results/ablations.csv'); fig,ax=plt.subplots(); kinds=list(d.kind.unique()); controls=list(d.control.unique()); x=np.arange(len(kinds)); width=.25
    for j,control in enumerate(controls):
        means=[]; errors=[]
        for kind in kinds:
            v=d.query('control==@control and kind==@kind').accuracy_drop
            means.append(v.mean() if len(v) else np.nan); errors.append(t.ppf(.975,len(v)-1)*v.std()/np.sqrt(len(v)) if len(v)>1 else 0)
        ax.bar(x+(j-1)*width,means,width,yerr=errors,capsize=3,label=control.replace('_',' '))
    ax.axhline(0,color='gray',linewidth=.8); ax.set_xticks(x,[k.replace('_',' ') for k in kinds]); ax.set(ylabel='accuracy drop (negative = improvement)'); ax.legend(fontsize=8); save(7,fig,d)
    d=pd.read_csv(root/'results/patching.csv'); fig,ax=plt.subplots(); curves(ax,d,'round','logit_recovery','control'); save(8,fig,d)
    d=pd.read_csv(root/'results/robustness.csv'); fig,ax=plt.subplots(); curves(ax,d,'fraction','degradation','kind'); ax.set_xlabel('failure fraction / message-noise standard deviation'); save(9,fig,d)
    d=df.query("rounds==4 and reliability==0.75 and model in ['gnn','gat','average','oracle']"); fig,ax=plt.subplots(); curves(ax,d,'n','accuracy','model'); save(10,fig,d)
