"""Populate manuscript only from completed run tables; no significance claims."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t
root=Path(__file__).resolve().parents[1]; out=root/'results'; meta=json.loads((out/'manifest.json').read_text())
def estimate(values):
 v=np.asarray(values,dtype=float); v=v[np.isfinite(v)]
 if len(v)<2: return f'{v.mean():.4f} (one seed; CI unavailable)' if len(v) else 'unavailable'
 half=t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
 return f'{v.mean():.4f} (SD {v.std(ddof=1):.4f}; 95\\% CI [{v.mean()-half:.4f}, {v.mean()+half:.4f}])'
d=pd.read_csv(out/'evaluation.csv').query("n==10 and topology=='random' and reliability==0.75 and rounds==4")
lines=[f"The completed {meta['mode'].upper()} run used {meta['config']['seeds']} training seeds on {meta['device']}. The full primary grid contains {len(pd.read_csv(out/'evaluation.csv'))} model-condition-seed rows. Numerical intervals below use training seeds, not agents.\\par\n",'Primary condition: ten agents, random connected graphs, reliability 0.75 with heterogeneity 0.1, four rounds.\\par\n']
for name,g in d.groupby('model'):
 lines.append(name.replace('_',r'\_')+': collective accuracy '+estimate(g.accuracy)+'; individual accuracy '+estimate(g.individual_accuracy)+'.\\par\n')
p=pd.read_csv(out/'paired_contrasts.csv'); lines.append('The paired GNN--MLP collective-accuracy contrast is '+estimate(d.set_index(['seed','model']).accuracy.unstack().gnn-d.set_index(['seed','model']).accuracy.unstack().mlp)+'. This does not by itself establish H1 because the collective readout pools private predictions.\\par\n')
(root/'paper/results_generated.tex').write_text(''.join(lines))
a=pd.read_csv(out/'ablations.csv'); wide=a.query("kind=='channels'").pivot(index='seed',columns='control',values='accuracy_drop')
probe=pd.read_csv(out/'probes.csv'); patch=pd.read_csv(out/'patching.csv')
text='Targeted minus random channel-ablation accuracy drop: '+estimate(wide.targeted-wide.random)+'. Targeted minus magnitude-matched: '+estimate(wide.targeted-wide.magnitude_matched)+'. These are paired seed-level descriptive effects; selection used independent discovery episodes.\\par\n'
for target,g in probe.query("control=='actual' and round==4").groupby('target'):
 text+=target.replace('_',r'\_')+' final-state probe $R^2$: '+estimate(g.r2)+'.\\par\n'
for control,g in patch.query('round==0').groupby('control'):
 text+=control.replace('_',r'\_')+' initial-round final signed-logit recovery: '+estimate(g.logit_recovery)+'.\\par\n'
text+='Initial-state restoration of the flipped sensor is an expected positive control. H2 remains unverified by descriptive controlled regressions; H3 concerns decodability only; H4 and H5 require interpretation of matched controls, later-round effects, and low-powered seed-level uncertainty. No hypothesis is declared statistically confirmed.\n'
(root/'paper/mechanisms_generated.tex').write_text(text)
report=dict(mode=meta['mode'],seeds=meta['config']['seeds'],status='Descriptive benchmark completed; no confirmatory hypothesis declarations',H1='Interpret individual accuracy separately from globally pooled collective readout',H2='Unverified; controlled regression is descriptive',H3='Inspect held-out R2 and permuted controls; probes are not causal',H4='Inspect paired targeted-minus-random and magnitude-matched seed differences',H5='Initial sensor patch is positive control; inspect later-round paired effects')
(out/'hypothesis_status.json').write_text(json.dumps(report,indent=2))
print('Generated manuscript tables and hypothesis status from saved data.')
