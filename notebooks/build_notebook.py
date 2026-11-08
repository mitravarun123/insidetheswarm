"""Build a self-contained Kaggle notebook from current repository source."""
from pathlib import Path
import base64,json,zlib
root=Path(__file__).resolve().parents[1]
files={str(p.relative_to(root)):p.read_text() for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.txt','.md','.tex') and not any(v in p.parts for v in ('results','checkpoints','__pycache__'))}
payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
def cell(kind,source):
 d=dict(cell_type=kind,metadata={},source=source.splitlines(True))
 if kind=='code': d.update(execution_count=None,outputs=[])
 return d
cells=[cell('markdown','# Inside the Swarm\nSelf-contained supervised multi-agent research pipeline. STANDARD runs three independent training seeds. All plots use actual simulator executions; this notebook does not contain precomputed numerical claims. Enable Internet to install dependencies and optionally select a GPU.'),cell('code','%pip install "torch>=2.2,<3" "numpy>=1.24,<3" "pandas>=2,<3" "matplotlib>=3.7,<4" "scipy>=1.10,<2" "scikit-learn>=1.3,<2"'),cell('code',f'''import base64,zlib,json,os,sys
from pathlib import Path
ROOT=Path('/kaggle/working/swarm-mech-interp') if Path('/kaggle').exists() else Path.cwd()/'swarm-notebook-run'
ROOT.mkdir(parents=True,exist_ok=True)
source=json.loads(zlib.decompress(base64.b64decode({payload!r})))
for name,content in source.items():
    path=ROOT/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(content)
for name in ('checkpoints','results'): (ROOT/name).mkdir(exist_ok=True)
os.chdir(ROOT); sys.path.insert(0,str(ROOT))
# To resume a previous Kaggle session, copy its checkpoints directory into ROOT first.
'''),cell('code',"import torch\nfrom environments.task import seed_all\nseed_all(0)\nDEVICE='cuda' if torch.cuda.is_available() else 'cpu'\nMODE='standard' # quick: 1 seed / 32 hidden; full: 5 seeds / 64 hidden\nprint(DEVICE, torch.__version__)"),cell('markdown','## Validate the simulation and intervention semantics'),cell('code',"import unittest\nsuite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))\nresult=unittest.TextTestRunner(verbosity=2).run(suite)\nassert result.wasSuccessful()"),cell('markdown','## Train, compare baselines, cache states, and run E1–E8\nThe orchestrator trains MLP/GNN/GAT; evaluates averaging, global/local oracles and message controls; runs probes, outgoing-message interventions, channel and edge ablations, paired patching, scaling and robustness; and writes all CSVs, figures and checkpoints. It resumes completed model checkpoints or the latest completed training epoch. Evaluation reruns deterministically.'),cell('code',"from experiments.run import run\nmanifest=run(MODE,DEVICE,ROOT)\nfrom experiments.supplement import run as supplement\nsupplement(ROOT)\nexec((ROOT/'paper/build_results.py').read_text(), {'__file__':str(ROOT/'paper/build_results.py')})\nprint(json.dumps(manifest,indent=2))"),cell('code',"import pandas as pd\nfrom IPython.display import display,Image\ndisplay(pd.read_csv(ROOT/'results/paired_contrasts.csv'))\nfor path in sorted((ROOT/'results/figures').glob('F*.png')):\n    display(Image(filename=str(path),width=800))"),cell('markdown','## Export and final summary\nThe archive includes raw tables, activation caches, trained checkpoints, source, and manuscript. Confidence intervals use training seeds. QUICK cannot estimate between-seed uncertainty.'),cell('code',"import shutil\nprint((ROOT/'results/summary.txt').read_text())\narchive=shutil.make_archive(str(ROOT.parent/'inside-the-swarm-results'),'zip',ROOT)\nprint('Saved:',archive)\nprint('Manuscript draft:',ROOT/'paper/main.tex')")]
nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.10')),nbformat=4,nbformat_minor=5)
(root/'notebooks/master_kaggle.ipynb').write_text(json.dumps(nb,indent=1))
