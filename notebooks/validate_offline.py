import os,json,traceback,contextlib,io
import nbformat
from pathlib import Path
root=Path(__file__).resolve().parents[1]
nb=nbformat.read(root/'notebooks/master_kaggle.ipynb',as_version=4)
nb.cells[1].source='import torch, numpy, pandas, matplotlib, scipy, sklearn\nprint("Dependencies already installed; installation cell skipped for offline validation")'
import tempfile
os.chdir(tempfile.mkdtemp(prefix='swarm-notebook-')); namespace={'__name__':'__main__'}
for i,cell in enumerate(nb.cells):
 if cell.cell_type!='code': continue
 cell.source=cell.source.replace("MODE='standard'","MODE='quick'")
 print('Executing notebook cell',i,flush=True)
 capture=io.StringIO()
 with contextlib.redirect_stdout(capture): exec(compile(cell.source,f'notebook_cell_{i}','exec'),namespace)
 cell.execution_count=i+1; cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=capture.getvalue())]
 print(capture.getvalue(),flush=True)
nbformat.write(nb,root/'results/notebook_smoke_executed.ipynb')
(root/'results/notebook_validation.json').write_text(json.dumps({'status':'passed','mode':'quick','method':'Python code cells in order; offline dependency import substituted for pip install','jupyter_kernel':'Sandbox disallowed binding local sockets; no live kernel started'},indent=2))
print('Notebook QUICK execution passed (offline Python-cell validation)')
