import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from training.engine import train

class TestTrainingResume(unittest.TestCase):
    def test_epoch_resume(self):
        cfg=dict(hidden=32,validation=16,epochs=2,steps=1,batch=8,patience=3)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); a=root/'full'; b=root/'resume'
            for directory in (a,b):
                (directory/'checkpoints').mkdir(parents=True); (directory/'results').mkdir()
            full=train('gnn',0,cfg,a,'cpu')
            original_save=torch.save
            def interrupt_after_epoch(value,path,*args,**kwargs):
                original_save(value,path,*args,**kwargs)
                if str(path).endswith('_latest.pt.tmp'):
                    # Complete the atomic replacement as though interrupted just
                    # after epoch commit, before entering the next epoch.
                    Path(path).replace(Path(str(path)[:-4]))
                    raise InterruptedError('simulated interruption after epoch')
            with patch('torch.save',side_effect=interrupt_after_epoch):
                with self.assertRaises(InterruptedError): train('gnn',0,cfg,b,'cpu')
            resumed=train('gnn',0,cfg,b,'cpu')
            for key,value in full.state_dict().items(): torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
if __name__=='__main__': unittest.main()
