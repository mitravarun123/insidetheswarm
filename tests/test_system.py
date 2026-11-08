import tempfile
import unittest
import torch
from environments.task import sample,oracle,reachable,seed_all
from models.networks import SwarmNet
from experiments.metrics import metrics

class TestSystem(unittest.TestCase):
    def setUp(self):
        seed_all(7); self.b=sample(64,n=5,topology='ring',seed=13); self.m=SwarmNet(hidden=32).eval()
    def test_signals(self):
        b=sample(20000,n=5,reliability=.75,heterogeneity=0,seed=2)
        self.assertLess(abs(float(((b['x'][...,0]>0)==b['y'][:,None].bool()).float().mean())-.75),.01)
    def test_oracle(self):
        b=self.b; expected=((b['x'][...,0])*torch.log(b['x'][...,1]/(1-b['x'][...,1]))).sum(-1)
        torch.testing.assert_close(oracle(b),expected)
    def test_masks(self):
        r=reachable(self.b['a'],1); self.assertEqual(float(r[0,0,2]),0); self.assertEqual(float(r[0,0,1]),1)
    def test_permutation(self):
        b=self.b; p=torch.tensor([2,4,1,0,3]); z=self.m(b['x'],b['a'],2)
        torch.testing.assert_close(self.m(b['x'][:,p],b['a'][:,p][:,:,p],2),z[:,p],atol=1e-6,rtol=1e-5)
    def test_no_leakage(self):
        b=self.b; x=b['x'].clone(); x[:,2,0]*=-1
        z=self.m(b['x'],b['a'],1); q=self.m(x,b['a'],1)
        torch.testing.assert_close(z[:,0],q[:,0])
        z=self.m(b['x'],b['a'],0); q=self.m(x,b['a'],0); torch.testing.assert_close(z[:,0],q[:,0])
    def test_patch(self):
        b=self.b; z,s,_=self.m(b['x'],b['a'],2,cache=True)
        q=self.m(b['x'],b['a'],2,{'patch_round':1,'patch_nodes':[0],'patch_value':s[1]})
        torch.testing.assert_close(z,q)
        x=-b['x']; q=self.m(x,b['a'],2,{'patch_round':0,'patch_nodes':list(range(5)),'patch_value':s[0]})
        torch.testing.assert_close(z,q)
    def test_ablation(self):
        b=self.b; zero=torch.zeros_like(b['a'])
        z=self.m(b['x'],b['a'],2,{'nodes':list(range(5))})
        torch.testing.assert_close(z,self.m(b['x'],zero,2))
    def test_deterministic(self):
        b=self.b; torch.testing.assert_close(self.m(b['x'],b['a']),self.m(b['x'],b['a']))
        torch.testing.assert_close(sample(64,n=5,topology='ring',seed=13)['x'],b['x'])
    def test_metrics(self):
        m=metrics(torch.tensor([[20.,20.],[-20.,-20.]]),torch.tensor([1.,0.]))
        self.assertEqual(m['accuracy'],1); self.assertEqual(m['consensus'],1); self.assertLess(m['brier'],1e-10)
    def test_checkpoint(self):
        with tempfile.NamedTemporaryFile() as f:
            torch.save(self.m.state_dict(),f.name); m=SwarmNet(hidden=32); m.load_state_dict(torch.load(f.name,weights_only=True))
            torch.testing.assert_close(m(self.b['x'],self.b['a']),self.m(self.b['x'],self.b['a']))
if __name__=='__main__': unittest.main()
