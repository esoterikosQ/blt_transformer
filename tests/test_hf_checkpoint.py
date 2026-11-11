import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from blt_hf.checkpoint import resolve_checkpoint
from blt_hf.select_best import select_best

class CheckpointContracts(unittest.TestCase):
    def test_pointer_cannot_escape_run_directory(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'latest.json';p.write_text('{"checkpoint":"../outside"}')
            with self.assertRaises(ValueError): resolve_checkpoint(p)

    def test_full_validation_gleu_selects_epoch_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';run.mkdir()
            (run/'run.json').write_text(json.dumps({'run_id':'outputs/blt_hf/native/example'}))
            epochs=[{'epoch':1,'checkpoint':'step-a','global_step':10,'val_loss':.4},
                    {'epoch':2,'checkpoint':'step-b','global_step':20,'val_loss':.3}]
            (run/'epoch_checkpoints.json').write_text(json.dumps(epochs))
            evaluations=[]
            for epoch,step,gleu in ((1,10,40.),(2,20,45.)):
                folder=root/f'eval-{epoch}';(folder/'scored').mkdir(parents=True)
                manifest={'split':'val','dataset':'native','num_beams':1,'length_penalty':1.,
                          'max_new_bytes':768,'training_run_id':'outputs/blt_hf/native/example',
                          'checkpoint_step':step,'fingerprint':f'f{epoch}'}
                metrics={'status':'complete','fingerprint':f'f{epoch}','gleu':gleu,'m2':{'f0.5':.5}}
                (folder/'run.json').write_text(json.dumps(manifest))
                (folder/'scored/metrics.json').write_text(json.dumps(metrics));evaluations.append(folder)
            best=select_best(run,evaluations)
            self.assertEqual((best['epoch'],best['checkpoint']),(2,'step-b'))

@unittest.skipUnless(importlib.util.find_spec('torch'), 'prepared HF environment required')
class CheckpointTensorTests(unittest.TestCase):
    def test_immutable_publication_strict_tensor_reload_and_tamper_detection(self):
        import torch
        from safetensors.torch import load_file
        from blt_hf.checkpoint import save_checkpoint
        class FakeOptimizer:
            def state_dict(self): return {'state':{},'param_groups':[]}
        with tempfile.TemporaryDirectory() as d:
            model=torch.nn.Linear(2,3)
            state={'global_step':1,'epoch':1,'next_batch':0,'last_val_loss':.5,'run_signature':'abc'}
            name=save_checkpoint(d,model,FakeOptimizer(),state,[{'cpu':torch.get_rng_state()}],best=True)
            path,meta=resolve_checkpoint(Path(d)/'best.json')
            self.assertEqual(path.name,name)
            restored=torch.nn.Linear(2,3)
            restored.load_state_dict(load_file(str(path/'model.safetensors')),strict=True)
            for a,b in zip(model.parameters(),restored.parameters()): self.assertTrue(torch.equal(a,b))
            self.assertEqual(torch.load(path/'training.pt',weights_only=True)['optimizer']['state'],{})
            epochs=json.loads((Path(d)/'epoch_checkpoints.json').read_text())
            self.assertEqual(epochs[0]['epoch'],1)
            (path/'model.safetensors').write_bytes(b'corrupted')
            with self.assertRaises(ValueError): resolve_checkpoint(Path(d)/'latest.json')
