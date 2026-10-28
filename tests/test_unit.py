"""Fast CPU tests: splits, strict checkpoint compatibility and VTU structure."""
import json
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from forging.model import CoinEquivalentStateModel
from forging.settings import SPLITS,PROJECT,MODES
from forging.scatter import scatter_add
from forging.vtk_io import write_vtu,write_pvd
from forging.checkpoints import load_checkpoint


class UnitTests(unittest.TestCase):
    def test_split(self):
        from forging.dataset import MPMFirstFrameDataset
        import ast,inspect
        tree=ast.parse(inspect.getsource(MPMFirstFrameDataset))
        actual=[ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Attribute) and t.attr=='split_map' for t in n.targets)][0]
        self.assertEqual(SPLITS,actual)
        documented=json.loads((PROJECT/'configs/dataset_split.json').read_text())
        self.assertEqual(SPLITS,{k:documented[k] for k in SPLITS})
        combined=sum(SPLITS.values(),[])
        self.assertEqual(len(combined),len(set(combined)))

    def test_scatter(self):
        ids=torch.tensor([0,0,2,1,2,2])
        values=torch.ones(6,1)
        self.assertEqual(scatter_add(values,ids,dim_size=4).flatten().tolist(),[2,1,3,0])

    def test_pretrained(self):
        for name,mode in MODES.items():
            saved=load_checkpoint(PROJECT/'pretrained'/name/'best_model.pth')
            model=CoinEquivalentStateModel(decoder_mode=mode)
            model.load_state_dict(saved['model_state_dict'],strict=True)
            self.assertEqual(model.net.out.out_channels,5 if name=='E0' else 96)

    def test_vtk(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.vtu'
            points=np.array([[0,1,2],[3,4,5]],dtype='f4')
            write_vtu(path,points,{'stress_MPa':np.array([1,2],dtype='f4'),'ids':np.array([0,1])})
            binary=path.read_bytes()
            self.assertIn(b'NumberOfPoints="2"',binary)
            self.assertIn(b'encoding="raw">_',binary)
            write_pvd(Path(d)/'sequence.pvd',[(.25,'test.vtu')])
            import xml.etree.ElementTree as ET
            self.assertEqual(ET.parse(Path(d)/'sequence.pvd').find('.//DataSet').attrib['timestep'],'0.25')


if __name__=='__main__':unittest.main(verbosity=2)
