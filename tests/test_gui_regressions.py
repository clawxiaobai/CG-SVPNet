"""GUI regressions; uses small temporary fixtures, never starts/stops training."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from forging.gui import Application
from forging.gui_history import read_history
from forging.settings import PROJECT


class GuiRegression(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.base=Path(self.temp.name)
        self.run=self.base/'train'
        self.run.mkdir()
        (self.run/'request.json').write_text(json.dumps(dict(model='E0',profile='paper',split='val',format='both')),encoding='utf-8')
        (self.run/'status.json').write_text(json.dumps(dict(stage='train',epoch=2,epochs=500,batch=50,batches=370)),encoding='utf-8')
        (self.run/'train_log.csv').write_text('epoch,train_total,disp_rmse,equiv_strain_rmse,effective_stress_rmse\n1,4.6,1.33,0.268,23.49\n2,2.7,1.08,0.263,\n',encoding='utf-8')
        (self.run/'worker.log').write_text('E001 batch=50/370 loss=4.5\nE003 train=2.31 validation=deferred\n',encoding='utf-8')
        self.results=self.base/'results'
        self.results.mkdir()
        (self.results/'request.json').write_text(json.dumps(dict(model='E2',profile='smoke')),encoding='utf-8')
        for case,step in [('case_06',1),('case_06',2),('case_09',3)]:
            directory=self.results/'predictions'/'val'/case/f'step_{step:06d}'
            directory.mkdir(parents=True)
            for field,color in [('displacement','red'),('strain','blue'),('stress','green')]:
                Image.new('RGB',(1200,360),color).save(directory/(field+'.png'))
        self.root=tk.Tk()
        self.root.geometry('1000x740+20+20')
        self.app=Application(self.root)
        self.root.geometry('1000x740+20+20')
        self.pump()

    def pump(self):
        deadline=time.monotonic()+.25
        while time.monotonic()<deadline:
            self.root.update();time.sleep(.01)

    def tearDown(self):
        for job in self.root.tk.splitlist(self.root.tk.call('after','info')):
            self.root.after_cancel(job)
        self.root.destroy();self.temp.cleanup()

    def test_partial_csv_one_point_and_skipped_validation_epoch(self):
        rows,epochs,batches=read_history(self.run)
        self.assertEqual(len(rows),2)
        self.assertEqual(epochs,[(1.,4.6),(2.,2.7),(3.,2.31)])
        self.app.draw_curves(rows,epochs,batches)
        self.pump()
        self.assertEqual(len(self.app.axes[1].lines[0].get_ydata()),2)
        self.assertEqual(list(self.app.axes[3].lines[0].get_ydata()),[23.49])
        self.assertEqual(self.app.axes[3].lines[0].get_marker(),'o')

    def test_selector_resize_clear_and_monitor_isolation(self):
        self.app.attach(self.run)
        self.app.set_preview_source(self.results)
        self.app.notebook.select(2);self.pump()
        self.assertEqual(len(self.app.case_box['values']),2)
        self.assertEqual(len(self.app.step_box['values']),2)
        self.assertEqual(self.app.selected_preview().name,'step_000001')
        self.app.next_button.invoke();self.pump()
        self.assertEqual(self.app.selected_preview().name,'step_000002')
        self.app.field.set('等效应力');self.app.show_preview()
        self.assertEqual(self.app.preview_key[0].name,'stress.png')
        self.assertLessEqual(self.app.photo.width(),self.app.preview.winfo_width())
        self.assertLessEqual(self.app.photo.height(),self.app.preview.winfo_height())
        self.assertIn('非精度结果',self.app.preview_note.get())
        self.app.case_choice.set('val / case_09');self.app.select_case()
        self.assertEqual(self.app.selected_preview().name,'step_000003')
        self.assertEqual(self.app.output,self.run)
        self.app.stop()
        self.assertFalse((self.run/'STOP').exists())
        self.assertEqual(str(self.app.train_button['state']),'disabled')
        self.app.set_preview_source(self.run)
        self.assertIsNone(self.app.photo)
        self.assertFalse(self.app.preview['image'])
        self.assertEqual(len(self.app.step_box['values']),0)
        self.assertIn('尚无',self.app.preview['text'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GuiRegression))
    report=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),training_started=False,training_stopped=False)
    (PROJECT/'qa/gui_regression_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    raise SystemExit(not result.wasSuccessful())
