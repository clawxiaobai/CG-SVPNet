"""Exercise the actual GUI data-check button and its background-worker lifecycle."""
import json
from pathlib import Path
import sys
import time
import tkinter as tk
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from forging.gui import Application
from forging.settings import PROJECT

root=tk.Tk();root.withdraw()
app=Application(root)
app.output_root.set(str(PROJECT/'qa/gui_check'))
app.check_button.invoke()
deadline=time.time()+180
while app.process is not None and time.time()<deadline:
    root.update();time.sleep(.1)
if app.process is not None:
    app.stop()
    raise RuntimeError('GUI worker timed out; stop requested')
root.update()
status=json.loads((app.output/'status.json').read_text(encoding='utf-8'))
assert status['stage']=='completed',status
assert str(app.train_button['state'])=='normal'
report=dict(data_check_button=True,background_process_completed=True,controls_reenabled=True,
            output=str(app.output.relative_to(PROJECT)),status=status)
(PROJECT/'qa/gui_test.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('GUI data selection, check action, progress/log polling and process completion passed.')
root.destroy()
