"""Process-local progress and cancellation, separate from scientific kernels."""
import json
import os
from pathlib import Path
import time

OUTPUT = None
START = time.time()


class Cancelled(Exception):
    pass


def configure(output):
    global OUTPUT, START
    OUTPUT = Path(output)
    START = time.time()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    for attempt in range(10):
        try:
            os.replace(temp,path)
            return
        except PermissionError:
            time.sleep(.05)
    os.replace(temp,path)


def check_stop():
    if OUTPUT and (OUTPUT/'STOP').exists():
        raise Cancelled('Stopped by user at a safe batch boundary.')


def event(stage, **values):
    if OUTPUT:
        atomic_json(OUTPUT/'status.json',dict(stage=stage,elapsed_seconds=time.time()-START,**values))
