"""Print environment and CPU/CUDA availability without starting training."""
import json
import platform
import sys
import tkinter
import numpy
import torch

print(json.dumps(dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,
    numpy=numpy.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available(),
    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    tkinter=tkinter.TkVersion),ensure_ascii=False,indent=2))
