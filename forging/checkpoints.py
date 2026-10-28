"""Restricted loading for original checkpoints with NumPy scalar metrics."""
import numpy as np
import torch


def load_checkpoint(path, device='cpu'):
    # Original metrics contain NumPy scalars; allow only these fixed numeric
    # constructors, never arbitrary globals and never silently retry unsafe pickle.
    allowed=[np._core.multiarray.scalar,np.dtype,np.dtypes.Float64DType,
             np.dtypes.Float32DType,np.dtypes.Int64DType,np.dtypes.Int32DType]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path,map_location=device,weights_only=True)
