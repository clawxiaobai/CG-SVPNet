"""Native scatter-add replacement: avoids unrelated compiled extensions.

The active E0/E1/E2 path only scatters ones for binary occupancy. Integer counts
and occupancy match torch_scatter; see the archived equivalence tests.
"""
import torch


def scatter_add(src, index, dim=0, dim_size=None):
    if dim != 0 or index.ndim != 1:
        raise ValueError('This project supports the original dim=0, 1-D index use only.')
    size = int(index.max())+1 if dim_size is None and index.numel() else (dim_size or 0)
    result = src.new_zeros((size,)+src.shape[1:])
    return result.index_add_(0, index, src)
