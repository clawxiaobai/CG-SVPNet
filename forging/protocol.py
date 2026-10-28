# Extracted scientific kernel; provenance: reference/source_provenance.json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dataset import MPMFirstFrameDataset
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    case_value = out["case_name"]
    case = case_value[0] if isinstance(case_value, (list, tuple)) else case_value
    scale = out["w_scale"].float().reshape(-1, 1)
    half_scale = scale * 0.5
    progress = out["target_condition"][:, 3:4].float().clamp(0.0, 1.0)
    return {
        "case": case,
        "pos": out["p_pos_norm"].float(),
        "target_disp": out["target_disp"].float() / half_scale[:, None, :],
        "current_punch": out["pc_t"].float() / half_scale,
        "target_punch": out["pn_t"].float() / half_scale,
        "raw_current_punch": out["pc_t"].float(),
        "raw_target_punch": out["pn_t"].float(),
        "center": out["w_center"].float(),
        "scale": scale,
        "progress": progress,
        "point_indices": out["point_indices"],
        "point_surface_focus": out.get("point_surface_focus"),
        "sample_key": out.get("sample_key"),
        "p_pos_raw": out["p_pos_raw"].float(),
        "p_pos_next": out["p_pos_next"].float(),
    }

STATE_FIELDS = ("effstn", "effsts")


STATE_NAMES = ("equiv_strain", "effective_stress")


def batch_one(raw):
    batch = {}
    for key, value in raw.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        elif key in ("case_name", "sample_key"):
            batch[key] = [value]
        else:
            batch[key] = value
    return batch


def move_state_batch(batch, device):
    state = move_batch(batch, device)
    state["target_state"] = batch["target_state"].to(
        device, non_blocking=True
    ).float()
    state["target_state_raw"] = batch["target_state_raw"].to(
        device, non_blocking=True
    ).float()
    state["target_state_scale"] = batch["target_state_scale"].to(
        device, non_blocking=True
    ).float()
    return state


def make_state_dataset(mode, args, train=False):
    max_points = args.max_points if train else args.eval_max_points
    if max_points <= 0:
        max_points = None
    return MPMFirstFrameDataset(
        str(args.data_root),
        mode=mode,
        phys_dim=6,
        max_points=max_points,
        num_source_frames=1,
        random_pair_ratio=0.0,
        point_sampling=args.sampling if train else "uniform",
        focus_ratio=args.focus_ratio,
        focus_quantile=args.focus_quantile,
        sampling_seed_offset=args.seed,
        sampling_seed_mode="sample" if train else "source",
        surface_root=str(args.surface_root),
        surface_ratio=args.surface_ratio,
        surface_contact_ratio=0.0,
        surface_feature_mode=False,
        geometry_root=str(args.geometry_root),
        geometry_feature_mode=False,
        minimal_io=True,
        target_state_mode=True,
        target_state_fields=STATE_FIELDS,
        sampling_repeats_per_sample=1,
        target_progress_min=0.0,
        target_progress_max=1.1,
        jitter_range=args.jitter if train else 0.0,
    )


def _new_accumulator():
    return {
        "count": 0,
        "state_sq": np.zeros(2, dtype=np.float64),
        "state_abs": np.zeros(2, dtype=np.float64),
        "state_target_sq": np.zeros(2, dtype=np.float64),
        "state_neg": np.zeros(2, dtype=np.int64),
        "disp_sq": 0.0,
        "disp_count": 0,
    }


def _update_accumulator(acc, pred_raw, target_raw, disp_diff_world):
    diff = pred_raw - target_raw
    acc["count"] += int(diff.shape[0] * diff.shape[1])
    acc["state_sq"] += diff.square().sum(dim=(0, 1)).double().cpu().numpy()
    acc["state_abs"] += diff.abs().sum(dim=(0, 1)).double().cpu().numpy()
    acc["state_target_sq"] += target_raw.square().sum(
        dim=(0, 1)
    ).double().cpu().numpy()
    acc["state_neg"] += (pred_raw < 0).sum(
        dim=(0, 1)
    ).long().cpu().numpy()
    acc["disp_sq"] += float(disp_diff_world.square().sum().cpu())
    acc["disp_count"] += int(disp_diff_world.numel())


def _finish_accumulator(acc, prefix=""):
    count = max(acc["count"], 1)
    result = {}
    for channel, name in enumerate(STATE_NAMES):
        result[f"{prefix}{name}_rmse"] = math.sqrt(
            acc["state_sq"][channel] / count
        )
        result[f"{prefix}{name}_mae"] = (
            acc["state_abs"][channel] / count
        )
        result[f"{prefix}{name}_zero_rmse"] = math.sqrt(
            acc["state_target_sq"][channel] / count
        )
        result[f"{prefix}{name}_negative_fraction"] = (
            acc["state_neg"][channel] / count
        )
    result[f"{prefix}disp_rmse"] = math.sqrt(
        acc["disp_sq"] / max(acc["disp_count"], 1)
    )
    return result

