# Extracted scientific kernel; provenance: reference/source_provenance.json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, glob, re, json
from torch.utils.data import Dataset
MPM_FEATURE_SCALES = {
    "effstn": 0.2,     # 1/5
    "effsts": 0.02,    # 1/50
    "displx": 0.0667,  # 1/15
    "disply": 0.0667,
    "displz": 0.0667,
    "pttemp": 0.002,   # 1/500
}


MPM_STATE_TARGET_SCALES = {
    "effstn": 0.2,
    "effsts": 0.02,
}


def build_target_condition(meta, current_translation, target_translation):
    """Return normalized target punch travel (xyz) plus case progress."""
    current_translation = np.asarray(current_translation, dtype=np.float32)
    target_translation = np.asarray(target_translation, dtype=np.float32)
    delta = target_translation - current_translation

    half_scale = max(float(meta.get('world_scale', 1.0)) * 0.5, 1e-6)
    delta_norm = delta / half_scale

    translations = meta.get('punch_translations', {})
    if translations:
        values = np.asarray(list(translations.values()), dtype=np.float32)
        reference = values[0]
        max_travel = float(np.linalg.norm(values - reference, axis=1).max())
    else:
        max_travel = float(np.linalg.norm(delta))
    progress = np.linalg.norm(delta) / max(max_travel, 1e-6)

    return np.concatenate([
        delta_norm,
        np.asarray([progress], dtype=np.float32),
    ]).astype(np.float32)


def get_final_punch_translation(meta):
    translations = meta.get('punch_translations', {})
    step_ids = meta.get('step_ids', [])
    if step_ids and str(step_ids[-1]) in translations:
        return np.asarray(translations[str(step_ids[-1])], dtype=np.float32)
    if translations:
        return np.asarray(list(translations.values())[-1], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


class MPMFirstFrameDataset(Dataset):
    """
    First-frame-to-any-frame dataset for MPM forging.

    Each sample pairs a source frame (always among the first few steps of
    each case) with a subsequent target frame. This matches the prediction
    use case where the model predicts from step_00001 to any later step.

    Parameters
    ----------
    processed_root : str
    mode : str — 'train', 'val', 'test', or 'all'
    phys_dim : int — physical feature channels (default 6)
    max_points : int or None — point subsampling cap per sample
    max_step : int — maximum step index
    jitter_range : float — spatial jitter for augmentation (default 1.0)
    num_source_frames : int — number of early frames used as sources (default 1)
    random_pair_ratio : float — fraction of samples that are random pairs (default 0.0)
    random_pair_weight : float — loss weight for random-pair samples (default 0.3)
    """
    def __init__(self, processed_root, mode='train',
                 phys_dim=6, max_points=20000, max_step=1000,
                 jitter_range=1.0, num_source_frames=1,
                 random_pair_ratio=0.0, random_pair_weight=0.3,
                 point_sampling='uniform', focus_ratio=0.3,
                 focus_quantile=0.9, sampling_seed_offset=0,
                 sampling_seed_mode='sample',
                 contact_ratio=0.3, contact_band_ratio=0.2,
                 contact_quantile=0.8, contact_band_quantile=0.2,
                 surface_root=None, surface_ratio=0.2,
                 surface_contact_ratio=0.1,
                 surface_feature_mode=False,
                 surface_contac_scale=20.0,
                 surface_fricwk_scale=1.0,
                 surface_only=False,
                 geometry_root=None,
                 geometry_feature_mode=False,
                 sampling_repeats_per_sample=1,
                 late_sampling_start=1.1,
                 late_sampling_extra_repeats=0,
                 final_sampling_start=1.1,
                 final_sampling_extra_repeats=0,
                 target_progress_min=0.0,
                 target_progress_max=1.1,
                 final_focus_ratio=0.13,
                 remaining_focus_ratio=0.13,
                 current_focus_ratio=None,
                 late_final_focus_ratio=0.04,
                 late_final_window=5,
                 late_region_focus_ratio=0.04,
                 late_region_radius=10.0,
                 late_region_y_half=5.0,
                 late_region_centers_xz=None,
                 late_region_axisymmetric=True,
                 mold_sdf_focus_ratio=0.04,
                 mold_sdf_grid_res=64,
                 mold_sdf_tau=3.0,
                 mold_curvature_weight=0.5,
                 mold_penetration_weight=1.0,
                 contact_corner_near_pool=600000,
                 minimal_io=False,
                 target_state_mode=False,
                 target_state_fields=('effstn', 'effsts'),
                 case_subset=None):
        self.processed_root = processed_root
        self.mode = mode
        self.phys_dim = phys_dim
        self.minimal_io = bool(minimal_io)
        self.target_state_mode = bool(target_state_mode)
        self.target_state_fields = tuple(target_state_fields)
        if self.target_state_mode and self.target_state_fields != (
            'effstn', 'effsts'
        ):
            raise ValueError(
                "target_state_fields currently must be ('effstn', 'effsts')"
            )
        self.max_points = max_points
        self.max_step = max_step
        self.jitter_range = jitter_range
        self.num_source_frames = num_source_frames
        self.random_pair_ratio = random_pair_ratio
        self.random_pair_weight = random_pair_weight
        self.point_sampling = point_sampling
        self.focus_ratio = focus_ratio
        self.focus_quantile = focus_quantile
        self.sampling_seed_offset = int(sampling_seed_offset)
        self.sampling_seed_mode = sampling_seed_mode
        self._focus_pool_cache = {}
        self.contact_ratio = float(contact_ratio)
        self.contact_band_ratio = float(contact_band_ratio)
        self.contact_quantile = float(contact_quantile)
        self.contact_band_quantile = float(contact_band_quantile)
        self._sdf_asset_cache = {}
        self._contact_pool_cache = {}
        self.surface_root = surface_root
        self.surface_ratio = float(surface_ratio)
        self.surface_contact_ratio = float(surface_contact_ratio)
        self.surface_feature_mode = bool(surface_feature_mode)
        self.surface_contac_scale = float(surface_contac_scale)
        self.surface_fricwk_scale = float(surface_fricwk_scale)
        self.surface_only = bool(surface_only)
        self.geometry_root = geometry_root
        self.geometry_feature_mode = bool(geometry_feature_mode)
        self._surface_index_cache = {}
        self._surface_step_cache = {}
        self._surface_feature_cache = {}
        self._geometry_feature_cache = {}
        self._final_contact_high_corner_cache = {}
        self._late_final_motion_cache = {}
        self.sampling_repeats_per_sample = max(
            1, int(sampling_repeats_per_sample)
        )
        self.late_sampling_start = float(late_sampling_start)
        self.late_sampling_extra_repeats = max(
            0, int(late_sampling_extra_repeats)
        )
        self.final_sampling_start = float(final_sampling_start)
        self.final_sampling_extra_repeats = max(
            0, int(final_sampling_extra_repeats)
        )
        self.target_progress_min = float(target_progress_min)
        self.target_progress_max = float(target_progress_max)
        self.final_focus_ratio = float(final_focus_ratio)
        self.remaining_focus_ratio = float(remaining_focus_ratio)
        self.current_focus_ratio = (
            None if current_focus_ratio is None else float(current_focus_ratio)
        )
        self.late_final_focus_ratio = float(late_final_focus_ratio)
        self.late_final_window = max(1, int(late_final_window))
        self.late_region_focus_ratio = float(late_region_focus_ratio)
        self.late_region_radius = float(late_region_radius)
        self.late_region_y_half = float(late_region_y_half)
        if late_region_centers_xz is None:
            late_region_centers_xz = ((75.0, 0.0), (-75.0, 0.0))
        self.late_region_centers_xz = tuple(
            (float(x), float(z)) for x, z in late_region_centers_xz
        )
        self.late_region_axisymmetric = bool(late_region_axisymmetric)
        self.mold_sdf_focus_ratio = float(mold_sdf_focus_ratio)
        self.mold_sdf_grid_res = max(2, int(mold_sdf_grid_res))
        self.mold_sdf_tau = max(1e-6, float(mold_sdf_tau))
        self.mold_curvature_weight = float(mold_curvature_weight)
        self.mold_penetration_weight = float(mold_penetration_weight)
        self.contact_corner_near_pool = max(
            1, int(contact_corner_near_pool)
        )

        self.split_map = {
            'train': ['case_01', 'case_02', 'case_03', 'case_04',
                      'case_07', 'case_08', 'case_10', 'case_11',
                      'case_12', 'case_13', 'case_14',
                      'case_18', 'case_19', 'case_20', 'case_21'],
            'val':   ['case_09', 'case_06', 'case_15'],
            'test':  ['case_05', 'case_16', 'case_17'],
        }

        if case_subset is not None:
            if isinstance(case_subset, str):
                case_subset = [case_subset]
            self.target_cases = [
                c if str(c).startswith('case_') else f"case_{int(c):02d}"
                for c in case_subset
            ]
        elif mode == 'all':
            dirs = [d for d in os.listdir(processed_root)
                    if os.path.isdir(os.path.join(processed_root, d))]
            self.target_cases = sorted(dirs)
        else:
            self.target_cases = self.split_map.get(mode, [])

        self.samples = []
        self.case_metas = {}

        for case_name in self.target_cases:
            case_path = os.path.join(processed_root, case_name)
            meta_path = os.path.join(case_path, 'metadata.json')
            if not os.path.exists(meta_path):
                print(f"  [SKIP] {case_name}: no metadata.json")
                continue

            with open(meta_path, 'r') as f:
                self.case_metas[case_name] = json.load(f)

            step_files = sorted(glob.glob(os.path.join(case_path, 'step_*.npz')))
            step_map = {}
            point_counts = {}

            for f in step_files:
                match = re.search(r'step_(\d+)', os.path.basename(f))
                if not match:
                    continue
                step_id = int(match.group(1))
                if step_id >= self.max_step:
                    continue
                try:
                    data = np.load(f)
                    point_counts[step_id] = data['points'].shape[0]
                    data.close()
                except Exception as exc:
                    print(f"  [SKIP_BAD_NPZ] {f}: {exc}")
                    continue
                step_map[step_id] = f

            if not step_map:
                continue

            sorted_steps = sorted(step_map.keys())
            num_steps = len(sorted_steps)

            # ── Primary: first N frames → all later frames ────────────
            max_src = min(self.num_source_frames, num_steps - 1)
            first_frame_samples = []

            for src_pos in range(max_src):
                src_step = sorted_steps[src_pos]
                src_npts = point_counts[src_step]
                compatible_steps = [
                    s for s in sorted_steps
                    if point_counts.get(s, -1) == src_npts
                ]
                if not compatible_steps or compatible_steps[-1] <= src_step:
                    continue
                final_valid_step = compatible_steps[-1]
                late_ref_valid_steps = compatible_steps[
                    max(0, len(compatible_steps) - 1
                        - self.late_final_window):
                    max(0, len(compatible_steps) - 1)
                ]
                for tgt_pos in range(src_pos + 1, num_steps):
                    denom = max(1, num_steps - 1)
                    progress_pos = (tgt_pos - src_pos) / denom
                    if (
                        progress_pos < self.target_progress_min
                        or progress_pos > self.target_progress_max
                    ):
                        continue
                    tgt_step = sorted_steps[tgt_pos]
                    if point_counts[tgt_step] != src_npts:
                        continue
                    first_frame_samples.append({
                        'case_name': case_name,
                        'curr_path': step_map[src_step],
                        'target_path': step_map[tgt_step],
                        'final_path': step_map[final_valid_step],
                        'late_final_ref_path': step_map[
                            late_ref_valid_steps[0]
                            if late_ref_valid_steps else final_valid_step
                        ],
                        'late_final_ref_paths': [
                            step_map[s]
                            for s in late_ref_valid_steps
                        ],
                        'prev_path': step_map[
                            sorted_steps[max(src_pos, tgt_pos - 1)]
                        ],
                        'physical_step_diff': tgt_step - src_step,
                        'index_stride': tgt_pos - src_pos,
                        'prev_index_stride': max(0, tgt_pos - 1 - src_pos),
                        'target_progress_pos': float(progress_pos),
                        'displacement_norm': 0.0,
                        'sample_weight': 1.0,
                    })

            expanded_first_frame_samples = []
            for sample in first_frame_samples:
                progress_pos = float(sample.get('target_progress_pos', 0.0))
                repeat_count = self.sampling_repeats_per_sample
                if self.mode == 'train':
                    if progress_pos >= self.late_sampling_start:
                        repeat_count += self.late_sampling_extra_repeats
                    if progress_pos >= self.final_sampling_start:
                        repeat_count += self.final_sampling_extra_repeats
                for repeat in range(repeat_count):
                    repeated = dict(sample)
                    repeated['sampling_repeat'] = repeat
                    repeated['target_progress_pos'] = float(progress_pos)
                    expanded_first_frame_samples.append(repeated)
            self.samples.extend(expanded_first_frame_samples)

            # ── Optional: random pairs for regularization ──────────────
            if self.random_pair_ratio > 0 and num_steps > 2:
                n_ff = len(first_frame_samples)
                if n_ff > 0:
                    n_random = int(n_ff * self.random_pair_ratio
                                   / max(1 - self.random_pair_ratio, 1e-6))
                else:
                    n_random = max(1, int(num_steps * 0.1))

                for _ in range(n_random):
                    src_pos = random.randint(0, num_steps - 2)
                    tgt_pos = random.randint(src_pos + 1, num_steps - 1)
                    src_step = sorted_steps[src_pos]
                    tgt_step = sorted_steps[tgt_pos]
                    if point_counts[tgt_step] != point_counts[src_step]:
                        continue
                    src_npts = point_counts[src_step]
                    compatible_steps = [
                        s for s in sorted_steps
                        if point_counts.get(s, -1) == src_npts
                    ]
                    if not compatible_steps or compatible_steps[-1] <= src_step:
                        continue
                    final_valid_step = compatible_steps[-1]
                    late_ref_valid_steps = compatible_steps[
                        max(0, len(compatible_steps) - 1
                            - self.late_final_window):
                        max(0, len(compatible_steps) - 1)
                    ]
                    for repeat in range(self.sampling_repeats_per_sample):
                        self.samples.append({
                            'case_name': case_name,
                            'curr_path': step_map[src_step],
                            'target_path': step_map[tgt_step],
                            'final_path': step_map[final_valid_step],
                            'late_final_ref_path': step_map[
                                late_ref_valid_steps[0]
                                if late_ref_valid_steps else final_valid_step
                            ],
                            'late_final_ref_paths': [
                                step_map[s]
                                for s in late_ref_valid_steps
                            ],
                            'prev_path': step_map[
                                sorted_steps[max(src_pos, tgt_pos - 1)]
                            ],
                            'physical_step_diff': tgt_step - src_step,
                            'index_stride': tgt_pos - src_pos,
                            'prev_index_stride': max(0, tgt_pos - 1 - src_pos),
                            'displacement_norm': 0.0,
                            'sample_weight': self.random_pair_weight,
                            'sampling_repeat': repeat,
                        })

        # Shuffle
        random.shuffle(self.samples)

        n_ff = sum(1 for s in self.samples if s.get('sample_weight', 1.0) >= 0.99)
        n_rp = len(self.samples) - n_ff
        print(f"Dataset [{mode.upper()}] | {len(self.samples)} samples "
              f"({n_ff} first-frame, {n_rp} random-pair) | "
              f"cases: {len(self.target_cases)}")

    def __len__(self):
        return len(self.samples)

    def _get_die_translation(self, case_name, step_id):
        meta = self.case_metas.get(case_name, {})
        translations = meta.get('die_translations', {})
        key = str(step_id)
        if key in translations:
            return np.array(translations[key], dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    def _get_punch_translation(self, case_name, step_id):
        meta = self.case_metas.get(case_name, {})
        translations = meta.get('punch_translations', {})
        key = str(step_id)
        if key in translations:
            return np.array(translations[key], dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    def _load_sdf_asset_np(self, case_name, part):
        key = (case_name, part)
        cached = self._sdf_asset_cache.get(key)
        if cached is not None:
            return cached
        path = os.path.join(
            self.processed_root, case_name, f'{part}_asset.npz'
        )
        data = np.load(path)
        asset = {
            'sdf': data['sdf'].astype(np.float32, copy=False),
            'curvature': data['curvature'].astype(np.float32, copy=False),
            'center': data['center'].astype(np.float32, copy=False),
            'scale': float(data['scale']),
        }
        self._sdf_asset_cache[key] = asset
        return asset

    def _load_contact_asset_np(self, case_name, part):
        key = (case_name, f'{part}_contact')
        cached = self._sdf_asset_cache.get(key)
        if cached is not None:
            return cached
        path = os.path.join(
            self.processed_root, case_name, f'{part}_contact_asset.npz'
        )
        data = np.load(path)
        distance = data['distance'] if 'distance' in data.files else data['sdf']
        asset = {
            # Use non-negative projected distance as the stable contact field.
            'distance': distance.astype(np.float32, copy=False),
            # Keep sdf alias for compatibility with generic samplers.
            'sdf': distance.astype(np.float32, copy=False),
            'grad': data['grad'].astype(np.float32, copy=False),
            'curvature': data['curvature'].astype(np.float32, copy=False),
            'center': data['center'].astype(np.float32, copy=False),
            'scale': float(data['scale']),
        }
        self._sdf_asset_cache[key] = asset
        return asset

    @staticmethod
    def _sample_volume_np(points, asset, translation, field='sdf'):
        vol = asset[field]
        res = vol.shape[0]
        coords = (
            points
            - (asset['center'][None, :] + translation[None, :])
        ) / (asset['scale'] / 2.0 + 1e-8)
        idx = (coords + 1.0) * 0.5 * (res - 1)
        idx = np.clip(idx, 0.0, res - 1.0)
        i0 = np.floor(idx).astype(np.int32)
        i1 = np.clip(i0 + 1, 0, res - 1)
        w = idx - i0
        x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
        x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
        wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
        c000 = vol[x0, y0, z0]
        c100 = vol[x1, y0, z0]
        c010 = vol[x0, y1, z0]
        c110 = vol[x1, y1, z0]
        c001 = vol[x0, y0, z1]
        c101 = vol[x1, y0, z1]
        c011 = vol[x0, y1, z1]
        c111 = vol[x1, y1, z1]
        c00 = c000 * (1 - wx) + c100 * wx
        c10 = c010 * (1 - wx) + c110 * wx
        c01 = c001 * (1 - wx) + c101 * wx
        c11 = c011 * (1 - wx) + c111 * wx
        c0 = c00 * (1 - wy) + c10 * wy
        c1 = c01 * (1 - wy) + c11 * wy
        return (c0 * (1 - wz) + c1 * wz).astype(np.float32)

    @staticmethod
    def _sample_sdf_np(points, asset, translation):
        return MPMFirstFrameDataset._sample_volume_np(
            points, asset, translation, field='sdf'
        )

    @staticmethod
    def _sample_vector_volume_np(points, asset, translation, field='grad'):
        vol = asset[field]
        if vol.ndim != 4:
            raise ValueError(f"Expected vector volume (C,R,R,R), got {vol.shape}")
        comps = []
        for c in range(vol.shape[0]):
            tmp = {
                'value': vol[c],
                'center': asset['center'],
                'scale': asset['scale'],
            }
            comps.append(MPMFirstFrameDataset._sample_volume_np(
                points, tmp, translation, field='value'
            ))
        return np.stack(comps, axis=1).astype(np.float32, copy=False)

    def _get_contact_pools(self, case_name, curr_step, target_step, points):
        key = (
            case_name, int(curr_step), int(target_step),
            float(self.contact_quantile), float(self.contact_band_quantile),
        )
        cached = self._contact_pool_cache.get(key)
        if cached is not None:
            return cached

        die = self._load_sdf_asset_np(case_name, 'die')
        punch = self._load_sdf_asset_np(case_name, 'punch')
        die_t = self._get_die_translation(case_name, curr_step).astype(np.float32)
        pc_t = self._get_punch_translation(case_name, curr_step).astype(np.float32)
        pn_t = self._get_punch_translation(case_name, target_step).astype(np.float32)
        pf_t = get_final_punch_translation(self.case_metas[case_name]).astype(np.float32)

        sdf_die = self._sample_sdf_np(points, die, die_t)
        sdf_pc = self._sample_sdf_np(points, punch, pc_t)
        sdf_pn = self._sample_sdf_np(points, punch, pn_t)
        sdf_pf = self._sample_sdf_np(points, punch, pf_t)

        sdf_init = np.minimum(sdf_die, sdf_pc)
        sdf_now = np.minimum(sdf_die, sdf_pn)
        sdf_final = np.minimum(sdf_die, sdf_pf)
        penetration_now = np.maximum(0.0, -sdf_now)
        penetration_final = np.maximum(0.0, -sdf_final)
        sweep_drop = np.maximum(0.0, sdf_init - sdf_now)
        risk = penetration_now + 0.5 * penetration_final + 0.25 * sweep_drop
        risk_threshold = np.quantile(risk, self.contact_quantile)
        risk_pool = np.flatnonzero(risk >= risk_threshold)

        band_distance = np.abs(sdf_now)
        band_threshold = np.quantile(band_distance, self.contact_band_quantile)
        band_pool = np.flatnonzero(band_distance <= band_threshold)

        result = (risk_pool, band_pool)
        self._contact_pool_cache[key] = result
        return result

    def _final_mold_voxel_score(self, case_name, final_points,
                                motion_score=None):
        """Score particles by final-position voxel center mold SDF/curvature.

        This is a training-only importance sampler.  Final particle positions
        are mapped to a coarse world grid; each occupied voxel center queries
        final die/punch SDF and curvature.  The voxel score is assigned back to
        all particles in that voxel.
        """
        meta = self.case_metas[case_name]
        w_center = np.asarray(meta['world_center'], dtype=np.float32)
        w_scale = float(meta['world_scale'])
        res = int(self.mold_sdf_grid_res)
        p_norm = (final_points.astype(np.float32, copy=False)
                  - w_center[None, :]) / (w_scale / 2.0 + 1e-8)
        coords = np.floor((p_norm + 1.0) * 0.5 * (res - 1)).astype(np.int32)
        coords = np.clip(coords, 0, res - 1)
        flat = coords[:, 0] * (res ** 2) + coords[:, 1] * res + coords[:, 2]
        unique_flat, inverse = np.unique(flat, return_inverse=True)
        ux = unique_flat // (res ** 2)
        uy = (unique_flat // res) % res
        uz = unique_flat % res
        center_norm = np.stack([ux, uy, uz], axis=1).astype(np.float32)
        center_norm = center_norm / max(res - 1, 1) * 2.0 - 1.0
        center_world = center_norm * (w_scale / 2.0) + w_center[None, :]

        die = self._load_sdf_asset_np(case_name, 'die')
        punch = self._load_sdf_asset_np(case_name, 'punch')
        die_t = self._get_die_translation(case_name, 1).astype(np.float32)
        pf_t = get_final_punch_translation(meta).astype(np.float32)

        sdf_die = self._sample_volume_np(center_world, die, die_t, 'sdf')
        sdf_punch = self._sample_volume_np(center_world, punch, pf_t, 'sdf')
        curv_die = self._sample_volume_np(
            center_world, die, die_t, 'curvature'
        )
        curv_punch = self._sample_volume_np(
            center_world, punch, pf_t, 'curvature'
        )
        use_die = np.abs(sdf_die) <= np.abs(sdf_punch)
        sdf = np.where(use_die, sdf_die, sdf_punch)
        curv = np.where(use_die, curv_die, curv_punch)

        # Importance sampling should emphasize the contact/filling *band*,
        # not deep negative SDF.  A previous version added max(0, -sdf),
        # which selected points far inside the mold solid / punch volume and
        # produced visibly wrong focus points.  Use |SDF| as a narrow-band
        # gate, then modulate by local mold curvature and optional late/final
        # particle motion.
        near = np.exp(-((np.abs(sdf) / self.mold_sdf_tau) ** 2))
        abs_curv = np.abs(curv)
        curv_scale = np.quantile(abs_curv, 0.95) + 1e-6
        curv_score = np.clip(abs_curv / curv_scale, 0.0, 2.0)
        voxel_score = (
            near * (1.0 + self.mold_curvature_weight * curv_score)
        ).astype(np.float32)
        particle_score = voxel_score[inverse]
        if motion_score is not None:
            motion_score = np.asarray(motion_score, dtype=np.float32)
            if motion_score.shape[0] == final_points.shape[0]:
                motion_scale = np.quantile(motion_score, 0.95) + 1e-6
                motion_norm = np.clip(motion_score / motion_scale, 0.0, 2.0)
                particle_score = particle_score * (0.5 + motion_norm)
        return particle_score.astype(np.float32, copy=False)

    def _final_contact_gradient_score(
        self, case_name, final_points, motion_score=None, motion_vector=None
    ):
        """Score final particles using corrected contact distance + gradient.

        This sampler uses the new open-surface contact assets.  Particles are
        mapped to a coarse world grid, each occupied voxel center queries final
        die/punch projected contact distance and nearest-cell normal.  The
        returned score prefers:

          1. small final contact distance (near-band),
          2. large late/final motion,
          3. motion direction aligned with the contact normal/gradient.

        The sign of the local half-space distance is intentionally not used as
        a global penetration indicator.
        """
        meta = self.case_metas[case_name]
        w_center = np.asarray(meta['world_center'], dtype=np.float32)
        w_scale = float(meta['world_scale'])
        res = int(self.mold_sdf_grid_res)
        p_norm = (
            final_points.astype(np.float32, copy=False) - w_center[None, :]
        ) / (w_scale / 2.0 + 1e-8)
        coords = np.floor((p_norm + 1.0) * 0.5 * (res - 1)).astype(np.int32)
        coords = np.clip(coords, 0, res - 1)
        flat = coords[:, 0] * (res ** 2) + coords[:, 1] * res + coords[:, 2]
        unique_flat, inverse = np.unique(flat, return_inverse=True)
        ux = unique_flat // (res ** 2)
        uy = (unique_flat // res) % res
        uz = unique_flat % res
        center_norm = np.stack([ux, uy, uz], axis=1).astype(np.float32)
        center_norm = center_norm / max(res - 1, 1) * 2.0 - 1.0
        center_world = center_norm * (w_scale / 2.0) + w_center[None, :]

        die = self._load_contact_asset_np(case_name, 'die')
        punch = self._load_contact_asset_np(case_name, 'punch')
        die_t = self._get_die_translation(case_name, 1).astype(np.float32)
        pf_t = get_final_punch_translation(meta).astype(np.float32)

        dist_die = self._sample_volume_np(center_world, die, die_t, 'distance')
        dist_punch = self._sample_volume_np(center_world, punch, pf_t, 'distance')
        grad_die = self._sample_vector_volume_np(center_world, die, die_t, 'grad')
        grad_punch = self._sample_vector_volume_np(center_world, punch, pf_t, 'grad')
        use_die = dist_die <= dist_punch
        dist = np.where(use_die, dist_die, dist_punch).astype(np.float32)
        grad = np.where(use_die[:, None], grad_die, grad_punch).astype(np.float32)
        grad = grad / (np.linalg.norm(grad, axis=1, keepdims=True) + 1e-8)

        near = np.exp(-((dist / self.mold_sdf_tau) ** 2)).astype(np.float32)
        voxel_score = near
        particle_score = voxel_score[inverse]
        particle_grad = grad[inverse]

        if motion_score is not None:
            motion_score = np.asarray(motion_score, dtype=np.float32)
            if motion_score.shape[0] == final_points.shape[0]:
                motion_scale = np.quantile(motion_score, 0.95) + 1e-6
                motion_norm = np.clip(motion_score / motion_scale, 0.0, 2.0)
                particle_score = particle_score * (0.5 + motion_norm)

        if motion_vector is not None:
            motion_vector = np.asarray(motion_vector, dtype=np.float32)
            if motion_vector.shape == final_points.shape:
                mv_norm = np.linalg.norm(motion_vector, axis=1, keepdims=True)
                mv_dir = motion_vector / (mv_norm + 1e-8)
                # Surface normal orientation can be arbitrary for open contact
                # surfaces, so use absolute alignment.
                align = np.abs(np.sum(mv_dir * particle_grad, axis=1))
                align = np.where(mv_norm[:, 0] > 1e-6, align, 0.0)
                particle_score = particle_score * (0.5 + align.astype(np.float32))

        return particle_score.astype(np.float32, copy=False)

    def _final_contact_corner_score(self, case_name, final_points):
        """Score final particles by final contact-surface corner criticality.

        Uses only final billet positions and final mold contact geometry:

          - projected distance to nearest die/final-punch contact surface;
          - local contact normal variation around the point.

        This intentionally does not use late-step displacement magnitude.  It
        targets final formed geometry corners/turning regions where contact
        surface normal changes quickly and details were under-sampled before.
        """
        meta = self.case_metas[case_name]
        w_center = np.asarray(meta['world_center'], dtype=np.float32)
        w_scale = float(meta['world_scale'])
        res = int(self.mold_sdf_grid_res)
        p_norm = (
            final_points.astype(np.float32, copy=False) - w_center[None, :]
        ) / (w_scale / 2.0 + 1e-8)
        coords = np.floor((p_norm + 1.0) * 0.5 * (res - 1)).astype(np.int32)
        coords = np.clip(coords, 0, res - 1)
        flat = coords[:, 0] * (res ** 2) + coords[:, 1] * res + coords[:, 2]
        unique_flat, inverse = np.unique(flat, return_inverse=True)
        ux = unique_flat // (res ** 2)
        uy = (unique_flat // res) % res
        uz = unique_flat % res
        center_norm = np.stack([ux, uy, uz], axis=1).astype(np.float32)
        center_norm = center_norm / max(res - 1, 1) * 2.0 - 1.0
        center_world = center_norm * (w_scale / 2.0) + w_center[None, :]

        die = self._load_contact_asset_np(case_name, 'die')
        punch = self._load_contact_asset_np(case_name, 'punch')
        die_t = self._get_die_translation(case_name, 1).astype(np.float32)
        pf_t = get_final_punch_translation(meta).astype(np.float32)

        dist_die = self._sample_volume_np(center_world, die, die_t, 'distance')
        dist_punch = self._sample_volume_np(center_world, punch, pf_t, 'distance')
        grad_die = self._sample_vector_volume_np(center_world, die, die_t, 'grad')
        grad_punch = self._sample_vector_volume_np(center_world, punch, pf_t, 'grad')
        use_die = dist_die <= dist_punch
        dist = np.where(use_die, dist_die, dist_punch).astype(np.float32)
        grad = np.where(use_die[:, None], grad_die, grad_punch).astype(np.float32)
        grad = grad / (np.linalg.norm(grad, axis=1, keepdims=True) + 1e-8)

        # Estimate local normal variation by sampling final mold normals around
        # each occupied world voxel.  High variation corresponds to edge/corner
        # or surface-turning regions.
        eps = max(w_scale / max(res - 1, 1), 1e-3)
        corner = np.zeros(center_world.shape[0], dtype=np.float32)
        for axis in range(3):
            off = np.zeros(3, dtype=np.float32)
            off[axis] = eps
            p_plus = center_world + off[None, :]
            p_minus = center_world - off[None, :]

            gd_p = self._sample_vector_volume_np(p_plus, die, die_t, 'grad')
            gp_p = self._sample_vector_volume_np(p_plus, punch, pf_t, 'grad')
            dd_p = self._sample_volume_np(p_plus, die, die_t, 'distance')
            dp_p = self._sample_volume_np(p_plus, punch, pf_t, 'distance')
            use_d_p = dd_p <= dp_p
            g_p = np.where(use_d_p[:, None], gd_p, gp_p)
            g_p = g_p / (np.linalg.norm(g_p, axis=1, keepdims=True) + 1e-8)

            gd_m = self._sample_vector_volume_np(p_minus, die, die_t, 'grad')
            gp_m = self._sample_vector_volume_np(p_minus, punch, pf_t, 'grad')
            dd_m = self._sample_volume_np(p_minus, die, die_t, 'distance')
            dp_m = self._sample_volume_np(p_minus, punch, pf_t, 'distance')
            use_d_m = dd_m <= dp_m
            g_m = np.where(use_d_m[:, None], gd_m, gp_m)
            g_m = g_m / (np.linalg.norm(g_m, axis=1, keepdims=True) + 1e-8)

            corner = np.maximum(
                corner,
                np.linalg.norm(g_p - g_m, axis=1).astype(np.float32),
            )

        near = np.exp(-((dist / self.mold_sdf_tau) ** 2)).astype(np.float32)
        corner_scale = np.quantile(corner, 0.95) + 1e-6
        corner_norm = np.clip(corner / corner_scale, 0.0, 2.0)
        voxel_score = near * (0.25 + corner_norm)
        particle_score = voxel_score[inverse].astype(np.float32, copy=False)

        # The raw corner score tends to pick the outer/bottom boundary of the
        # die because it is also a sharp contact-surface turn.  For the forging
        # accuracy problem here, the critical missing details are the final
        # filled cavity/shoulder turns, not the already well-defined outside
        # perimeter.  Use soft gates derived from the final billet distribution
        # itself: de-emphasize the outer radial tail and the lowest z tail.
        r_xy = np.sqrt(final_points[:, 0] ** 2 + final_points[:, 1] ** 2)
        r_cut = np.quantile(r_xy, 0.70)
        z_cut = np.quantile(final_points[:, 2], 0.20)
        radial_gate = 1.0 / (1.0 + np.exp((r_xy - r_cut) / 3.0))
        z_gate = 1.0 / (1.0 + np.exp((z_cut - final_points[:, 2]) / 2.0))
        particle_score = particle_score * radial_gate.astype(np.float32)
        particle_score = particle_score * z_gate.astype(np.float32)
        return particle_score.astype(np.float32, copy=False)

    def _final_contact_high_corner_score(self, case_name, final_points):
        """Score final particles by high contact-normal variation near contact.

        This implements the accepted diagnostic strategy:

          1. query final die/final-punch projected contact distance;
          2. keep only the nearest ``contact_corner_near_pool`` particles;
          3. inside that near-distance pool, rank by local mold-normal
             variation.

        Points outside the near-distance pool receive ``-inf`` so the generic
        top-k picker cannot select them unless the requested quota exceeds the
        candidate pool.
        """
        cache_key = (
            case_name,
            int(final_points.shape[0]),
            int(self.mold_sdf_grid_res),
            int(self.contact_corner_near_pool),
        )
        cached = self._final_contact_high_corner_cache.get(cache_key)
        if cached is not None:
            return cached

        meta = self.case_metas[case_name]
        w_center = np.asarray(meta['world_center'], dtype=np.float32)
        w_scale = float(meta['world_scale'])
        res = int(self.mold_sdf_grid_res)
        p_norm = (
            final_points.astype(np.float32, copy=False) - w_center[None, :]
        ) / (w_scale / 2.0 + 1e-8)
        coords = np.floor((p_norm + 1.0) * 0.5 * (res - 1)).astype(np.int32)
        coords = np.clip(coords, 0, res - 1)
        flat = coords[:, 0] * (res ** 2) + coords[:, 1] * res + coords[:, 2]
        unique_flat, inverse = np.unique(flat, return_inverse=True)
        ux = unique_flat // (res ** 2)
        uy = (unique_flat // res) % res
        uz = unique_flat % res
        center_norm = np.stack([ux, uy, uz], axis=1).astype(np.float32)
        center_norm = center_norm / max(res - 1, 1) * 2.0 - 1.0
        center_world = center_norm * (w_scale / 2.0) + w_center[None, :]

        die = self._load_contact_asset_np(case_name, 'die')
        punch = self._load_contact_asset_np(case_name, 'punch')
        die_t = self._get_die_translation(case_name, 1).astype(np.float32)
        pf_t = get_final_punch_translation(meta).astype(np.float32)

        dist_die = self._sample_volume_np(center_world, die, die_t, 'distance')
        dist_punch = self._sample_volume_np(center_world, punch, pf_t, 'distance')
        dist_voxel = np.minimum(dist_die, dist_punch).astype(np.float32)

        eps = max(w_scale / max(res - 1, 1), 1e-3)
        corner = np.zeros(center_world.shape[0], dtype=np.float32)
        for axis in range(3):
            off = np.zeros(3, dtype=np.float32)
            off[axis] = eps
            p_plus = center_world + off[None, :]
            p_minus = center_world - off[None, :]

            gd_p = self._sample_vector_volume_np(p_plus, die, die_t, 'grad')
            gp_p = self._sample_vector_volume_np(p_plus, punch, pf_t, 'grad')
            dd_p = self._sample_volume_np(p_plus, die, die_t, 'distance')
            dp_p = self._sample_volume_np(p_plus, punch, pf_t, 'distance')
            use_d_p = dd_p <= dp_p
            g_p = np.where(use_d_p[:, None], gd_p, gp_p)
            g_p = g_p / (np.linalg.norm(g_p, axis=1, keepdims=True) + 1e-8)

            gd_m = self._sample_vector_volume_np(p_minus, die, die_t, 'grad')
            gp_m = self._sample_vector_volume_np(p_minus, punch, pf_t, 'grad')
            dd_m = self._sample_volume_np(p_minus, die, die_t, 'distance')
            dp_m = self._sample_volume_np(p_minus, punch, pf_t, 'distance')
            use_d_m = dd_m <= dp_m
            g_m = np.where(use_d_m[:, None], gd_m, gp_m)
            g_m = g_m / (np.linalg.norm(g_m, axis=1, keepdims=True) + 1e-8)

            corner = np.maximum(
                corner,
                np.linalg.norm(g_p - g_m, axis=1).astype(np.float32),
            )

        dist = dist_voxel[inverse].astype(np.float32, copy=False)
        variation = corner[inverse].astype(np.float32, copy=False)
        near_n = min(self.contact_corner_near_pool, final_points.shape[0])
        if near_n >= final_points.shape[0]:
            near_pool = np.arange(final_points.shape[0], dtype=np.int64)
        else:
            near_pool = np.argpartition(dist, near_n - 1)[:near_n]
        score = np.full(final_points.shape[0], -np.inf, dtype=np.float32)
        score[near_pool] = variation[near_pool]
        self._final_contact_high_corner_cache[cache_key] = score
        return score

    def _get_surface_index(self, case_name):
        if not self.surface_root:
            return None
        cached = self._surface_index_cache.get(case_name)
        if cached is not None:
            return cached
        path = os.path.join(
            self.surface_root, case_name, 'surface_index.npz'
        )
        if not os.path.exists(path):
            return None
        data = np.load(path)
        full_index = data['full_index'].astype(np.int64, copy=False)
        self._surface_index_cache[case_name] = full_index
        data.close()
        return full_index

    def _get_surface_contact_pool(self, case_name, target_step):
        if not self.surface_root:
            return None
        key = (case_name, int(target_step))
        cached = self._surface_step_cache.get(key)
        if cached is not None:
            return cached
        surface_index = self._get_surface_index(case_name)
        if surface_index is None:
            return None
        path = os.path.join(
            self.surface_root, case_name,
            f'surface_step_{int(target_step):06d}.npz',
        )
        if not os.path.exists(path):
            return None
        data = np.load(path)
        if 'contac' in data:
            contact_mask = data['contac'].astype(np.float32, copy=False) > 0
            pool = surface_index[contact_mask]
            if pool.size == 0:
                pool = surface_index
        else:
            pool = surface_index
        pool = pool.astype(np.int64, copy=False)
        self._surface_step_cache[key] = pool
        data.close()
        return pool

    def _get_surface_features(self, case_name, target_step, n_total):
        """Return full-particle aligned surface/contact/friction features.

        Channels:
          0. is_surface: first-frame surface membership.
          1. target surface contac value, written only on surface particles.
          2. target surface fricwk value, written only on surface particles.
        """
        if not self.surface_root:
            return None
        key = (case_name, int(target_step), int(n_total))
        cached = self._surface_feature_cache.get(key)
        if cached is not None:
            return cached

        surface_index = self._get_surface_index(case_name)
        if surface_index is None:
            return None

        feats = np.zeros((n_total, 3), dtype=np.float32)
        valid = surface_index[
            (surface_index >= 0) & (surface_index < int(n_total))
        ]
        feats[valid, 0] = 1.0

        path = os.path.join(
            self.surface_root, case_name,
            f'surface_step_{int(target_step):06d}.npz',
        )
        if os.path.exists(path):
            data = np.load(path)
            step_index = data.get('full_index', surface_index)
            step_index = step_index.astype(np.int64, copy=False)
            valid_mask = (step_index >= 0) & (step_index < int(n_total))
            step_valid = step_index[valid_mask]
            if 'contac' in data:
                contac = data['contac'].astype(np.float32, copy=False)
                feats[step_valid, 1] = (
                    contac[valid_mask] * self.surface_contac_scale
                )
            if 'fricwk' in data:
                fricwk = data['fricwk'].astype(np.float32, copy=False)
                feats[step_valid, 2] = (
                    fricwk[valid_mask] * self.surface_fricwk_scale
                )
            data.close()

        self._surface_feature_cache[key] = feats
        return feats

    def _get_geometry_features(self, case_name, n_total):
        """Return full-particle aligned initial blank geometry features.

        Channels:
          0 blank_signed_dist / 20, clipped to [-1, 1]
          1 blank_unsigned_dist / 20, clipped to [0, 1]
          2 blank_edge_dist / 80, clipped to [0, 1]
          3:6 nearest initial blank surface normal
          6 is initial surface point
          7 isUp from varSurfMatpoints
          8 idDie from varSurfMatpoints
        """
        if not self.geometry_root:
            return None
        cached = self._geometry_feature_cache.get(case_name)
        if cached is not None:
            return cached
        path = os.path.join(
            self.geometry_root, case_name, 'blank_features.npz'
        )
        if not os.path.exists(path):
            return None
        data = np.load(path)
        signed = data['blank_signed_dist'].astype(np.float32, copy=False)
        unsigned = data['blank_unsigned_dist'].astype(np.float32, copy=False)
        edge = data['blank_edge_dist'].astype(np.float32, copy=False)
        normal = data['blank_normal'].astype(np.float32, copy=False)
        is_surface = data['blank_is_surface'].astype(np.float32, copy=False)
        is_up = data['blank_is_up'].astype(np.float32, copy=False)
        id_die = data['blank_id_die'].astype(np.float32, copy=False)
        if signed.shape[0] != int(n_total):
            data.close()
            raise RuntimeError(
                f"Geometry feature count mismatch for {case_name}: "
                f"{signed.shape[0]} vs {n_total}"
            )
        feats = np.concatenate([
            np.clip(signed[:, None] / 20.0, -1.0, 1.0),
            np.clip(unsigned[:, None] / 20.0, 0.0, 1.0),
            np.clip(edge[:, None] / 80.0, 0.0, 1.0),
            normal,
            is_surface[:, None],
            is_up[:, None],
            id_die[:, None],
        ], axis=1).astype(np.float32, copy=False)
        data.close()
        self._geometry_feature_cache[case_name] = feats
        return feats

    def __getitem__(self, idx):
        sample = self.samples[idx]
        case_name = sample['case_name']
        meta = self.case_metas[case_name]

        w_center = torch.tensor(meta['world_center'], dtype=torch.float32)
        w_scale  = torch.tensor(meta['world_scale'],  dtype=torch.float32)

        d_curr = np.load(sample['curr_path'])
        d_next = np.load(sample['target_path'])
        curr_points = np.asarray(d_curr['points'], dtype=np.float32)
        next_points = np.asarray(d_next['points'], dtype=np.float32)
        if self.minimal_io:
            d_prev = None
            prev_points = curr_points
        else:
            d_prev = np.load(sample.get('prev_path', sample['curr_path']))
            prev_points = np.asarray(d_prev['points'], dtype=np.float32)

        curr_step = int(d_curr.get('step_id', 0))
        target_step = int(d_next.get('step_id', 0))
        prev_step = (
            curr_step if d_prev is None
            else int(d_prev.get('step_id', curr_step))
        )

        p_pos = torch.from_numpy(curr_points).float()

        # Point subsampling
        n_total = p_pos.shape[0]
        if self.surface_only:
            surface_pool = self._get_surface_index(case_name)
            if surface_pool is None:
                raise RuntimeError(
                    f"surface_only=True but no surface_index for {case_name}"
                )
            surface_pool = surface_pool[
                (surface_pool >= 0) & (surface_pool < n_total)
            ].astype(np.int64, copy=False)
            if surface_pool.size == 0:
                raise RuntimeError(
                    f"surface_only=True but empty surface_index for {case_name}"
                )
            if self.sampling_seed_mode == 'source':
                source_key = (
                    sum(ord(ch) for ch in case_name) * 1009
                    + curr_step * 9176
                )
            else:
                source_key = idx * 137 + curr_step
            seed = (
                source_key
                + self.sampling_seed_offset
                + int(sample.get('sampling_repeat', 0)) * 1000003
            ) % (2 ** 32 - 1)
            rng = np.random.RandomState(seed)
            focus_members = np.zeros(n_total, dtype=bool)
            contact_members = np.zeros(n_total, dtype=bool)
            surface_members = np.zeros(n_total, dtype=bool)
            late_final_members = np.zeros(n_total, dtype=bool)
            surface_members[surface_pool] = True

            target_count = (
                min(self.max_points, surface_pool.size)
                if self.max_points is not None else surface_pool.size
            )
            if target_count < surface_pool.size:
                if self.mode == 'train' and self.point_sampling in (
                    'deformation', 'surface_only_deformation',
                    'surface_deformation',
                ):
                    disp_mag = np.linalg.norm(
                        next_points[surface_pool]
                        - curr_points[surface_pool],
                        axis=1,
                    )
                    threshold = np.quantile(disp_mag, self.focus_quantile)
                    focus_pool = surface_pool[disp_mag >= threshold]
                    focus_count = min(
                        int(round(target_count * self.focus_ratio)),
                        len(focus_pool),
                    )
                    focus_indices = (
                        rng.choice(focus_pool, focus_count, replace=False)
                        if focus_count > 0 else np.empty(0, dtype=np.int64)
                    )
                    focus_members[focus_indices] = True
                    available = np.setdiff1d(
                        surface_pool, focus_indices, assume_unique=False
                    )
                    uniform_count = target_count - focus_indices.shape[0]
                    uniform_indices = (
                        rng.choice(available, uniform_count, replace=False)
                        if uniform_count > 0 else np.empty(0, dtype=np.int64)
                    )
                    indices_np = np.concatenate([
                        uniform_indices, focus_indices
                    ])
                else:
                    indices_np = rng.choice(
                        surface_pool, target_count, replace=False
                    )
                rng.shuffle(indices_np)
            else:
                indices_np = surface_pool.copy()
            surface_contact_pool = self._get_surface_contact_pool(
                case_name, target_step
            )
            if surface_contact_pool is not None:
                contact_members[surface_contact_pool] = True
            indices = torch.from_numpy(indices_np).long()
            p_pos = p_pos[indices]
            p_pos_next_full = torch.from_numpy(next_points).float()
            p_pos_prev_full = torch.from_numpy(prev_points).float()
            p_pos_next = p_pos_next_full[indices]
            p_pos_prev = p_pos_prev_full[indices]
            del p_pos_next_full
            del p_pos_prev_full
            _subsample = True
            point_focus = torch.zeros(
                len(indices_np), 1, dtype=torch.float32
            )
            point_focus[:, 0] = torch.from_numpy(
                focus_members[indices_np].astype(np.float32)
            )
            point_contact_focus = torch.zeros(
                len(indices_np), 1, dtype=torch.float32
            )
            point_contact_focus[:, 0] = torch.from_numpy(
                contact_members[indices_np].astype(np.float32)
            )
            point_surface_focus = torch.ones(
                len(indices_np), 1, dtype=torch.float32
            )
            point_late_final_focus = torch.zeros(
                len(indices_np), 1, dtype=torch.float32
            )
        elif self.max_points is not None and n_total > self.max_points:
            if self.sampling_seed_mode == 'source':
                source_key = (
                    sum(ord(ch) for ch in case_name) * 1009
                    + curr_step * 9176
                )
            else:
                source_key = idx * 137 + curr_step
            seed = (
                source_key
                + self.sampling_seed_offset
                + int(sample.get('sampling_repeat', 0)) * 1000003
            ) % (2 ** 32 - 1)
            rng = np.random.RandomState(seed)
            focus_count = 0
            focus_members = np.zeros(n_total, dtype=bool)
            contact_members = np.zeros(n_total, dtype=bool)
            surface_members = np.zeros(n_total, dtype=bool)
            late_final_members = np.zeros(n_total, dtype=bool)
            if (
                self.mode == 'train'
                and self.point_sampling == 'contact_deformation'
            ):
                disp_mag = np.linalg.norm(
                    next_points - curr_points, axis=1
                )
                threshold = np.quantile(disp_mag, self.focus_quantile)
                focus_pool = np.flatnonzero(disp_mag >= threshold)
                focus_count = min(
                    int(round(self.max_points * self.focus_ratio)),
                    len(focus_pool),
                )
                focus_indices = rng.choice(
                    focus_pool, focus_count, replace=False
                )
                focus_members[focus_indices] = True

                risk_pool, band_pool = self._get_contact_pools(
                    case_name, curr_step, target_step, curr_points
                )
                risk_count = min(
                    int(round(self.max_points * self.contact_ratio)),
                    len(risk_pool),
                )
                risk_indices = rng.choice(
                    risk_pool, risk_count, replace=False
                )
                band_count = min(
                    int(round(self.max_points * self.contact_band_ratio)),
                    len(band_pool),
                )
                band_indices = rng.choice(
                    band_pool, band_count, replace=False
                )
                contact_members[risk_indices] = True
                contact_members[band_indices] = True

                selected = np.unique(np.concatenate([
                    focus_indices, risk_indices, band_indices
                ]))
                remaining_count = self.max_points - selected.shape[0]
                if remaining_count > 0:
                    available = np.ones(n_total, dtype=bool)
                    available[selected] = False
                    uniform_pool = np.flatnonzero(available)
                    uniform_indices = rng.choice(
                        uniform_pool, remaining_count, replace=False
                    )
                    indices_np = np.concatenate([selected, uniform_indices])
                else:
                    indices_np = rng.choice(
                        selected, self.max_points, replace=False
                    )
                rng.shuffle(indices_np)
            elif (
                self.mode == 'train'
                and self.point_sampling == 'all_surface_deformation'
            ):
                surface_pool = self._get_surface_index(case_name)
                if surface_pool is None:
                    surface_pool = np.empty(0, dtype=np.int64)
                surface_pool = surface_pool[
                    (surface_pool >= 0) & (surface_pool < n_total)
                ].astype(np.int64, copy=False)
                surface_members[surface_pool] = True
                if self.point_sampling == 'review_stratified_40_20_20_20':
                    surface_count = min(
                        int(round(self.max_points * self.surface_ratio)),
                        surface_pool.size,
                    )
                    if surface_pool.size > surface_count:
                        surface_pool = rng.choice(
                            surface_pool, surface_count, replace=False
                        ).astype(np.int64, copy=False)

                if surface_pool.size >= self.max_points:
                    indices_np = rng.choice(
                        surface_pool, self.max_points, replace=False
                    )
                    rng.shuffle(indices_np)
                else:
                    disp_mag = np.linalg.norm(
                        next_points - curr_points, axis=1
                    )
                    threshold = np.quantile(disp_mag, self.focus_quantile)
                    focus_pool = np.flatnonzero(disp_mag >= threshold)
                    if surface_pool.size:
                        surface_mask = np.zeros(n_total, dtype=bool)
                        surface_mask[surface_pool] = True
                        focus_pool = focus_pool[~surface_mask[focus_pool]]
                    focus_count = min(
                        int(round(self.max_points * self.focus_ratio)),
                        len(focus_pool),
                        self.max_points - surface_pool.size,
                    )
                    focus_indices = (
                        rng.choice(focus_pool, focus_count, replace=False)
                        if focus_count > 0 else np.empty(0, dtype=np.int64)
                    )
                    focus_members[focus_indices] = True

                    selected = np.unique(np.concatenate([
                        surface_pool, focus_indices
                    ]))
                    remaining_count = self.max_points - selected.shape[0]
                    if remaining_count > 0:
                        available = np.ones(n_total, dtype=bool)
                        available[selected] = False
                        uniform_pool = np.flatnonzero(available)
                        uniform_indices = rng.choice(
                            uniform_pool, remaining_count, replace=False
                        )
                        indices_np = np.concatenate([
                            selected, uniform_indices
                        ])
                    else:
                        indices_np = selected
                    rng.shuffle(indices_np)
            elif (
                self.mode == 'train'
                and self.point_sampling in (
                    'all_surface_deformation_latefinal',
                    'all_surface_deformation_late_region',
                    'all_surface_deformation_mold_sdf_curv',
                    'all_surface_deformation_contact_grad',
                    'all_surface_deformation_final_corner',
                    'all_surface_deformation_contact_high_corner',
                    'review_stratified_40_20_20_20',
                )
            ):
                surface_pool = self._get_surface_index(case_name)
                if surface_pool is None:
                    surface_pool = np.empty(0, dtype=np.int64)
                surface_pool = surface_pool[
                    (surface_pool >= 0) & (surface_pool < n_total)
                ].astype(np.int64, copy=False)
                surface_members[surface_pool] = True
                if self.point_sampling == 'review_stratified_40_20_20_20':
                    surface_count = min(
                        int(round(self.max_points * self.surface_ratio)),
                        surface_pool.size,
                    )
                    if surface_pool.size > surface_count:
                        surface_pool = rng.choice(
                            surface_pool, surface_count, replace=False
                        ).astype(np.int64, copy=False)

                if surface_pool.size >= self.max_points:
                    indices_np = rng.choice(
                        surface_pool, self.max_points, replace=False
                    )
                    rng.shuffle(indices_np)
                else:
                    current_disp_mag = np.linalg.norm(
                        next_points - curr_points, axis=1
                    )

                    final_path = sample.get(
                        'final_path', sample['target_path']
                    )
                    ref_paths = list(sample.get('late_final_ref_paths') or [])
                    if not ref_paths and sample.get('late_final_ref_path'):
                        ref_paths = [sample['late_final_ref_path']]
                    motion_cache_key = (
                        str(sample['curr_path']), str(final_path),
                        tuple(map(str, ref_paths)),
                    )
                    cached_motion = self._late_final_motion_cache.get(
                        motion_cache_key
                    )
                    if cached_motion is not None:
                        final_points, late_final_score, late_final_vector = (
                            cached_motion
                        )
                    else:
                        d_final = np.load(final_path)
                        final_points = np.asarray(
                            d_final['points'], dtype=np.float32
                        )
                        late_final_score = np.zeros(
                            n_total, dtype=np.float32
                        )
                        late_final_vector = np.zeros(
                            (n_total, 3), dtype=np.float32
                        )
                        valid_ref_paths = []
                        valid_weights = []
                        if ref_paths:
                            raw_weights = np.arange(
                                1, len(ref_paths) + 1, dtype=np.float32
                            )
                            for path, weight in zip(ref_paths, raw_weights):
                                d_ref = np.load(path)
                                ref_points = d_ref['points']
                                if ref_points.shape == final_points.shape:
                                    valid_ref_paths.append(path)
                                    valid_weights.append(float(weight))
                                d_ref.close()
                        if valid_ref_paths:
                            weight_sum = float(np.sum(valid_weights))
                            for path, weight in zip(
                                valid_ref_paths, valid_weights
                            ):
                                d_ref = np.load(path)
                                ref_points = d_ref['points']
                                late_final_score += (
                                    float(weight)
                                    * np.linalg.norm(
                                        final_points - ref_points, axis=1
                                    ).astype(np.float32, copy=False)
                                )
                                late_final_vector += (
                                    float(weight)
                                    * (final_points - ref_points).astype(
                                        np.float32, copy=False
                                    )
                                )
                                d_ref.close()
                            late_final_score /= max(weight_sum, 1.0)
                            late_final_vector /= max(weight_sum, 1.0)
                        else:
                            late_final_vector = (
                                final_points - curr_points
                            ).astype(np.float32, copy=False)
                            late_final_score = np.linalg.norm(
                                late_final_vector, axis=1
                            ).astype(np.float32, copy=False)
                        d_final.close()
                        self._late_final_motion_cache[motion_cache_key] = (
                            final_points, late_final_score,
                            late_final_vector,
                        )

                    blocked = np.zeros(n_total, dtype=bool)
                    if surface_pool.size:
                        blocked[surface_pool] = True
                    selected_parts = [surface_pool]

                    def pick_top_pool(score, count, already_blocked):
                        count = min(int(count), self.max_points)
                        if count <= 0:
                            return np.empty(0, dtype=np.int64)
                        pool = np.flatnonzero(~already_blocked)
                        if pool.size == 0:
                            return np.empty(0, dtype=np.int64)
                        count = min(count, pool.size)
                        if pool.size <= count:
                            chosen = pool
                        else:
                            local = score[pool]
                            kth = pool.size - count
                            part = np.argpartition(local, kth)[kth:]
                            chosen = pool[part]
                        if chosen.size > count:
                            chosen = rng.choice(chosen, count, replace=False)
                        return chosen.astype(np.int64, copy=False)

                    # Reserve the late/final-focused quota before filling the
                    # rest with current-step deformation points.  Otherwise,
                    # "all surface + deformation" can consume the entire
                    # max_points budget and silently squeeze out the SDF/curv
                    # focus points that this sampling mode is meant to add.
                    remaining_focus_budget = max(
                        0,
                        self.max_points
                        - np.unique(np.concatenate(selected_parts)).size,
                    )
                    if (
                        self.point_sampling
                        == 'all_surface_deformation_mold_sdf_curv'
                    ):
                        final_score = self._final_mold_voxel_score(
                            case_name, final_points,
                            motion_score=late_final_score,
                        )
                        late_count = min(
                            int(round(
                                self.max_points * self.mold_sdf_focus_ratio
                            )),
                            remaining_focus_budget,
                        )
                        late_indices = pick_top_pool(
                            final_score, late_count, blocked
                        )
                    elif (
                        self.point_sampling
                        == 'all_surface_deformation_contact_grad'
                    ):
                        final_score = self._final_contact_gradient_score(
                            case_name, final_points,
                            motion_score=late_final_score,
                            motion_vector=late_final_vector,
                        )
                        late_count = min(
                            int(round(
                                self.max_points * self.mold_sdf_focus_ratio
                            )),
                            remaining_focus_budget,
                        )
                        late_indices = pick_top_pool(
                            final_score, late_count, blocked
                        )
                    elif (
                        self.point_sampling
                        == 'all_surface_deformation_final_corner'
                    ):
                        final_score = self._final_contact_corner_score(
                            case_name, final_points,
                        )
                        late_count = min(
                            int(round(
                                self.max_points * self.mold_sdf_focus_ratio
                            )),
                            remaining_focus_budget,
                        )
                        late_indices = pick_top_pool(
                            final_score, late_count, blocked
                        )
                    elif (
                        self.point_sampling
                        in (
                            'all_surface_deformation_contact_high_corner',
                            'review_stratified_40_20_20_20',
                        )
                    ):
                        final_score = self._final_contact_high_corner_score(
                            case_name, final_points,
                        )
                        late_count = min(
                            int(round(
                                self.max_points * self.mold_sdf_focus_ratio
                            )),
                            remaining_focus_budget,
                        )
                        late_indices = pick_top_pool(
                            final_score, late_count, blocked
                        )
                    elif (
                        self.point_sampling
                        == 'all_surface_deformation_late_region'
                    ):
                        final_score = None
                        late_count = min(
                            int(round(
                                self.max_points * self.late_region_focus_ratio
                            )),
                            remaining_focus_budget,
                        )
                        final_x = final_points[:, 0]
                        final_y = final_points[:, 1]
                        final_z = final_points[:, 2]
                        if self.late_region_axisymmetric:
                            # Rotate the XZ-section target points (±R, z0)
                            # around the z-axis.  This forms a torus-like
                            # tube: (sqrt(x^2+y^2)-R)^2 + (z-z0)^2 <= r^2.
                            radial_xy = np.sqrt(
                                final_x ** 2 + final_y ** 2
                            ).astype(np.float32, copy=False)
                            region_dist2 = np.full(
                                n_total, np.inf, dtype=np.float32
                            )
                            for cx, cz in self.late_region_centers_xz:
                                target_r = abs(float(cx))
                                dist2 = (
                                    (radial_xy - target_r) ** 2
                                    + (final_z - cz) ** 2
                                ).astype(np.float32, copy=False)
                                region_dist2 = np.minimum(
                                    region_dist2, dist2
                                )
                            region_mask = (
                                region_dist2 <= self.late_region_radius ** 2
                            )
                        else:
                            # Non-axisymmetric diagnostic fallback: two
                            # cylinders through y at XZ centers.
                            region_mask = (
                                np.abs(final_y) <= self.late_region_y_half
                            )
                            region_dist2 = np.full(
                                n_total, np.inf, dtype=np.float32
                            )
                            for cx, cz in self.late_region_centers_xz:
                                dist2 = (
                                    (final_x - cx) ** 2
                                    + (final_z - cz) ** 2
                                ).astype(np.float32, copy=False)
                                region_dist2 = np.minimum(
                                    region_dist2, dist2
                                )
                            region_mask &= (
                                region_dist2
                                <= self.late_region_radius ** 2
                            )
                        region_mask &= ~blocked
                        region_pool = np.flatnonzero(region_mask)
                        if (
                            not self.late_region_axisymmetric
                            and region_pool.size < late_count
                        ):
                            # If the strict middle slice has too few points,
                            # relax only the y-band while keeping the same XZ
                            # target circles.
                            relaxed = (
                                region_dist2 <= self.late_region_radius ** 2
                            )
                            relaxed &= ~blocked
                            region_pool = np.flatnonzero(relaxed)
                        if region_pool.size <= late_count:
                            late_indices = region_pool.astype(
                                np.int64, copy=False
                            )
                        else:
                            # Prefer the points deepest inside the target
                            # circles, and break ties with late-final motion.
                            local_score = (
                                -region_dist2[region_pool]
                                / max(self.late_region_radius ** 2, 1e-6)
                                + 0.01 * late_final_score[region_pool]
                            )
                            kth = region_pool.size - late_count
                            part = np.argpartition(local_score, kth)[kth:]
                            late_indices = region_pool[part].astype(
                                np.int64, copy=False
                            )
                    else:
                        final_score = None
                        late_count = min(
                            int(round(
                                self.max_points * self.late_final_focus_ratio
                            )),
                            remaining_focus_budget,
                        )
                        late_indices = pick_top_pool(
                            late_final_score, late_count, blocked
                        )
                    if late_indices.size:
                        focus_members[late_indices] = True
                        late_final_members[late_indices] = True
                        selected_parts.append(late_indices)
                        blocked[late_indices] = True

                    if (
                        self.point_sampling
                        in (
                            'all_surface_deformation_contact_high_corner',
                            'review_stratified_40_20_20_20',
                        )
                    ):
                        remaining_focus_budget = max(
                            0,
                            self.max_points
                            - np.unique(np.concatenate(selected_parts)).size,
                        )
                        bigdisp_count = min(
                            int(round(
                                self.max_points * self.late_final_focus_ratio
                            )),
                            remaining_focus_budget,
                        )
                        bigdisp_indices = pick_top_pool(
                            late_final_score, bigdisp_count, blocked
                        )
                        if bigdisp_indices.size:
                            focus_members[bigdisp_indices] = True
                            late_final_members[bigdisp_indices] = True
                            selected_parts.append(bigdisp_indices)
                            blocked[bigdisp_indices] = True

                    adaptive_cache = getattr(self, 'adaptive_error_cache', None)
                    adaptive_ratio = float(
                        getattr(self, 'adaptive_sampling_ratio', 0.0) or 0.0
                    )
                    adaptive_active = bool(
                        getattr(self, 'adaptive_sampling_active', True)
                    )
                    if (
                        adaptive_active
                        and adaptive_cache
                        and adaptive_ratio > 0.0
                    ):
                        sample_key = f"{case_name}|{curr_step}|{target_step}"
                        adaptive_pool = adaptive_cache.get(sample_key)
                        if adaptive_pool is not None:
                            adaptive_pool = np.asarray(
                                adaptive_pool, dtype=np.int64
                            )
                            adaptive_pool = adaptive_pool[
                                (adaptive_pool >= 0)
                                & (adaptive_pool < n_total)
                                & (~blocked[adaptive_pool])
                            ]
                            if adaptive_pool.size:
                                remaining_focus_budget = max(
                                    0,
                                    self.max_points
                                    - np.unique(
                                        np.concatenate(selected_parts)
                                    ).size,
                                )
                                adaptive_count = min(
                                    int(round(
                                        self.max_points * adaptive_ratio
                                    )),
                                    remaining_focus_budget,
                                    adaptive_pool.size,
                                )
                                if adaptive_count > 0:
                                    if adaptive_pool.size > adaptive_count:
                                        adaptive_indices = rng.choice(
                                            adaptive_pool,
                                            adaptive_count,
                                            replace=False,
                                        )
                                    else:
                                        adaptive_indices = adaptive_pool
                                    focus_members[adaptive_indices] = True
                                    selected_parts.append(adaptive_indices)
                                    blocked[adaptive_indices] = True

                    remaining_focus_budget = max(
                        0,
                        self.max_points
                        - np.unique(np.concatenate(selected_parts)).size,
                    )
                    current_count = min(
                        int(round(self.max_points * self.focus_ratio)),
                        remaining_focus_budget,
                    )
                    current_indices = pick_top_pool(
                        current_disp_mag, current_count, blocked
                    )
                    if current_indices.size:
                        focus_members[current_indices] = True
                        selected_parts.append(current_indices)
                        blocked[current_indices] = True

                    selected = np.unique(np.concatenate(selected_parts))
                    remaining_count = self.max_points - selected.shape[0]
                    if remaining_count > 0:
                        available = np.ones(n_total, dtype=bool)
                        available[selected] = False
                        uniform_pool = np.flatnonzero(available)
                        uniform_indices = rng.choice(
                            uniform_pool, remaining_count, replace=False
                        )
                        indices_np = np.concatenate([
                            selected, uniform_indices
                        ])
                    else:
                        indices_np = selected
                    rng.shuffle(indices_np)
            elif (
                self.mode == 'train'
                and self.point_sampling == 'all_surface_finalaware'
            ):
                surface_pool = self._get_surface_index(case_name)
                if surface_pool is None:
                    surface_pool = np.empty(0, dtype=np.int64)
                surface_pool = surface_pool[
                    (surface_pool >= 0) & (surface_pool < n_total)
                ].astype(np.int64, copy=False)
                surface_members[surface_pool] = True
                if self.point_sampling == 'review_stratified_40_20_20_20':
                    surface_count = min(
                        int(round(self.max_points * self.surface_ratio)),
                        surface_pool.size,
                    )
                    if surface_pool.size > surface_count:
                        surface_pool = rng.choice(
                            surface_pool, surface_count, replace=False
                        ).astype(np.int64, copy=False)

                if surface_pool.size >= self.max_points:
                    indices_np = rng.choice(
                        surface_pool, self.max_points, replace=False
                    )
                    rng.shuffle(indices_np)
                else:
                    d_final = np.load(
                        sample.get('final_path', sample['target_path'])
                    )
                    final_points = d_final['points']
                    current_disp_mag = np.linalg.norm(
                        next_points - curr_points, axis=1
                    )
                    final_disp_mag = np.linalg.norm(
                        final_points - curr_points, axis=1
                    )
                    remaining_mag = np.linalg.norm(
                        final_points - next_points, axis=1
                    )
                    d_final.close()

                    blocked = np.zeros(n_total, dtype=bool)
                    if surface_pool.size:
                        blocked[surface_pool] = True
                    selected_parts = [surface_pool]
                    remaining_budget = self.max_points - surface_pool.size

                    def pick_top_pool(score, ratio, already_blocked):
                        count = min(
                            int(round(self.max_points * ratio)),
                            max(0, remaining_budget),
                        )
                        if count <= 0:
                            return np.empty(0, dtype=np.int64)
                        pool = np.flatnonzero(~already_blocked)
                        if pool.size == 0:
                            return np.empty(0, dtype=np.int64)
                        if pool.size <= count:
                            chosen = pool
                        else:
                            # Top-k without sorting the full million-point array.
                            local = score[pool]
                            kth = pool.size - count
                            part = np.argpartition(local, kth)[kth:]
                            chosen = pool[part]
                        if chosen.size > count:
                            chosen = rng.choice(chosen, count, replace=False)
                        return chosen.astype(np.int64, copy=False)

                    for score, ratio in (
                        (final_disp_mag, self.final_focus_ratio),
                        (remaining_mag, self.remaining_focus_ratio),
                        (
                            current_disp_mag,
                            self.focus_ratio if self.current_focus_ratio is None
                            else self.current_focus_ratio,
                        ),
                    ):
                        chosen = pick_top_pool(score, ratio, blocked)
                        if chosen.size:
                            focus_members[chosen] = True
                            selected_parts.append(chosen)
                            blocked[chosen] = True
                            remaining_budget = (
                                self.max_points
                                - np.unique(np.concatenate(selected_parts)).size
                            )

                    selected = np.unique(np.concatenate(selected_parts))
                    remaining_count = self.max_points - selected.shape[0]
                    if remaining_count > 0:
                        available = np.ones(n_total, dtype=bool)
                        available[selected] = False
                        uniform_pool = np.flatnonzero(available)
                        uniform_indices = rng.choice(
                            uniform_pool, remaining_count, replace=False
                        )
                        indices_np = np.concatenate([
                            selected, uniform_indices
                        ])
                    else:
                        indices_np = selected
                    rng.shuffle(indices_np)
            elif (
                self.mode == 'train'
                and self.point_sampling == 'surface_deformation'
            ):
                disp_mag = np.linalg.norm(
                    next_points - curr_points, axis=1
                )
                threshold = np.quantile(disp_mag, self.focus_quantile)
                focus_pool = np.flatnonzero(disp_mag >= threshold)
                focus_count = min(
                    int(round(self.max_points * self.focus_ratio)),
                    len(focus_pool),
                )
                focus_indices = rng.choice(
                    focus_pool, focus_count, replace=False
                )
                focus_members[focus_indices] = True

                surface_pool = self._get_surface_index(case_name)
                if surface_pool is None:
                    surface_pool = np.empty(0, dtype=np.int64)
                surface_count = min(
                    int(round(self.max_points * self.surface_ratio)),
                    len(surface_pool),
                )
                surface_indices = (
                    rng.choice(surface_pool, surface_count, replace=False)
                    if surface_count > 0 else np.empty(0, dtype=np.int64)
                )

                surface_contact_pool = self._get_surface_contact_pool(
                    case_name, target_step
                )
                if surface_contact_pool is None:
                    surface_contact_pool = np.empty(0, dtype=np.int64)
                surface_contact_count = min(
                    int(round(
                        self.max_points * self.surface_contact_ratio
                    )),
                    len(surface_contact_pool),
                )
                surface_contact_indices = (
                    rng.choice(
                        surface_contact_pool, surface_contact_count,
                        replace=False,
                    )
                    if surface_contact_count > 0
                    else np.empty(0, dtype=np.int64)
                )

                surface_members[surface_indices] = True
                surface_members[surface_contact_indices] = True
                contact_members[surface_contact_indices] = True

                selected = np.unique(np.concatenate([
                    focus_indices, surface_indices, surface_contact_indices
                ]))
                remaining_count = self.max_points - selected.shape[0]
                if remaining_count > 0:
                    available = np.ones(n_total, dtype=bool)
                    available[selected] = False
                    uniform_pool = np.flatnonzero(available)
                    uniform_indices = rng.choice(
                        uniform_pool, remaining_count, replace=False
                    )
                    indices_np = np.concatenate([selected, uniform_indices])
                else:
                    indices_np = rng.choice(
                        selected, self.max_points, replace=False
                    )
                rng.shuffle(indices_np)
            elif self.mode == 'train' and self.point_sampling == 'deformation':
                focus_key = (
                    sample['curr_path'],
                    sample['target_path'],
                    float(self.focus_quantile),
                )
                focus_pool = self._focus_pool_cache.get(focus_key)
                if focus_pool is None:
                    disp_mag = np.linalg.norm(
                        next_points - curr_points, axis=1
                    )
                    threshold = np.quantile(disp_mag, self.focus_quantile)
                    focus_pool = np.flatnonzero(disp_mag >= threshold)
                    self._focus_pool_cache[focus_key] = focus_pool
                focus_count = min(
                    int(round(self.max_points * self.focus_ratio)),
                    len(focus_pool),
                )
                focus_indices = rng.choice(
                    focus_pool, focus_count, replace=False
                )
                focus_members[focus_indices] = True
                available = np.ones(n_total, dtype=bool)
                available[focus_indices] = False
                uniform_pool = np.flatnonzero(available)
                uniform_indices = rng.choice(
                    uniform_pool,
                    self.max_points - focus_count,
                    replace=False,
                )
                indices_np = np.concatenate([
                    uniform_indices, focus_indices
                ])
            else:
                indices_np = rng.choice(
                    n_total, self.max_points, replace=False
                )
            indices = torch.from_numpy(indices_np).long()
            p_pos = p_pos[indices]
            p_pos_next_full = torch.from_numpy(next_points).float()
            p_pos_prev_full = torch.from_numpy(prev_points).float()
            p_pos_next = p_pos_next_full[indices]
            p_pos_prev = p_pos_prev_full[indices]
            del p_pos_next_full
            del p_pos_prev_full
            _subsample = True
            point_focus = torch.zeros(
                self.max_points, 1, dtype=torch.float32
            )
            point_focus[:, 0] = torch.from_numpy(
                focus_members[indices_np].astype(np.float32)
            )
            point_contact_focus = torch.zeros(
                self.max_points, 1, dtype=torch.float32
            )
            point_contact_focus[:, 0] = torch.from_numpy(
                contact_members[indices_np].astype(np.float32)
            )
            point_surface_focus = torch.zeros(
                self.max_points, 1, dtype=torch.float32
            )
            point_surface_focus[:, 0] = torch.from_numpy(
                surface_members[indices_np].astype(np.float32)
            )
            point_late_final_focus = torch.zeros(
                self.max_points, 1, dtype=torch.float32
            )
            point_late_final_focus[:, 0] = torch.from_numpy(
                late_final_members[indices_np].astype(np.float32)
            )
        else:
            p_pos_next = torch.from_numpy(next_points).float()
            p_pos_prev = torch.from_numpy(prev_points).float()
            _subsample = False
            indices_np = np.arange(n_total, dtype=np.int64)
            point_focus = torch.zeros(
                n_total, 1, dtype=torch.float32
            )
            point_contact_focus = torch.zeros(
                n_total, 1, dtype=torch.float32
            )
            point_surface_focus = torch.zeros(
                n_total, 1, dtype=torch.float32
            )
            point_late_final_focus = torch.zeros(
                n_total, 1, dtype=torch.float32
            )

        # Occupancy-only surrogate: preserve the return contract without
        # decompressing velocity/history arrays that this model never uses.
        if self.minimal_io:
            point_count = int(p_pos.shape[0])
            p_vel = torch.zeros((point_count, 3), dtype=torch.float32)
            p_feat = torch.zeros(
                (point_count, self.phys_dim), dtype=torch.float32
            )
        else:
            p_vel_raw = torch.stack([
                torch.from_numpy(d_curr['velovx']).float(),
                torch.from_numpy(d_curr['velovy']).float(),
                torch.from_numpy(d_curr['velovz']).float(),
            ], dim=-1)
            p_vel = p_vel_raw[indices] if _subsample else p_vel_raw
            feat_list = []
            for fname in [
                'effstn', 'effsts', 'displx', 'disply', 'displz', 'pttemp'
            ]:
                arr = torch.from_numpy(d_curr[fname]).float()
                if _subsample:
                    arr = arr[indices]
                scale = MPM_FEATURE_SCALES.get(fname, 1.0)
                feat_list.append(arr * scale)
            p_feat = torch.stack(feat_list, dim=-1)
        if self.surface_feature_mode:
            surface_feat = self._get_surface_features(
                case_name, target_step, n_total
            )
            if surface_feat is None:
                surface_feat_t = torch.zeros(
                    (n_total, 3), dtype=torch.float32
                )
            else:
                surface_feat_t = torch.from_numpy(surface_feat).float()
            if _subsample:
                surface_feat_t = surface_feat_t[indices]
            p_feat = torch.cat([p_feat, surface_feat_t], dim=-1)
        if self.geometry_feature_mode:
            geom_feat = self._get_geometry_features(case_name, n_total)
            if geom_feat is None:
                geom_feat_t = torch.zeros(
                    (n_total, 9), dtype=torch.float32
                )
            else:
                geom_feat_t = torch.from_numpy(geom_feat).float()
            if _subsample:
                geom_feat_t = geom_feat_t[indices]
            p_feat = torch.cat([p_feat, geom_feat_t], dim=-1)

        # Optional target-only supervision for the equivalent-state task.
        # These arrays are deliberately never concatenated to p_feat: doing so
        # would leak target-frame information into an initial-to-target query.
        if self.target_state_mode:
            state_raw_cols = []
            state_norm_cols = []
            for field_name in self.target_state_fields:
                if field_name not in d_next.files:
                    raise KeyError(
                        f"missing target state field {field_name!r} in "
                        f"{sample['target_path']}"
                    )
                full = np.asarray(d_next[field_name], dtype=np.float32)
                selected = full[indices_np] if _subsample else full
                raw = torch.from_numpy(
                    np.asarray(selected, dtype=np.float32).copy()
                )
                state_raw_cols.append(raw)
                state_norm_cols.append(
                    raw * float(MPM_STATE_TARGET_SCALES[field_name])
                )
            target_state_raw = torch.stack(state_raw_cols, dim=-1)
            target_state = torch.stack(state_norm_cols, dim=-1)
            target_state_scale = torch.tensor(
                [MPM_STATE_TARGET_SCALES[k]
                 for k in self.target_state_fields],
                dtype=torch.float32,
            )
        else:
            target_state_raw = torch.empty(
                (p_pos.shape[0], 0), dtype=torch.float32
            )
            target_state = target_state_raw
            target_state_scale = torch.empty((0,), dtype=torch.float32)

        # Mold translations
        die_t = torch.from_numpy(
            self._get_die_translation(case_name, curr_step)
        ).float()
        pc_t = torch.from_numpy(
            self._get_punch_translation(case_name, curr_step)
        ).float()
        pn_t = torch.from_numpy(
            self._get_punch_translation(case_name, target_step)
        ).float()
        pp_t = torch.from_numpy(
            self._get_punch_translation(case_name, prev_step)
        ).float()
        target_condition = torch.from_numpy(
            build_target_condition(meta, pc_t.numpy(), pn_t.numpy())
        ).float()
        prev_condition = torch.from_numpy(
            build_target_condition(meta, pc_t.numpy(), pp_t.numpy())
        ).float()
        pf_t = torch.from_numpy(get_final_punch_translation(meta)).float()

        # Data augmentation: spatial jitter
        if self.mode == 'train':
            jitter = (torch.rand(3) * 2 - 1) * self.jitter_range
            p_pos = p_pos + jitter
            p_pos_next = p_pos_next + jitter
            p_pos_prev = p_pos_prev + jitter
            die_t = die_t + jitter
            pc_t = pc_t + jitter
            pn_t = pn_t + jitter
            pp_t = pp_t + jitter
            pf_t = pf_t + jitter

        p_pos_raw = p_pos.clone()
        target_disp = p_pos_next - p_pos_raw
        prev_disp = p_pos_prev - p_pos_raw
        delta_disp = target_disp - prev_disp

        # Normalize
        p_pos_norm = (p_pos - w_center) / (w_scale / 2.0)

        d_curr.close()
        d_next.close()
        if d_prev is not None:
            d_prev.close()

        return {
            'case_name': case_name,
            'p_pos_raw': p_pos_raw,
            'p_pos_norm': p_pos_norm,
            'p_pos_next': p_pos_next,
            'p_vel': p_vel,
            'p_feat': p_feat,
            'die_t': die_t,
            'pc_t': pc_t,
            'pn_t': pn_t,
            'pp_t': pp_t,
            'pf_t': pf_t,
            'target_condition': target_condition,
            'prev_condition': prev_condition,
            'w_center': w_center,
            'w_scale': w_scale,
            'target_disp': target_disp,
            'target_state': target_state,
            'target_state_raw': target_state_raw,
            'target_state_scale': target_state_scale,
            'prev_disp': prev_disp,
            'delta_disp': delta_disp,
            'point_focus': point_focus,
            'point_contact_focus': point_contact_focus,
            'point_surface_focus': point_surface_focus,
            'point_late_final_focus': point_late_final_focus,
            'point_indices': torch.from_numpy(indices_np).long(),
            'sample_key': f"{case_name}|{curr_step}|{target_step}",
            'index_stride': sample.get('index_stride', 0),
            'prev_index_stride': sample.get('prev_index_stride', 0),
            'physical_step_diff': sample.get('physical_step_diff', 0),
            'displacement_norm': sample.get('displacement_norm', 0.0),
            'sample_weight': sample.get('sample_weight', 1.0),
            'sampling_repeat': sample.get('sampling_repeat', 0),
        }


def load_case_assets_mpm(processed_root, device, asset_variant='sdf'):
    """
    Load all case mold assets.

    asset_variant='sdf'     -> die_asset.npz / punch_asset.npz
    asset_variant='contact' -> die_contact_asset.npz / punch_contact_asset.npz

    The contact assets keep the same tensor keys (`sdf`, `grad`,
    `curvature`, `center`, `scale`) but `sdf` is a local projected distance
    signed by nearest contact-surface cell normal, not a closed-solid SDF.
    Same interface as the original load_case_assets.
    """
    case_assets = {}
    dirs = [d for d in os.listdir(processed_root)
            if os.path.isdir(os.path.join(processed_root, d))]
    for case in dirs:
        case_path = os.path.join(processed_root, case)
        assets = {}
        for part in ['die', 'punch']:
            if asset_variant == 'contact':
                path = os.path.join(case_path, f'{part}_contact_asset.npz')
            else:
                path = os.path.join(case_path, f'{part}_asset.npz')
            if not os.path.exists(path):
                print(f"  [WARN] Missing asset: {path}")
                continue
            data = np.load(path)
            # Contact assets include a non-negative projected distance field.
            # Use it as the first "sdf" channel so existing model plumbing sees
            # a stable near-contact distance instead of the invalid old signed
            # SDF from open surfaces.
            sdf_np = data['distance'] if (
                asset_variant == 'contact' and 'distance' in data.files
            ) else data['sdf']
            assets[part] = {
                'sdf': torch.from_numpy(sdf_np).float().unsqueeze(0).unsqueeze(0).to(device),
                'grad': torch.from_numpy(data['grad']).float().unsqueeze(0).to(device),
                'curvature': torch.from_numpy(data['curvature']).float().unsqueeze(0).unsqueeze(0).to(device),
                'center': torch.from_numpy(data['center']).float().to(device),
                'scale': torch.tensor(float(data['scale'])).float().to(device),
            }
        case_assets[case] = assets
    return case_assets

