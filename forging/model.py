# Extracted scientific kernel; provenance: reference/source_provenance.json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .scatter import scatter_add
from .components import GridUNet, RigidMoldLayer, MoldGridMixin
DECODER_MODES = {
    "E0_direct": "grid displacement followed by trilinear G2P",
    "E1_no_local": "particle MLP without local sub-voxel coordinates",
    "E2_full_subvoxel": "particle MLP with xi, xi^2 and sin/cos(pi*xi)",
    "E2c_continuous_fourier": (
        "particle MLP with boundary-continuous periodic grid-coordinate encoding"
    ),
    "E3_linear_local": "particle MLP with linear xi only",
    "E4_no_sincos": "particle MLP with xi and xi^2 but no Fourier terms",
    "E5_residual_subvoxel": "direct G2P base plus a zero-initialized sub-voxel residual",
    "E5_residual_no_local": (
        "direct G2P base plus a gated zero-initialized residual without local coordinates"
    ),
    "E5_residual_fixed_gate": (
        "direct G2P base plus a unit-gain zero-initialized sub-voxel residual"
    ),
    "E5_residual_random_init": (
        "direct G2P base plus a gated randomly initialized sub-voxel residual"
    ),
    "E5c_residual_continuous_fourier": (
        "direct G2P base plus a gated zero-initialized boundary-continuous residual"
    ),
    "E6_basepos_targetmold": (
        "direct G2P base plus a residual MLP using target-mold features "
        "queried only at the base-predicted particle position"
    ),
}


RESIDUAL_DECODER_MODES = {
    "E5_residual_subvoxel",
    "E5_residual_no_local",
    "E5_residual_fixed_gate",
    "E5_residual_random_init",
    "E5c_residual_continuous_fourier",
}


CONTINUOUS_FOURIER_MODES = {
    "E2c_continuous_fourier",
    "E5c_residual_continuous_fourier",
}


LOCAL_LINEAR_MODES = {
    "E2_full_subvoxel", "E3_linear_local", "E4_no_sincos",
    "E5_residual_subvoxel", "E5_residual_fixed_gate",
    "E5_residual_random_init",
}


LOCAL_QUADRATIC_MODES = {
    "E2_full_subvoxel", "E4_no_sincos", "E5_residual_subvoxel",
    "E5_residual_fixed_gate", "E5_residual_random_init",
}


LOCAL_FOURIER_MODES = {
    "E2_full_subvoxel", "E5_residual_subvoxel",
    "E5_residual_fixed_gate", "E5_residual_random_init",
}


class CoinInitialMoldModel(nn.Module, MoldGridMixin):
    """Initial-position encoder with interchangeable grid-to-particle decoder."""

    def __init__(self, res=64, width=24, decoder_mode="E0_direct", hidden_dim=96,
                 point_chunk=200000, mold_input_mode="dense_full",
                 friction_coefficient=0.30):
        super().__init__()
        if decoder_mode not in DECODER_MODES:
            raise ValueError(f"unknown decoder_mode={decoder_mode}")
        self.res = int(res)
        self.decoder_mode = decoder_mode
        self.hidden_dim = int(hidden_dim)
        self.point_chunk = int(point_chunk)
        self.mold_input_mode = str(mold_input_mode)
        self.friction_coefficient = float(friction_coefficient)
        if self.mold_input_mode not in {
            "dense_full", "occupied_only", "no_target_punch_field",
            "distance_only"
        }:
            raise ValueError(f"unknown mold_input_mode={self.mold_input_mode}")
        self.mold = RigidMoldLayer(self.res)
        self.init_grid(self.res)
        # occupancy + 3x(11-channel mold field) + two translations + progress
        self.in_channels = 1 + 11 + 11 + 11 + 3 + 3 + 1
        if decoder_mode == "E0_direct":
            self.net = GridUNet(self.in_channels, 3, width)
            self.point_head = None
        elif decoder_mode in RESIDUAL_DECODER_MODES:
            self.net = GridUNet(self.in_channels, 3 + self.hidden_dim, width)
            self.point_head = self._make_head(self.hidden_dim + self._extra_dim(decoder_mode))
            if decoder_mode != "E5_residual_random_init":
                # Start exactly from direct G2P; residual capacity is learned gradually.
                nn.init.zeros_(self.point_head[-1].weight)
                nn.init.zeros_(self.point_head[-1].bias)
            if decoder_mode != "E5_residual_fixed_gate":
                self.residual_gain = nn.Parameter(torch.tensor(-2.0))
        elif decoder_mode == "E6_basepos_targetmold":
            self.net = GridUNet(self.in_channels, 3, width)
            # base displacement(3) + die target geometry(11)
            # + target-punch geometry(11)
            self.point_head = self._make_head(25)
            nn.init.zeros_(self.point_head[-1].weight)
            nn.init.zeros_(self.point_head[-1].bias)
            self.residual_gain = nn.Parameter(torch.tensor(-2.0))
        else:
            self.net = GridUNet(self.in_channels, self.hidden_dim, width)
            self.point_head = self._make_head(self.hidden_dim + self._extra_dim(decoder_mode))

    @staticmethod
    def _make_head(in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 3),
        )

    @staticmethod
    def _extra_dim(mode):
        # Absolute normalized position + target progress are always retained.
        dim = 4
        if mode in CONTINUOUS_FOURIER_MODES:
            # sin(2*pi*q) and cos(2*pi*q), three axes each.  Unlike the
            # cell-reset xi features below, these are continuous when q
            # crosses an integer grid-cell boundary.
            dim += 6
        if mode in LOCAL_LINEAR_MODES:
            dim += 3
        if mode in LOCAL_QUADRATIC_MODES:
            dim += 3
        if mode in LOCAL_FOURIER_MODES:
            dim += 6
        return dim

    def local_coords(self, pos):
        q = (pos + 1.0) * 0.5 * (self.res - 1)
        base = torch.floor(q).clamp(0, self.res - 2)
        return q - base - 0.5

    def point_extra(self, pos, progress):
        pieces = [pos, progress[:, None, :].expand(pos.shape[0], pos.shape[1], 1)]
        mode = self.decoder_mode
        if mode in CONTINUOUS_FOURIER_MODES:
            q = (pos + 1.0) * 0.5 * (self.res - 1)
            pieces.extend([
                torch.sin(2.0 * math.pi * q),
                torch.cos(2.0 * math.pi * q),
            ])
        elif mode in LOCAL_LINEAR_MODES:
            xi = self.local_coords(pos)
            pieces.append(xi)
            if mode in LOCAL_QUADRATIC_MODES:
                pieces.append(xi * xi)
            if mode in LOCAL_FOURIER_MODES:
                pieces.extend([torch.sin(math.pi * xi), torch.cos(math.pi * xi)])
        return torch.cat(pieces, dim=-1)

    def occupancy_grid(self, pos):
        b, n, _ = pos.shape
        r = self.res
        ijk = torch.floor((pos + 1.0) * 0.5 * (r - 1)).long().clamp(0, r - 1)
        flat = (ijk[..., 0] * r * r + ijk[..., 1] * r + ijk[..., 2]).reshape(-1)
        offset = torch.arange(b, device=pos.device).view(b, 1).expand(b, n).reshape(-1) * r ** 3
        flat = flat + offset
        count = scatter_add(torch.ones(b * n, 1, device=pos.device), flat,
                            dim=0, dim_size=b * r ** 3)
        return (count > 0).to(pos.dtype).view(b, r, r, r, 1).permute(0, 4, 1, 2, 3)

    def input_grid(self, s, assets):
        asset = assets[s["case"]]
        fixed_mu = torch.full((s["pos"].shape[0], 1), self.friction_coefficient, device=s["pos"].device,
                              dtype=s["pos"].dtype)
        zero = torch.zeros_like(s["raw_current_punch"])
        occupancy = self.occupancy_grid(s["pos"])
        die_grid = self.mold_grid(
            zero, s["center"], s["scale"], asset["die"], fixed_mu)
        initial_punch_grid = self.mold_grid(
            s["raw_current_punch"], s["center"], s["scale"],
            asset["punch"], fixed_mu)
        target_punch_grid = self.mold_grid(
            s["raw_target_punch"], s["center"], s["scale"],
            asset["punch"], fixed_mu)
        if self.mold_input_mode == "occupied_only":
            die_grid = die_grid * occupancy
            initial_punch_grid = initial_punch_grid * occupancy
            target_punch_grid = target_punch_grid * occupancy
        elif self.mold_input_mode == "no_target_punch_field":
            target_punch_grid = torch.zeros_like(target_punch_grid)
        elif self.mold_input_mode == "distance_only":
            distance_grids = []
            for mold_grid in (die_grid, initial_punch_grid, target_punch_grid):
                distance_grid = torch.zeros_like(mold_grid)
                distance_grid[:, 0:3] = mold_grid[:, 0:3]
                distance_grids.append(distance_grid)
            die_grid, initial_punch_grid, target_punch_grid = distance_grids
        return torch.cat([
            occupancy,
            die_grid,
            initial_punch_grid,
            target_punch_grid,
            self.global_grid(s["current_punch"]),
            self.global_grid(s["target_punch"]),
            self.global_grid(s["progress"]),
        ], dim=1)

    def _decode_chunk(self, grid, pos, progress):
        if self.decoder_mode == "E0_direct":
            return self.g2p(grid, pos)
        if self.decoder_mode in RESIDUAL_DECODER_MODES:
            base = self.g2p(grid[:, :3], pos)
            latent = self.g2p(grid[:, 3:], pos)
            residual = self.point_head(torch.cat([latent, self.point_extra(pos, progress)], -1))
            gain = (1.0 if self.decoder_mode == "E5_residual_fixed_gate"
                    else torch.sigmoid(self.residual_gain))
            return base + gain * residual
        latent = self.g2p(grid, pos)
        return self.point_head(torch.cat([latent, self.point_extra(pos, progress)], -1))

    def _decode_target_mold_chunk(self, grid, pos, s, assets):
        base = self.g2p(grid, pos)
        base_pos = pos + base
        asset = assets[s["case"]]
        zero = torch.zeros_like(s["raw_target_punch"])
        fixed_mu = torch.full(
            (pos.shape[0], 1), self.friction_coefficient,
            device=pos.device, dtype=pos.dtype)
        die_feat = self.mold.query_particles(
            base_pos, zero, zero, s["center"], s["scale"],
            asset["die"], fixed_mu)
        punch_feat = self.mold.query_particles(
            base_pos, s["raw_target_punch"], zero, s["center"], s["scale"],
            asset["punch"], fixed_mu)
        residual_input = torch.cat([
            base, die_feat.to(base.dtype), punch_feat.to(base.dtype)
        ], dim=-1)
        residual = self.point_head(residual_input)
        return base + torch.sigmoid(self.residual_gain) * residual

    def forward(self, s, assets):
        grid = self.net(self.input_grid(s, assets))
        if self.decoder_mode == "E6_basepos_targetmold":
            if self.point_chunk <= 0 or s["pos"].shape[1] <= self.point_chunk:
                return self._decode_target_mold_chunk(
                    grid, s["pos"], s, assets)
            chunks = []
            for start in range(0, s["pos"].shape[1], self.point_chunk):
                chunks.append(self._decode_target_mold_chunk(
                    grid, s["pos"][:, start:start + self.point_chunk],
                    s, assets))
            return torch.cat(chunks, dim=1)
        if self.point_chunk <= 0 or s["pos"].shape[1] <= self.point_chunk:
            return self._decode_chunk(grid, s["pos"], s["progress"])
        chunks = []
        for start in range(0, s["pos"].shape[1], self.point_chunk):
            chunks.append(self._decode_chunk(
                grid, s["pos"][:, start:start + self.point_chunk], s["progress"]
            ))
        return torch.cat(chunks, dim=1)

REQUESTED_MODES = {
    "E0_direct": "E0",
    "E1_no_local": "E1",
    "E2_full_subvoxel": "E2",
    "E2c_continuous_fourier": "E2c",
    "E5_residual_subvoxel": "E5",
}


def _softplus_inverse(value: float) -> float:
    return math.log(math.expm1(value))


class CoinEquivalentStateModel(CoinInitialMoldModel):
    """E0--E5 decoder family extended from three to five outputs."""

    def __init__(
        self,
        res=64,
        width=24,
        decoder_mode="E0_direct",
        hidden_dim=96,
        point_chunk=160000,
        mold_input_mode="dense_full",
        friction_coefficient=0.30,
        state_init=0.05,
        residual_gate_mode="shared",
    ):
        if decoder_mode not in REQUESTED_MODES:
            raise ValueError(
                f"unsupported decoder_mode={decoder_mode!r}; "
                f"choose from {sorted(REQUESTED_MODES)}"
            )
        super().__init__(
            res=res,
            width=width,
            decoder_mode=decoder_mode,
            hidden_dim=hidden_dim,
            point_chunk=point_chunk,
            mold_input_mode=mold_input_mode,
            friction_coefficient=friction_coefficient,
        )
        self.state_init = float(state_init)
        self.residual_gate_mode = str(residual_gate_mode)
        if self.residual_gate_mode not in {"shared", "three"}:
            raise ValueError(
                "residual_gate_mode must be 'shared' or 'three', got "
                f"{self.residual_gate_mode!r}"
            )
        if (
            self.residual_gate_mode == "three"
            and decoder_mode != "E5_residual_subvoxel"
        ):
            raise ValueError(
                "three-gate residual decoding is only defined for "
                "E5_residual_subvoxel"
            )

        if self.residual_gate_mode == "three":
            # The base class creates one shared residual gate.  Replace it
            # with task-specific gates so state losses cannot directly alter
            # the displacement residual amplitude.
            del self.residual_gain
            self.residual_gain_disp = nn.Parameter(torch.tensor(-2.0))
            self.residual_gain_strain = nn.Parameter(torch.tensor(-2.0))
            self.residual_gain_stress = nn.Parameter(torch.tensor(-2.0))

        if decoder_mode == "E0_direct":
            self.net = GridUNet(self.in_channels, 5, width)
            self.point_head = None
            self._initialize_grid_state_channels()
        elif decoder_mode in RESIDUAL_DECODER_MODES:
            self.net = GridUNet(
                self.in_channels, 5 + self.hidden_dim, width
            )
            self.point_head = self._make_field_head(
                self.hidden_dim + self._extra_dim(decoder_mode)
            )
            # E5 starts exactly from its direct grid prediction.
            nn.init.zeros_(self.point_head[-1].weight)
            nn.init.zeros_(self.point_head[-1].bias)
            self._initialize_grid_state_channels()
        else:
            self.net = GridUNet(self.in_channels, self.hidden_dim, width)
            self.point_head = self._make_field_head(
                self.hidden_dim + self._extra_dim(decoder_mode)
            )
            self._initialize_point_state_channels()

    @staticmethod
    def _make_field_head(in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 5),
        )

    def _initialize_grid_state_channels(self):
        bias = _softplus_inverse(self.state_init)
        with torch.no_grad():
            self.net.out.weight[3:5].zero_()
            self.net.out.bias[3:5].fill_(bias)

    def _initialize_point_state_channels(self):
        bias = _softplus_inverse(self.state_init)
        with torch.no_grad():
            self.point_head[-1].weight[3:5].zero_()
            self.point_head[-1].bias[3:5].fill_(bias)

    def _decode_fields_chunk(self, grid, pos, progress):
        if self.decoder_mode == "E0_direct":
            return self.g2p(grid, pos)
        if self.decoder_mode in RESIDUAL_DECODER_MODES:
            base = self.g2p(grid[:, :5], pos)
            latent = self.g2p(grid[:, 5:], pos)
            residual = self.point_head(
                torch.cat(
                    [latent, self.point_extra(pos, progress)], dim=-1
                )
            )
            if self.residual_gate_mode == "three":
                gain_disp = torch.sigmoid(self.residual_gain_disp)
                gain_strain = torch.sigmoid(self.residual_gain_strain)
                gain_stress = torch.sigmoid(self.residual_gain_stress)
                return torch.cat([
                    base[..., :3] + gain_disp * residual[..., :3],
                    base[..., 3:4] + gain_strain * residual[..., 3:4],
                    base[..., 4:5] + gain_stress * residual[..., 4:5],
                ], dim=-1)
            gain = (
                1.0
                if self.decoder_mode == "E5_residual_fixed_gate"
                else torch.sigmoid(self.residual_gain)
            )
            return base + gain * residual
        latent = self.g2p(grid, pos)
        return self.point_head(
            torch.cat([latent, self.point_extra(pos, progress)], dim=-1)
        )

    def forward(self, sample, assets):
        grid = self.net(self.input_grid(sample, assets))
        if self.point_chunk <= 0 or sample["pos"].shape[1] <= self.point_chunk:
            fields = self._decode_fields_chunk(
                grid, sample["pos"], sample["progress"]
            )
        else:
            chunks = []
            for start in range(0, sample["pos"].shape[1], self.point_chunk):
                chunks.append(
                    self._decode_fields_chunk(
                        grid,
                        sample["pos"][:, start:start + self.point_chunk],
                        sample["progress"],
                    )
                )
            fields = torch.cat(chunks, dim=1)
        return {
            "disp": fields[..., :3],
            "state": F.softplus(fields[..., 3:5]),
        }

