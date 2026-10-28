# Extracted scientific kernel; provenance: reference/source_provenance.json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .scatter import scatter_add
class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        groups = 8 if cout >= 8 else 1
        self.a = nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.GELU())
        self.b = nn.Sequential(nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(groups, cout))
        self.skip = nn.Identity() if cin == cout else nn.Conv3d(cin, cout, 1)
    def forward(self, x):
        return F.gelu(self.b(self.a(x)) + self.skip(x))


class GridUNet(nn.Module):
    def __init__(self, cin, cout, width=24):
        super().__init__()
        self.e1 = ResBlock(cin, width)
        self.d1 = nn.Conv3d(width, width * 2, 3, stride=2, padding=1)
        self.e2 = ResBlock(width * 2, width * 2)
        self.d2 = nn.Conv3d(width * 2, width * 4, 3, stride=2, padding=1)
        self.e3 = ResBlock(width * 4, width * 4)
        self.d3 = nn.Conv3d(width * 4, width * 8, 3, stride=2, padding=1)
        self.mid = ResBlock(width * 8, width * 8)
        self.u3 = nn.ConvTranspose3d(width * 8, width * 4, 2, stride=2)
        self.r3 = ResBlock(width * 8, width * 4)
        self.u2 = nn.ConvTranspose3d(width * 4, width * 2, 2, stride=2)
        self.r2 = ResBlock(width * 4, width * 2)
        self.u1 = nn.ConvTranspose3d(width * 2, width, 2, stride=2)
        self.r1 = ResBlock(width * 2, width)
        self.out = nn.Conv3d(width, cout, 1)
    def forward(self, x):
        a = self.e1(x); b = self.e2(self.d1(a)); c = self.e3(self.d2(b)); z = self.mid(self.d3(c))
        z = self.r3(torch.cat([self.u3(z), c], 1)); z = self.r2(torch.cat([self.u2(z), b], 1))
        return self.out(self.r1(torch.cat([self.u1(z), a], 1)))

class RigidMoldLayer(nn.Module):
    """Query SDF/gradient/curvature volumes for mold contact features."""
    def __init__(self, grid_res):
        super().__init__()
        self.res = grid_res
        self.voxel_size = 2.0 / grid_res
        self.default_threshold = self.voxel_size * 1.5

    def _safe_expand(self, tensor, B, last_dim):
        if tensor is None:
            return None
        if not torch.is_tensor(tensor):
            tensor = torch.tensor(tensor)
        flat = tensor.reshape(-1)
        num_el = flat.numel()
        if num_el == 1:
            return flat.expand(B).view(B, 1, 1).expand(B, 1, last_dim)
        elif num_el == last_dim:
            return flat.view(1, 1, last_dim).expand(B, 1, last_dim)
        elif num_el == B * last_dim:
            return flat.view(B, 1, last_dim)
        else:
            return flat[:last_dim].view(1, 1, last_dim).expand(B, 1, last_dim)

    def query_particles(self, p_pos_norm, trans, vel, w_center, w_scale,
                        asset, friction_mu=0.3):
        """
        Query mold features for each particle.

        Returns (B, N, 11): [sdf, relu(thresh-sdf), exp(-sens*|sdf|),
                             normalized_grad(3), curvature, velocity(3), friction]
        """
        B, N, _ = p_pos_norm.shape
        device = p_pos_norm.device
        w_s = self._safe_expand(w_scale, B, 1).to(device)
        w_c = self._safe_expand(w_center, B, 3).to(device)
        m_center = self._safe_expand(asset['center'], B, 3).to(device)
        m_scale = self._safe_expand(asset['scale'], B, 1).to(device)
        m_t = self._safe_expand(trans, B, 3).to(device)
        m_v = self._safe_expand(vel, B, 3).to(device)

        # Transform particles to mold-local normalized coordinates
        p_world = p_pos_norm * (w_s / 2.0) + w_c
        m_curr_pos = m_center + m_t
        # Asset `scale` is the full side length of the cubic volume
        # generated over [center - scale/2, center + scale/2].  grid_sample
        # expects normalized coordinates in [-1, 1], so divide by scale/2,
        # not by scale.  The old formula effectively doubled the mold volume
        # and misaligned contact/SDF features.
        coords = (p_world - m_curr_pos) / (m_scale / 2.0 + 1e-8)
        # Asset arrays are stored as [x, y, z], while 5-D grid_sample expects
        # coordinates ordered for [D, H, W], i.e. [W, H, D] = [z, y, x].
        sample_coords = coords.flip(-1).view(B, 1, N, 1, 3)

        def _sample(vol):
            vol_f = vol.float().to(device)
            if vol_f.dim() == 4:
                vol_f = vol_f.unsqueeze(0)
            if vol_f.shape[0] == 1 and B > 1:
                vol_f = vol_f.expand(B, -1, -1, -1, -1)
            return F.grid_sample(vol_f, sample_coords,
                                 align_corners=True, padding_mode='border')

        sdf = _sample(asset['sdf']).view(B, 1, N)
        grad = _sample(asset['grad']).view(B, 3, N)
        curv = _sample(asset['curvature']).view(B, 1, N)

        sensitivity = 2.0 / self.voxel_size
        contact_mask = torch.sigmoid((self.default_threshold - sdf) * sensitivity)

        # SDF features
        f_sdf = torch.cat([
            sdf,
            torch.relu(self.default_threshold - sdf),
            torch.exp(-sensitivity * torch.abs(sdf)),
        ], dim=1).permute(0, 2, 1)  # (B, N, 3)

        # Normalized gradient
        v_grad = grad.permute(0, 2, 1)  # (B, N, 3)
        v_grad = v_grad / (torch.norm(v_grad, dim=-1, keepdim=True) + 1e-9)

        # Curvature
        f_curv = curv.permute(0, 2, 1)  # (B, N, 1)

        # Mold velocity
        f_vel = m_v.expand(-1, N, -1)  # (B, N, 3)

        # Friction
        f_fric = (friction_mu * contact_mask).permute(0, 2, 1)  # (B, N, 1)

        return torch.cat([f_sdf, v_grad, f_curv, f_vel, f_fric], dim=-1)

    def query_grid_single_batch(self, v_coords_norm, trans, vel, w_center,
                                w_scale, asset, friction_mu=0.3):
        """Query mold features for voxel centers (single batch)."""
        feats = self.query_particles(
            v_coords_norm.unsqueeze(0),
            trans.reshape(1, 3), vel.reshape(1, 3),
            w_center.reshape(1, 3), w_scale.reshape(1, 1),
            asset, friction_mu
        )
        return feats.view(-1, 11)

class MoldGridMixin:
    def init_grid(self, res):
        vals = torch.linspace(-1.0, 1.0, res)
        xx, yy, zz = torch.meshgrid(vals, vals, vals, indexing="ij")
        self.register_buffer("grid_points", torch.stack([xx, yy, zz], -1).reshape(1, -1, 3))

    def p2g(self, pos, feat):
        b, n, c = feat.shape
        r = self.res
        ijk = torch.floor((pos + 1.0) * 0.5 * (r - 1)).long().clamp(0, r - 1)
        flat = (ijk[..., 0] * r * r + ijk[..., 1] * r + ijk[..., 2]).reshape(-1)
        offset = torch.arange(b, device=pos.device).view(b, 1).expand(b, n).reshape(-1) * (r ** 3)
        flat = flat + offset
        sums = scatter_add(feat.reshape(-1, c), flat, dim=0, dim_size=b * r ** 3)
        count = scatter_add(torch.ones(b * n, 1, device=pos.device), flat, dim=0, dim_size=b * r ** 3)
        grid = (sums / count.clamp_min(1.0)).view(b, r, r, r, c).permute(0, 4, 1, 2, 3)
        den = torch.log1p(count).view(b, r, r, r, 1).permute(0, 4, 1, 2, 3)
        return grid, den

    @staticmethod
    def g2p(grid, pos):
        coords = pos.flip(-1).view(pos.shape[0], 1, pos.shape[1], 1, 3)
        out = F.grid_sample(grid, coords, align_corners=True, padding_mode="border")
        return out.view(pos.shape[0], grid.shape[1], pos.shape[1]).permute(0, 2, 1)

    def mold_grid(self, translation, center, scale, asset, mu, chunk=65536):
        b = translation.shape[0]
        pts = self.grid_points.expand(b, -1, -1)
        vel = torch.zeros_like(translation)
        outs = []
        for a in range(0, pts.shape[1], chunk):
            q = self.mold.query_particles(pts[:, a:a + chunk], translation, vel, center, scale, asset, mu.view(-1))
            outs.append(q)
        f = torch.cat(outs, 1)
        r = self.res
        return f.view(b, r, r, r, -1).permute(0, 4, 1, 2, 3)

    def global_grid(self, vec):
        b = vec.shape[0]
        r = self.res
        return vec.view(b, -1, 1, 1, 1).expand(b, vec.shape[-1], r, r, r)

