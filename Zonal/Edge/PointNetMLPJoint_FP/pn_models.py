# [patch_pointnet_features] patched
from typing import List, Tuple, Optional, Dict, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PointNetMLPJoint", "PointNetMLPJoint_FP", "PointNet2Encoder2D", "SetAbstraction", "MLP"]


def farthest_point_sampling(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    # Farthest Point Sampling (FPS) indices
    device = xyz.device
    B, N, _ = xyz.shape
    n_samples = min(n_samples, N)
    idx = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    distances = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device)
    for i in range(n_samples):
        idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # [B,1,2]
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # [B,N]
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = torch.max(distances, dim=1).indices
    return idx


def ball_query(
    xyz: torch.Tensor, centers: torch.Tensor, radius: float, max_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Naive radius (ball) query capped at max_k with fallback to KNN when no neighbors
    _, N, _ = xyz.shape
    d2 = torch.cdist(centers, xyz, p=2) ** 2  # [B,M,N]
    within = d2 <= (radius**2)
    d2_masked = d2.clone()
    d2_masked[~within] = float("inf")
    k_init = min(max_k, N)
    idx = torch.topk(-d2_masked, k=k_init, dim=-1).indices
    gathered = torch.gather(d2_masked, dim=2, index=idx)
    mask = torch.isfinite(gathered)
    zero_mask = mask.sum(dim=-1) == 0  # [B,M]
    if zero_mask.any():
        idx_knn = torch.topk(-d2, k=k_init, dim=-1).indices
        zm_exp = zero_mask.unsqueeze(-1).expand_as(idx)
        idx = torch.where(zm_exp, idx_knn, idx)
        mask = torch.where(zm_exp, torch.ones_like(mask, dtype=torch.bool), mask)
    if idx.shape[-1] < max_k:
        pad = max_k - idx.shape[-1]
        idx = F.pad(idx, (0, pad), value=0)
        mask = F.pad(mask, (0, pad), value=False)
    elif idx.shape[-1] > max_k:
        idx = idx[..., :max_k]
        mask = mask[..., :max_k]
    return idx, mask


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: List[int],
        out_dim: int,
        act=nn.SiLU,
        norm: str = "batch",  # 'batch' | 'layer' | 'group' | 'none'
        num_groups: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [in_dim] + hidden
        layers: List[nn.Module] = []

        def _norm(ch: int) -> Optional[nn.Module]:
            if norm == "batch":
                return nn.BatchNorm1d(ch)
            if norm == "layer":
                return nn.LayerNorm(ch)
            if norm == "group":
                g = max(1, min(num_groups, ch))
                # ensure divisible
                while ch % g != 0 and g > 1:
                    g -= 1
                return nn.GroupNorm(g, ch)
            return None

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            n = _norm(dims[i + 1])
            if n is not None:
                layers.append(n)
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierFeatures(nn.Module):
    """Fourier positional encoding for 2D inputs.

    Produces [sin, cos] features at exponentially increasing frequencies.
    If include_input=True, original (x,y) are concatenated as well.
    """

    def __init__(self, n_freqs: int, scale: float = 1.0, include_input: bool = True):
        super().__init__()
        self.n = max(0, int(n_freqs))
        self.scale = float(scale)
        self.include_input = bool(include_input)

    @property
    def out_dim(self) -> int:
        base = 2 if self.include_input else 0
        # For 2D coords: for each freq we add sin/cos for both x and y => 4 dims per freq
        return base + self.n * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 2]
        if self.n == 0:
            return x if self.include_input else torch.zeros_like(x)
        orig_shape = x.shape
        x2 = x.view(-1, 2)
        feats: List[torch.Tensor] = []
        if self.include_input:
            feats.append(x2)
        for k in range(self.n):
            w = (2.0**k) * self.scale * math.pi
            s = torch.sin(w * x2)
            c = torch.cos(w * x2)
            feats.extend([s, c])
        y = torch.cat(feats, dim=-1)
        return y.view(*orig_shape[:-1], y.shape[-1])


class SetAbstraction(nn.Module):
    # Simple PointNet++ style set abstraction for 2D point sets
    def __init__(
        self,
        n_samples: int,
        radius: float,
        max_k: int,
        in_ch: int,
        out_ch: int,
        mlp_hidden: Optional[List[int]] = None,
        norm: str = "batch",
        num_groups: int = 16,
        pool: str = "max",  # 'max' | 'max+mean'
    ):
        super().__init__()
        self.n_samples = n_samples
        self.radius = radius
        self.max_k = max_k
        self.pool = pool
        if mlp_hidden is None:
            mlp_hidden = [out_ch // 2]
        self.mlp = MLP(in_ch + 2, mlp_hidden, out_ch, norm=norm, num_groups=num_groups)

    def forward(
        self, xyz: torch.Tensor, feats: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # xyz: [B,N,2]; feats: [B,N,C] or None
        B, N, _ = xyz.shape
        if feats is None:
            feats = xyz
        C = feats.shape[-1]
        # If n_samples <= 0 or >= N, treat all points as centers (no subsampling) for full coverage
        if self.n_samples <= 0 or self.n_samples >= N:
            centers = xyz  # [B,N,2]
            S = N
            idx = torch.arange(N, device=xyz.device).unsqueeze(0).repeat(B, 1)
        else:
            idx = farthest_point_sampling(xyz, self.n_samples)  # [B,S]
            centers = torch.gather(
                xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 2)
            )  # [B,S,2]
            S = centers.shape[1]
        knn_idx, knn_mask = ball_query(xyz, centers, self.radius, self.max_k)  # [B,S,K]
        grouped_xyz = torch.gather(
            xyz.unsqueeze(1).expand(B, S, N, 2),
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, 2),
        )  # [B,S,K,2]
        grouped_feats = torch.gather(
            feats.unsqueeze(1).expand(B, S, N, C),
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, C),
        )  # [B,S,K,C]
        rel = grouped_xyz - centers.unsqueeze(2)  # [B,S,K,2]
        x = torch.cat([rel, grouped_feats], dim=-1)  # [B,S,K,2+C]
        x = x.view(B * S * self.max_k, -1)
        x = self.mlp(x)
        x = x.view(B, S, self.max_k, -1)  # [B,S,K,OC]
        mask = knn_mask.unsqueeze(-1)
        # Use a large negative finite value for masked entries to avoid -inf propagation issues
        fill_val = -1e4
        x = torch.where(mask.expand_as(x), x, torch.full_like(x, fill_val))
        if self.pool == "max+mean":
            x_max = torch.max(x, dim=2).values  # [B,S,OC]
            # fill masked with 0 for mean
            x_masked = torch.where(mask.expand_as(x), x, torch.zeros_like(x))
            denom = torch.clamp(mask.sum(dim=2, keepdim=False).to(x.dtype), min=1.0)
            x_mean = x_masked.sum(dim=2) / denom  # [B,S,OC]
            x = torch.cat([x_max, x_mean], dim=-1)  # [B,S,2*OC]
        else:
            x = torch.max(x, dim=2).values  # [B,S,OC]
        return centers, x


class GlobalFeature(nn.Module):
    def __init__(
        self,
        in_ch: int,
        latent_dim: int,
        hidden: Optional[List[int]] = None,
        norm: str = "batch",
        num_groups: int = 16,
    ):
        super().__init__()
        if hidden is None:
            hidden = [max(8, latent_dim // 2)]
        self.mlp = MLP(in_ch, hidden, latent_dim, norm=norm, num_groups=num_groups)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B, S, C = feats.shape
        x = feats.reshape(B * S, C)
        x = self.mlp(x)
        x = x.view(B, S, -1)
        x = torch.max(x, dim=1).values  # [B,latent]
        return x


class PointNet2Encoder2D(nn.Module):
    """Configurable PointNet++-style encoder for 2D point sets.

    encoder_cfg schema (defaults shown):
      {
        'latent_dim': 128,
        'pre_hidden': [64, 64],
        'sa_blocks': [
          {'n_samples': 256, 'radius': 0.10, 'max_k': 32, 'out_ch': 128, 'mlp_hidden': [64]},
          {'n_samples': 64,  'radius': 0.25, 'max_k': 32, 'out_ch': 256, 'mlp_hidden': [128]}
        ],
        'gf_hidden': [64]
      }
    """

    def __init__(
        self,
        latent_dim: int = 128,
        encoder_cfg: Optional[Dict[str, Any]] = None,
        in_channels: int = 0,
    ):
        super().__init__()
        cfg = dict(encoder_cfg) if encoder_cfg is not None else {}
        self.latent_dim = int(cfg.get("latent_dim", latent_dim))
        self.in_channels = int(cfg.get("in_channels", in_channels))
        pre_hidden: List[int] = list(cfg.get("pre_hidden", [64, 64]))
        sa_blocks_cfg: List[Dict[str, Any]] = list(
            cfg.get(
                "sa_blocks",
                [
                    {
                        "n_samples": 256,
                        "radius": 0.10,
                        "max_k": 32,
                        "out_ch": 128,
                        "mlp_hidden": [64],
                    },
                    {
                        "n_samples": 64,
                        "radius": 0.25,
                        "max_k": 32,
                        "out_ch": 256,
                        "mlp_hidden": [128],
                    },
                ],
            )
        )
        gf_hidden: List[int] = list(
            cfg.get("gf_hidden", [max(8, self.latent_dim // 2)])
        )
        # Normalization and pooling options
        norm_type: str = str(cfg.get("norm", "batch"))  # for SA and GF MLPs
        num_groups: int = int(cfg.get("num_groups", 16))
        pool_type: str = str(cfg.get("pool", "max"))  # 'max' | 'max+mean'

        # Optional Fourier positional encoding on xyz only
        posenc_cfg = cfg.get("posenc", None)
        self.posenc: Optional[FourierFeatures] = None
        xyz_dim = 2
        xyz_feat_dim = xyz_dim
        if isinstance(posenc_cfg, dict):
            n_freqs = int(posenc_cfg.get("n_freqs", 0))
            scale = float(posenc_cfg.get("scale", 1.0))
            if n_freqs > 0:
                self.posenc = FourierFeatures(
                    n_freqs=n_freqs, scale=scale, include_input=True
                )
                xyz_feat_dim = self.posenc.out_dim

        # Pre pointwise MLP on [xyz_encoding, extra_feats]
        in_ch_pre = xyz_feat_dim + self.in_channels
        pre_layers: List[nn.Module] = []
        dims = [in_ch_pre] + pre_hidden
        for i in range(len(dims) - 1):
            pre_layers.append(nn.Linear(dims[i], dims[i + 1]))
            pre_layers.append(nn.SiLU())
        self.pre = nn.Sequential(*pre_layers) if pre_layers else nn.Identity()
        current_in = dims[-1] if pre_layers else in_ch_pre

        # Set abstraction layers
        self.sa_layers = nn.ModuleList()
        for block in sa_blocks_cfg:
            n_samples = int(block.get("n_samples", 128))
            radius = float(block.get("radius", 0.1))
            max_k = int(block.get("max_k", 32))
            out_ch = int(block.get("out_ch", 128))
            mlp_hidden = block.get("mlp_hidden", None)
            if mlp_hidden is not None:
                mlp_hidden = [int(h) for h in mlp_hidden]
            sa = SetAbstraction(
                n_samples=n_samples,
                radius=radius,
                max_k=max_k,
                in_ch=current_in,
                out_ch=out_ch,
                mlp_hidden=mlp_hidden,
                norm=norm_type,
                num_groups=num_groups,
                pool=pool_type,
            )
            self.sa_layers.append(sa)
            current_in = out_ch * 2 if pool_type == "max+mean" else out_ch

        # Global feature aggregator
        self.glob = GlobalFeature(
            in_ch=current_in,
            latent_dim=self.latent_dim,
            hidden=gf_hidden,
            norm=norm_type,
            num_groups=num_groups,
        )

        # Persist resolved config
        self.encoder_cfg: Dict[str, Any] = {
            "latent_dim": self.latent_dim,
            "in_channels": self.in_channels,
            "pre_hidden": pre_hidden,
            "sa_blocks": [
                {
                    "n_samples": int(b.get("n_samples", 128)),
                    "radius": float(b.get("radius", 0.1)),
                    "max_k": int(b.get("max_k", 32)),
                    "out_ch": int(b.get("out_ch", 128)),
                    "mlp_hidden": list(
                        b.get("mlp_hidden", [int(b.get("out_ch", 128)) // 2])
                    ),
                }
                for b in sa_blocks_cfg
            ],
            "gf_hidden": gf_hidden,
            "norm": norm_type,
            "num_groups": num_groups,
            "pool": pool_type,
        }
        if isinstance(posenc_cfg, dict):
            self.encoder_cfg["posenc"] = {
                "n_freqs": int(posenc_cfg.get("n_freqs", 0)),
                "scale": float(posenc_cfg.get("scale", 1.0)),
            }

    def forward(
        self, xyz: torch.Tensor, feats: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        xyz_in = self.posenc(xyz) if self.posenc is not None else xyz
        if feats is not None and feats.numel() > 0:
            x_in = torch.cat([xyz_in, feats], dim=-1)
        else:
            x_in = xyz_in
        feats = self.pre(x_in)
        centers = xyz
        for sa in self.sa_layers:
            centers, feats = sa(centers, feats)
        latent = self.glob(feats)
        return latent


class PointNetMLPJoint(nn.Module):
    # Joint model: PointNet++ encoder + MLP head conditioned on query (x,y)
    def __init__(
        self,
        latent_dim: int = 128,
        mlp_hidden: Optional[List[int]] = None,
        out_dim: int = 1,
        encoder_cfg: Optional[Dict[str, Any]] = None,
        in_channels: int = 0,
    ):
        super().__init__()
        # Encoder (cfg latent_dim overrides arg if provided)
        self.encoder = PointNet2Encoder2D(
            latent_dim=latent_dim, encoder_cfg=encoder_cfg, in_channels=in_channels
        )
        eff_latent = self.encoder.latent_dim
        if mlp_hidden is None:
            mlp_hidden = [256, 256, 128]
        self.head_hidden = list(mlp_hidden)
        self.out_dim = int(out_dim)
        # Optional Fourier features for query points in head
        head_posenc_cfg = None
        if encoder_cfg is not None:
            head_posenc_cfg = encoder_cfg.get("head_posenc", None)
        self.head_posenc: Optional[FourierFeatures] = None
        q_in_dim = 2
        if isinstance(head_posenc_cfg, dict):
            n_freqs = int(head_posenc_cfg.get("n_freqs", 0))
            scale = float(head_posenc_cfg.get("scale", 1.0))
            if n_freqs > 0:
                self.head_posenc = FourierFeatures(
                    n_freqs=n_freqs, scale=scale, include_input=True
                )
                q_in_dim = self.head_posenc.out_dim

        # Head normalization/dropout options
        head_norm = "batch"
        head_dropout = 0.0
        if encoder_cfg is not None:
            head_norm = str(encoder_cfg.get("head_norm", "batch"))
            head_dropout = float(encoder_cfg.get("head_dropout", 0.0))

        in_dim = eff_latent + q_in_dim
        self.head = MLP(
            in_dim,
            self.head_hidden,
            out_dim=self.out_dim,
            norm=head_norm,
            num_groups=16,
            dropout=head_dropout,
        )

        # Persist arch for checkpoints
        enc_cfg_persist = dict(self.encoder.encoder_cfg)
        if isinstance(head_posenc_cfg, dict) and head_posenc_cfg.get("n_freqs", 0) > 0:
            enc_cfg_persist["head_posenc"] = {
                "n_freqs": int(head_posenc_cfg.get("n_freqs", 0)),
                "scale": float(head_posenc_cfg.get("scale", 1.0)),
            }
        # persist head normalization/dropout
        enc_cfg_persist["head_norm"] = head_norm
        enc_cfg_persist["head_dropout"] = head_dropout
        self._arch: Dict[str, Any] = {
            "encoder_cfg": enc_cfg_persist,
            "head_hidden": list(self.head_hidden),
            "out_dim": self.out_dim,
        }

    def get_arch(self) -> Dict[str, Any]:
        # Return the persisted architecture (including optional head_posenc if present).
        # self._arch was constructed in __init__ with encoder_cfg (plus head_posenc when used)
        return dict(self._arch)

    def forward(
        self,
        geom_xyz: torch.Tensor,
        query_points: torch.Tensor,
        geom_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.encoder(geom_xyz, geom_feats)  # [B,L]
        B, Q, _ = query_points.shape
        q_feat = (
            self.head_posenc(query_points)
            if self.head_posenc is not None
            else query_points
        )
        z_exp = z.unsqueeze(1).expand(-1, Q, -1)
        x = torch.cat([z_exp, q_feat], dim=-1)
        x = x.reshape(B * Q, -1)
        y = self.head(x)
        y = y.view(B, Q, self.out_dim)
        return y

class FeaturePropagation(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 mlp_hidden: List[int], norm: str = "layer", num_groups: int = 16):
        super().__init__()
        # in_ch: channels from the coarser layer being upsampled
        # skip_ch: channels from the skip connection (same resolution as target)
        self.mlp = MLP(in_ch + skip_ch, mlp_hidden, out_ch,
                       norm=norm, num_groups=num_groups)

    def forward(self, xyz_dense: torch.Tensor, xyz_sparse: torch.Tensor,
                feats_sparse: torch.Tensor, feats_skip: torch.Tensor) -> torch.Tensor:
        # xyz_dense:    [B, N, 2]  — points to interpolate TO
        # xyz_sparse:   [B, M, 2]  — points to interpolate FROM
        # feats_sparse: [B, M, C1] — features at sparse points
        # feats_skip:   [B, N, C2] — skip connection features at dense points
        B, N, _ = xyz_dense.shape
        B, M, C1 = feats_sparse.shape

        # 3-nearest-neighbour inverse distance weighting
        dists = torch.cdist(xyz_dense, xyz_sparse)          # [B, N, M]
        k = min(3, M)
        knn_dists, knn_idx = torch.topk(-dists, k=k, dim=-1)  # [B, N, k]
        knn_dists = -knn_dists                                  # positive distances

        # Avoid division by zero for coincident points
        knn_dists = torch.clamp(knn_dists, min=1e-8)
        weights = 1.0 / knn_dists                           # [B, N, k]
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Gather and weighted-sum features
        # knn_idx_exp = knn_idx.unsqueeze(-1).expand(-1, -1, -1, C1)  # [B,N,k,C1]
        # gathered = torch.gather(
        #     feats_sparse.unsqueeze(2).expand(B, M, k, C1).permute(0,2,1,3)
        #     .reshape(B, k, M, C1),
        #     2,
        #     knn_idx_exp.permute(0,2,1,3).reshape(B, k, N, C1)
        # )
        # Simpler gather:
        fs_exp = feats_sparse.unsqueeze(1).expand(B, N, M, C1)  # [B,N,M,C1]
        knn_idx_c = knn_idx.unsqueeze(-1).expand(-1, -1, -1, C1)  # [B,N,k,C1]
        gathered = torch.gather(fs_exp, 2, knn_idx_c)              # [B,N,k,C1]
        interp = (weights.unsqueeze(-1) * gathered).sum(dim=2)      # [B,N,C1]

        # Concatenate skip and run MLP
        x = torch.cat([interp, feats_skip], dim=-1)                 # [B,N,C1+C2]
        B2, N2, Cin = x.shape
        x = self.mlp(x.reshape(B2 * N2, Cin)).reshape(B2, N2, -1)
        return x


def build_model_from_arch(arch: Dict[str, Any]) -> PointNetMLPJoint:
    """Reconstruct a PointNetMLPJoint from an 'arch' dict saved in a checkpoint.

    Expected keys:
        arch = {
            'encoder_cfg': {...},
            'head_hidden': [...],
            'out_dim': 1
        }
    """
    encoder_cfg = arch.get("encoder_cfg", None)
    head_hidden = arch.get("head_hidden", [256, 256, 128])
    out_dim = int(arch.get("out_dim", 1))
    latent_dim = int(encoder_cfg.get("latent_dim", 128)) if encoder_cfg else 128
    in_channels = int(encoder_cfg.get("in_channels", 2)) if encoder_cfg else 2
    return PointNetMLPJoint(
        latent_dim=latent_dim,
        mlp_hidden=head_hidden,
        out_dim=out_dim,
        encoder_cfg=encoder_cfg,
        in_channels=in_channels,
    )



class PointNetMLPJoint_FP(nn.Module):
    def __init__(self, latent_dim=128, mlp_hidden=None, out_dim=1,
                 encoder_cfg=None, in_channels=0):
        super().__init__()
        cfg = dict(encoder_cfg) if encoder_cfg else {}

        # --- identical config parsing to PointNet2Encoder2D ---
        self.latent_dim = int(cfg.get("latent_dim", latent_dim))
        self.in_channels = int(cfg.get("in_channels", in_channels))
        pre_hidden: List[int] = list(cfg.get("pre_hidden", [64, 64]))
        sa_blocks_cfg = list(cfg.get("sa_blocks", [
            {"n_samples": 256, "radius": 0.10, "max_k": 32, "out_ch": 128, "mlp_hidden": [64]},
            {"n_samples": 64,  "radius": 0.25, "max_k": 32, "out_ch": 256, "mlp_hidden": [128]},
        ]))
        gf_hidden: List[int] = list(cfg.get("gf_hidden", [max(8, self.latent_dim // 2)]))
        norm_type: str = str(cfg.get("norm", "layer"))
        num_groups: int = int(cfg.get("num_groups", 16))
        pool_type: str = str(cfg.get("pool", "max"))

        # Fourier posenc on encoder input
        posenc_cfg = cfg.get("posenc", None)
        self.posenc: Optional[FourierFeatures] = None
        xyz_feat_dim = 2
        if isinstance(posenc_cfg, dict):
            n_freqs = int(posenc_cfg.get("n_freqs", 0))
            scale = float(posenc_cfg.get("scale", 1.0))
            if n_freqs > 0:
                self.posenc = FourierFeatures(n_freqs=n_freqs, scale=scale, include_input=True)
                xyz_feat_dim = self.posenc.out_dim

        # Pre MLP
        in_ch_pre = xyz_feat_dim + self.in_channels
        pre_layers: List[nn.Module] = []
        dims = [in_ch_pre] + pre_hidden
        for i in range(len(dims) - 1):
            pre_layers.append(nn.Linear(dims[i], dims[i + 1]))
            pre_layers.append(nn.SiLU())
        self.pre = nn.Sequential(*pre_layers) if pre_layers else nn.Identity()
        pre_out_ch = dims[-1] if pre_layers else in_ch_pre  # channel count after pre MLP

        # SA layers — track output channels per layer for FP skip connections
        self.sa_layers = nn.ModuleList()
        sa_out_chs: List[int] = []
        current_in = pre_out_ch
        for block in sa_blocks_cfg:
            out_ch = int(block.get("out_ch", 128))
            mlp_hidden_sa = block.get("mlp_hidden", None)
            if mlp_hidden_sa is not None:
                mlp_hidden_sa = [int(h) for h in mlp_hidden_sa]
            sa = SetAbstraction(
                n_samples=int(block.get("n_samples", 128)),
                radius=float(block.get("radius", 0.1)),
                max_k=int(block.get("max_k", 32)),
                in_ch=current_in, out_ch=out_ch,
                mlp_hidden=mlp_hidden_sa,
                norm=norm_type, num_groups=num_groups, pool=pool_type,
            )
            self.sa_layers.append(sa)
            effective_ch = out_ch * 2 if pool_type == "max+mean" else out_ch
            sa_out_chs.append(effective_ch)
            current_in = effective_ch

        # Global feature (keep for z global context)
        self.glob = GlobalFeature(
            in_ch=current_in, latent_dim=self.latent_dim,
            hidden=gf_hidden, norm=norm_type, num_groups=num_groups,
        )

        # FP layers — budget-aware channel sizes
        fp_cfg = cfg.get("fp", {})
        fp2_out = int(fp_cfg.get("fp2_out", 128))
        fp1_out = int(fp_cfg.get("fp1_out", 64))

        # FP2: SA2 output → SA1 resolution (skip = SA1 features)
        self.fp2 = FeaturePropagation(
            in_ch=sa_out_chs[1], skip_ch=sa_out_chs[0],
            out_ch=fp2_out, mlp_hidden=[fp2_out], norm=norm_type, num_groups=num_groups,
        )
        # FP1: FP2 output → original point resolution (skip = pre MLP features)
        self.fp1 = FeaturePropagation(
            in_ch=fp2_out, skip_ch=pre_out_ch,
            out_ch=fp1_out, mlp_hidden=[fp1_out], norm=norm_type, num_groups=num_groups,
        )

        # Head posenc
        head_posenc_cfg = cfg.get("head_posenc", None)
        self.head_posenc: Optional[FourierFeatures] = None
        q_in_dim = 2
        if isinstance(head_posenc_cfg, dict):
            n_freqs = int(head_posenc_cfg.get("n_freqs", 0))
            scale = float(head_posenc_cfg.get("scale", 1.0))
            if n_freqs > 0:
                self.head_posenc = FourierFeatures(n_freqs=n_freqs, scale=scale, include_input=True)
                q_in_dim = self.head_posenc.out_dim

        head_norm = str(cfg.get("head_norm", "layer"))
        head_dropout = float(cfg.get("head_dropout", 0.0))
        self.out_dim = int(out_dim)
        self.head_hidden = list(mlp_hidden) if mlp_hidden else [256, 256, 128]

        # Head input = per-node FP features + global z + query posenc
        head_in_dim = fp1_out + self.latent_dim + q_in_dim
        self.head = MLP(head_in_dim, self.head_hidden, out_dim,
                        norm=head_norm, num_groups=num_groups, dropout=head_dropout)

        # Persist arch
        self._arch: Dict[str, Any] = {
            "encoder_cfg": dict(cfg),
            "head_hidden": self.head_hidden,
            "out_dim": self.out_dim,
        }

    def get_arch(self) -> Dict[str, Any]:
        return dict(self._arch)

    def forward(self, geom_xyz: torch.Tensor, query_points: torch.Tensor,
                geom_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = geom_xyz.shape

        # 1. Pre MLP
        xyz_enc = self.posenc(geom_xyz) if self.posenc is not None else geom_xyz
        x_in = torch.cat([xyz_enc, geom_feats], dim=-1) if (geom_feats is not None and geom_feats.numel() > 0) else xyz_enc
        pre_feats = self.pre(x_in.reshape(B * N, -1)).reshape(B, N, -1)  # [B, N, pre_out_ch]

        # 2. SA1 — save xyz and features for FP skip
        xyz1, feats1 = self.sa_layers[0](geom_xyz, pre_feats)   # [B, n1, sa_out_chs[0]]

        # 3. SA2
        xyz2, feats2 = self.sa_layers[1](xyz1, feats1)           # [B, n2, sa_out_chs[1]]

        # 4. Global z for context
        z = self.glob(feats2)                                     # [B, latent_dim]

        # 5. FP2: interpolate SA2 → SA1 resolution
        fp2_feats = self.fp2(xyz1, xyz2, feats2, feats1)          # [B, n1, fp2_out]

        # 6. FP1: interpolate SA1 → original N resolution
        fp1_feats = self.fp1(geom_xyz, xyz1, fp2_feats, pre_feats)  # [B, N, fp1_out]

        # 7. For each query point get its corresponding per-node FP feature
        #    Assumes query_points are a subset of geom_xyz — use nearest neighbour
        Q = query_points.shape[1]
        dists = torch.cdist(query_points, geom_xyz)               # [B, Q, N]
        nn_idx = dists.argmin(dim=-1)                             # [B, Q]
        nn_idx_exp = nn_idx.unsqueeze(-1).expand(-1, -1, fp1_feats.shape[-1])
        node_feats = torch.gather(fp1_feats, 1, nn_idx_exp)       # [B, Q, fp1_out]

        # 8. Head: [node_feats | z | posenc(query)]
        q_enc = self.head_posenc(query_points) if self.head_posenc is not None else query_points
        z_exp = z.unsqueeze(1).expand(-1, Q, -1)
        x = torch.cat([node_feats, z_exp, q_enc], dim=-1)        # [B, Q, head_in_dim]
        x = x.reshape(B * Q, -1)
        y = self.head(x).reshape(B, Q, self.out_dim)
        assert y.shape == (B, Q, self.out_dim)
        return y

def build_fp_model_from_arch(arch: Dict[str, Any]) -> PointNetMLPJoint_FP:
    encoder_cfg = arch.get("encoder_cfg", None)
    head_hidden = arch.get("head_hidden", [256, 256, 128])
    out_dim = int(arch.get("out_dim", 1))
    latent_dim = int(encoder_cfg.get("latent_dim", 128)) if encoder_cfg else 128
    in_channels = int(encoder_cfg.get("in_channels", 0)) if encoder_cfg else 0
    return PointNetMLPJoint_FP(
        latent_dim=latent_dim, mlp_hidden=head_hidden,
        out_dim=out_dim, encoder_cfg=encoder_cfg, in_channels=in_channels,
    )