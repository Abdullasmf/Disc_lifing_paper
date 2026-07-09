#!/usr/bin/env python3
"""
patch_pointnet_features.py

Correct PointNet feature-path patch for Disc_lifing_paper.

What it changes
---------------
1) pn_models.py
   - SetAbstraction.forward now accepts:
       xyz:   [B,N,2]      spatial coordinates only
       feats: [B,N,C]      point features
   - PointNet2Encoder2D.forward now accepts:
       xyz:   [B,N,2]
       feats: [B,N,F] or None
   - The encoder uses xyz for FPS / ball query / relative coordinates,
     and concatenates grouped point features separately.
   - PointNetMLPJoint.forward now accepts:
       geom_xyz, query_points, geom_feats=None

2) Training_script*.py under */PointNetMLPJoint/
   - Keep QUERY_COLS = [0, 1]
   - Replace dataset outputs / collates / train-val calls so:
       geom_xyz   = t[:, [0,1]]
       geom_feats = t[:, extra feature cols]   (all cols 2 .. width-3)
       query_xy   = t[:, [0,1]]
   - Model call becomes:
       model(geom_xyz, query_xy, geom_feats)

This is the architecturally correct setup for your model style:
- extra features go into the PointNet encoder only
- query MLP gets only query coordinates
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path


PN_MODELS_SENTINEL = "# [patch_pointnet_features] patched"
TRAINING_SENTINEL = "# [patch_pointnet_features] dataset/model path patched"


def find_files(root):
    pn_models = sorted(glob.glob(os.path.join(root, "**", "pn_models.py"), recursive=True))
    training = sorted(glob.glob(os.path.join(root, "**", "Training_script*.py"), recursive=True))
    training = [p for p in training if f"{os.sep}PointNetMLPJoint{os.sep}" in p]
    return pn_models, training


def write_if_changed(path, content, dry_run):
    old = Path(path).read_text(encoding="utf-8")
    if old == content:
        return False
    if not dry_run:
        Path(path).write_text(content, encoding="utf-8", newline="")
    return True


def patch_pn_models(content, path):
    if PN_MODELS_SENTINEL in content:
        return content, False

    original = content

    # 1) SetAbstraction.forward signature
    content = re.sub(
        r"def forward\(\n\s*self, xyz: torch\.Tensor, feats: torch\.Tensor\n\s*\) -> Tuple\[torch\.Tensor, torch\.Tensor\]:",
        "def forward(\n        self, xyz: torch.Tensor, feats: Optional[torch.Tensor] = None\n    ) -> Tuple[torch.Tensor, torch.Tensor]:",
        content,
    )

    # 2) SetAbstraction.forward internals
    old_block = """        # xyz: [B,N,2]; feats: [B,N,C]
        B, N, _ = xyz.shape
        C = feats.shape[-1]"""
    new_block = """        # xyz: [B,N,2]; feats: [B,N,C] or None
        B, N, _ = xyz.shape
        if feats is None:
            feats = xyz
        C = feats.shape[-1]"""
    content = content.replace(old_block, new_block)

    old_mlp = "        self.mlp = MLP(in_ch + 2, mlp_hidden, out_ch, norm=norm, num_groups=num_groups)"
    new_mlp = "        self.mlp = MLP(in_ch + 2, mlp_hidden, out_ch, norm=norm, num_groups=num_groups)"
    content = content.replace(old_mlp, new_mlp)

    # 3) PointNet2Encoder2D __init__ defaults
    content = re.sub(
        r"in_channels: int = 2,",
        "in_channels: int = 0,",
        content,
        count=1,
    )

    old_inch = """        self.in_channels = int(cfg.get("in_channels", in_channels))"""
    new_inch = """        self.in_channels = int(cfg.get("in_channels", in_channels))"""
    content = content.replace(old_inch, new_inch)

    # 4) Replace pre-MLP input logic from using all channels as xyz to xyz + feats
    old_pre = """        # Optional Fourier positional encoding before pre-MLP
        posenc_cfg = cfg.get("posenc", None)
        self.posenc: Optional[FourierFeatures] = None
        in_ch_pre = self.in_channels
        if isinstance(posenc_cfg, dict):
            n_freqs = int(posenc_cfg.get("n_freqs", 0))
            scale = float(posenc_cfg.get("scale", 1.0))
            if n_freqs > 0:
                self.posenc = FourierFeatures(
                    n_freqs=n_freqs, scale=scale, include_input=True
                )
                in_ch_pre = self.posenc.out_dim

        # Pre pointwise MLP on coords
        pre_layers: List[nn.Module] = []
        dims = [in_ch_pre] + pre_hidden"""
    new_pre = """        # Optional Fourier positional encoding on xyz only
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
        dims = [in_ch_pre] + pre_hidden"""
    content = content.replace(old_pre, new_pre)

    # 5) PointNet2Encoder2D.forward signature and body
    old_forward = """    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        x_in = self.posenc(xyz) if self.posenc is not None else xyz
        feats = self.pre(x_in)
        centers = xyz
        for sa in self.sa_layers:
            centers, feats = sa(centers, feats)
        latent = self.glob(feats)
        return latent"""
    new_forward = """    def forward(
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
        return latent"""
    content = content.replace(old_forward, new_forward)

    # 6) PointNetMLPJoint __init__ default in_channels
    content = re.sub(
        r"in_channels: int = 2,",
        "in_channels: int = 0,",
        content,
        count=1,
    )

    # 7) PointNetMLPJoint.forward signature/body
    old_joint = """    def forward(
        self, geom_points: torch.Tensor, query_points: torch.Tensor
    ) -> torch.Tensor:
        z = self.encoder(geom_points)  # [B,L]
        B, Q, _ = query_points.shape"""
    new_joint = """    def forward(
        self,
        geom_xyz: torch.Tensor,
        query_points: torch.Tensor,
        geom_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.encoder(geom_xyz, geom_feats)  # [B,L]
        B, Q, _ = query_points.shape"""
    content = content.replace(old_joint, new_joint)

    if content != original:
        content = PN_MODELS_SENTINEL + "\n" + content
        return content, True
    return content, False


def patch_training_script(content, path):
    if TRAINING_SENTINEL in content:
        return content, False

    original = content

    # 1) config block: INPUT_COLS -> EXTRA_FEAT_COLS placeholder
    content = content.replace(
        'INPUT_COLS: List[int] = [0, 1]',
        'INPUT_COLS: List[int] = [0, 1]  # legacy, kept for compatibility\nEXTRA_FEAT_COLS: List[int] = []',
    )

    # 2) build_enc_norm over EXTRA_FEAT_COLS instead of INPUT_COLS
    content = content.replace("for c in INPUT_COLS:", "for c in EXTRA_FEAT_COLS:")

    # 3) GeomLifeDataset item packing
    old_dataset = """            required_cols = max(INPUT_COLS) + 1
            if width < required_cols:
                missing = [c for c in INPUT_COLS if c >= width]
                raise RuntimeError(
                    f"Tensor width {width} too small for INPUT_COLS={INPUT_COLS}; "
                    f"missing columns {missing}. Check H5 representation matches '{EXPECTED_REPR}'."
                )
            tcols = target_cols_for_width(width)
            enc_feats = t[:, INPUT_COLS].contiguous()
            query_xy = t[:, QUERY_COLS].contiguous()
            target = t[:, list(tcols)].contiguous()
            self.items.append((enc_feats, query_xy, target))"""
    new_dataset = """            tcols = target_cols_for_width(width)
            geom_xyz = t[:, [0, 1]].contiguous()
            if EXTRA_FEAT_COLS:
                geom_feats = t[:, EXTRA_FEAT_COLS].contiguous()
            else:
                geom_feats = t[:, 0:0].contiguous()
            query_xy = t[:, QUERY_COLS].contiguous()
            target = t[:, list(tcols)].contiguous()
            self.items.append((geom_xyz, geom_feats, query_xy, target))"""
    content = content.replace(old_dataset, new_dataset)

    old_getitem = """        enc_feats, xy, s = self.items[idx]
        # Normalize with GLOBAL stats computed from training set
        enc_n = (enc_feats - self.enc_mean) / self.enc_std
        xyn = (xy - self.coord_center) / self.coord_half_range
        zn = (s - self.target_mean) / self.target_std
        return {
            "enc_feats": enc_n,  # [N, len(INPUT_COLS)] normalized encoder input
            "points": xyn,  # [N,2] normalized (x,r) query coords
            "target": zn,  # [N, NUM_TARGETS] standardized targets
            # Provide also unnormalized for potential analysis if needed
            "enc_feats_raw": enc_feats,
            "points_raw": xy,
            "target_raw": s,
        }"""
    new_getitem = """        geom_xyz, geom_feats, xy, s = self.items[idx]
        # Normalize with GLOBAL stats computed from training set
        if geom_feats.numel() > 0:
            geom_feats_n = (geom_feats - self.enc_mean) / self.enc_std
        else:
            geom_feats_n = geom_feats
        xyn = (xy - self.coord_center) / self.coord_half_range
        zn = (s - self.target_mean) / self.target_std
        return {
            "geom_xyz": xyn,  # [N,2] normalized (x,r) for encoder neighborhoods
            "geom_feats": geom_feats_n,  # [N,F] normalized extra encoder features
            "points": xyn,  # [N,2] normalized (x,r) query coords
            "target": zn,  # [N, NUM_TARGETS] standardized targets
            "geom_xyz_raw": geom_xyz,
            "geom_feats_raw": geom_feats,
            "points_raw": xy,
            "target_raw": s,
        }"""
    content = content.replace(old_getitem, new_getitem)

    # 4) collates
    content = content.replace(
        'enc = item["enc_feats"]  # [N, len(INPUT_COLS)]',
        'geom_xyz = item["geom_xyz"]  # [N,2]\n            geom_feats = item["geom_feats"]  # [N,F]'
    )
    content = content.replace(
        "gp_b.append(enc[idx])",
        "gp_b.append(geom_xyz[idx])\n            gf_b.append(geom_feats[idx])"
    )
    content = content.replace(
        'return {\n            "geom_points": torch.stack(gp_b, dim=0),',
        'return {\n            "geom_points": torch.stack(gp_b, dim=0),\n            "geom_feats": torch.stack(gf_b, dim=0),'
    )

    # insert gf_b declarations in collates
    content = content.replace(
        "        gp_b: List[torch.Tensor] = []\n        qp_b: List[torch.Tensor] = []",
        "        gp_b: List[torch.Tensor] = []\n        gf_b: List[torch.Tensor] = []\n        qp_b: List[torch.Tensor] = []"
    )

    # dual sampler replacements
    content = content.replace(
        "            gp_b.append(enc[idx_enc])",
        "            gp_b.append(geom_xyz[idx_enc])\n            gf_b.append(geom_feats[idx_enc])"
    )

    # all nodes collate replacements
    content = content.replace(
        "            enc = item[\"enc_feats\"]  # [N, len(INPUT_COLS)]",
        "            geom_xyz = item[\"geom_xyz\"]  # [N,2]\n            geom_feats = item[\"geom_feats\"]  # [N,F]"
    )
    content = content.replace(
        "            gp_b.append(enc[enc_idx])",
        "            gp_b.append(geom_xyz[enc_idx])\n            gf_b.append(geom_feats[enc_idx])"
    )

    # 5) compute_global_normalization only for extra features, not x/r
    old_norm = """    extra_feat_stats: Dict[int, Dict[str, float]] = {}
    for c in INPUT_COLS:
        if c in (0, 1):
            continue"""
    new_norm = """    extra_feat_stats: Dict[int, Dict[str, float]] = {}
    for c in EXTRA_FEAT_COLS:"""
    content = content.replace(old_norm, new_norm)

    # 6) inject runtime extra-feature cols after PS_list_whole load
    marker = '    print(f"Loaded {len(PS_list_whole)} datasets from the HDF5 file.")\n'
    inject = (
        '    print(f"Loaded {len(PS_list_whole)} datasets from the HDF5 file.")\n'
        '    width0 = int(PS_list_whole[0].shape[1])\n'
        '    EXTRA_FEAT_COLS = list(range(2, width0 - 2))\n'
        '    print(f"[patch_pointnet_features] EXTRA_FEAT_COLS={EXTRA_FEAT_COLS} '
        '(n={len(EXTRA_FEAT_COLS)})")\n'
    )
    content = content.replace(marker, inject)

    # 7) model creation: in_channels should be number of extra feats, not len(INPUT_COLS)
    content = content.replace(
        "        in_channels=len(INPUT_COLS),",
        "        in_channels=len(EXTRA_FEAT_COLS),",
    )

    # 8) train loop batched path needs geom_feats
    content = content.replace(
        '                gp: torch.Tensor = batch["geom_points"].to(device)  # [B,Kenc,2]',
        '                gp: torch.Tensor = batch["geom_points"].to(device)  # [B,Kenc,2]\n'
        '                gf: torch.Tensor = batch["geom_feats"].to(device)  # [B,Kenc,F]'
    )

    content = content.replace(
        "                pred_z = model(gp, query_xy)  # [B,Kq,2] standardized",
        "                pred_z = model(gp, query_xy, gf)  # [B,Kq,2] standardized",
    )

    # non-batched fallback path
    content = content.replace(
        '                pts: torch.Tensor = batch["points"].to(device)  # [N,2] or [B,K,2]',
        '                pts: torch.Tensor = batch["points"].to(device)  # [N,2] or [B,K,2]\n'
        '                geom_xyz_b: torch.Tensor = batch["geom_xyz"].to(device)\n'
        '                geom_feats_b: torch.Tensor = batch["geom_feats"].to(device)'
    )
    content = content.replace(
        "                    gp = pts.unsqueeze(0)  # [1,N,2]",
        "                    gp = geom_xyz_b.unsqueeze(0)  # [1,N,2]\n                    gf = geom_feats_b.unsqueeze(0)"
    )
    content = content.replace(
        "                    gp = pts  # use same points for encoder",
        "                    gp = geom_xyz_b  # use same xyz for encoder\n                    gf = geom_feats_b"
    )

    # validation path
    content = content.replace(
        '                enc: torch.Tensor = batch["enc_feats"].to(device)  # [N, len(INPUT_COLS)]',
        '                geom_xyz: torch.Tensor = batch["geom_xyz"].to(device)  # [N,2]\n'
        '                geom_feats: torch.Tensor = batch["geom_feats"].to(device)  # [N,F]'
    )
    content = content.replace(
        '                    pred = model(enc.unsqueeze(0), pts.unsqueeze(0)).squeeze(',
        '                    pred = model(geom_xyz.unsqueeze(0), pts.unsqueeze(0), geom_feats.unsqueeze(0)).squeeze('
    )

    # 9) arch hash should use EXTRA_FEAT_COLS
    content = content.replace(
        '"input_cols": INPUT_COLS,',
        '"extra_feat_cols": EXTRA_FEAT_COLS,',
    )

    if content != original:
        content = TRAINING_SENTINEL + "\n" + content
        return content, True
    return content, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    print(f"Scanning under: {root}")

    pn_files, tr_files = find_files(root)

    if not pn_files and not tr_files:
        print("No target files found.")
        sys.exit(1)

    changed = 0

    for path in pn_files:
        text = Path(path).read_text(encoding="utf-8")
        new_text, did = patch_pn_models(text, path)
        if did:
            write_if_changed(path, new_text, args.dry_run)
            changed += 1
            print(f"{'[DRY-RUN] ' if args.dry_run else ''}{path}")
            print("  [pn_models] patched")

    for path in tr_files:
        text = Path(path).read_text(encoding="utf-8")
        new_text, did = patch_training_script(text, path)
        if did:
            write_if_changed(path, new_text, args.dry_run)
            changed += 1
            print(f"{'[DRY-RUN] ' if args.dry_run else ''}{path}")
            print("  [training] patched")

    print(
        f"\n{'[DRY-RUN] ' if args.dry_run else ''}Done. "
        f"{changed} file(s) {'would be' if args.dry_run else 'were'} modified."
    )
    if args.dry_run:
        print("Re-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()