import random
import json
import hashlib
from typing import List, Tuple, Dict, Optional

import h5py
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from benchmarks import ArGEnTDeepONet

project_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)
parent_dir = Path(project_dir).parent

# Defer device prints and data loading to main() to avoid re-exec in worker processes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== PER-ABLATION CONFIG ====
TARGET_NAMES: List[str] = ["Stress", "LogLife"]
INPUT_COLS: List[int] = [0, 1, 2]
H5_FILENAME: str = "disc_dataset_edge_deriv_zonal.h5"
EXPECTED_REPR: str = "edge"
# ==== END PER-ABLATION CONFIG ====

NUM_TARGETS: int = len(TARGET_NAMES)
QUERY_COLS: List[int] = [0, 1]  # head query always uses (x, r)


def target_cols_for_width(width: int) -> Tuple[int, ...]:
    """Loader appends stress at width-2 and log10(life) at width-1.

    Returns both when NUM_TARGETS == 2, else only the life column.
    """
    stress_col = width - 2
    life_col = width - 1
    if NUM_TARGETS == 2:
        return (stress_col, life_col)
    return (life_col,)


def build_enc_norm(
    coord_center: torch.Tensor,
    coord_half_range: torch.Tensor,
    extra_feat_stats: Dict[int, Dict[str, float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build per-INPUT_COL normalization (mean, std) vectors aligned to INPUT_COLS.

    Cols 0/1 use coord min-max (center, half_range); other cols use z-score stats.
    """
    means: List[float] = []
    stds: List[float] = []
    for c in INPUT_COLS:
        if c == 0:
            means.append(float(coord_center[0]))
            stds.append(float(coord_half_range[0]))
        elif c == 1:
            means.append(float(coord_center[1]))
            stds.append(float(coord_half_range[1]))
        else:
            st = extra_feat_stats[c]
            means.append(float(st["mean"]))
            stds.append(float(st["std"]))
    enc_mean = torch.tensor(means, dtype=torch.float32)
    enc_std = torch.clamp(torch.tensor(stds, dtype=torch.float32), min=1e-8)
    return enc_mean, enc_std


# ---------------------------------------------------------------------------
# SDF computation helpers (used when SDF is not pre-stored in HDF5)
# ---------------------------------------------------------------------------

def _dist_point_to_segment_batch(
    px: np.ndarray, py: np.ndarray,
    ax: float, ay: float, bx: float, by: float,
) -> np.ndarray:
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-12:
        return np.hypot(px - ax, py - ay)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / len2, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))





def load_h5_pointsets(path: Path) -> List[torch.Tensor]:
    """
    Returns list of tensors, one per sample.
    
    Input columns (always):
        0: x (mm)
        1: r (mm)
        2: zone_id
    
    Input columns (edge only — arc_length present and fully valid):
        3: arc_length_mm
        4: tangent_x          (if derivatives stored)
        5: tangent_r          (if derivatives stored)
        6: curvature          (if derivatives stored)
        7: curvature_gradient (if derivatives stored)
    
    Target columns (always last):
        -2: stress_max_vm
        -1: life_raw (log10 scale)
    """
    sets: List[torch.Tensor] = []

    with h5py.File(path, "r") as f:
        print("N samples:", len(f["samples"]))
        for name in sorted(f["samples"].keys()):
            grp = f["samples"][name]

            coords   = grp["node_coords_mm"][:]
            zone_id  = grp["zone_id"][:].astype(np.float32)
            stress   = grp["stress_max_vm"][:]
            life     = grp["life_raw"][:]

            columns = [
                coords[:, 0],  # x
                coords[:, 1],  # r
                zone_id,
            ]

            # Arc length only if present and fully valid (i.e. pure edge representation)
            if "arc_length_mm" in grp:
                arc = grp["arc_length_mm"][:].astype(np.float32)
                if not np.any(np.isnan(arc)):
                    columns.append(arc)

            # Derivative features only if stored and non-empty
            node_features = grp["node_features"][:]
            if node_features.shape[1] > 0:
                for i in range(node_features.shape[1]):
                    columns.append(node_features[:, i].astype(np.float32))
            life = np.log10(life).astype(np.float32)  # Convert life to log10 scale
            columns.append(stress)
            columns.append(life)

            arr = np.stack(columns, axis=-1).astype(np.float32)
            sets.append(torch.tensor(arr, dtype=torch.float32))

    return sets


# Loaded in main()


class GeomLifeDataset(Dataset):
    """
    Geometry-level dataset. Each item is one simulation geometry with variable number of points.
    Encoder features come from INPUT_COLS; head query is always (x, r); targets per NUM_TARGETS.
    Applies global normalization using provided stats.
    """

    def __init__(
        self,
        tensors: List[torch.Tensor],
        coord_center: torch.Tensor,
        coord_half_range: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        extra_feat_stats: Dict[int, Dict[str, float]],
    ) -> None:
        super().__init__()
        self.items: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.coord_center = coord_center
        self.coord_half_range = torch.clamp(coord_half_range, min=1e-8)
        self.target_mean = target_mean
        self.target_std = torch.clamp(target_std, min=1e-8)
        self.extra_feat_stats = extra_feat_stats
        self.enc_mean, self.enc_std = build_enc_norm(
            coord_center, coord_half_range, extra_feat_stats
        )
        for t in tensors:
            width = t.shape[1]
            required_cols = max(INPUT_COLS) + 1
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
            self.items.append((enc_feats, query_xy, target))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc_feats, xy, target = self.items[idx]
        # Normalize with GLOBAL stats computed from training set
        enc_n = (enc_feats - self.enc_mean) / self.enc_std
        xyn = (xy - self.coord_center) / self.coord_half_range
        target_n = (target - self.target_mean) / self.target_std
        return {
            "enc_feats": enc_n,   # [N, len(INPUT_COLS)] normalized encoder input
            "points": xyn,   # [N, 2] normalized (x, r) query coords
            "target": target_n,    # [N, NUM_TARGETS] standardized targets
            # Provide also unnormalized for potential analysis if needed
            "enc_feats_raw": enc_feats,
            "points_raw": xy,
            "target_raw": target,
        }


def default_collate_variable(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    # We expect batch_size=1; keep interface flexible
    assert len(batch) == 1, "Use batch_size=1 for variable-size point sets."
    return batch[0]


def make_collate_fixed_points(k: int):
    """Return a collate function that samples exactly k points per geometry (with replacement if N<k)
    and stacks a batch: points [B,k,2], target [B,k,2].
    """

    def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        gp_b: List[torch.Tensor] = []
        qp_b: List[torch.Tensor] = []
        t_b: List[torch.Tensor] = []
        for item in batch:
            enc = item["enc_feats"]  # [N, len(INPUT_COLS)]
            pts = item["points"]  # [N,2]
            target = item["target"]
            N = pts.shape[0]
            if N >= k:
                idx = torch.randperm(N)[:k]
            else:
                # sample with replacement to reach k
                idx = torch.randint(0, N, (k,))
            gp_b.append(enc[idx])
            qp_b.append(pts[idx])
            t_b.append(target[idx])
        return {
            "geom_points": torch.stack(gp_b, dim=0),
            "query_points": torch.stack(qp_b, dim=0),
            "target": torch.stack(t_b, dim=0),
        }

    return _collate


class DualSamplerCollate:
    """Pickle-safe callable collate that samples two point sets per geometry.

    Returns dict with keys:
      - 'geom_points': [B,K_enc,2]
      - 'query_points': [B,K_q,2]
            - 'target': [B,K_q,2]
    """

    def __init__(self, k_enc: int, k_q: int) -> None:
        self.k_enc = int(k_enc)
        self.k_q = int(k_q)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        gp_b: List[torch.Tensor] = []
        qp_b: List[torch.Tensor] = []
        t_b: List[torch.Tensor] = []
        for item in batch:
            enc = item["enc_feats"]  # [N, len(INPUT_COLS)]
            pts = item["points"]  # [N,2]
            target = item["target"]
            N = pts.shape[0]
            # Encoder samples
            if N >= self.k_enc:
                idx_enc = torch.randperm(N)[: self.k_enc]
            else:
                idx_enc = torch.randint(0, N, (self.k_enc,))
            # Query samples
            if N >= self.k_q:
                idx_q = torch.randperm(N)[: self.k_q]
            else:
                idx_q = torch.randint(0, N, (self.k_q,))
            gp_b.append(enc[idx_enc])
            qp_b.append(pts[idx_q])
            t_b.append(target[idx_q])
        return {
            "geom_points": torch.stack(gp_b, dim=0),
            "query_points": torch.stack(qp_b, dim=0),
            "target": torch.stack(t_b, dim=0),
        }


class AllNodesPadCollate:
    """Pickle-safe collate that uses ALL nodes per geometry.

    Pads smaller point clouds in the batch with zeros up to maxN and returns a
    boolean mask of shape [B, maxN] (True for real points, False for padded zeros):
      - 'geom_points': [B,maxN,C]
      - 'query_points': [B,maxN,C]
            - 'target': [B,maxN,2]
      - 'mask': [B,maxN] bool, True=real point, False=zero-padded
    """

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        Ns = [item["points"].shape[0] for item in batch]
        maxN = max(Ns)
        enc_C = batch[0]["enc_feats"].shape[1]
        q_C = batch[0]["points"].shape[1]
        gp_b: List[torch.Tensor] = []
        qp_b: List[torch.Tensor] = []
        t_b: List[torch.Tensor] = []
        mask_b: List[torch.Tensor] = []
        for item, N in zip(batch, Ns):
            enc = item["enc_feats"]  # [N, len(INPUT_COLS)]
            pts = item["points"]  # [N, q_C]
            target = item["target"]    # [N, NUM_TARGETS]
            if N < maxN:
                pad = maxN - N
                enc_pad = torch.cat([enc, torch.zeros(pad, enc_C, dtype=enc.dtype)], dim=0)
                pts_pad = torch.cat([pts, torch.zeros(pad, q_C, dtype=pts.dtype)], dim=0)
                target_pad = torch.cat([target, torch.zeros(pad, target.shape[1], dtype=target.dtype)], dim=0)
            else:
                enc_pad = enc
                pts_pad = pts
                target_pad = target
            m = torch.zeros(maxN, dtype=torch.bool)
            m[:N] = True
            gp_b.append(enc_pad)
            qp_b.append(pts_pad)
            t_b.append(target_pad)
            mask_b.append(m)
        stacked_mask = torch.stack(mask_b, dim=0)  # [B, maxN] bool
        return {
            "geom_points": torch.stack(gp_b, dim=0),
            "query_points": torch.stack(qp_b, dim=0),
            "target": torch.stack(t_b, dim=0),
            "mask": stacked_mask,     # [B, maxN] bool query padding mask
            "kv_mask": stacked_mask,  # geometry mask == query mask (same padded node set)
        }


def compute_global_normalization(
    train_tensors: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, Dict[str, float]]]:
    """Compute coord center/half-range, target mean/std, and extra feature stats.

    Coords (x, r) use min-max; targets and extra features (cols 2-7) use z-score.
    zone_id (col 2) is mapped via fixed {mean=0, std=4} to scale {0..4} -> {0..1}.
    """
    width = train_tensors[0].shape[1]
    tcols = target_cols_for_width(width)

    all_xy = torch.cat([t[:, :2]  for t in train_tensors], dim=0)
    all_targets = torch.cat([t[:, list(tcols)] for t in train_tensors], dim=0)
    xy_min = all_xy.min(dim=0).values
    xy_max = all_xy.max(dim=0).values
    coord_center     = 0.5 * (xy_min + xy_max)
    coord_half_range = torch.clamp(0.5 * (xy_max - xy_min), min=1e-6)
    target_mean = all_targets.mean(dim=0)
    target_std = all_targets.std(dim=0, unbiased=False).clamp(min=1e-6)

    extra_feat_stats: Dict[int, Dict[str, float]] = {}
    for c in INPUT_COLS:
        if c in (0, 1):
            continue
        if c == 2:
            # zone_id: fixed scaling, divide by 4.0
            extra_feat_stats[c] = {"mean": 0.0, "std": 4.0}
        else:
            vals = torch.cat([t[:, c] for t in train_tensors], dim=0)
            mean_c = float(vals.mean())
            std_c = float(vals.std(unbiased=False))
            extra_feat_stats[c] = {"mean": mean_c, "std": max(std_c, 1e-6)}

    return coord_center, coord_half_range, target_mean, target_std, extra_feat_stats


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    coord_center: torch.Tensor,
    coord_half_range: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    extra_feat_stats: Dict[int, Dict[str, float]],
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_points_per_geom: Optional[int] = None,
    grad_clip_norm: Optional[float] = 1.0,
    save_path: Optional[Path] = None,
    early_stopping_patience: Optional[int] = 20,
    early_stopping_min_delta: float = 0.0,
    use_amp: bool = False,
) -> None:
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_amp))
    best_val = float("inf")
    # For raw-space validation logging
    target_mean_d = target_mean.to(device)
    target_std_d = target_std.to(device)

    # Learning rate schedule (optional OneCycleLR based on rough steps)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
        min_lr=1e-6,
    )

    t0 = time.time()
    epochs_since_improve = 0
    start_epoch = 1
    train_history: list = []

    # Resume from checkpoint if the file already exists
    if save_path is not None and Path(save_path).exists():
        print(f"Found existing checkpoint at {save_path}. Resuming training...")
        try:
            resume_ckpt = torch.load(str(save_path), map_location=device, weights_only=False)
            model.load_state_dict(resume_ckpt["model_state"])
            if "optimizer_state" in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            if "scheduler_state" in resume_ckpt:
                scheduler.load_state_dict(resume_ckpt["scheduler_state"])
            if "scaler_state" in resume_ckpt:
                scaler.load_state_dict(resume_ckpt["scaler_state"])
            best_val = resume_ckpt.get("best_val_loss", float("inf"))
            epochs_since_improve = resume_ckpt.get("epochs_since_improve", 0)
            start_epoch = resume_ckpt.get("config", {}).get("epochs_trained", 0) + 1
            print(
                f"Resumed: best_val={best_val:.6f}, start_epoch={start_epoch}, "
                f"epochs_since_improve={epochs_since_improve}"
            )
            train_history = resume_ckpt.get("train_history", [])
        except (KeyError, RuntimeError, ValueError) as exc:
            print(f"Checkpoint incompatible ({exc}). Starting fresh.")

    for epoch in range(start_epoch, epochs + 1):
        epoch_t0 = time.time()
        model.train()
        train_loss = 0.0
        ntrain = 0
        for batch in train_loader:
            pad_mask: Optional[torch.Tensor] = None
            kv_mask: Optional[torch.Tensor] = None
            if "geom_points" in batch:
                # Dual-sampling batched mode
                gp: torch.Tensor = batch["geom_points"].to(device)  # [B,Kenc,2]
                query_xy: torch.Tensor = batch["query_points"].to(device)  # [B,Kq,2]
                target: torch.Tensor = batch["target"].to(device)  # [B,Kq,2] [Stress, LogLife]
                B, Kq, _ = query_xy.shape
                Bmul = B * Kq
                # Extract zero-padding mask from batched_all collate
                if "mask" in batch:
                    pad_mask = batch["mask"].to(device)  # [B, maxN] bool
                kv_mask = batch.get("kv_mask", None)
                if kv_mask is not None:
                    kv_mask = kv_mask.to(device)  # [B, maxN] bool
            else:
                # Full-geometry single/batched mode from earlier
                pts: torch.Tensor = batch["points"].to(device)  # [N,2] or [B,K,2] — fallback path (train_mode != batched_all)
                targets: torch.Tensor = batch["target"].to(device)  # [N,2] or [B,K,2]
                if pts.dim() == 2:
                    # Single geometry
                    N = pts.shape[0]
                    if max_points_per_geom is None:
                        query_xy = pts
                        target = targets
                    else:
                        q = min(N, max_points_per_geom)
                        idxs = torch.randperm(N, device=device)[:q]
                        query_xy = pts[idxs]
                        target = targets[idxs]
                    gp = pts.unsqueeze(0)  # [1,N,2]
                    query_xy = query_xy.unsqueeze(0)  # [1,q,2] or [1,N,2]
                    target = target.unsqueeze(0)
                    Bmul = query_xy.shape[1]
                else:
                    # Batched [B,K,*]
                    B, K, _ = pts.shape
                    if max_points_per_geom is None or max_points_per_geom >= K:
                        query_xy = pts
                        target = targets
                        Bmul = B * K
                    else:
                        q = max_points_per_geom
                        # Build per-batch indices
                        idxs = torch.stack(
                            [torch.randperm(K, device=device)[:q] for _ in range(B)],
                            dim=0,
                        )  # [B,q]
                        bidx = (
                            torch.arange(B, device=device).unsqueeze(-1).expand(-1, q)
                        )
                        query_xy = pts[bidx, idxs, :]
                        target = targets[bidx, idxs, :]
                        Bmul = B * q
                    gp = pts  # use same points for encoder

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=(device.type == "cuda" and use_amp)
            ):
                pred = model(gp, query_xy, pad_mask, kv_mask)  # [B,Kq,2]
                if pred.shape[-1] != target.shape[-1]:
                    raise RuntimeError(
                        f"Model/data mismatch: pred has {pred.shape[-1]} outputs but target has {target.shape[-1]}"
                    )
                diff2 = (pred - target) ** 2

                # Unweighted loss with mask support for padded nodes.
                if pad_mask is not None:
                    mask = pad_mask.unsqueeze(-1).to(diff2.dtype)
                    denom = (mask.sum() * diff2.shape[-1]).clamp(min=1.0)
                    loss = (diff2 * mask).sum() / denom
                else:
                    loss = diff2.mean()


            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * Bmul
            ntrain += Bmul

        train_loss /= max(1, ntrain)

        # Validation
        model.eval()
        val_loss = 0.0
        nval = 0
        n_targets = int(target_mean_d.numel())
        target_names = TARGET_NAMES if len(TARGET_NAMES) == n_targets else [f"Target_{i}" for i in range(n_targets)]
        se_sum = torch.zeros(n_targets, dtype=torch.float64)
        sum_y = torch.zeros(n_targets, dtype=torch.float64)
        sum_y2 = torch.zeros(n_targets, dtype=torch.float64)
        count_val_points = 0
        with torch.no_grad():
            for batch in val_loader:
                enc: torch.Tensor = batch["enc_feats"].to(device)  # [N, len(INPUT_COLS)]
                pts: torch.Tensor = batch["points"].to(device)  # [N,2] (x_norm, r_norm)
                target: torch.Tensor = batch["target"].to(device)  # [N, NUM_TARGETS]
                N = pts.shape[0]
                with torch.amp.autocast(
                    "cuda", enabled=(device.type == "cuda" and use_amp)
                ):
                    pred = model(enc.unsqueeze(0), pts.unsqueeze(0)).squeeze(0)  # [N, NUM_TARGETS]
                if pred.shape[-1] != target.shape[-1]:
                    raise RuntimeError(
                        f"Model/data mismatch in validation: pred has {pred.shape[-1]} outputs but target has {target.shape[-1]}"
                    )
                loss = (pred - target).pow(2).mean()
                val_loss += loss.item() * N

                target_std_v = target_std_d.view(1, -1)
                target_mean_v = target_mean_d.view(1, -1)
                pred_raw = pred * target_std_v + target_mean_v
                true_raw = target * target_std_v + target_mean_v

                d = pred_raw - true_raw
                se_sum += torch.sum(d**2, dim=0).double().cpu()
                sum_y += torch.sum(true_raw, dim=0).double().cpu()
                sum_y2 += torch.sum(true_raw**2, dim=0).double().cpu()

                count_val_points += int(N)
                nval += N
        val_loss /= max(1, nval)
        scheduler.step(val_loss)
        if count_val_points > 0:
            val_mse = (se_sum / count_val_points).tolist()
            r2_vals: List[float] = []
            for j in range(n_targets):
                mean_j = float(sum_y[j] / count_val_points)
                ss_tot_j = max(1e-12, float(sum_y2[j] - count_val_points * (mean_j**2)))
                r2_vals.append(1.0 - float(se_sum[j]) / ss_tot_j)
        else:
            val_mse = [0.0] * n_targets
            r2_vals = [float("nan")] * n_targets
        epoch_dt = time.time() - epoch_t0

        metric_parts: List[str] = []
        for name, mse_j, r2_j in zip(target_names, val_mse, r2_vals):
            metric_parts.append(f"MSE({name}): {mse_j:.2f}")
            metric_parts.append(f"R2({name}): {r2_j:.4f}")
        metrics_str = " | ".join(metric_parts)

        print(
            f"Epoch {epoch:03d} | train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f} | "
            f"{metrics_str} | lr: {optimizer.param_groups[0]['lr']:.2e} | epoch: {epoch_dt:.1f}s"
        )

        train_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # Checkpoint best
        if val_loss < (best_val - early_stopping_min_delta):
            best_val = val_loss
            epochs_since_improve = 0
            if save_path is not None:
                ckpt = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "arch": model.get_arch() if hasattr(model, "get_arch") else None,
                    "coord_center": coord_center.cpu(),
                    "coord_half_range": coord_half_range.cpu(),
                    "target_mean": target_mean.cpu(),
                    "target_std": target_std.cpu(),
                    "extra_feat_stats": extra_feat_stats,
                    "epochs_since_improve": epochs_since_improve,
                    "train_history": train_history,
                    "config": {
                        "epochs_trained": epoch,
                        "best_val": best_val,
                    },
                }
                # Convenience for validators expecting this key name
                ckpt["best_val_loss"] = best_val
                torch.save(ckpt, str(save_path))
                print(f"Saved best model to: {save_path}")
        else:
            epochs_since_improve += 1
            if (
                early_stopping_patience is not None
                and epochs_since_improve >= early_stopping_patience
            ):
                print(
                    f"Early stopping triggered after {epochs_since_improve} epochs without improvement. Best val MSE: {best_val:.6f}"
                )
                break

    dt = time.time() - t0
    print(f"Training finished in {dt/60:.1f} min. Best val MSE: {best_val:.6f}")


def main(preset_name: str = "S0", batch=8) -> None:
    # preset_name = "S0"
    # batch = 8
    print(
        f"Starting training script with preset '{preset_name}' and batch size {batch}"
    )
    # Device/backend setup

    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    print(f"Using device: {device}")

    # Locate HDF5 file based on dataset choice

    grandparent_dir = parent_dir.parent
    repo_dir = grandparent_dir.parent
    h5_dir = Path(repo_dir, "Data_gen", "output")
    h5py_path = Path(h5_dir, H5_FILENAME)
    if not h5py_path.exists():
        raise FileNotFoundError(
            f"HDF5 file not found at {h5py_path}. Please ensure the data generation step has been completed and the file is in the expected location."
        )
    with h5py.File(h5py_path, "r") as _h5f:
        _repr = _h5f.attrs.get("representation")
        if isinstance(_repr, bytes):
            _repr = _repr.decode("utf-8")
        if _repr != EXPECTED_REPR:
            raise RuntimeError(
                f"H5 representation mismatch at {h5py_path}: expected '{EXPECTED_REPR}', "
                f"found '{_repr}'. Wrong dataset file for this ablation."
            )

    print(f"Loading data from: {h5py_path}")
    PS_list_whole = load_h5_pointsets(h5py_path)
    print(f"Loaded {len(PS_list_whole)} datasets from the HDF5 file.")

    # Load external presets JSON to allow expanding model zoo without editing this script
    presets_path = Path(project_dir, "model_presets.json")
    if not presets_path.exists():
        raise FileNotFoundError(
            f"Preset file 'model_presets.json' not found at {presets_path}. Please create it or copy the provided template."
        )
    with open(presets_path, "r", encoding="utf-8") as f:
        try:
            PRESETS = json.load(f)
        except Exception as exc:
            raise RuntimeError(
                "Failed to parse model_presets.json (invalid JSON)"
            ) from exc
    if preset_name not in PRESETS:
        raise KeyError(
            f"Preset '{preset_name}' not found. Available presets: {', '.join(sorted(PRESETS.keys()))}"
        )

    _cfg = PRESETS[preset_name]
    # In-file configuration (no CLI needed)
    epochs: int = 50000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    # Use all points every step (no subsampling of queries)
    max_points_per_geom: Optional[int] = (
        None  # set to an int to sample per-geometry points per step
    )
    # Early stopping
    early_stopping_patience: int = 100
    early_stopping_min_delta: float = 0.0
    # Architecture – ArGEnT DeepONet parameters
    hidden_dim: int = int(_cfg.get("hidden_dim", 128))
    num_heads: int = int(_cfg.get("num_heads", 4))
    num_layers: int = int(_cfg.get("num_layers", 2))
    output_dim: int = int(_cfg.get("output_dim", 128))
    # Optional human-readable model name (prefix for the file); set to None to use default
    model_name: Optional[str] = _cfg.get("model_name")

    # Save path (unique per-architecture; overwrites across runs for the same arch)
    arch_for_hash = {
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "output_dim": output_dim,
        "out_channels": NUM_TARGETS,
        "attention_type": "cross",
        "use_sdf": False,
        "in_ch_geom": len(INPUT_COLS),
        "in_channels": len(INPUT_COLS),
        "input_cols": INPUT_COLS,
        "target_names": TARGET_NAMES,
    }
    arch_hash = hashlib.md5(
        json.dumps(arch_for_hash, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    save_dir = Path(project_dir, "Trained_models")
    base_name = model_name if model_name else "argent_cross_nosdf"
    save_path = save_dir / f"{base_name}_{arch_hash}.pt"

    set_seed(42)

    # Split geometries into train/val
    n_geoms = len(PS_list_whole)
    idxs = list(range(n_geoms))
    train_idx, val_idx = train_test_split(idxs, test_size=0.2, random_state=42)

    train_tensors = [PS_list_whole[i] for i in train_idx]
    val_tensors = [PS_list_whole[i] for i in val_idx]

    coord_center, coord_half_range, target_mean, target_std, extra_feat_stats = (
        compute_global_normalization(train_tensors)
    )
    print(
        f"Coord center={coord_center.numpy()}, half_range={coord_half_range.numpy()} | "
        f"target_mean={target_mean.numpy()}, target_std={target_std.numpy()}"
    )

    train_ds = GeomLifeDataset(
        train_tensors, coord_center, coord_half_range, target_mean, target_std,
        extra_feat_stats,
    )
    val_ds = GeomLifeDataset(
        val_tensors, coord_center, coord_half_range, target_mean, target_std,
        extra_feat_stats,
    )

    # Training mode: "full" (encoder sees ALL points, batch_size=1),
    #                "batched" (dual sampling), or
    #                "batched_all" (all nodes per geometry with padded repeats to batch max)
    train_mode = "batched_all"  # default per request
    if train_mode == "full":
        train_batch_size: int = 1
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=default_collate_variable,
            pin_memory=(device.type == "cuda"),
            persistent_workers=False,
        )
        # keep model forward using all points
        max_points_per_geom: Optional[int] = None
    elif train_mode == "batched":
        # Dual-sampling high-utilization settings (tune to your VRAM)
        train_batch_size: int = 4
        k_enc: int = 8192  # encoder points per geometry
        k_q: int = 8192  # query points per geometry
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=DualSamplerCollate(k_enc, k_q),
            pin_memory=(device.type == "cuda"),
            persistent_workers=True,
        )
        # no per-geometry query subsampling in the loop (dual provides fixed K)
        max_points_per_geom = None
    else:
        # batched_all: use ALL nodes per geometry; pad to batch max by repeating real indices (no zero pads)
        train_batch_size: int = batch
        print(f"Using 'batched_all' training with batch size {train_batch_size}")
        # Use top-level AllNodesPadCollate (pickle-safe). No local redefinition.
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=AllNodesPadCollate(),
            pin_memory=(device.type == "cuda"),
            persistent_workers=True,
        )
        max_points_per_geom = None
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=default_collate_variable,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    # Build architecture config from flags
    print(
        f"Building ArGEnTDeepONet (cross-attention, no SDF): "
        f"hidden_dim={hidden_dim}, num_heads={num_heads}, "
        f"num_layers={num_layers}, output_dim={output_dim}, in_ch_geom={len(INPUT_COLS)}."
    )

    model = ArGEnTDeepONet(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        output_dim=output_dim,
        out_channels=NUM_TARGETS,
        attention_type="cross",
        use_sdf=False,
        in_ch_geom=len(INPUT_COLS),
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {parameter_count:,}")
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {n_param:,}")
    # Ensure save directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving best checkpoint to: {save_path}")
    train(
        model,
        train_loader,
        val_loader,
        coord_center,
        coord_half_range,
        target_mean,
        target_std,
        extra_feat_stats,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        max_points_per_geom=max_points_per_geom,
        grad_clip_norm=0.5,
        save_path=save_path,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        use_amp=(device.type == "cuda"),
    )


if __name__ == "__main__":
    try:
        # Choose dataset: "L_bracket" for L-bracket geometry or "Plate_hole" for hole plate geometry
        main("S", batch=1)
    except Exception as e:
        print(f"Error during training: {e}")
        raise
