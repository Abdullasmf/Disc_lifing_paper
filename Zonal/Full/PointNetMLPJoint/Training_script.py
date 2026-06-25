import random
import json
import hashlib
from typing import List, Tuple, Dict, Optional, Any

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

from pn_models import PointNetMLPJoint

project_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)

# Defer device prints and data loading to main() to avoid re-exec in worker processes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_NAMES: List[str] = ["Stress", "LogLife"]
TARGET_COLS: Tuple[int, int] = (2, 4)  # [x, y, stress, temp, log(life)]
NUM_TARGETS: int = len(TARGET_NAMES)

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

    x: [N,2] coordinates, y: [N,2] = [Stress, LogLife].
    Applies global normalization using provided stats.
    """

    def __init__(
        self,
        tensors: List[torch.Tensor],
        coord_center: torch.Tensor,
        coord_half_range: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.items: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.coord_center = coord_center
        self.coord_half_range = torch.clamp(coord_half_range, min=1e-8)
        self.target_mean = target_mean
        self.target_std = torch.clamp(target_std, min=1e-8)
        for t in tensors:
            required_cols = max(TARGET_COLS) + 1
            if t.shape[1] < required_cols:
                raise ValueError(
                    "Each tensor must be [N,5]: x,y,stress,temp,log(life)"
                )
            xy = t[:, :2].contiguous()
            target = t[:, list(TARGET_COLS)].contiguous()
            self.items.append((xy, target))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        xy, s = self.items[idx]
        # Normalize with GLOBAL stats computed from training set
        xyn = (xy - self.coord_center) / self.coord_half_range
        # s is [N,2], target_mean/std are [2]
        zn = (s - self.target_mean) / self.target_std
        return {
            "points": xyn,  # [N,2]
            "target": zn,  # [N,2] standardized (Stress, LogLife)
            # Provide also unnormalized for potential analysis if needed
            "points_raw": xy,
            "target_raw": s,  # [N,2] raw (Stress, LogLife)
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
        pts_b: List[torch.Tensor] = []
        t_b: List[torch.Tensor] = []
        for item in batch:
            pts = item["points"]  # [N,2] on CPU
            target = item["target"]  # [N,2]
            N = pts.shape[0]
            if N >= k:
                idx = torch.randperm(N)[:k]
            else:
                # sample with replacement to reach k
                idx = torch.randint(0, N, (k,))
            pts_b.append(pts[idx])
            t_b.append(target[idx])
        points = torch.stack(pts_b, dim=0)  # [B,k,2]
        target = torch.stack(t_b, dim=0)  # [B,k,2]
        return {"points": points, "target": target}

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
            pts = item["points"]  # [N,2] on CPU
            target = item["target"]  # [N,2]
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
            gp_b.append(pts[idx_enc])
            qp_b.append(pts[idx_q])
            t_b.append(target[idx_q])
        return {
            "geom_points": torch.stack(gp_b, dim=0),
            "query_points": torch.stack(qp_b, dim=0),
            "target": torch.stack(t_b, dim=0),
        }


class AllNodesPadCollate:
    """Pickle-safe collate that uses ALL nodes per geometry.

    Pads each geometry in the batch to the maximum node count by repeating valid indices
    (no zero-padding), and returns a dual-set dict compatible with the model forward:
      - 'geom_points': [B,maxN,2]
      - 'query_points': [B,maxN,2]
        - 'target': [B,maxN,2]
            - 'mask': [B,maxN,1] with 1 for real nodes and 0 for repeated pad slots
    """

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        Ns = [item["points"].shape[0] for item in batch]
        maxN = max(Ns)
        gp_b: List[torch.Tensor] = []
        qp_b: List[torch.Tensor] = []
        t_b: List[torch.Tensor] = []
        mask_b: List[torch.Tensor] = []
        for item, N in zip(batch, Ns):
            pts = item["points"]  # [N,2]
            target = item["target"]  # [N,2]
            idx_all = torch.arange(N)
            if N < maxN:
                extra = torch.randint(0, N, (maxN - N,))
                enc_idx = torch.cat([idx_all, extra], dim=0)
                qry_idx = enc_idx
                mask = torch.cat(
                    [torch.ones(N, dtype=torch.float32), torch.zeros(maxN - N, dtype=torch.float32)],
                    dim=0,
                )
            else:
                enc_idx = idx_all
                qry_idx = idx_all
                mask = torch.ones(maxN, dtype=torch.float32)
            gp_b.append(pts[enc_idx])
            qp_b.append(pts[qry_idx])
            t_b.append(target[qry_idx])
            mask_b.append(mask.unsqueeze(-1))
        return {
            "geom_points": torch.stack(gp_b, dim=0),
            "query_points": torch.stack(qp_b, dim=0),
            "target": torch.stack(t_b, dim=0),
            "mask": torch.stack(mask_b, dim=0),  # [B,maxN,1]
        }


def compute_global_normalization(
    train_tensors: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute coord center/half-range and target (Stress, LogLife) mean/std from training set.

    Expects each tensor row as [x, y, stress, temp, log(life)].
    """
    all_xy = torch.cat([t[:, :2] for t in train_tensors], dim=0)
    all_targets = torch.cat([t[:, list(TARGET_COLS)] for t in train_tensors], dim=0)

    xy_min = all_xy.min(dim=0).values
    xy_max = all_xy.max(dim=0).values
    coord_center = 0.5 * (xy_min + xy_max)
    coord_half_range = torch.clamp(0.5 * (xy_max - xy_min), min=1e-6)

    target_mean = all_targets.mean(dim=0)
    target_std = all_targets.std(dim=0, unbiased=False).clamp(min=1e-6)

    return coord_center, coord_half_range, target_mean, target_std


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
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_points_per_geom: Optional[int] = None,
    grad_clip_norm: Optional[float] = 1.0,
    save_path: Optional[Path] = None,
    early_stopping_patience: Optional[int] = 20,
    early_stopping_min_delta: float = 0.0,
    use_amp: bool = False,
    resume_checkpoint: Optional[Dict] = None,
    model_name: Optional[str] = None,
) -> None:
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and use_amp))
    best_val = float("inf")
    start_epoch = 1
    history: List[Dict[str, Any]] = []

    if resume_checkpoint is not None:
        print("Resuming training from checkpoint...")
        model.load_state_dict(resume_checkpoint["model_state"])
        if "config" in resume_checkpoint:
            start_epoch = resume_checkpoint["config"].get("epochs_trained", 0) + 1
            best_val = resume_checkpoint["config"].get("best_val", float("inf"))
        # Fallback for legacy keys
        if best_val == float("inf") and "best_val_loss" in resume_checkpoint:
            best_val = resume_checkpoint["best_val_loss"]
        if "optimizer_state" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if "scaler_state" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler_state"])
        if isinstance(resume_checkpoint.get("history"), list):
            history = list(resume_checkpoint["history"])
        print(f"Resumed state: start_epoch={start_epoch}, best_val={best_val:.6f}")

    # For raw-space validation logging
    target_mean_d = target_mean.to(device)  # [2]
    target_std_d = target_std.to(device)  # [2]

    # Learning rate schedule (optional OneCycleLR based on rough steps)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=10.0,
        anneal_strategy="cos",
        three_phase=False,
    )

    if resume_checkpoint is not None and "scheduler_state" in resume_checkpoint:
        try:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        except (KeyError, ValueError, RuntimeError) as exc:
            print(f"Warning: failed to restore scheduler state: {exc}")

    t0 = time.time()
    epochs_since_improve = 0
    for epoch in range(start_epoch, epochs + 1):
        epoch_t0 = time.time()
        model.train()
        train_loss = 0.0
        ntrain = 0.0
        for batch in train_loader:
            node_mask: Optional[torch.Tensor] = None
            if "geom_points" in batch:
                # Dual-sampling batched mode
                gp: torch.Tensor = batch["geom_points"].to(device)  # [B,Kenc,2]
                query_xy: torch.Tensor = batch["query_points"].to(device)  # [B,Kq,2]
                target_z: torch.Tensor = batch["target"].to(
                    device
                )  # [B,Kq,2] standardized (Stress, LogLife)
                B, Kq, _ = query_xy.shape
                Bmul = B * Kq
                if "mask" in batch:
                    node_mask = batch["mask"].to(device)  # [B,Kq,1]
            else:
                # Full-geometry single/batched mode from earlier
                pts: torch.Tensor = batch["points"].to(device)  # [N,2] or [B,K,2]
                target: torch.Tensor = batch["target"].to(device)  # [N,2] or [B,K,2]
                if pts.dim() == 2:
                    # Single geometry
                    N = pts.shape[0]
                    if max_points_per_geom is None:
                        query_xy = pts
                        target_z = target
                    else:
                        q = min(N, max_points_per_geom)
                        idxs = torch.randperm(N, device=device)[:q]
                        query_xy = pts[idxs]
                        target_z = target[idxs]
                    gp = pts.unsqueeze(0)  # [1,N,2]
                    query_xy = query_xy.unsqueeze(0)  # [1,q,2] or [1,N,2]
                    target_z = target_z.unsqueeze(0)
                    Bmul = query_xy.shape[1]
                else:
                    # Batched [B,K,*]
                    B, K, _ = pts.shape
                    if max_points_per_geom is None or max_points_per_geom >= K:
                        query_xy = pts
                        target_z = target
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
                        target_z = target[bidx, idxs, :]
                        Bmul = B * q
                    gp = pts  # use same points for encoder

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and use_amp)):
                pred_z = model(gp, query_xy)  # [B,Kq,2] standardized
                if pred_z.shape[-1] != target_z.shape[-1]:
                    raise RuntimeError(
                        f"Model/data mismatch: pred has {pred_z.shape[-1]} outputs but target has {target_z.shape[-1]}"
                    )

                diff2 = (pred_z - target_z) ** 2
                if isinstance(node_mask, torch.Tensor):
                    mask = node_mask.to(diff2.dtype)
                    denom = torch.clamp(mask.sum() * diff2.shape[-1], min=1.0)
                    loss = (diff2 * mask).sum() / denom
                    Bmul = int(node_mask.sum().item())
                else:
                    loss = diff2.mean()

            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)

            # Only step scheduler if optimizer step was not skipped (scale didn't decrease)
            scale_before = scaler.get_scale()
            scaler.update()
            scale_after = scaler.get_scale()

            if scale_after >= scale_before:
                scheduler.step()

            train_loss += loss.item() * Bmul
            ntrain += float(Bmul)

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
                pts: torch.Tensor = batch["points"].to(device)  # [N,2]
                target: torch.Tensor = batch["target"].to(
                    device
                )  # [N,2] standardized [Stress, LogLife]
                N = pts.shape[0]
                with torch.cuda.amp.autocast(
                    enabled=(device.type == "cuda" and use_amp)
                ):
                    pred = model(pts.unsqueeze(0), pts.unsqueeze(0)).squeeze(
                        0
                    )  # [N,2] standardized
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

                if epoch == 1 and count_val_points == 0:
                    print(
                        f"DEBUG: target_mean={target_mean_d.cpu().numpy()}, target_std={target_std_d.cpu().numpy()}"
                    )
                    for j, name in enumerate(target_names):
                        print(
                            f"DEBUG: {name} Pred range: {pred_raw[:, j].min().item():.2f} - {pred_raw[:, j].max().item():.2f}"
                        )

                d = pred_raw - true_raw
                se_sum += torch.sum(d**2, dim=0).double().cpu()
                sum_y += torch.sum(true_raw, dim=0).double().cpu()
                sum_y2 += torch.sum(true_raw**2, dim=0).double().cpu()

                count_val_points += int(N)
                nval += N

        val_loss /= max(1, nval)

        # Compute per-target R2/MSE in raw spaces
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
            metric_parts.append(f"{name} R2: {r2_j:.4f}")
            metric_parts.append(f"MSE({name}): {mse_j:.2f}")
        metrics_str = " | ".join(metric_parts)

        print(
            f"Ep {epoch:03d} | L_tot: {train_loss:.4f}/{val_loss:.4f} | {metrics_str} | lr: {scheduler.get_last_lr()[0]:.2e} | {epoch_dt:.1f}s"
        )

        val_mse_map = {name: float(mse_j) for name, mse_j in zip(target_names, val_mse)}
        val_r2_map = {name: float(r2_j) for name, r2_j in zip(target_names, r2_vals)}

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "target_names": target_names,
                "val_r2": val_r2_map,
                "val_mse": val_mse_map,
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )

        # Checkpoint best
        if val_loss < (best_val - early_stopping_min_delta):
            best_val = val_loss
            epochs_since_improve = 0
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

        # Save single checkpoint file each epoch (weights + optimizer/scheduler/scaler + history)
        if save_path is not None and epochs_since_improve == 0:
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
                "model_name": model_name,
                "history": history,
                "config": {
                    "epochs_trained": epoch,
                    "best_val": best_val,
                },
            }
            # Convenience for validators expecting this key name
            ckpt["best_val_loss"] = best_val
            torch.save(ckpt, str(save_path))
            print(f"Saved checkpoint to: {save_path}")

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

    # Locate HDF5 file only in main
    parent_dir = Path(project_dir).parent
    grandparent_dir = parent_dir.parent
    repo_dir = grandparent_dir.parent
    h5_dir = Path(repo_dir, "Data_gen", "output")
    h5py_path = Path(h5_dir, "disc_dataset_full_zonal.h5")
    if not h5py_path.exists():
        raise FileNotFoundError(
            f"HDF5 file not found at {h5py_path}. Please ensure the data generation step has been completed and the file is in the expected location."
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
    epochs: int = int(_cfg.get("epochs", 10000))
    lr: float = float(_cfg.get("lr", 3e-4))
    weight_decay: float = float(_cfg.get("weight_decay", 1e-4))
    # Use all points every step (no subsampling of queries)
    max_points_per_geom: Optional[int] = (
        None  # set to an int to sample per-geometry points per step
    )
    # Early stopping
    early_stopping_patience: int = 100
    early_stopping_min_delta: float = 0.0
    # Architecture
    latent_dim: int = int(_cfg["latent_dim"])  # encoder latent size
    pre_hidden: List[int] = list(_cfg["pre_hidden"])  # pre-MLP on coords
    sa_blocks: List[dict] = list(_cfg["sa_blocks"])  # set abstraction blocks
    gf_hidden: List[int] = list(_cfg["gf_hidden"])  # global feature head
    head_hidden: List[int] = list(_cfg["head_hidden"])  # MLP head sizes
    # Optional human-readable model name (prefix for the file); set to None to use default
    model_name: Optional[str] = _cfg.get("model_name", None)  # e.g., "pn_small_r0p08"

    # Fourier positional encodings to enhance spatial/detail sensitivity
    # Allow overriding positional encodings per preset; default to 4 freqs if unspecified
    posenc = _cfg.get("posenc", {"n_freqs": 4, "scale": 1.0})
    head_posenc = _cfg.get("head_posenc", {"n_freqs": 4, "scale": 1.0})

    # Normalization/pooling flags (encoder + head). Defaults keep backward compatibility
    enc_norm: str = str(_cfg.get("norm", "batch"))
    enc_num_groups: int = int(_cfg.get("num_groups", 16))
    enc_pool: str = str(_cfg.get("pool", "max"))  # 'max' | 'max+mean'
    head_norm: str = str(_cfg.get("head_norm", "batch"))
    head_dropout: float = float(_cfg.get("head_dropout", 0.0))

    # Save path (unique per-architecture; overwrites across runs for the same arch)
    arch_for_hash = {
        "latent_dim": latent_dim,
        "pre_hidden": pre_hidden,
        "sa_blocks": sa_blocks,
        "gf_hidden": gf_hidden,
        "head_hidden": head_hidden,
        "out_dim": NUM_TARGETS,
        "target_names": TARGET_NAMES,
        "posenc": posenc,
        "head_posenc": head_posenc,
        "norm": enc_norm,
        "num_groups": enc_num_groups,
        "pool": enc_pool,
        "head_norm": head_norm,
        "head_dropout": head_dropout,
        "normalization_fix": "v2",
    }
    arch_hash = hashlib.md5(
        json.dumps(arch_for_hash, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    save_dir = Path(project_dir, "Trained_models")
    base_name = model_name if model_name else "pnmlp"
    save_path = save_dir / f"{base_name}_{arch_hash}.pt"

    resume_checkpoint = None
    if save_path.exists():
        print(f"Found existing checkpoint at {save_path}. Loading...")
        try:
            resume_checkpoint = torch.load(save_path, map_location="cpu")
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting fresh.")
            resume_checkpoint = None
    else:
        print(f"No existing checkpoint at {save_path}. Initializing new model.")

    set_seed(42)

    # Split geometries into train/val
    n_geoms = len(PS_list_whole)
    idxs = list(range(n_geoms))
    train_idx, val_idx = train_test_split(idxs, test_size=0.2, random_state=42)

    train_tensors = [PS_list_whole[i] for i in train_idx]
    val_tensors = [PS_list_whole[i] for i in val_idx]

    coord_center, coord_half_range, target_mean, target_std = (
        compute_global_normalization(train_tensors)
    )

    if resume_checkpoint is not None:
        print(
            "Overwriting normalization stats with values from checkpoint to ensure consistency."
        )
        coord_center = resume_checkpoint["coord_center"]
        coord_half_range = resume_checkpoint["coord_half_range"]
        target_mean = resume_checkpoint["target_mean"]
        target_std = resume_checkpoint["target_std"]

    print(
        "Using per-target z-score normalization so Stress (~200-1200) and LogLife (~3-7) are balanced during training."
    )
    print(
        f"Coord center={coord_center.numpy()}, half_range={coord_half_range.numpy()} | target_mean={target_mean.numpy()}, target_std={target_std.numpy()}"
    )

    train_ds = GeomLifeDataset(
        train_tensors,
        coord_center,
        coord_half_range,
        target_mean,
        target_std,
    )
    val_ds = GeomLifeDataset(
        val_tensors,
        coord_center,
        coord_half_range,
        target_mean,
        target_std,
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
    encoder_cfg = {
        "latent_dim": latent_dim,
        "pre_hidden": pre_hidden,
        "sa_blocks": sa_blocks,
        "gf_hidden": gf_hidden,
        "posenc": posenc,
        "head_posenc": head_posenc,
        # new flags
        "norm": enc_norm,
        "num_groups": enc_num_groups,
        "pool": enc_pool,
        "head_norm": head_norm,
        "head_dropout": head_dropout,
    }

    model = PointNetMLPJoint(
        latent_dim=latent_dim,
        mlp_hidden=head_hidden,
        out_dim=NUM_TARGETS,
        encoder_cfg=encoder_cfg,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {param_count:,} parameters.")
    if resume_checkpoint is not None:
        try:
            model.load_state_dict(resume_checkpoint["model_state"], strict=True)
            print("Checkpoint model state is compatible and will be resumed.")
        except (KeyError, RuntimeError, ValueError) as exc:
            print(
                f"Checkpoint model state is incompatible ({exc}). Initializing new model."
            )
            resume_checkpoint = None

    # Ensure save directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to: {save_path}")
    train(
        model,
        train_loader,
        val_loader,
        coord_center,
        coord_half_range,
        target_mean,
        target_std,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        max_points_per_geom=max_points_per_geom,
        grad_clip_norm=0.5,
        save_path=save_path,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        use_amp=(device.type == "cuda"),
        resume_checkpoint=resume_checkpoint,
        model_name=model_name,
    )


if __name__ == "__main__":
    try:
        main("S_full_ln_pos12", 1)
    except Exception as e:
        print(f"Error during training: {e}")
        raise
