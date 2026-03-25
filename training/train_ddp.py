"""
DDP training loop for APD Intelligibility Estimator.

Usage (8x A6000):
    # Full resume (model + optimizer + scheduler):
    torchrun --nproc_per_node=8 -m training.train_ddp \
        --manifest_dir data/manifests \
        --checkpoint_dir checkpoints \
        --resume checkpoints/best_model.pt

    # Weights-only resume (model weights only, fresh optimizer/scheduler):
    torchrun --nproc_per_node=8 -m training.train_ddp \
        --manifest_dir data/manifests \
        --checkpoint_dir checkpoints \
        --resume checkpoints/checkpoint_epoch5.pt --weights_only
"""

import argparse
import json
import math
import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from scipy.stats import spearmanr
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model_definition import create_model
from training.config import AugmentationConfig, AudioConfig, Config, TrainConfig
from training.dataset import APDManifestDataset, collate_fn, collate_with_mixup
from training.loss import APDLoss


def is_main():
    return dist.get_rank() == 0


def log(msg: str):
    if is_main():
        print(msg, flush=True)


def set_seed(seed: int, rank: int):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _worker_init_fn(worker_id: int):
    """Seed each DataLoader worker uniquely per rank + worker."""
    worker_info = torch.utils.data.get_worker_info()
    rank = dist.get_rank() if dist.is_initialized() else 0
    seed = worker_info.dataset.__dict__.get("_base_seed", 42) + rank * 100 + worker_id
    random.seed(seed)
    np.random.seed(seed % (2**32))


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int,
                                     total_steps: int, min_lr: float = 1e-6):
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def lr_lambda(step, base_lr):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, partial(lr_lambda, base_lr=base_lrs[0])
    )


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: APDLoss,
             device: torch.device) -> dict:
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        preds = model(audio).squeeze(-1)
        loss, _ = criterion(preds, labels)

        total_loss += loss.item()
        n_batches += 1

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Gather from all ranks
    world_size = dist.get_world_size()
    gathered_preds = [None] * world_size
    gathered_labels = [None] * world_size
    dist.all_gather_object(gathered_preds, all_preds.tolist())
    dist.all_gather_object(gathered_labels, all_labels.tolist())

    all_preds = np.concatenate([np.array(p) for p in gathered_preds])
    all_labels = np.concatenate([np.array(l) for l in gathered_labels])

    # Gather loss
    loss_tensor = torch.tensor([total_loss, n_batches], device=device)
    dist.all_reduce(loss_tensor)
    avg_loss = loss_tensor[0].item() / max(loss_tensor[1].item(), 1)

    spearman_corr, _ = spearmanr(all_preds, all_labels)
    mae = np.mean(np.abs(all_preds - all_labels))

    return {
        "loss": avg_loss,
        "spearman": spearman_corr,
        "mae": mae,
        "pred_mean": float(all_preds.mean()),
        "pred_std": float(all_preds.std()),
        "label_mean": float(all_labels.mean()),
        "label_std": float(all_labels.std()),
    }


def log_bitnet_stats(model: nn.Module) -> dict:
    # Unwrap DDP
    m = model.module if hasattr(model, "module") else model
    stats = {}
    for name, module in m.named_modules():
        if hasattr(module, "scale") and hasattr(module, "binarize"):
            w = module.weight.data
            stats[name] = {
                "w_scale": float(module.scale.data.item()),
                "w_abs_mean": float(w.abs().mean().item()),
                "w_std": float(w.std().item()),
            }
    return stats


def save_checkpoint(model: nn.Module, optimizer, scheduler, epoch: int,
                    global_step: int, best_spearman: float, cfg: Config,
                    path: Path):
    m = model.module if hasattr(model, "module") else model
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": m.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_spearman": best_spearman,
        "config": cfg,
    }, path)


def train(cfg: Config, resume_path: str | None = None,
          weights_only: bool = False):
    # DDP setup
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")

    set_seed(cfg.train.seed, rank)

    # Model — detect architecture from checkpoint if resuming
    start_epoch = 0
    global_step = 0
    best_spearman = -1.0
    history = []
    ckpt = None

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"]

        # Auto-detect model size from state dict
        if "model_config" in ckpt:
            mc = ckpt["model_config"]
            enc_dim = mc["encoder_dim"]
            bn_dim = mc["bottleneck_dim"]
            tcn_dim = mc["tcn_channels"]
        else:
            enc_dim = sd["encoder.conv.weight"].shape[0]
            bn_dim = sd["bottleneck.weight"].shape[0]
            tcn_dim = sd["tcn_input.weight"].shape[0]

        n_repeats = 0
        while f"tcn_blocks.{n_repeats}.layers.0.depthwise.weight" in sd:
            n_repeats += 1
        n_layers_tcn = 0
        if n_repeats > 0:
            while f"tcn_blocks.0.layers.{n_layers_tcn}.depthwise.weight" in sd:
                n_layers_tcn += 1
        use_bitnet = "head.fc_out.scale" in sd

        from model.model_definition import APDIntelligibilityEstimator
        model = APDIntelligibilityEstimator(
            encoder_dim=enc_dim, bottleneck_dim=bn_dim,
            tcn_channels=tcn_dim, n_repeats=n_repeats,
            n_layers=n_layers_tcn, use_bitnet_output=use_bitnet,
        )
        model.load_state_dict(sd)
        log(f"Model from checkpoint: enc={enc_dim}, bn={bn_dim}, tcn={tcn_dim}, "
            f"repeats={n_repeats}, layers={n_layers_tcn}")

        if weights_only:
            log(f"Loaded weights from {resume_path} "
                f"(epoch {ckpt.get('epoch', '?')}, optimizer/scheduler reset)")
        else:
            start_epoch = ckpt.get("epoch", 0)
            global_step = ckpt.get("global_step", 0)
            best_spearman = ckpt.get("best_spearman", -1.0)
            log(f"Resuming from {resume_path}: epoch {start_epoch}, "
                f"step {global_step}, best_spearman={best_spearman:.4f}")
    else:
        model = create_model(overparameterized=True, use_bitnet_output=True)

    model = model.to(device)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])

    total_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {total_params:,}")
    log(f"DDP: {world_size} GPUs, per-GPU batch_size={cfg.train.batch_size}, "
        f"effective batch_size={cfg.train.batch_size * world_size}")

    # Datasets
    train_manifest = Path(cfg.data.manifest_dir) / cfg.data.train_manifest
    val_manifest = Path(cfg.data.manifest_dir) / cfg.data.val_manifest

    train_dataset = APDManifestDataset(
        train_manifest,
        audio_config=cfg.audio,
        augmentation=cfg.augmentation,
    )
    train_dataset._base_seed = cfg.train.seed
    val_dataset = APDManifestDataset(
        val_manifest,
        audio_config=cfg.audio,
        augmentation=None,
    )

    # Samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False,
    )

    mixup_collate = partial(
        collate_with_mixup,
        alpha=cfg.augmentation.mixup_alpha,
        prob=cfg.augmentation.mixup_prob,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        collate_fn=mixup_collate,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size * 2,
        sampler=val_sampler,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )

    # Loss
    criterion = APDLoss(
        ranking_weight=cfg.train.ranking_loss_weight,
        boundary_weight=cfg.train.boundary_loss_weight,
        boundary_thresholds=cfg.train.boundary_thresholds,
        boundary_sigma=cfg.train.boundary_sigma,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.train.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, cfg.train.warmup_steps, total_steps, cfg.train.min_lr,
    )

    # Restore optimizer/scheduler state if full resume
    if resume_path and not weights_only and ckpt and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        log("  Restored optimizer & scheduler state")

    # Load history if exists
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = ckpt_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    patience_counter = 0

    log(f"Training for {cfg.train.epochs} epochs ({start_epoch} done), "
        f"{steps_per_epoch} steps/epoch")
    log(f"Total steps: {total_steps}, Warmup: {cfg.train.warmup_steps}")

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_components = {"mse": 0.0, "ranking": 0.0, "boundary": 0.0}
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            preds = model(audio).squeeze(-1)
            loss, components = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.train.grad_clip_max_norm,
            )
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            for k in epoch_components:
                epoch_components[k] += components[k]
            n_batches += 1
            global_step += 1

            if global_step % cfg.train.log_every_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = epoch_loss / n_batches
                log(
                    f"  step {global_step} | loss={avg_loss:.4f} "
                    f"(mse={epoch_components['mse']/n_batches:.4f} "
                    f"rank={epoch_components['ranking']/n_batches:.4f} "
                    f"bound={epoch_components['boundary']/n_batches:.4f}) "
                    f"| lr={lr:.2e}"
                )

        epoch_time = time.time() - t0
        avg_epoch_loss = epoch_loss / max(n_batches, 1)

        # Validation (all ranks participate, rank 0 logs)
        val_metrics = evaluate(model, val_loader, criterion, device)
        bitnet_stats = log_bitnet_stats(model)

        log(
            f"Epoch {epoch+1}/{cfg.train.epochs} "
            f"| train_loss={avg_epoch_loss:.4f} "
            f"| val_loss={val_metrics['loss']:.4f} "
            f"| spearman={val_metrics['spearman']:.4f} "
            f"| mae={val_metrics['mae']:.4f} "
            f"| time={epoch_time:.0f}s"
        )

        # Check improvement (all ranks, using gathered val_metrics)
        is_best = val_metrics["spearman"] > best_spearman
        if is_best:
            best_spearman = val_metrics["spearman"]
            patience_counter = 0
        else:
            patience_counter += 1

        # Checkpointing (rank 0 only)
        if is_main():
            epoch_record = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": avg_epoch_loss,
                "val_loss": val_metrics["loss"],
                "spearman": val_metrics["spearman"],
                "mae": val_metrics["mae"],
                "lr": optimizer.param_groups[0]["lr"],
                "bitnet_stats": bitnet_stats,
            }
            history.append(epoch_record)
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)

            if is_best:
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1,
                    global_step, best_spearman, cfg,
                    ckpt_dir / "best_model.pt",
                )
                log(f"  -> New best model (spearman={best_spearman:.4f})")

            if (epoch + 1) % cfg.train.save_every_epochs == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1,
                    global_step, best_spearman, cfg,
                    ckpt_dir / f"checkpoint_epoch{epoch+1}.pt",
                )

        # Early stopping (all ranks have same patience_counter)
        if patience_counter >= cfg.train.patience:
            log(f"Early stopping at epoch {epoch+1} (patience={cfg.train.patience})")
            break

        dist.barrier()

    log(f"\nTraining complete. Best Spearman: {best_spearman:.4f}")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train APD (DDP)")
    parser.add_argument("--manifest_dir", type=str, default="data/manifests")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-GPU batch size")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--weights_only", action="store_true",
                        help="Load model weights only, reset optimizer/scheduler/epoch")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Override warmup steps (default: from config)")
    args = parser.parse_args()

    cfg = Config()
    cfg.data.manifest_dir = Path(args.manifest_dir)
    cfg.data.num_workers = args.num_workers
    cfg.train.checkpoint_dir = Path(args.checkpoint_dir)
    cfg.train.batch_size = args.batch_size
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.device = "cuda"
    cfg.train.seed = args.seed
    if args.warmup_steps is not None:
        cfg.train.warmup_steps = args.warmup_steps

    train(cfg, resume_path=args.resume, weights_only=args.weights_only)


if __name__ == "__main__":
    main()
