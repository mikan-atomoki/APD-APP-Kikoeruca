"""
Prune V3 best model and fine-tune with knowledge distillation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m training.prune_finetune \
        --teacher checkpoints_v3/best_model.pt \
        --output_dir checkpoints_v3 \
        --manifest_dir data/manifests_v3 \
        --prune_rounds 4 --finetune_epochs 25
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model_definition import APDIntelligibilityEstimator, create_model
from training.config import Config
from training.dataset import APDManifestDataset, collate_fn
from training.pruning import prune_model, distillation_loss


def detect_architecture(state_dict):
    enc_dim = state_dict["encoder.conv.weight"].shape[0]
    bn_dim = state_dict["bottleneck.weight"].shape[0]
    tcn_dim = state_dict["tcn_input.weight"].shape[0]
    n_repeats = 0
    while f"tcn_blocks.{n_repeats}.layers.0.depthwise.weight" in state_dict:
        n_repeats += 1
    n_layers = 0
    if n_repeats > 0:
        while f"tcn_blocks.0.layers.{n_layers}.depthwise.weight" in state_dict:
            n_layers += 1
    use_bitnet = "head.fc_out.scale" in state_dict
    return enc_dim, bn_dim, tcn_dim, n_repeats, n_layers, use_bitnet


def build_model(state_dict):
    enc, bn, tcn, reps, layers, bitnet = detect_architecture(state_dict)
    model = APDIntelligibilityEstimator(
        encoder_dim=enc, bottleneck_dim=bn, tcn_channels=tcn,
        n_repeats=reps, n_layers=layers, use_bitnet_output=bitnet,
    )
    model.load_state_dict(state_dict)
    return model


def evaluate(model, loader, device):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["audio"].to(device)).squeeze(-1)
            preds.extend(out.cpu().numpy().tolist())
            labs.extend(batch["label"].numpy().tolist())
    preds, labs = np.array(preds), np.array(labs)
    sp, _ = spearmanr(preds, labs)
    mae = np.mean(np.abs(preds - labs))
    return sp, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints_v3")
    parser.add_argument("--manifest_dir", type=str, default="data/manifests_v3")
    parser.add_argument("--prune_rounds", type=int, default=4)
    parser.add_argument("--prune_ratio", type=float, default=0.15)
    parser.add_argument("--finetune_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load teacher
    t_ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
    teacher = create_model(overparameterized=True, use_bitnet_output=True).to(device)
    teacher.load_state_dict(t_ckpt["model_state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher: {sum(p.numel() for p in teacher.parameters()):,} params, "
          f"spearman={t_ckpt.get('best_spearman', '?')}")

    # Prune
    student = create_model(overparameterized=True, use_bitnet_output=True)
    student.load_state_dict(t_ckpt["model_state_dict"])
    n_before = sum(p.numel() for p in student.parameters())

    for r in range(args.prune_rounds):
        student = prune_model(student, args.prune_ratio)
        n = sum(p.numel() for p in student.parameters())
        print(f"  Prune round {r+1}: {n:,} params ({100*(1-n/n_before):.1f}% reduction)")

    student = student.to(device)
    n_after = sum(p.numel() for p in student.parameters())
    print(f"Pruned: {n_before:,} -> {n_after:,} ({100*(1-n_after/n_before):.1f}%)")

    # Data
    cfg = Config()
    train_ds = APDManifestDataset(
        Path(args.manifest_dir) / "train.jsonl", audio_config=cfg.audio)
    val_ds = APDManifestDataset(
        Path(args.manifest_dir) / "val.jsonl", audio_config=cfg.audio)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.finetune_epochs
    warmup = 200

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(1e-6 / args.lr, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Pre-prune evaluation
    sp0, mae0 = evaluate(student, val_loader, device)
    print(f"Pre-finetune: spearman={sp0:.4f}, mae={mae0:.4f}")

    # Fine-tune
    best_spearman = -1
    history = []

    for epoch in range(args.finetune_epochs):
        student.train()
        total_loss = 0
        n = 0
        t0 = time.time()

        for batch in train_loader:
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            s_out = student(audio).squeeze(-1)
            with torch.no_grad():
                t_out = teacher(audio).squeeze(-1)

            loss = distillation_loss(s_out, t_out, labels, alpha=0.7)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n += 1

        sp, mae = evaluate(student, val_loader, device)
        elapsed = time.time() - t0
        is_best = sp > best_spearman
        if is_best:
            best_spearman = sp

        history.append({
            "epoch": epoch + 1,
            "loss": total_loss / n,
            "spearman": sp,
            "mae": mae,
        })

        flag = " *BEST*" if is_best else ""
        print(f"  Epoch {epoch+1}/{args.finetune_epochs} | "
              f"loss={total_loss/n:.4f} | spearman={sp:.4f} | "
              f"mae={mae:.4f} | {elapsed:.0f}s{flag}")

        if is_best:
            torch.save({
                "model_state_dict": student.state_dict(),
                "model_config": {
                    "encoder_dim": student.encoder.conv.out_channels,
                    "bottleneck_dim": student.bottleneck.weight.shape[0],
                    "tcn_channels": student.tcn_input.weight.shape[0],
                },
                "best_spearman": best_spearman,
            }, output_dir / "pruned_finetuned.pt")

    with open(output_dir / "pruning_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best spearman: {best_spearman:.4f}")
    print(f"Params: {n_before:,} -> {n_after:,} ({100*(1-n_after/n_before):.1f}% reduction)")


if __name__ == "__main__":
    main()
