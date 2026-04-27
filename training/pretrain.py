#!/usr/bin/env python3
"""
L-JEPA self-supervised pretraining.

Stage 1 of the L-JEPA pipeline: trains the ContextEncoder + Predictor
(with a frozen EMA TargetEncoder as target) on offline LiDAR trajectory data.

The encoder is NOT conditioned on reward or task — it only learns to predict
future latent states from past context and actions (the JEPA non-generative
objective). The resulting encoder is then frozen for the RL stage.

Usage:
    python training/pretrain.py
    python training/pretrain.py --config ../config.yaml --data-dir ../data
"""
import sys
import os
import argparse
import time

import torch
import torch.optim as optim
import yaml

# Allow running from the training/ subdirectory or from the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import LJEPA
from training.dataset import make_dataloader


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg):
    mc = cfg['model']
    pc = cfg['pretrain']
    model = LJEPA(
        lidar_dim=mc['lidar_dim'],
        vel_dim=mc['vel_dim'],
        context_window=mc['context_window'],
        latent_dim=mc['latent_dim'],
        action_dim=mc['action_dim'],
        prediction_horizon=mc['prediction_horizon'] if 'prediction_horizon' in mc else pc.get('prediction_horizon', 5),
        conv_channels=tuple(mc['encoder']['channels']),
        conv_kernel=mc['encoder']['kernel_size'],
        conv_stride=mc['encoder']['stride'],
        predictor_hidden=tuple(mc['predictor']['hidden_dims']),
        ema_decay=pc['ema_decay'],
    )
    return model


def train(cfg_path):
    cfg = load_config(cfg_path)
    mc = cfg['model']
    pc = cfg['pretrain']
    dc = cfg['data_collection']

    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # explicitly NVIDIA, not Intel iGPU
        torch.backends.cudnn.benchmark = True  # faster convolutions
        torch.backends.cuda.matmul.allow_tf32 = True  # faster matmul on Ampere+
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("WARNING: CUDA not available, falling back to CPU")
    print(f"Device: {device}")

    # Data
    data_dir = dc['output_dir']
    loader, dataset = make_dataloader(
        data_dir=data_dir,
        context_window=mc['context_window'],
        prediction_horizon=pc.get('prediction_horizon', dc.get('prediction_horizon', 5)),
        batch_size=pc['batch_size'],
        num_workers=min(4, os.cpu_count() or 1),
    )
    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    # Model
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    # Optimiser — only context_encoder + predictor; target_encoder has no grad
    optim_params = (
        list(model.context_encoder.parameters()) +
        list(model.predictor.parameters())
    )
    optimizer = optim.AdamW(optim_params, lr=pc['lr'], weight_decay=pc['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pc['epochs'])

    checkpoint_dir = pc['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float('inf')

    for epoch in range(1, pc['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (context_obs, future_obs, actions) in enumerate(loader):
            context_obs = context_obs.to(device, non_blocking=True)
            future_obs   = future_obs.to(device, non_blocking=True)
            actions      = actions.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss, z_ctx, z_pred = model(context_obs, future_obs, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optim_params, max_norm=1.0)
            optimizer.step()

            # EMA update of target encoder (after each optimiser step)
            model.update_ema()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0

        if epoch % pc['log_every'] == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d}/{pc['epochs']} | loss: {avg_loss:.6f} | lr: {lr:.2e} | {elapsed:.1f}s")

        # Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.context_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': cfg,
            }, os.path.join(checkpoint_dir, 'best.pt'))

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.context_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': cfg,
            }, os.path.join(checkpoint_dir, f'epoch_{epoch:04d}.pt'))

    print(f"\nPretraining done. Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Best encoder weights: {checkpoint_dir}/best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(PROJECT_ROOT, 'config.yaml'))
    args = parser.parse_args()
    train(args.config)


if __name__ == '__main__':
    main()
