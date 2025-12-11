#!/usr/bin/env python3
# train_diffusion_endoscopy.py
"""
Train diffusion pipeline for endoscopy images using medical_diffusion repo components.
Features:
 - Load dataset with AIROGSDataset (folder-of-classes)
 - Support latent diffusion using a pretrained VAE checkpoint (optional)
 - If no latent_ckpt provided -> train pixel-space diffusion (in_ch/out_ch = 3)
 - CLI args for dataset path, ckpt, training params
"""

import argparse
from pathlib import Path
from datetime import datetime

import torch
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# medical_diffusion imports (assumes repo is on PYTHONPATH or installed)
from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import AIROGSDataset, CheXpert_2_Dataset
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.models.estimators import UNet
from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medical_diffusion.models.embedders import LabelEmbedder, TimeEmbbeding
from medical_diffusion.models.embedders.latent_embedders import VAE

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="/home/labcoha/datasets/original", help="Path to dataset root (ImageFolder with class subfolders)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--min_epochs", type=int, default=100)
    p.add_argument("--save_every_steps", type=int, default=100)
    p.add_argument("--latent_ckpt", type=str, default=None, help="Path to pretrained VAE checkpoint. If not set, runs pixel-space diffusion.")
    p.add_argument("--emb_channels", type=int, default=16, help="latent channels produced by VAE (only used if latent_ckpt set)")
    p.add_argument("--outdir", type=str, default="runs", help="runs directory")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = build_args()

    # ---------------- prepare paths and device ----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / args.outdir / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    # if GPU requested but not available, it will fallback
    devices = [args.gpu_id] if torch.cuda.is_available() else None

    # ---------------- DataModule ----------------
    # Use AIROGSDataset to load images from folder-of-classes
    ds = AIROGSDataset(
        crawler_ext='jpg',
        augment_horizontal_flip=True,
        augment_vertical_flip=False,
        path_root=str(args.data_root),
        image_resize=args.img_size,
        image_crop=None
    )
    dm = SimpleDataModule(
        ds_train=ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # ---------------- Embedders / Settings ----------------
    # Label embedder (optional) — set as None if not using labels
    cond_embedder = None  # For endoscopy, often no labels; set to LabelEmbedder if you have classes
    cond_embedder_kwargs = None

    time_embedder = TimeEmbbeding
    time_embedder_kwargs = {'emb_dim': 1024}

    # Decide if using latent diffusion (VAE) or pixel-space
    if args.latent_ckpt:
        print(f"[INFO] Using latent diffusion with VAE checkpoint: {args.latent_ckpt}")
        latent_embedder = VAE
        latent_embedder_checkpoint = str(args.latent_ckpt)
        latent_channels = args.emb_channels
        in_out_ch = latent_channels
    else:
        print("[INFO] No latent_ckpt provided -> using pixel-space diffusion (RGB)")
        latent_embedder = None
        latent_embedder_checkpoint = None
        in_out_ch = 3

    # ---------------- Noise Estimator (UNet) ----------------
    noise_estimator = UNet
    noise_estimator_kwargs = {
        'in_ch': in_out_ch,
        'out_ch': in_out_ch,
        'spatial_dims': 2,
        # adjust UNet capacity for latent or pixel:
        'hid_chs': [256, 256, 512, 1024] if in_out_ch > 3 else [128, 256, 512, 1024],
        'kernel_sizes': [3, 3, 3, 3],
        'strides': [1, 2, 2, 2],
        'time_embedder': time_embedder,
        'time_embedder_kwargs': time_embedder_kwargs,
        'cond_embedder': cond_embedder,
        'cond_embedder_kwargs': cond_embedder_kwargs,
        'deep_supervision': False,
        'use_res_block': True,
        'use_attention': 'none',
    }

    # ---------------- Noise Scheduler ----------------
    noise_scheduler = GaussianNoiseScheduler
    noise_scheduler_kwargs = {
        'timesteps': 1000,
        'beta_start': 0.002,
        'beta_end': 0.02,
        'schedule_strategy': 'scaled_linear'
    }

    # ---------------- Build pipeline ----------------
    pipeline = DiffusionPipeline(
        noise_estimator=noise_estimator,
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=noise_scheduler,
        noise_scheduler_kwargs=noise_scheduler_kwargs,
        latent_embedder=latent_embedder,
        latent_embedder_checkpoint=latent_embedder_checkpoint,
        estimator_objective='x_T',
        estimate_variance=False,
        use_self_conditioning=False,
        use_ema=False,
        classifier_free_guidance_dropout=0.0,  # set 0 for pure training; use >0 for cfg later
        do_input_centering=False,
        clip_x0=False,
        sample_every_n_steps=args.save_every_steps
    )

    # ---------------- Callbacks & Trainer ----------------
    to_monitor = "train/loss"
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        monitor=to_monitor,
        every_n_train_steps=args.save_every_steps,
        save_last=True,
        save_top_k=2,
        mode="min"
    )

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0,
        patience=30,
        mode="min"
    )

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=args.save_every_steps,
        auto_lr_find=False,
        limit_val_batches=0,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=2,
    )

    # ---------------- Execute Training ----------------
    trainer.fit(pipeline, datamodule=dm)

    # ---------------- Save final best checkpoint info ----------------
    try:
        pipeline.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
    except Exception as e:
        print("[WARN] Could not save best checkpoint via pipeline.save_best_checkpoint():", e)
        print("Best ckpt path (ModelCheckpoint):", checkpointing.best_model_path)


if __name__ == "__main__":
    main()
