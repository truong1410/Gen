#!/usr/bin/env python3
# train_vae_endoscopy.py
"""
Train a VAE for endoscopy images using PyTorch Lightning.
- Expects dataset in ImageFolder structure: root/class_x/xxx.png
- Produces checkpoint and last.ckpt in runs/<timestamp>/
"""

import os
from pathlib import Path
from datetime import datetime
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor


# ------------------------
# Encoder / Decoder Blocks
# ------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------
# VAE Module
# ------------------------
class VAEpl(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        emb_channels: int = 16,
        hid_chs: List[int] = [64, 128, 256, 512],
        kernel_sizes: List[int] = None,
        strides: List[int] = None,
        lr: float = 1e-4,
        kl_weight: float = 1e-4,
        recon_loss: str = "l1",   # "l1" or "mse"
        img_size: int = 256,
    ):
        super().__init__()
        self.save_hyperparameters()

        if kernel_sizes is None:
            kernel_sizes = [3] * len(hid_chs)
        if strides is None:
            # first layer keep spatial, others downsample
            strides = [1] + [2] * (len(hid_chs) - 1)

        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.hid_chs = hid_chs
        self.lr = lr
        self.kl_weight = kl_weight
        self.recon_loss = recon_loss
        self.img_size = img_size

        # ---------------- Encoder ----------------
        enc_layers = []
        ch = in_channels
        for i, out_ch in enumerate(hid_chs):
            enc_layers.append(ConvBlock(ch, out_ch, kernel=kernel_sizes[i], stride=strides[i]))
            ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        # compute flattened dimension after encoder by passing dummy
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            e = self.encoder(dummy)
            self._enc_out_shape = e.shape[1:]   # (C, H, W)
            flat_dim = e.numel() // e.shape[0]

        # latent projection
        self.fc_mu = nn.Linear(flat_dim, emb_channels)
        self.fc_logvar = nn.Linear(flat_dim, emb_channels)
        self.fc_from_z = nn.Linear(emb_channels, flat_dim)

        # ---------------- Decoder ----------------
        # Mirror encoder upsampling so final output matches img_size
        self.flat_dim = flat_dim
        self.C_enc, self.H_enc, self.W_enc = self._enc_out_shape

        dec_layers = []
        ch = self.C_enc

        # We want to invert the encoder downsample steps.
        # Encoder strides: [1, 2, 2, 2] (example). Encoder channels: [64,128,256,512]
        # After encoder, ch=C_enc==hid_chs[-1] (512). We should upsample to 256 -> 128 -> 64 and then final conv -> 3
        # So choose upsample_channels = reversed(hid_chs[:-1]) -> [256,128,64]
        upsample_channels = list(reversed(hid_chs[:-1]))

        for out_ch in upsample_channels:
            dec_layers.append(
                nn.ConvTranspose2d(
                    ch,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            )
            dec_layers.append(nn.BatchNorm2d(out_ch))
            dec_layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch = out_ch

        # final conv (no upsampling)
        dec_layers.append(nn.Conv2d(ch, in_channels, kernel_size=3, stride=1, padding=1))

        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        e = self.encoder(x)
        batch = e.shape[0]
        flat = e.view(batch, -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # z -> flat -> reshape -> decoder
        flat = self.fc_from_z(z)
        batch = flat.shape[0]
        x = flat.view(batch, self.C_enc, self.H_enc, self.W_enc)
        x = self.decoder(x)
        # final activation: tanh or sigmoid depending on input normalization
        # we assume inputs in range [-1, 1] (use Normalize([0.5]*3, [0.5]*3) in transforms)
        x = torch.tanh(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar

    def loss_function(self, x, x_rec, mu, logvar):
        # Ensure shapes match (in case user gives wrong img_size)
        if x_rec.shape != x.shape:
            # Try to adapt by center-cropping or resizing x_rec to x if only minor mismatch occurs.
            # But best is to make transforms produce same size as model.img_size.
            raise RuntimeError(f"Shape mismatch: x_rec {x_rec.shape} vs x {x.shape}. "
                               "Check img_size and datamodule transforms.")
        if self.recon_loss == "l1":
            recon = F.l1_loss(x_rec, x, reduction="mean")
        else:
            recon = F.mse_loss(x_rec, x, reduction="mean")
        # KL divergence between q(z|x) and N(0,1)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + self.kl_weight * kl
        return {"loss": loss, "recon": recon.detach(), "kl": kl.detach()}

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_rec, mu, logvar = self.forward(x)
        losses = self.loss_function(x, x_rec, mu, logvar)
        self.log("train/loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon", losses["recon"], on_step=True, on_epoch=True)
        self.log("train/kl", losses["kl"], on_step=True, on_epoch=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_rec, mu, logvar = self.forward(x)
        losses = self.loss_function(x, x_rec, mu, logvar)
        self.log("val/loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=True)
        return losses

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt


# ------------------------
# DataModule (ImageFolder)
# ------------------------
class EndoDataModule(pl.LightningDataModule):
    def __init__(self, path_root, batch_size=16, img_size=256, num_workers=4):
        super().__init__()
        self.path_root = Path(path_root)
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers

        # transforms: normalize to [-1,1]
        self.train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)   # range [-1,1]
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def setup(self, stage=None):
        # Use ImageFolder: if labels not needed, files can be in a single folder w/ dummy class
        self.train_dataset = datasets.ImageFolder(str(self.path_root), transform=self.train_transform)
        # If you have a separate val, change this
        self.val_dataset = datasets.ImageFolder(str(self.path_root), transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


# ------------------------
# CLI and run
# ------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="path to dataset root (ImageFolder)")
    p.add_argument("--outdir", type=str, default="runs", help="output runs dir")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--emb_channels", type=int, default=16)
    p.add_argument("--kl_weight", type=float, default=1e-4)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_dir = Path(args.outdir) / current_time
    run_dir.mkdir(parents=True, exist_ok=True)

    dm = EndoDataModule(path_root=args.data_root, batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers)
    model = VAEpl(
        in_channels=3,
        emb_channels=args.emb_channels,
        hid_chs=[64, 128, 256, 512],
        lr=args.lr,
        kl_weight=args.kl_weight,
        recon_loss="l1",
        img_size=args.img_size
    )

    checkpoint_cb = ModelCheckpoint(dirpath=str(run_dir), filename="last", save_last=True, save_top_k=3, monitor="train/loss", mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early = EarlyStopping(monitor="train/loss", patience=30, mode="min", verbose=True)

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        devices=args.gpus if args.gpus and torch.cuda.is_available() else None,
        accelerator="gpu" if args.gpus and torch.cuda.is_available() else "cpu",
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=50,
        precision=16 if (args.gpus and torch.cuda.is_available()) else 32,
    )

    trainer.fit(model, datamodule=dm)
    print("Training finished. Best ckpt:", checkpoint_cb.best_model_path)


if __name__ == "__main__":
    main()
