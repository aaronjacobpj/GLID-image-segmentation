import os

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import random

import math
import torch
import sklearn
import torchvision
import torch.nn as nn
from pathlib import Path
from sklearn import metrics
from torchvision import transforms, models
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import logging

logging.basicConfig(
    filename="app.log",  # log file name
    level=logging.INFO,  # minimum level to log
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def setup_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


device = setup_device()


class GLIDDataset(torch.utils.data.Dataset):

    imConvert = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Grayscale(),
        ]
    )

    def __init__(
        self, folder, mu=0.5, std=0.5, sample=float("inf"), hflip=0.5, train=False
    ):
        self.mu = mu
        self.std = std
        self.sample = sample
        self.files, self.labels = self.load_files(folder)
        self.train = train
        self.hflip = hflip
        self.grayscale = transforms.Grayscale()

    def __len__(self):
        return min(self.sample, len(self.labels))

    def __getitem__(self, idx):

        if self.train and torch.rand(1) > self.hflip:
            image, label = self.h_flip(
                Image.open(self.files[idx]), Image.open(self.labels[idx])
            )
        else:
            image = Image.open(self.files[idx])
            label = Image.open(self.labels[idx])

        return self.transform(image).type(torch.float32), self.imConvert(label)

    @classmethod
    def load_files(cls, folder):
        filenames = []
        labels = []

        for filename in os.listdir(os.path.join(folder, "images")):
            if not Path(filename).suffix in {".jpeg", ".jpg", ".tiff", ".png"}:
                continue
            filenames.append(os.path.join(folder, "images", filename))
            labels.append(os.path.join(folder, "labels", filename))

        return filenames, labels

    def transform(self, images):
        kernels = [transforms.Resize((224, 224))]

        if self.train:
            kernels += [
                transforms.GaussianBlur((3, 3), (0.001, 0.05)),
                transforms.RandomCrop(224, 7),
            ]
        normalise = transforms.Compose(
            kernels
            + [
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mu], std=[self.std]),
            ]
        )
        return normalise(images)

    @classmethod
    def h_flip(cls, image, label):
        return TF.hflip(image), TF.hflip(label)

    def get_balance_weights(self):
        o = torch.Tensor(self.labels).sum().item()
        N = len(self.labels)
        z = 0.5 / (N - o)
        o = 0.5 / o

        weights = []
        for x in self.labels:
            weights.append(o if x == 1 else z)

        return weights


def mask(X, y):
    return (X * y.squeeze().repeat(3, 1, 1)).permute(1, 2, 0)


class DiceBCELoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5, smooth=1e-5):
        super(DiceBCELoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([10]).to(device))

    def forward(self, logits, targets):
        """
        logits:  (B, 1, H, W)  -> raw output (NO sigmoid)
        targets: (B, 1, H, W)  -> binary mask (0 or 1)
        """

        # BCE Loss
        bce_loss = self.bce(logits, targets)

        # Apply sigmoid
        probs = torch.sigmoid(logits)

        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)

        # Dice Loss
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        dice_loss = 1 - dice

        # Combined Loss
        total_loss = self.weight_bce * bce_loss + self.weight_dice * dice_loss

        return total_loss


def get_pred_mask(logits):
    return (torch.sigmoid(logits) > 0.5).float()


class RunManager:
    def __init__(self, model, optimizer, loss_fn, device, out_file, log_file):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.out_file = out_file
        self.log_file = log_file

        self.df = pd.DataFrame(
            columns=[
                "epoch",
                "train_loss",
                "val_loss",
                "train_acc",
                "val_acc",
                "train_iou",
                "val_iou",
            ]
        )

    # -------------------------
    # Train One Epoch
    # -------------------------
    def train_epoch(self, dataloader):
        self.model.train()

        total_loss = 0
        total_correct = 0
        total_pixels = 0

        all_preds = []
        all_targets = []

        N = len(dataloader)

        for X, y in tqdm.tqdm(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            logits = self.model(X)
            loss = self.loss_fn(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred_mask = get_pred_mask(logits)

            total_loss += loss.item()
            total_correct += (pred_mask == y).sum().item()
            total_pixels += y.numel()

            # Store for global IoU
            all_preds.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        avg_loss = total_loss / N
        acc = total_correct / total_pixels
        iou = iou_score_global(all_preds, all_targets)

        return avg_loss, acc, iou

    # -------------------------
    # Evaluation
    # -------------------------
    def evaluate(self, dataloader):
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_pixels = 0

        all_preds = []
        all_targets = []

        N = len(dataloader)

        with torch.inference_mode():
            for X, y in tqdm.tqdm(dataloader):
                X, y = X.to(self.device), y.to(self.device)

                logits = self.model(X)
                loss = self.loss_fn(logits, y)

                pred_mask = get_pred_mask(logits)

                total_loss += loss.item()
                total_correct += (pred_mask == y).sum().item()
                total_pixels += y.numel()

                all_preds.append(logits.cpu())
                all_targets.append(y.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        avg_loss = total_loss / N
        acc = total_correct / total_pixels
        iou = iou_score_global(all_preds, all_targets)

        return avg_loss, acc, iou

    # -------------------------
    # Training Loop
    # -------------------------
    def train(self, epochs, train_dataloader, val_dataloader, iter_save=2):
        best_iou = -float("inf")

        for epoch in range(epochs):

            train_loss, train_acc, train_iou = self.train_epoch(train_dataloader)
            val_loss, val_acc, val_iou = self.evaluate(val_dataloader)

            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_iou": train_iou,
                "val_iou": val_iou,
            }

            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(self.model.state_dict(), self.out_file)

            if epoch % iter_save == 0:
                self.df.to_csv(self.log_file, index=False)

            msg = (
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f}"
            )

            print(msg)
            logging.info(msg)

        self.df.to_csv(self.log_file, index=False)


def iou_score(pred, target, smooth=1e-6):

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def iou_score_global(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item() - intersection

    return (intersection + smooth) / (union + smooth)


def pixel_accuracy(pred, target):
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total


def set_seed(seed=42):
    random.seed(seed)  # Python random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU (single)
    torch.cuda.manual_seed_all(seed)  # GPU (multi)


def train_models(
    modelfn,
    lossfc,
    optimiser,
    train_loader,
    val_loader,
    device,
    epochs,
    lrs,
    weight_decay,
    filename,
):
    logging.info(
        f"Train| size {train_loader.batch_size}, batches: {len(train_loader)} "
        f"Val| size {val_loader.batch_size}, batches: {len(val_loader)} "
    )
    for i, lr in enumerate(lrs):
        msg = (
            f"Config| weight: {filename}-{i+1}.pth,"
            f" log: {filename}-{i+1}.csv,  learning rate:{lr},"
            f" epochs: {epochs}"
        )
        print(msg)
        logging.info(msg)
        model = modelfn()
        model.to(device)

        optim = optimiser(model.parameters(), lr=lr, weight_decay=weight_decay)
        manager = RunManager(
            model,
            optim,
            lossfc,
            device,
            f"{filename}-{i+1}.pth",
            f"{filename}-{i+1}.csv",
        )
        manager.train(epochs, train_loader, val_loader)
