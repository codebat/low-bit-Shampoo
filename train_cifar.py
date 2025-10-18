#!/usr/bin/env python3

import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import vgg, resnet, vit, swin
from optimizers.sgd import SGD
from optimizers.adamw import AdamW
from optimizers.shampoo1 import (
    ShampooSGD as Shampoo1SGD,
    ShampooAdamW as Shampoo1AdamW,
)
from optimizers.shampoo2 import (
    ShampooSGD as Shampoo2SGD,
    ShampooAdamW as Shampoo2AdamW,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CIFAR-100 models with baseline and quantized Shampoo optimizers."
    )
    parser.add_argument("--data", default="./data", type=str, help="dataset root directory")
    parser.add_argument("--download", action="store_true", help="download CIFAR-100 if missing")
    parser.add_argument("--model", default="resnet34", choices=["vgg19", "resnet34", "vit-small", "swin-tiny"], help="backbone to train")
    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=[
            "adamw",
            "sgd",
            "shampoo1-adamw",
            "shampoo1-sgd",
            "shampoo2-adamw",
            "shampoo2-sgd",
        ],
        help="optimization algorithm",
    )
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=8, type=int, help="dataloader worker processes")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--weight-decay", default=5e-2, type=float)
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum for SGD variants")
    parser.add_argument("--betas", default=(0.9, 0.999), type=float, nargs=2, metavar=("BETA1", "BETA2"))
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--stat-compute-steps", default=100, type=int, help="Shampoo T1")
    parser.add_argument("--prec-compute-steps", default=500, type=int, help="Shampoo T2")
    parser.add_argument("--start-prec-step", default=1, type=int)
    parser.add_argument("--stat-decay", default=0.95, type=float)
    parser.add_argument("--matrix-eps", default=1e-6, type=float)
    parser.add_argument("--prec-maxorder", default=1200, type=int)
    parser.add_argument("--prec-bits", default=4, type=int, choices=[4, 8, 16, 32], help="quantizer bitwidth")
    parser.add_argument("--min-lowbit-size", default=4096, type=int)
    parser.add_argument("--quan-blocksize", default=64, type=int)
    parser.add_argument("--rect-t1", default=1, type=int, help="Shampoo2 SVD rectification t1")
    parser.add_argument("--rect-t2", default=4, type=int, help="Shampoo2 SVD rectification t2")
    parser.add_argument("--inv-root-mode", default=0, type=int, choices=[0, 1], help="Shampoo2 inverse root method")
    parser.add_argument("--lr-scheduler", default="cosine", choices=["none", "cosine"])
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="compute device to target; auto prefers CUDA, then MPS, then CPU",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="enable automatic mixed precision when supported (CUDA/MPS)",
    )
    parser.add_argument("--print-freq", default=50, type=int)
    parser.add_argument("--output", default="./checkpoints", type=str, help="directory to store checkpoints and logs")
    return parser.parse_args()


def resolve_device(device_flag: str) -> Tuple[torch.device, bool, bool, str]:
    if device_flag == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
        elif torch.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
    elif device_flag == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        resolved = "cpu"
    elif device_flag == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available, falling back to CPU.")
        resolved = "cpu"
    else:
        resolved = device_flag

    device = torch.device(resolved)
    return device, resolved == "cuda", resolved == "mps", resolved


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=True,
    )


def build_model(name: str) -> nn.Module:
    if name == "vgg19":
        return vgg.vgg19(num_classes=100)
    if name == "resnet34":
        return resnet.resnet34(num_classes=100)
    if name == "vit-small":
        return vit.vit_small(img_size=32, num_classes=100, drop_path_rate=0.1)
    if name == "swin-tiny":
        return swin.swin_tiny(img_size=32, num_classes=100, drop_path_rate=0.1)
    raise ValueError(f"Unknown model: {name}")


def build_dataloaders(
    root: str, batch_size: int, workers: int, download: bool, device: torch.device
) -> Tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_set = datasets.CIFAR100(root=root, train=True, download=download, transform=train_tf)
    test_set = datasets.CIFAR100(root=root, train=False, download=download, transform=val_tf)

    pin_memory = device.type in {"cuda", "mps"}
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )
    return train_loader, test_loader


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    params = model.parameters()

    if args.optimizer == "adamw":
        return AdamW(params, lr=args.lr, betas=tuple(args.betas), eps=args.eps, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    shampoo_kwargs = dict(
        start_prec_step=args.start_prec_step,
        stat_compute_steps=args.stat_compute_steps,
        prec_compute_steps=args.prec_compute_steps,
        stat_decay=args.stat_decay,
        matrix_eps=args.matrix_eps,
        prec_maxorder=args.prec_maxorder,
        prec_bits=args.prec_bits,
        min_lowbit_size=args.min_lowbit_size,
        quan_blocksize=args.quan_blocksize,
    )
    if args.optimizer == "shampoo1-adamw":
        return Shampoo1AdamW(
            params,
            lr=args.lr,
            betas=tuple(args.betas),
            eps=args.eps,
            weight_decay=args.weight_decay,
            **shampoo_kwargs,
        )
    if args.optimizer == "shampoo1-sgd":
        return Shampoo1SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
            **shampoo_kwargs,
        )
    if args.optimizer == "shampoo2-adamw":
        return Shampoo2AdamW(
            params,
            lr=args.lr,
            betas=tuple(args.betas),
            eps=args.eps,
            weight_decay=args.weight_decay,
            rect_t1=args.rect_t1,
            rect_t2=args.rect_t2,
            inv_root_mode=args.inv_root_mode,
            **shampoo_kwargs,
        )
    if args.optimizer == "shampoo2-sgd":
        return Shampoo2SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
            rect_t1=args.rect_t1,
            rect_t2=args.rect_t2,
            inv_root_mode=args.inv_root_mode,
            **shampoo_kwargs,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> Iterable[torch.Tensor]:
    with torch.no_grad():
        maxk = max(topk)
        pred = output.topk(maxk, dim=1)[1].t()
        correct = pred.eq(target.unsqueeze(0))
        return [(correct[:k].reshape(-1).float().sum(0) * (100.0 / target.size(0))) for k in topk]


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int,
    amp_enabled: bool,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Tuple[float, float]:
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    start = time.time()
    for step, (images, targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc1.item(), images.size(0))
        top5_meter.update(acc5.item(), images.size(0))

        if step % print_freq == 0:
            elapsed = time.time() - start
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch}] Step [{step}/{len(data_loader)}] "
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                f"Top1 {top1_meter.val:.2f} ({top1_meter.avg:.2f}) "
                f"Top5 {top5_meter.val:.2f} ({top5_meter.avg:.2f}) "
                f"LR {lr:.3e} "
                f"Time {elapsed:.1f}s",
                flush=True,
            )
            start = time.time()

    return top1_meter.avg, loss_meter.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast_context(device, amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc1.item(), images.size(0))
        top5_meter.update(acc5.item(), images.size(0))

    print(
        f"Validation  Loss {loss_meter.avg:.4f}  Top1 {top1_meter.avg:.2f}  Top5 {top5_meter.avg:.2f}",
        flush=True,
    )
    return top1_meter.avg, loss_meter.avg


def main() -> None:
    args = parse_args()

    overall_start = time.time()

    device, use_cuda, use_mps, resolved = resolve_device(args.device)
    if args.device == "auto":
        print(f"Auto-selected device: {resolved}")
    else:
        print(f"Using device: {resolved}")

    torch.backends.cudnn.benchmark = use_cuda

    model = build_model(args.model).to(device)
    optimizer = build_optimizer(args, model)
    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, val_loader = build_dataloaders(
        root=args.data,
        batch_size=args.batch_size,
        workers=args.workers,
        download=args.download,
        device=device,
    )

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    amp_enabled = args.amp and (use_cuda or use_mps)
    if args.amp and not amp_enabled:
        print("Automatic mixed precision requested, but it is not supported on the selected device.")
    scaler = torch.cuda.amp.GradScaler() if amp_enabled and use_cuda else None

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_top1 = 0.0
    for epoch in range(1, args.epochs + 1):
        top1_train, loss_train = train_one_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            device,
            epoch,
            args.print_freq,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )

        top1_val, loss_val = evaluate(model, criterion, val_loader, device, amp_enabled=amp_enabled)
        best_top1 = max(best_top1, top1_val)

        if scheduler is not None:
            scheduler.step()

        ckpt_path = output_dir / f"{args.model}_{args.optimizer}_epoch{epoch}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "top1_train": top1_train,
                "top1_val": top1_val,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")
        print(f"Best validation Top1 so far: {best_top1:.2f}%")

    total_elapsed = time.time() - overall_start
    print(f"Training completed in {total_elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
