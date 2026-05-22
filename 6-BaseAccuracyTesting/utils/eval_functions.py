from __future__ import annotations
import os, math, time, argparse, random
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision
from torchvision import datasets, transforms, models


IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]

def pick_device(arg_device: str, local_rank: int) -> torch.device:
    if arg_device.lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")

def get_datasets(dataset: str, data_root: str, imagenet_root: Optional[str]=None):
    ds = dataset.lower()
    tf_tr = build_transforms(ds, train=True)
    tf_te = build_transforms(ds, train=False)

    if ds == "cifar10":
        tr = datasets.CIFAR10(data_root, train=True,  transform=tf_tr, download=True)
        te = datasets.CIFAR10(data_root, train=False, transform=tf_te, download=True)
        nc = 10

    elif ds == "cifar100":
        tr = datasets.CIFAR100(data_root, train=True,  transform=tf_tr, download=True)
        te = datasets.CIFAR100(data_root, train=False, transform=tf_te, download=True)
        nc = 100

    elif ds == "mnist":
        tr = datasets.MNIST(data_root, train=True,  transform=tf_tr, download=True)
        te = datasets.MNIST(data_root, train=False, transform=tf_te, download=True)
        nc = 10

    elif ds == "imagenet":
        if imagenet_root is None or imagenet_root == "":
            raise ValueError("For dataset=imagenet you MUST pass --imagenet-root pointing to ImageNet folder (train or val).")



        train_dir_try1 = os.path.join(imagenet_root, "train")
        train_dir_try2 = imagenet_root
        train_dir = train_dir_try1 if os.path.isdir(train_dir_try1) else train_dir_try2

        val_dir_try1 = os.path.join(imagenet_root, "val")
        val_dir_try2 = imagenet_root
        val_dir = val_dir_try1 if os.path.isdir(val_dir_try1) else val_dir_try2

        tr = datasets.ImageFolder(train_dir, transform=tf_tr)
        te = datasets.ImageFolder(val_dir,   transform=tf_te)
        nc = 1000  # standard ImageNet-1k

    else:
        raise ValueError(dataset)

    return tr, te, nc

def get_dataloaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    dist_mode: bool,
    imagenet_root: Optional[str]=None,
    seed: int = 42
):
    tr_set, te_set, nc = get_datasets(dataset, data_root, imagenet_root)
    sampler = DistributedSampler(tr_set, shuffle=True, seed=seed) if dist_mode else None
    tr_loader = DataLoader(
        tr_set,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    te_loader = DataLoader(
        te_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return tr_loader, te_loader, nc, sampler

def cifar_mean_std(dataset: str):
    ds = dataset.lower()
    if ds == "cifar10":
        return [0.4914,0.4822,0.4465],[0.2470,0.2435,0.2616]
    elif ds == "cifar100":
        return [0.5071,0.4867,0.4408],[0.2675,0.2565,0.2761]
    else:
        
        return [0.4914,0.4822,0.4465],[0.2470,0.2435,0.2616]

def build_transforms(dataset: str, train: bool):
    ds = dataset.lower()
    # ImageNet-style (224)
    if ds == "imagenet":
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMNET_MEAN, IMNET_STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMNET_MEAN, IMNET_STD),
            ])

    # CIFAR10/100 (32x32 native)
    if ds in ("cifar10","cifar100"):
        mean,std = cifar_mean_std(ds)
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean,std),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean,std),
            ])

    # MNIST (grayscale -> 3ch 32x32)
    if ds == "mnist":
        mean,std = [0.1307]*3,[0.3081]*3
        base = [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3,-1,-1)),
            transforms.Normalize(mean,std),
        ]
        return transforms.Compose(base)

    raise ValueError(f"Unknown dataset {dataset}")

def build_model(arch: str, num_classes: int, use_pretrained: bool) -> nn.Module:
    a = arch.lower()

    TV_WEIGHTS = {
        "resnet18":      getattr(models, "ResNet18_Weights",      None).IMAGENET1K_V1 if hasattr(models, "ResNet18_Weights") else None,
        "resnet34":      getattr(models, "ResNet34_Weights",      None).IMAGENET1K_V1 if hasattr(models, "ResNet34_Weights") else None,
        "resnet50":      getattr(models, "ResNet50_Weights",      None).IMAGENET1K_V1 if hasattr(models, "ResNet50_Weights") else None,
        "resnet101":     getattr(models, "ResNet101_Weights",     None).IMAGENET1K_V1 if hasattr(models, "ResNet101_Weights") else None,
        "vgg16":         getattr(models, "VGG16_Weights",         None).IMAGENET1K_V1 if hasattr(models, "VGG16_Weights") else None,
        "alexnet":       getattr(models, "AlexNet_Weights",       None).IMAGENET1K_V1 if hasattr(models, "AlexNet_Weights") else None,
        "mobilenet_v2":  getattr(models, "MobileNet_V2_Weights",  None).IMAGENET1K_V1 if hasattr(models, "MobileNet_V2_Weights") else None,
        "efficientnet_b0": getattr(models, "EfficientNet_B0_Weights", None).IMAGENET1K_V1 if hasattr(models, "EfficientNet_B0_Weights") else None,
        "efficientnet_b1": getattr(models, "EfficientNet_B1_Weights", None).IMAGENET1K_V1 if hasattr(models, "EfficientNet_B1_Weights") else None,
        "efficientnet_b2": getattr(models, "EfficientNet_B2_Weights", None).IMAGENET1K_V1 if hasattr(models, "EfficientNet_B2_Weights") else None,
        "efficientnet_b3": getattr(models, "EfficientNet_B3_Weights", None).IMAGENET1K_V1 if hasattr(models, "EfficientNet_B3_Weights") else None,
        "efficientnet_b4": getattr(models, "EfficientNet_B4_Weights", None).IMAGENET1K_V1 if hasattr(models, "EfficientNet_B4_Weights") else None,
        "convnext_base":   getattr(models, "ConvNeXt_Base_Weights",   None).IMAGENET1K_V1 if hasattr(models, "ConvNeXt_Base_Weights") else None,
        "convnext_large":  getattr(models, "ConvNeXt_Large_Weights",  None).IMAGENET1K_V1 if hasattr(models, "ConvNeXt_Large_Weights") else None,
        # "mlp" doesn't have pretrained
    }

    # Special case: simple MLP for CIFAR etc.
    class CIFARMLP(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3*32*32, 1024), nn.ReLU(True),
                nn.Linear(1024, 512), nn.ReLU(True),
                nn.Linear(512, num_classes),
            )
        def forward(self, x):
            return self.net(x)

    if a == "mlp":
        return CIFARMLP(num_classes)

    # pick constructor
    ctor = getattr(models, a, None)
    if ctor is None:
        raise ValueError(f"Unknown arch {arch}")

    # choose weights if user asked for pretrained
    weights = TV_WEIGHTS.get(a, None) if use_pretrained else None

    m = ctor(weights=weights)

    # fix classifier head if num_classes != default
    # ResNet-style
    if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
        if m.fc.out_features != num_classes:
            m.fc = nn.Linear(m.fc.in_features, num_classes)

    # torchvision models like MobileNetV2 / EfficientNet / ConvNeXt expose .classifier
    if hasattr(m, "classifier"):
        if isinstance(m.classifier, nn.Sequential):
            last = m.classifier[-1]
            if isinstance(last, nn.Linear) and last.out_features != num_classes:
                in_f = last.in_features
                new_seq = list(m.classifier[:-1]) + [nn.Linear(in_f, num_classes)]
                m.classifier = nn.Sequential(*new_seq)
        elif isinstance(m.classifier, nn.Linear):
            if m.classifier.out_features != num_classes:
                m.classifier = nn.Linear(m.classifier.in_features, num_classes)

    return m

def ckpt_path(dataset: str, arch: str, tag: str) -> str:
    return f"artifacts/models/{dataset.lower()}/{arch.lower()}/model_{tag}.pth"

def dequantize_tensor(q: torch.Tensor, scale: float):
    """Per-tensor dequantization."""
    return q.to(torch.float32) * float(scale)


def dequantize_per_channel_conv(q: torch.Tensor, scales: torch.Tensor):
    """Per-channel dequantization for Conv2d weights."""
    qf = q.to(torch.float32)
    s = scales
    while s.ndim < qf.ndim:
        s = s.unsqueeze(-1)
    return qf * s

def strip_prefix_from_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        out[nk] = v
    return out

def load_quantized_into_model(model: nn.Module, dataset: str, arch: str,
                              num_bits: int, qtag: str, map_location: torch.device, weight_argument: str = None):
    """Load quantized checkpoint and rebuild float weights."""
    if(weight_argument is None):
        ck = ckpt_path(dataset, arch, f"int{num_bits}_{qtag}")
    
    ck = weight_argument
     
    payload = torch.load(ck, map_location=map_location)
    qsd, scales = payload["qstate_dict"], payload["meta"]["scales"]

    dsd = {}
    for k, v in qsd.items():
        sinfo = scales.get(k, None)
        if sinfo is None:
            dsd[k] = v
        else:
            t = sinfo.get("type", None)
            if t == "per_tensor":
                dsd[k] = dequantize_tensor(v, sinfo["scale"])
            elif t == "per_channel":
                dsd[k] = dequantize_per_channel_conv(v, sinfo["scales"].to(v.device))
            else:
                dsd[k] = v

    dsd = strip_prefix_from_state_dict(dsd)
    model.load_state_dict(dsd, strict=True)
    return payload["meta"]

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            total += targets.size(0)

            # Get top-k predictions (k = 5)
            _, pred = outputs.topk(5, dim=1, largest=True, sorted=True)

            # Top-1 check
            top1_correct += (pred[:, 0] == targets).sum().item()

            # Top-5 check
            top5_correct += (
                pred.eq(targets.view(-1, 1)).sum().item()
            )

    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total

    print(f"[Eval] Top-1 = {top1_acc:.2f}% | Top-5 = {top5_acc:.2f}%")
    return top1_acc, top5_acc