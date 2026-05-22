# Developed By Habibur Rahaman
# University of Florida
# ECE Department

# Developed By Habibur Rahaman
# University of Florida
# ECE Department


from __future__ import annotations
import os, math, time, argparse, random, urllib.request, urllib.error, tarfile, hashlib, shutil, subprocess, sys
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset, Subset
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision
from torchvision import datasets, transforms, models


IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]


def now_str() -> str:
    return time.strftime("%H:%M:%S")

def log(rank: int, *msg: Any):
    if rank == 0:
        print("[" + now_str() + "]", *msg, flush=True)

def pick_device(arg_device: str, local_rank: int) -> torch.device:
    if arg_device.lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")

def maybe_compile(model: nn.Module, use_compile: bool) -> nn.Module:
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, fullgraph=False, mode="max-autotune")
        except Exception:
            pass
    return model

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


# =========================================================
# CIFAR data acquisition with two-stage fallback
#
#   Stage 1: toronto.edu canonical URL with 5 retries + exponential backoff.
#            Handles transient 503s, which are common on this server.
#   Stage 2: HuggingFace `datasets` library (auto-installed if missing).
#            Wraps HF's hosted CIFAR-10/100 in a torch Dataset.
#
# The rest of the pipeline doesn't care which source produced the data.
# =========================================================

CIFAR10_TGZ_URL  = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_TGZ_MD5  = "c58f30108f718f92721af3b95e74349a"
CIFAR10_TGZ_NAME = "cifar-10-python.tar.gz"
CIFAR10_DIR      = "cifar-10-batches-py"

CIFAR100_TGZ_URL  = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR100_TGZ_MD5  = "eb9058c3a382ffc7106e4002c42a8d85"
CIFAR100_TGZ_NAME = "cifar-100-python.tar.gz"
CIFAR100_DIR      = "cifar-100-python"


def _md5_of_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _download_with_retries(url: str, dst: str,
                           max_attempts: int = 5,
                           base_timeout: int = 60) -> bool:
    """
    Try downloading url -> dst with retries on transient errors.
    Backoff sequence: 2s, 5s, 15s, 30s, 60s.
    """
    backoffs = [2, 5, 15, 30, 60]
    user_agents = [
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "curl/8.5.0",
        "Wget/1.21",
        "python-requests/2.31",
    ]
    for attempt in range(max_attempts):
        ua = user_agents[attempt % len(user_agents)]
        try:
            print(f"  attempt {attempt+1}/{max_attempts}  (UA: {ua[:30]}...)")
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=base_timeout) as resp, open(dst, "wb") as out:
                shutil.copyfileobj(resp, out)
            return True
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"    failed: {type(e).__name__}: {e}")
            if os.path.exists(dst):
                try:
                    os.remove(dst)
                except OSError:
                    pass
            if attempt < max_attempts - 1:
                wait = backoffs[attempt]
                print(f"    waiting {wait}s before retry ...")
                time.sleep(wait)
    return False


def _try_torchvision_cifar(data_root: str, url: str, expected_md5: str,
                           tgz_name: str, dir_name: str) -> bool:
    os.makedirs(data_root, exist_ok=True)
    extracted_path = os.path.join(data_root, dir_name)
    if os.path.isdir(extracted_path):
        return True

    tgz_path = os.path.join(data_root, tgz_name)

    if os.path.isfile(tgz_path):
        try:
            if _md5_of_file(tgz_path) == expected_md5:
                print(f"[CIFAR] found valid {tgz_name}, extracting ...")
                with tarfile.open(tgz_path, "r:gz") as tf:
                    tf.extractall(data_root)
                return True
        except Exception:
            pass
        try:
            os.remove(tgz_path)
        except OSError:
            pass

    print(f"[CIFAR] trying canonical source: {url}")
    if not _download_with_retries(url, tgz_path):
        print(f"[CIFAR] toronto.edu unreachable after retries.")
        return False

    try:
        got = _md5_of_file(tgz_path)
        if got != expected_md5:
            print(f"[CIFAR] md5 mismatch (got {got}); discarding")
            os.remove(tgz_path)
            return False
    except Exception as e:
        print(f"[CIFAR] md5 check failed: {e}")
        return False

    print(f"[CIFAR] md5 OK, extracting ...")
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(data_root)
    return True


def _ensure_hf_datasets_installed():
    try:
        import datasets as _hf
        return _hf
    except ImportError:
        print("[HF] `datasets` library not found, installing via pip ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "datasets"]
        )
        import datasets as _hf
        return _hf


class _HFCifarDataset(Dataset):
    def __init__(self, hf_split, transform=None):
        self.hf = hf_split
        self.transform = transform
        cols = list(self.hf.column_names)
        if "img" in cols:
            self._img_col = "img"
        elif "image" in cols:
            self._img_col = "image"
        else:
            raise ValueError(f"No image column found in HF dataset (have: {cols})")
        if "label" in cols:
            self._lbl_col = "label"
        elif "fine_label" in cols:
            self._lbl_col = "fine_label"
        else:
            raise ValueError(f"No label column found (have: {cols})")

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, idx):
        row = self.hf[int(idx)]
        img = row[self._img_col]
        lbl = int(row[self._lbl_col])
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl


def _load_hf_cifar(name: str, tf_tr, tf_te) -> Tuple[Dataset, Dataset]:
    hf = _ensure_hf_datasets_installed()
    if name == "cifar10":
        repos = ["uoft-cs/cifar10", "cifar10"]
    elif name == "cifar100":
        repos = ["uoft-cs/cifar100", "cifar100"]
    else:
        raise ValueError(name)

    last_err = None
    for repo in repos:
        try:
            print(f"[HF] loading {repo} ...")
            ds = hf.load_dataset(repo)
            train_split = ds["train"]
            test_split  = ds["test"] if "test" in ds else ds["validation"]
            return (
                _HFCifarDataset(train_split, transform=tf_tr),
                _HFCifarDataset(test_split,  transform=tf_te),
            )
        except Exception as e:
            print(f"[HF] {repo} failed: {type(e).__name__}: {e}")
            last_err = e
    raise RuntimeError(f"Could not load {name} from HuggingFace either: {last_err}")


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
    if ds == "imagenet":
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMNET_MEAN, IMNET_STD),
            ])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMNET_MEAN, IMNET_STD),
        ])

    if ds in ("cifar10","cifar100"):
        mean, std = cifar_mean_std(ds)
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    if ds == "mnist":
        mean, std = [0.1307]*3, [0.3081]*3
        return transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),
            transforms.Normalize(mean, std),
        ])

    raise ValueError(f"Unknown dataset {dataset}")


def _safe_image_folder(root: str, transform=None):
    """
    Like torchvision.datasets.ImageFolder, but ignores any entry in `root`
    that is not a directory. Handles the common case where the ImageNet
    val folder also contains a stray metadata file (e.g. dataset-metadata.json).
    """
    # Build the list of valid class subdirectories
    if not os.path.isdir(root):
        raise ValueError(f"_safe_image_folder: {root} is not a directory")

    classes = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )
    if not classes:
        raise ValueError(f"_safe_image_folder: no class subdirectories under {root}")

    # ImageFolder lets us override the discovery via find_classes
    class _Filtered(datasets.ImageFolder):
        def find_classes(self, directory):
            cls = sorted(
                d for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d))
            )
            class_to_idx = {c: i for i, c in enumerate(cls)}
            return cls, class_to_idx

    return _Filtered(root, transform=transform)


def get_datasets(dataset: str, data_root: str, imagenet_root: Optional[str]=None):
    ds = dataset.lower()
    tf_tr = build_transforms(ds, train=True)
    tf_te = build_transforms(ds, train=False)

    if ds == "cifar10":
        ok = _try_torchvision_cifar(data_root, CIFAR10_TGZ_URL, CIFAR10_TGZ_MD5,
                                     CIFAR10_TGZ_NAME, CIFAR10_DIR)
        if ok:
            tr = datasets.CIFAR10(data_root, train=True,  transform=tf_tr, download=False)
            te = datasets.CIFAR10(data_root, train=False, transform=tf_te, download=False)
        else:
            print("[CIFAR] falling back to HuggingFace datasets for CIFAR-10")
            tr, te = _load_hf_cifar("cifar10", tf_tr, tf_te)
        nc = 10

    elif ds == "cifar100":
        ok = _try_torchvision_cifar(data_root, CIFAR100_TGZ_URL, CIFAR100_TGZ_MD5,
                                     CIFAR100_TGZ_NAME, CIFAR100_DIR)
        if ok:
            tr = datasets.CIFAR100(data_root, train=True,  transform=tf_tr, download=False)
            te = datasets.CIFAR100(data_root, train=False, transform=tf_te, download=False)
        else:
            print("[CIFAR] falling back to HuggingFace datasets for CIFAR-100")
            tr, te = _load_hf_cifar("cifar100", tf_tr, tf_te)
        nc = 100

    elif ds == "mnist":
        tr = datasets.MNIST(data_root, train=True,  transform=tf_tr, download=True)
        te = datasets.MNIST(data_root, train=False, transform=tf_te, download=True)
        nc = 10

    elif ds == "imagenet":
        if not imagenet_root:
            raise ValueError("dataset=imagenet requires --imagenet-root")
        train_dir = os.path.join(imagenet_root, "train") if os.path.isdir(os.path.join(imagenet_root, "train")) else imagenet_root
        val_dir   = os.path.join(imagenet_root, "val")   if os.path.isdir(os.path.join(imagenet_root, "val"))   else imagenet_root
        tr = _safe_image_folder(train_dir, transform=tf_tr)
        te = _safe_image_folder(val_dir,   transform=tf_te)
        nc = 1000

    else:
        raise ValueError(dataset)

    return tr, te, nc


def get_dataloaders(dataset, data_root, batch_size, num_workers,
                    dist_mode, imagenet_root=None, seed=42):
    tr_set, te_set, nc = get_datasets(dataset, data_root, imagenet_root)
    sampler = DistributedSampler(tr_set, shuffle=True, seed=seed) if dist_mode else None
    tr_loader = DataLoader(tr_set, batch_size=batch_size,
                           shuffle=(sampler is None), sampler=sampler,
                           num_workers=num_workers, pin_memory=True, drop_last=True)
    te_loader = DataLoader(te_set, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    return tr_loader, te_loader, nc, sampler


# =========================================================
# Models
# =========================================================

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


def build_model(arch: str, num_classes: int, use_pretrained: bool) -> nn.Module:
    a = arch.lower()

    TV_W = {
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
    }

    if a == "mlp":
        return CIFARMLP(num_classes)

    ctor = getattr(models, a, None)
    if ctor is None:
        raise ValueError(f"Unknown arch {arch}")

    weights = TV_W.get(a, None) if use_pretrained else None
    m = ctor(weights=weights)

    if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
        if m.fc.out_features != num_classes:
            m.fc = nn.Linear(m.fc.in_features, num_classes)

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


# =========================================================
# Training / Eval
# =========================================================

@dataclass
class TrainCfg:
    epochs: int = 120
    lr: float = 0.1
    weight_decay: float = 5e-4
    optim: str = "sgd"
    momentum: float = 0.9
    scheduler: str = "cosine"
    step_size: int = 60
    gamma: float = 0.2
    label_smoothing: float = 0.0
    warmup_epochs: int = 0


def build_optimizer(model, cfg):
    if cfg.optim.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                         weight_decay=cfg.weight_decay, nesterov=True)
    if cfg.optim.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optim)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    if cfg.scheduler == "cosine":
        def lr_lambda(epoch):
            if cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs:
                return float(epoch + 1) / float(cfg.warmup_epochs)
            t = (epoch - cfg.warmup_epochs) / max(1, (cfg.epochs - cfg.warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * t))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if cfg.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    if cfg.scheduler == "none":
        return None
    raise ValueError(cfg.scheduler)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    top1c = top5c = total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        total += targets.size(0)
        _, pred = outputs.topk(5, dim=1, largest=True, sorted=True)
        top1c += (pred[:, 0] == targets).sum().item()
        top5c += pred.eq(targets.view(-1, 1)).sum().item()
    top1 = 100.0 * top1c / total
    top5 = 100.0 * top5c / total
    print(f"[Eval] Top-1 = {top1:.2f}% | Top-5 = {top5:.2f}%")
    return top1, top5


# =========================================================
# PTQ
# =========================================================

def should_quantize_param(name, tensor):
    if not tensor.dtype.is_floating_point:
        return False
    lname = name.lower()
    if "running_mean" in lname or "running_var" in lname or "num_batches_tracked" in lname:
        return False
    if lname.endswith(".bias") or lname.endswith("_bias"):
        return False
    return True


def quantize_per_tensor(t, num_bits):
    t32 = t.detach().to(torch.float32)
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    max_abs = t32.abs().max().item() + 1e-12
    scale = max_abs / qmax
    q = torch.clamp(torch.round(t32 / scale), qmin, qmax)
    q = q.to(torch.int8 if num_bits <= 8 else torch.int16)
    return q, {"type": "per_tensor", "scale": float(scale)}


def quantize_per_channel_conv(t, num_bits):
    t32 = t.detach().to(torch.float32)
    if t32.ndim != 4:
        return quantize_per_tensor(t32, num_bits)
    OC = t32.shape[0]
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    flat = t32.view(OC, -1)
    max_abs = flat.abs().max(dim=1).values + 1e-12
    scales = max_abs / qmax
    q_list = []
    for c in range(OC):
        q_c = torch.clamp(torch.round(t32[c] / scales[c]), qmin, qmax)
        q_list.append(q_c)
    q = torch.stack(q_list, dim=0)
    q = q.to(torch.int8 if num_bits <= 8 else torch.int16)
    return q, {"type": "per_channel", "scales": scales.cpu().to(torch.float32)}


def quantize_param_smart(name, param, num_bits):
    if not should_quantize_param(name, param):
        return param.cpu(), None
    lname = name.lower()
    shape = param.shape
    if lname.endswith(".weight") and len(shape) == 4:
        return quantize_per_channel_conv(param, num_bits)
    if lname.endswith(".weight") and len(shape) == 2:
        return quantize_per_tensor(param, num_bits)
    return quantize_per_tensor(param, num_bits)


def save_quantized_checkpoint(model, path, num_bits):
    qsd, scales = {}, {}
    with torch.no_grad():
        for k, p in model.state_dict().items():
            if p.dtype.is_floating_point:
                q_tensor, scale_info = quantize_param_smart(k, p, num_bits)
                qsd[k] = q_tensor.cpu()
                scales[k] = scale_info
            else:
                qsd[k] = p.cpu()
                scales[k] = None
    payload = {"qstate_dict": qsd, "meta": {"num_bits": num_bits, "scales": scales}}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def dequantize_tensor(q, scale):
    return q.to(torch.float32) * float(scale)


def dequantize_per_channel_conv(q, scales):
    qf = q.to(torch.float32)
    s = scales
    while s.ndim < qf.ndim:
        s = s.unsqueeze(-1)
    return qf * s


def load_quantized_into_model(model, dataset, arch, num_bits, qtag, map_location):
    ck = ckpt_path(dataset, arch, f"int{num_bits}_{qtag}")
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


# =========================================================
# Checkpoint I/O
# =========================================================

def ckpt_path(dataset, arch, tag):
    return f"artifacts/models/{dataset.lower()}/{arch.lower()}/model_{tag}.pth"


def save_float_checkpoint(model, dataset, arch, tag, extra=None):
    path = ckpt_path(dataset, arch, tag)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_float_into_model(model, dataset, arch, tag, map_location):
    path = ckpt_path(dataset, arch, tag)
    payload = torch.load(path, map_location=map_location)
    sd = payload.get("state_dict", payload)
    sd = strip_prefix_from_state_dict(sd)
    model.load_state_dict(sd, strict=True)


def maybe_load_float32_or_pretrained(model, dataset, arch, use_pretrained, map_location="cpu"):
    ck = ckpt_path(dataset, arch, "float32")
    if os.path.exists(ck):
        load_float_into_model(model, dataset, arch, "float32", map_location=map_location)
        return "artifact_float32", os.path.getsize(ck) / 1e6
    if use_pretrained:
        return "torchvision_pretrained", 0.0
    return "random_init", 0.0


# =========================================================
# Commands
# =========================================================

def cmd_train(args):
    use_ddp = args.dist and dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if use_ddp else 0
    device = pick_device(args.device, local_rank)

    log(rank, f"[Device] {device} (DDP={use_ddp}, rank={rank}, world={(dist.get_world_size() if use_ddp else 1)})")

    train_loader, test_loader, nc, train_sampler = get_dataloaders(
        args.dataset, args.data_root, args.batch_size, args.workers,
        use_ddp, imagenet_root=args.imagenet_root,
    )

    model = build_model(args.arch, nc, use_pretrained=bool(args.use_pretrained)).to(device)
    model = maybe_compile(model, args.compile)

    if use_ddp:
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)

    cfg = TrainCfg(
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        optim=args.optim, momentum=args.momentum, scheduler=args.scheduler,
        step_size=args.step_size, gamma=args.gamma,
        label_smoothing=args.label_smoothing, warmup_epochs=args.warmup_epochs,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        if use_ddp:
            train_sampler.set_epoch(epoch)
        model.train()
        run_loss = 0.0
        corr = tot = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and args.scheduler_step == 'step':
                scheduler.step()
            run_loss += loss.item() * x.size(0)
            corr += (logits.argmax(1) == y).sum().item()
            tot += y.size(0)

        if scheduler is not None and args.scheduler_step == 'epoch':
            scheduler.step()

        train_acc = 100.0 * corr / tot
        test_top1, test_top5 = evaluate(model if not use_ddp else model.module, test_loader, device)
        log(rank, f"[Epoch {epoch:03d}/{cfg.epochs}] loss={run_loss/tot:.4f} train={train_acc:.2f}% test_top1={test_top1:.2f}% test_top5={test_top5:.2f}%")

        if test_top1 > best_acc and rank == 0:
            path = save_float_checkpoint(model.module if use_ddp else model,
                                         args.dataset, args.arch, "float32")
            log(rank, f"[Checkpoint] -> {path} (top1={test_top1:.2f}%)")
            best_acc = test_top1

    if rank == 0:
        elapsed = time.time() - t0
        fp32_path = ckpt_path(args.dataset, args.arch, "float32")
        fp32_size = os.path.getsize(fp32_path) / 1e6 if os.path.exists(fp32_path) else 0
        log(rank, f"[Train Done] Best top-1: {best_acc:.2f}%, elapsed {elapsed:.1f}s")
        log(rank, f"[Size] float32: {fp32_size:.2f} MB")


def cmd_quantize(args):
    _, _, nc = get_datasets(args.dataset, args.data_root, args.imagenet_root)
    use_pt = bool(getattr(args, "use_pretrained", 0))
    model = build_model(args.arch, nc, use_pretrained=use_pt)

    src = maybe_load_float32_or_pretrained(model, args.dataset, args.arch,
                                            use_pretrained=use_pt, map_location="cpu")

    fp32_path = ckpt_path(args.dataset, args.arch, "float32")
    os.makedirs(os.path.dirname(fp32_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, fp32_path)

    out_ckpt = ckpt_path(args.dataset, args.arch, f"int{args.bits}_ptq")
    save_quantized_checkpoint(model, out_ckpt, args.bits)

    fp32_mb = os.path.getsize(fp32_path) / 1e6 if os.path.exists(fp32_path) else 0.0
    q_mb    = os.path.getsize(out_ckpt) / 1e6   if os.path.exists(out_ckpt) else 0.0

    print(f"[Quantize] base_weights={src}")
    print(f"[Quantize] Saved float32 -> {fp32_path}")
    print(f"[Quantize] Saved int{args.bits} -> {out_ckpt}")
    print(f"[Size] float32: {fp32_mb:.2f} MB | int{args.bits}: {q_mb:.2f} MB")


def cmd_eval(args):
    device = pick_device(args.device, local_rank=0)
    print(f"[Device] {device}")

    _, test_loader, nc, _ = get_dataloaders(
        args.dataset, args.data_root, args.batch_size, args.workers,
        dist_mode=False, imagenet_root=getattr(args, "imagenet_root", None),
    )

    model = build_model(args.arch, nc, use_pretrained=bool(args.use_pretrained)).to(device)

    if args.bits is None:
        ck = ckpt_path(args.dataset, args.arch, "float32")
        if os.path.exists(ck):
            load_float_into_model(model, args.dataset, args.arch, "float32", map_location=device)
        tag = "float32"
    else:
        meta = load_quantized_into_model(model, args.dataset, args.arch,
                                          args.bits, args.qtag, map_location=device)
        tag = f"int{meta['num_bits']}_{args.qtag}"

    top1, top5 = evaluate(model, test_loader, device)
    print(f"[Eval Summary] {tag} Top-1 = {top1:.2f}% | Top-5 = {top5:.2f}%")


@torch.no_grad()
def quick_eval_for_inspector(model, dataset, data_root, imagenet_root,
                             device, workers, batch_size):
    _, test_loader, _, _ = get_dataloaders(
        dataset, data_root, batch_size, workers,
        dist_mode=False, imagenet_root=imagenet_root,
    )
    return evaluate(model, test_loader, device)


def show_structure(model):
    print(model)


def show_float_weights(model, layer_filter=None, max_vals=16):
    print("\n[Float Weights Preview]")
    for n, p in model.named_parameters():
        if (layer_filter is None) or any(f in n for f in layer_filter):
            vals = p.detach().flatten()[:max_vals].tolist()
            print(f"{n:30s}  shape={tuple(p.shape)}  vals={[round(v,6) for v in vals]}")


def cmd_inspect(args):
    device = pick_device(args.device if hasattr(args, "device") else "cuda", local_rank=0)
    _, _, nc = get_datasets(args.dataset, args.data_root, args.imagenet_root)
    model = build_model(args.arch, nc, use_pretrained=bool(args.use_pretrained)).to(device)

    qinfo: Dict[str, Tuple[torch.Tensor, Optional[Dict[str, Any]]]] = {}
    tag = "float32"

    if args.weights:
        payload = torch.load(args.weights, map_location=device)
        if "state_dict" in payload:
            sd = strip_prefix_from_state_dict(payload["state_dict"])
            model.load_state_dict(sd, strict=True)
            tag = "float32"
        elif "qstate_dict" in payload and "meta" in payload and "scales" in payload["meta"]:
            qsd = payload["qstate_dict"]
            sc  = payload["meta"]["scales"]
            dsd = {}
            for k, v in qsd.items():
                sinfo = sc.get(k, None)
                qinfo[k] = (v, sinfo)
                if sinfo is None:
                    dsd[k] = v
                else:
                    qtype = sinfo.get("type", None)
                    if qtype == "per_tensor":
                        dsd[k] = dequantize_tensor(v, sinfo["scale"])
                    elif qtype == "per_channel":
                        dsd[k] = dequantize_per_channel_conv(v, sinfo["scales"].to(v.device))
                    else:
                        raise ValueError(f"Unsupported quant scale type for {k}: {qtype}")
            dsd = strip_prefix_from_state_dict(dsd)
            model.load_state_dict(dsd, strict=True)
            nb = payload["meta"].get("num_bits", 8)
            tag = f"int{nb}_ptq"
        else:
            raise ValueError("Unknown checkpoint format passed to --weights in inspect")
        print(f"[Loaded] {args.weights} -> {tag}")
    else:
        load_float_into_model(model, args.dataset, args.arch, "float32", map_location=device)

    show_structure(model)

    if args.eval_acc:
        acc = quick_eval_for_inspector(
            model, args.dataset, args.data_root, args.imagenet_root,
            device=device,
            workers=args.workers if hasattr(args, "workers") else 2,
            batch_size=args.batch_size if hasattr(args, "batch_size") else 256,
        )
        print(f"\n[Test accuracy] {acc} on {args.dataset}")

    layer_filter = args.layers if args.layers else None
    show_float_weights(model, layer_filter, args.max_vals)

    if args.show_raw:
        if not qinfo:
            print("\n[No quantized tensors / raw view is empty]")
        else:
            print("\n[Quantized INT Weights Preview]")
            for n, (q_tensor, sinfo) in qinfo.items():
                if (layer_filter is None) or any(f in n for f in layer_filter):
                    if sinfo is None:
                        vals = q_tensor.detach().flatten()[:args.max_vals].tolist()
                        print(f"{n:30s}  (not quantized)  shape={tuple(q_tensor.shape)}  vals={vals}")
                    else:
                        qtype = sinfo.get("type", None)
                        if qtype == "per_tensor":
                            vals = q_tensor.detach().flatten()[:args.max_vals].tolist()
                            print(f"{n:30s}  per_tensor  dtype={q_tensor.dtype}  scale={sinfo['scale']}  vals={vals}")
                        elif qtype == "per_channel":
                            scale_preview = sinfo["scales"].detach().flatten()[:args.max_vals].tolist()
                            vals = q_tensor.detach().flatten()[:args.max_vals].tolist()
                            print(f"{n:30s}  per_channel  dtype={q_tensor.dtype}  scales={scale_preview}  vals={vals}")
                        else:
                            print(f"{n:30s}  UNKNOWN qtype={qtype}")


# =========================================================
# Argparse / DDP
# =========================================================

def build_parser():
    p = argparse.ArgumentParser(
        "Unified Trainer / PTQ / Eval / Inspector with ImageNet + ConvNeXt/EfficientNet/MobileNet"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p.add_argument("--data-root", default="./data")
    p.add_argument("--imagenet-root", default="")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--use-pretrained", type=int, default=0)

    arch_choices = ["resnet18","resnet34","resnet50","resnet101","vgg16","alexnet",
                    "mobilenet_v2","efficientnet_b0","efficientnet_b1","efficientnet_b2",
                    "efficientnet_b3","efficientnet_b4","convnext_base","convnext_large","mlp"]
    ds_choices = ["CIFAR10", "CIFAR100", "MNIST", "IMAGENET"]

    tr = sub.add_parser("train")
    tr.add_argument("--dataset", required=True, choices=ds_choices)
    tr.add_argument("--arch", required=True, choices=arch_choices)
    tr.add_argument("--epochs", type=int, default=120)
    tr.add_argument("--lr", type=float, default=0.1)
    tr.add_argument("--weight-decay", type=float, default=5e-4)
    tr.add_argument("--optim", default="sgd", choices=["sgd","adamw"])
    tr.add_argument("--momentum", type=float, default=0.9)
    tr.add_argument("--scheduler", default="cosine", choices=["cosine","step","none"])
    tr.add_argument("--step-size", type=int, default=60)
    tr.add_argument("--gamma", type=float, default=0.2)
    tr.add_argument("--label-smoothing", type=float, default=0.0)
    tr.add_argument("--warmup-epochs", type=int, default=0)
    tr.add_argument("--scheduler-step", choices=["epoch","step"], default="epoch")
    tr.add_argument("--dist", action="store_true")

    qz = sub.add_parser("quantize")
    qz.add_argument("--dataset", required=True, choices=ds_choices)
    qz.add_argument("--arch", required=True, choices=arch_choices)
    qz.add_argument("--bits", type=int, required=True, choices=[4,8,16])

    ev = sub.add_parser("eval")
    ev.add_argument("--dataset", required=True, choices=ds_choices)
    ev.add_argument("--arch", required=True, choices=arch_choices)
    ev.add_argument("--bits", type=int, choices=[4,8,16])
    ev.add_argument("--qtag", default="ptq")

    ins = sub.add_parser("inspect")
    ins.add_argument("--dataset", required=True, choices=ds_choices)
    ins.add_argument("--arch", required=True, choices=arch_choices)
    ins.add_argument("--weights", default="")
    ins.add_argument("--layers", nargs="*")
    ins.add_argument("--max-vals", type=int, default=16)
    ins.add_argument("--show-raw", action="store_true")
    ins.add_argument("--eval-acc", action="store_true")

    return p


def ddp_init_if_needed(args):
    if args.cmd == "train" and getattr(args, "dist", False):
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

def ddp_cleanup_if_needed(args):
    if args.cmd == "train" and getattr(args, "dist", False):
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main():
    p = build_parser()
    args = p.parse_args()
    args.dataset = args.dataset.upper()

    ddp_init_if_needed(args)
    try:
        if args.cmd == "train":      cmd_train(args)
        elif args.cmd == "quantize": cmd_quantize(args)
        elif args.cmd == "eval":     cmd_eval(args)
        elif args.cmd == "inspect":  cmd_inspect(args)
        else: raise ValueError(args.cmd)
    finally:
        ddp_cleanup_if_needed(args)


if __name__ == "__main__":
    main()