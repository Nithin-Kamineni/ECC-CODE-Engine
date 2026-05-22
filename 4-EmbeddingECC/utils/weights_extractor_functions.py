import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import sys
import math

# ------------------------------
# CIFAR test loader
# ------------------------------
def cifar_test_loader(dataset: str, batch_size: int = 128):
    norm = transforms.Normalize(mean=[0.4914,0.4822,0.4465],
                                std =[0.2470,0.2435,0.2616])
    tf = transforms.Compose([transforms.ToTensor(), norm])
    if dataset.lower() == "cifar10":
        ds = datasets.CIFAR10("./data", train=False, download=True, transform=tf)
        num_classes = 10
    elif dataset.lower() == "cifar100":
        ds = datasets.CIFAR100("./data", train=False, download=True, transform=tf)
        num_classes = 100
    else:
        raise ValueError("dataset must be CIFAR10 or CIFAR100")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True), num_classes


class CIFARMLP(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32, 1024), nn.ReLU(True),
            nn.Linear(1024, 512), nn.ReLU(True),
            nn.Linear(512, nc),
        )
    def forward(self, x): return self.net(x)

def adapt_resnet(m, nc):
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, nc)
    return m

def adapt_vgg(m, nc):
    in_f = m.classifier[0].in_features
    m.classifier = nn.Sequential(
        nn.Linear(in_f, 512), nn.ReLU(True), nn.Dropout(0.5),
        nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
        nn.Linear(512, nc),
    )
    return m

def adapt_alexnet(m, nc):
    m.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    with torch.no_grad():
        d = torch.zeros(1,3,32,32)
        feat = m.features(d)
        if hasattr(m, "avgpool"): feat = m.avgpool(feat)
        in_f = int(feat.numel())
    m.classifier = nn.Sequential(
        nn.Dropout(), nn.Linear(in_f, 512), nn.ReLU(True),
        nn.Dropout(), nn.Linear(512, nc)
    )
    return m

def build_model(arch, nc):
    a = arch.lower()
    if a in ["resnet18","resnet34","resnet50"]:
        return adapt_resnet(getattr(models, a)(weights=None), nc)
    if a == "vgg16":
        return adapt_vgg(models.vgg16(weights=None), nc)
    if a == "alexnet":
        return adapt_alexnet(models.alexnet(weights=None), nc)
    if a == "mlp":
        return CIFARMLP(nc)
    raise ValueError(f"Unsupported arch: {arch}")


def _dequant(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q.to(torch.float32) * float(scale)

def load_any_checkpoint(model: nn.Module, path: str, map_location="cpu", strict=True):

    payload = torch.load(path, map_location=map_location)
    quant_info = {}

    if isinstance(payload, dict) and "state_dict" in payload:
        model.load_state_dict(payload["state_dict"], strict=strict)
        return "float32", quant_info

    if isinstance(payload, dict) and "qstate_dict" in payload and "meta" in payload and "scales" in payload["meta"]:
        qsd, scales = payload["qstate_dict"], payload["meta"]["scales"]
        dsd = {}
        for k, v in qsd.items():
            s = scales.get(k, None)
            quant_info[k] = (v, s)
            if s is None:
                dsd[k] = v if not torch.is_floating_point(v) else v.to(torch.float32)
            else:
                dsd[k] = _dequant(v, s)
        missing, unexpected = model.load_state_dict(dsd, strict=strict)
        if missing or unexpected:
            print(f"[Warn] missing={len(missing)} unexpected={len(unexpected)}")
        nb = payload["meta"].get("num_bits", "X")
        tag = payload["meta"].get("tag", "ptq")
        return f"int{nb}_{tag}", quant_info

    if isinstance(payload, dict) and all(isinstance(v, torch.Tensor) for v in payload.values()):
        model.load_state_dict(payload, strict=strict)
        return "float32(raw)", quant_info

    raise ValueError(f"Unrecognized checkpoint format: {path}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total

def _name_matches(name: str, filters):
    return (filters is None) or any(f in name for f in filters)

def show_float_weights_from_model(model: nn.Module, layers=None, max_vals: int = 20):
    print("\n[Dequantized/Float weights from model (first values)]")
    for name, p in model.named_parameters():
        if _name_matches(name, layers):
            flat = p.detach().cpu().view(-1)
            vals = flat[:max_vals].tolist()
            print(f"{name:40s} shape={str(tuple(p.shape)):18s} first{min(len(vals),max_vals)}={[round(v,6) for v in vals]} ...")

def show_quantized_weights_raw(quant_info: dict, layers=None, max_vals: int = 20):
    if not quant_info:
        print("\n[No quantized payload detected; nothing to show in raw INT format]")
        return
    print("\n[RAW quantized weights from checkpoint (integers) + scales (first values)]")
    for name, (q, s) in quant_info.items():
        if _name_matches(name, layers):
            qcpu = q.detach().cpu().view(-1)
            qvals = qcpu[:max_vals].tolist()
            dtype = str(q.dtype).replace("torch.", "")
            print(f"{name:40s} qdtype={dtype:6s} scale={s} shape={str(tuple(q.shape)):18s} first{min(len(qvals),max_vals)}={qvals} ...")

def count_learnable_layers(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if any(p.numel() for p in m.parameters(recurse=False)))

def param_counts(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def save_updated_quant_checkpoint(dst_path: str, editable_quant_info: dict, tag_suffix: str = "_edited"):
    payload = torch.load(dst_path, map_location="cpu")
    if not (isinstance(payload, dict) and "qstate_dict" in payload and "meta" in payload and "scales" in payload["meta"]):
        raise ValueError("Destination checkpoint missing qstate_dict/meta/scales.")

    # Update qstate_dict tensors from editable_quant_info
    qsd = payload["qstate_dict"]
    for name, (q_new, s_new) in editable_quant_info.items():
        if name in qsd:
            # Ensure dtype/device align; move if needed
            if qsd[name].dtype != q_new.dtype or qsd[name].device != q_new.device or qsd[name].shape != q_new.shape:
                qsd[name] = q_new.to(dtype=qsd[name].dtype, device="cpu").reshape_as(qsd[name])
            else:
                qsd[name].copy_(q_new)
        else:
            # If a new param appears, we can choose to add it (rare)
            qsd[name] = q_new.clone()

        # Keep the scale as-is unless you also changed it
        if s_new is not None:
            payload["meta"]["scales"][name] = s_new

    # mark edit in meta tag
    # old_tag = payload["meta"].get("tag", "ptq")
    # payload["meta"]["tag"] = str(old_tag) + str(tag_suffix)

    torch.save(payload, dst_path)
    return dst_path

def copy_quant_checkpoint(src_path: str, dst_path: str):
    """
    Copies a quantized checkpoint file (expects keys like 'qstate_dict' and 'meta').
    If it's already float-only, raises a ValueError.
    """
    payload = torch.load(src_path, map_location="cpu")
    if not (isinstance(payload, dict) and "qstate_dict" in payload and "meta" in payload):
        raise ValueError("Source checkpoint does not look quantized (missing 'qstate_dict'/'meta').")

    # Shallow copy the dict and tensors; we will *clone* them later when editing
    new_payload = {
        "qstate_dict": {k: v.clone() for k, v in payload["qstate_dict"].items()},  # copy tensors
        "meta": dict(payload["meta"]),  # scales, num_bits, tag, etc.
    }
    # Optionally carry other fields if present:
    for k in payload.keys():
        if k not in new_payload and k not in ("state_dict",):  # don't accidentally keep float state_dict
            new_payload[k] = payload[k]

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save(new_payload, dst_path)
    return dst_path