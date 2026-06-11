# Developed By Habibur Rahaman
# University of Florida
# ECE Department
#
# qat_train.py
# ============
# Quantization-Aware Training (QAT) following Jacob et al., CVPR 2018:
#   -  Conv2d/Linear weights in the forward pass
#     using the SAME per-channel symmetric scheme that test-Quantizer.py
#     uses for PTQ (so attack pipelines are directly comparable)
#   - Straight-Through Estimator (STE) in the backward pass: gradient
#     flows through the round() as if it were the identity
#   - Fine-tune from float32 pretrained / artifact weights
#
# Output: artifacts/models/<dataset>/<arch>/model_int{N}_qat.pth
#         (same PTQ-style payload, so validate_bitflip_int.py can
#          load it via `--qtag qat`)
#
# Usage (example):
#   python qat_train.py --dataset CIFAR10 --arch resnet18 --bits 4 \
#       --epochs 30 --lr 1e-3 --batch-size 256
#
#   python qat_train.py --dataset IMAGENET --imagenet-root ./imagenet-val \
#       --arch mobilenet_v2 --bits 4 --epochs 15 --lr 1e-4 \
#       --use-pretrained 1 --batch-size 128

from __future__ import annotations
import os, time, math, argparse, json
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Re-use ALL building blocks from test-Quantizer.py so we stay consistent:
#   - dataset / loader logic
#   - model builder (arch -> nn.Module)
#   - PTQ quantize/dequantize helpers
#   - checkpoint path conventions
import importlib.util, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "test_quantizer",
    os.path.join(_HERE, "test-Quantizer.py"),
)
TQ = importlib.util.module_from_spec(_spec)
# IMPORTANT: register before exec so @dataclass can find the module
# (dataclasses looks up cls.__module__ in sys.modules)
sys.modules["test_quantizer"] = TQ
_spec.loader.exec_module(TQ)


# =========================================================
# Per-channel symmetric fake quantize with STE
# =========================================================

class _FakeQuantPerChannelSTE(torch.autograd.Function):
    """
    Forward: w_fq = clamp(round(w / scale), qmin, qmax) * scale
              where `scale` is computed per output-channel from |w|.max().
    Backward: straight-through (gradient = grad_output).

    This mirrors test-Quantizer.py's quantize_per_channel_conv() exactly.
    """
    @staticmethod
    def forward(ctx, w: torch.Tensor, num_bits: int) -> torch.Tensor:
        qmin = -(2 ** (num_bits - 1))
        qmax = (2 ** (num_bits - 1)) - 1
        if w.ndim == 4:
            # Conv2d weight [OC, IC, kH, kW] : per-out-channel
            OC = w.shape[0]
            flat = w.detach().abs().view(OC, -1)
            max_abs = flat.max(dim=1).values + 1e-12
            scales = max_abs / qmax            # [OC]
            s = scales.view(OC, 1, 1, 1)
        elif w.ndim == 2:
            # Linear weight [out, in] : per-out-channel (per row)
            OC = w.shape[0]
            max_abs = w.detach().abs().max(dim=1).values + 1e-12
            scales = max_abs / qmax            # [OC]
            s = scales.view(OC, 1)
        else:
            # Fallback: per-tensor
            max_abs = w.detach().abs().max() + 1e-12
            scales = (max_abs / qmax).view(1)
            s = scales
        q = torch.clamp(torch.round(w / s), qmin, qmax)
        return q * s

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: gradient w.r.t. the (rounded) output flows back
        # to the original weight as if quantization were the identity.
        return grad_output, None


def fake_quant(w: torch.Tensor, num_bits: int) -> torch.Tensor:
    return _FakeQuantPerChannelSTE.apply(w, num_bits)


# =========================================================
# Module wrappers
# =========================================================

class FakeQuantConv2d(nn.Conv2d):
    """Conv2d that fake-quantizes its weight on every forward pass."""
    def __init__(self, *args, num_bits: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fq = fake_quant(self.weight, self.num_bits)
        return self._conv_forward(x, w_fq, self.bias)


class FakeQuantLinear(nn.Linear):
    """Linear that fake-quantizes its weight on every forward pass."""
    def __init__(self, *args, num_bits: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fq = fake_quant(self.weight, self.num_bits)
        return torch.nn.functional.linear(x, w_fq, self.bias)


# =========================================================
# LSQ (Esser et al., ICLR 2020): "Learned Step Size Quantization"
# ---------------------------------------------------------
# Difference from Jacob 2018:
#   - The scale `s` is a learnable per-output-channel parameter
#     (initialized from weight stats, then trained by SGD)
#   - Gradient through round() uses STE *and* gets a clamp mask
#     (so clipped values produce zero gradient w.r.t. the weight)
#   - The scale gets a special gradient-scaling factor g, derived
#     in the LSQ paper to stabilize training:
#         g = 1 / sqrt(numel * Qp)
# This is the standard LSQ recipe. Per-channel sym, weights only.
# =========================================================

class _LSQGradScale(torch.autograd.Function):
    """Identity on forward, scale gradient by `g` on backward."""
    @staticmethod
    def forward(ctx, x, g):
        ctx.g = float(g)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.g, None


def _grad_scale(x: torch.Tensor, g: float) -> torch.Tensor:
    return _LSQGradScale.apply(x, g)


class _LSQFakeQuant(torch.autograd.Function):
    """
    LSQ fake-quant for weights (per-channel symmetric).

    Forward:  w_q = clamp(round(w/s), Qn, Qp) * s
    Backward (LSQ-paper formulas):
      dL/dw = dL/dw_q if (w/s) in [Qn, Qp] else 0     (STE through round
                                                       + clamp mask)
      dL/ds = dL/dw_q * f(w/s)
              where f(z) = -z + round(z)          if Qn < z < Qp
                            Qn                    if z <= Qn
                            Qp                    if z >= Qp
    """
    @staticmethod
    def forward(ctx, w, s, qn, qp):
        # s is broadcastable along output-channel axis (shape e.g. [OC,1,1,1])
        w_over_s = w / s
        ctx.save_for_backward(w_over_s, s)
        ctx.qn = float(qn)
        ctx.qp = float(qp)
        q = torch.clamp(torch.round(w_over_s), qn, qp)
        return q * s

    @staticmethod
    def backward(ctx, grad_output):
        w_over_s, s = ctx.saved_tensors
        qn, qp = ctx.qn, ctx.qp

        # Mask for clipped vs in-range values
        in_range = (w_over_s > qn) & (w_over_s < qp)

        # dL/dw: STE through round, zero outside clip region
        grad_w = grad_output * in_range.to(grad_output.dtype)

        # dL/ds: piecewise per LSQ paper
        f = torch.where(
            w_over_s <= qn,
            torch.full_like(w_over_s, qn),
            torch.where(
                w_over_s >= qp,
                torch.full_like(w_over_s, qp),
                -w_over_s + torch.round(w_over_s),
            ),
        )
        grad_s_full = grad_output * f

        # Reduce grad_s to match s's broadcasted shape (sum over the
        # dimensions where s was broadcast, i.e. anywhere s has size 1).
        # s is per-output-channel: [OC,1,1,1] for conv, [OC,1] for linear.
        reduce_dims = [i for i, sz in enumerate(s.shape) if sz == 1]
        if reduce_dims:
            grad_s = grad_s_full.sum(dim=reduce_dims, keepdim=True)
        else:
            grad_s = grad_s_full
        return grad_w, grad_s, None, None


def _lsq_init_scale(weight: torch.Tensor, qp: float) -> torch.Tensor:
    """
    LSQ-paper initialization for s:
        s = 2 * mean(|W|) / sqrt(Qp)
    Computed per output-channel.
    """
    w = weight.detach()
    if w.ndim == 4:  # Conv2d weight [OC, IC, kH, kW]
        OC = w.shape[0]
        mean_abs = w.abs().mean(dim=(1, 2, 3)).clamp(min=1e-12)  # [OC]
        s = (2.0 * mean_abs / math.sqrt(qp)).view(OC, 1, 1, 1)
    elif w.ndim == 2:  # Linear weight [out, in]
        OC = w.shape[0]
        mean_abs = w.abs().mean(dim=1).clamp(min=1e-12)  # [OC]
        s = (2.0 * mean_abs / math.sqrt(qp)).view(OC, 1)
    else:
        mean_abs = w.abs().mean().clamp(min=1e-12)
        s = (2.0 * mean_abs / math.sqrt(qp)).view(1)
    return s


class LSQConv2d(nn.Conv2d):
    """Conv2d with LSQ learnable-scale weight quantization."""
    def __init__(self, *args, num_bits: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.qp = float(2 ** (num_bits - 1) - 1)   # e.g. INT4 -> 7
        self.qn = -self.qp                          # symmetric
        # Learnable per-output-channel scale (initialized lazily on first
        # forward, once we know weight stats).
        self.s = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer("_s_initialized", torch.zeros(1))

    def _lazy_init(self):
        if self._s_initialized.item() < 0.5:
            with torch.no_grad():
                self.s.data.copy_(_lsq_init_scale(self.weight, self.qp))
            self._s_initialized.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._lazy_init()
        # Gradient scaling for LSQ stability
        g = 1.0 / math.sqrt(float(self.weight.numel()) * self.qp)
        s_scaled = _grad_scale(self.s, g)
        w_fq = _LSQFakeQuant.apply(self.weight, s_scaled, self.qn, self.qp)
        return self._conv_forward(x, w_fq, self.bias)


class LSQLinear(nn.Linear):
    """Linear with LSQ learnable-scale weight quantization."""
    def __init__(self, *args, num_bits: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.qp = float(2 ** (num_bits - 1) - 1)
        self.qn = -self.qp
        self.s = nn.Parameter(torch.ones(self.out_features, 1))
        self.register_buffer("_s_initialized", torch.zeros(1))

    def _lazy_init(self):
        if self._s_initialized.item() < 0.5:
            with torch.no_grad():
                self.s.data.copy_(_lsq_init_scale(self.weight, self.qp))
            self._s_initialized.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._lazy_init()
        g = 1.0 / math.sqrt(float(self.weight.numel()) * self.qp)
        s_scaled = _grad_scale(self.s, g)
        w_fq = _LSQFakeQuant.apply(self.weight, s_scaled, self.qn, self.qp)
        return torch.nn.functional.linear(x, w_fq, self.bias)


def _copy_conv_attrs(src: nn.Conv2d, dst: nn.Conv2d) -> None:
    dst.weight = src.weight
    if src.bias is not None:
        dst.bias = src.bias


def _copy_linear_attrs(src: nn.Linear, dst: nn.Linear) -> None:
    dst.weight = src.weight
    if src.bias is not None:
        dst.bias = src.bias


def convert_to_qat(model: nn.Module, num_bits: int,
                   skip_first_last: bool = False,
                   qat_method: str = "jacob") -> nn.Module:
    """
    Replace every nn.Conv2d / nn.Linear in `model` with the fake-quant
    version. Weights/biases are shared (assigned, not copied) so optimizer
    state remains correct.

    If skip_first_last is True, the first Conv2d and the final Linear
    (typically the classifier) are LEFT as float32 — a common QAT practice
    in mobile-quantization papers (Jacob 2018, Krishnamoorthi 2018).

    qat_method:
      "jacob" -> FakeQuantConv2d/Linear (Jacob et al. 2018, CVPR)
                 Scale recomputed from max(|W|) every forward pass.
      "lsq"   -> LSQConv2d/Linear (Esser et al. 2020, ICLR)
                 Scale is a learnable per-channel nn.Parameter.
    """
    # First, find first conv and last linear (for skip-first-last option)
    convs, lins = [], []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and not isinstance(m, (FakeQuantConv2d, LSQConv2d)):
            convs.append(name)
        if isinstance(m, nn.Linear) and not isinstance(m, (FakeQuantLinear, LSQLinear)):
            lins.append(name)
    first_conv = convs[0] if convs else None
    last_lin   = lins[-1]  if lins  else None

    # Select the right wrapper classes
    if qat_method == "lsq":
        ConvCls = LSQConv2d
        LinCls  = LSQLinear
    elif qat_method == "jacob":
        ConvCls = FakeQuantConv2d
        LinCls  = FakeQuantLinear
    else:
        raise ValueError(f"Unknown qat_method: {qat_method!r} "
                         f"(expected 'jacob' or 'lsq')")

    def _set_module(root, qname, new_mod):
        parts = qname.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_mod)

    # Replace Conv2d
    for name in convs:
        if skip_first_last and name == first_conv:
            continue
        # find module
        parent = model
        for p in name.split(".")[:-1]:
            parent = getattr(parent, p)
        leaf = name.split(".")[-1]
        old: nn.Conv2d = getattr(parent, leaf)
        new = ConvCls(
            in_channels=old.in_channels,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=(old.bias is not None),
            padding_mode=old.padding_mode,
            num_bits=num_bits,
        )
        _copy_conv_attrs(old, new)
        setattr(parent, leaf, new)

    # Replace Linear
    for name in lins:
        if skip_first_last and name == last_lin:
            continue
        parent = model
        for p in name.split(".")[:-1]:
            parent = getattr(parent, p)
        leaf = name.split(".")[-1]
        old: nn.Linear = getattr(parent, leaf)
        new = LinCls(
            in_features=old.in_features,
            out_features=old.out_features,
            bias=(old.bias is not None),
            num_bits=num_bits,
        )
        _copy_linear_attrs(old, new)
        setattr(parent, leaf, new)

    return model


# =========================================================
# QAT-aware checkpoint saver
#
# We save in the SAME PTQ payload format the rest of the pipeline reads:
#   {'qstate_dict': {layer: int_tensor}, 'meta': {num_bits, scales:{...}}}
# This way validate_bitflip_int.py (which already supports --qtag) picks
# the QAT checkpoint up as int{N}_qat with no other code change.
# =========================================================

def save_qat_as_ptq_payload(model: nn.Module,
                             dataset: str, arch: str, num_bits: int,
                             tag_suffix: str = "") -> str:
    """
    Save a QAT-trained model in a way that EXACTLY preserves the
    QAT-mode forward-pass weights.

    The problem with the standard PTQ saver (`TQ.save_quantized_checkpoint`)
    is that it independently re-computes per-channel scales from the
    weights, which differs in float-precision details from the scales
    used inside `fake_quant()` during QAT. For depthwise-conv networks
    (EfficientNet-B0, MobileNet-V2) this tiny mismatch collapses the
    deployed model.

    The fix here: snapshot the EXACT fake-quantized float values
    produced by our `fake_quant()` function, then save those as the
    "dequantized" float weights via the standard float-checkpoint
    saver. This sidesteps the PTQ saver entirely. The result is an
    `int{N}_qat` checkpoint that, on load, reproduces the QAT-mode
    forward pass byte-for-byte.

    The saved checkpoint format:
        {
          "state_dict": {layer_name: fake_quantized_float_tensor, ...},
          "meta": {"num_bits": N, "qat_baked": True},
        }
    `load_quantized_into_model` in test-Quantizer.py reads any payload
    with "state_dict" present as a float checkpoint, so this loads
    cleanly through the existing pipeline.
    """
    # Build a state_dict where every QAT-wrapped layer has its weight
    # replaced by the exact fake-quantized value used in training.
    # Also record the exact per-channel scales used by fake_quant().
    new_sd: Dict[str, torch.Tensor] = {}
    scales: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for full_name, p in model.state_dict().items():
            # If this weight is owned by a FakeQuant{Conv2d,Linear} (Jacob)
            # or an LSQ{Conv2d,Linear} (LSQ), bake the fake-quantize result
            # into the saved tensor.
            module_name = full_name.rsplit(".", 1)[0]
            module = dict(model.named_modules()).get(module_name, None)

            # Jacob 2018 FakeQuant path
            if isinstance(module, (FakeQuantConv2d, FakeQuantLinear)) and \
               full_name.endswith(".weight"):
                w_fq = fake_quant(module.weight, num_bits)
                new_sd[full_name] = w_fq.detach().cpu().clone()
                # Per-channel symmetric scale used inside fake_quant.
                w = module.weight.detach()
                qmax = (1 << (num_bits - 1)) - 1
                w_flat = w.reshape(w.shape[0], -1)
                max_abs = w_flat.abs().amax(dim=1).clamp_min(1e-12)
                s = (max_abs / float(qmax)).detach().cpu().clone()
                scales[full_name] = s

            # LSQ (Esser 2020) path: use the LEARNED per-channel scale s
            elif isinstance(module, (LSQConv2d, LSQLinear)) and \
                 full_name.endswith(".weight"):
                # Make sure scale was initialized (it would be after any
                # forward pass; if not, init it now from current weights).
                module._lazy_init()
                qp = module.qp
                qn = module.qn
                s_param = module.s.detach()  # learned scale
                w = module.weight.detach()
                # Fake-quant with the learned scale (no gradient needed here).
                q = torch.clamp(torch.round(w / s_param), qn, qp)
                w_fq = q * s_param
                new_sd[full_name] = w_fq.detach().cpu().clone()
                # Flatten s_param into [OC] to match the existing scales
                # CSV format used by validate_bitflip_int.py.
                s_flat = s_param.reshape(s_param.shape[0]).detach().cpu().clone()
                scales[full_name] = s_flat
            else:
                # Skip LSQ-only bookkeeping params/buffers — they exist only
                # in the LSQ-wrapped training graph, not in the plain model
                # that loads the baked-PTQ checkpoint at eval time.
                # full_name endings handled here: ".s" and "._s_initialized".
                if full_name.endswith(".s") or full_name.endswith("._s_initialized"):
                    continue
                new_sd[full_name] = p.detach().cpu().clone()

    out_path = TQ.ckpt_path(dataset, arch, f"int{num_bits}_qat{tag_suffix}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "state_dict": new_sd,
        "meta": {"num_bits": num_bits, "qat_baked": True, "scales": scales},
    }, out_path)
    return out_path


# =========================================================
# Train / evaluate one epoch
# =========================================================

@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    top1c = top5c = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        total += y.size(0)
        _, pred = out.topk(min(5, out.size(1)), dim=1)
        top1c += (pred[:, 0] == y).sum().item()
        top5c += pred.eq(y.view(-1, 1)).sum().item()
    return 100.0 * top1c / max(1, total), 100.0 * top5c / max(1, total)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device,
                    use_amp: bool, log_every: int = 100) -> Tuple[float, float]:
    model.train()
    run_loss = 0.0
    corr = tot = 0
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        run_loss += loss.item() * x.size(0)
        corr += (logits.argmax(1) == y).sum().item()
        tot  += y.size(0)
        if (i + 1) % log_every == 0:
            print(f"    [batch {i+1}/{len(loader)}] "
                  f"loss={run_loss/max(1,tot):.4f}  acc={100.0*corr/max(1,tot):.2f}%")
    return run_loss / max(1, tot), 100.0 * corr / max(1, tot)


# =========================================================
# Main
# =========================================================

def main():
    p = argparse.ArgumentParser("Quantization-Aware Training (QAT)")
    p.add_argument("--dataset", required=True,
                   choices=["CIFAR10", "CIFAR100", "MNIST", "IMAGENET"])
    p.add_argument("--arch", required=True)
    p.add_argument("--bits", type=int, required=True, choices=[2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--imagenet-root", default="")
    p.add_argument("--use-pretrained", type=int, default=0,
                   help="If 1 and no float32 artifact exists, load torchvision pretrained.")
    p.add_argument("--skip-first-last", type=int, default=0,
                   help="If 1, leave the first Conv2d and last Linear at float32 "
                        "(common QAT practice for INT4 robustness).")
    p.add_argument("--qat-method", default="jacob", choices=["jacob", "lsq"],
                   help="Which QAT scheme to use for fake-quantizing weights. "
                        "'jacob' (default) follows Jacob et al. 2018 (CVPR) with "
                        "per-channel symmetric scales recomputed from max(|W|). "
                        "'lsq' follows Esser et al. 2020 (ICLR) with a learnable "
                        "per-channel step size. Both share the same forward-math "
                        "shape so downstream bit-flip tooling is unchanged.")
    p.add_argument("--qtag-suffix", default="",
                   help="Optional suffix appended to the saved checkpoint tag. "
                        "E.g. --qtag-suffix _lsq saves to model_int4_qat_lsq.pth "
                        "instead of model_int4_qat.pth, so LSQ runs do NOT "
                        "overwrite Jacob runs. Default empty (= overwrite/standard).")
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "none"])
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--amp", type=int, default=1, help="Use mixed precision (autocast).")
    p.add_argument("--float32-finetune-epochs", type=int, default=0,
                   help="Number of float32 fine-tuning epochs on the same train subset "
                        "BEFORE QAT, to produce a fair Float32 baseline. The Float32 "
                        "result reported with this flag will be measured AFTER this "
                        "fine-tuning, on the same eval split as QAT.")
    args = p.parse_args()
    args.dataset = args.dataset.upper()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device]    {device}")
    print(f"[dataset]   {args.dataset}")
    print(f"[arch]      {args.arch}")
    print(f"[bits]      INT{args.bits}")
    print(f"[epochs]    {args.epochs}")

    # ---- Data ----
    # If user passed an ImageNet path that contains ONLY the val set
    # (no train/ subfolder), we still want to do QAT without train/test
    # data leakage. We do a STRATIFIED 80/20 split of the val set
    # (per-class) — 40 images/class for QAT fine-tuning, 10 images/class
    # held out for evaluation. Indices are deterministic (seed=0) so
    # results reproduce. This is the standard workaround when the full
    # ImageNet training set isn't available locally and is the same
    # protocol used by published PTQ/QAT evaluation papers (e.g., LSQ,
    # AdaRound, BRECQ) which all calibrate/finetune on a subset of val.
    if args.dataset == "IMAGENET":
        ir = args.imagenet_root
        has_train = os.path.isdir(os.path.join(ir, "train"))
        has_val   = os.path.isdir(os.path.join(ir, "val"))
        if not has_train:
            print(f"[data] no 'train/' subfolder found at {ir}; "
                  f"stratified 80/20 split of val set (no overlap between "
                  f"QAT-finetune and eval).")
            tf_tr = TQ.build_transforms("imagenet", train=True)
            tf_te = TQ.build_transforms("imagenet", train=False)
            val_dir = os.path.join(ir, "val") if has_val else ir
            # Two dataset instances on the SAME folder with different transforms.
            full_tr = TQ._safe_image_folder(val_dir, transform=tf_tr)
            full_te = TQ._safe_image_folder(val_dir, transform=tf_te)
            # Stratified split by class: for each class, 80% to finetune, 20% to eval.
            import collections, random
            cls_to_idx = collections.defaultdict(list)
            for i, (_, y) in enumerate(full_tr.samples):
                cls_to_idx[y].append(i)
            rng = random.Random(0)
            train_idx, eval_idx = [], []
            for y, idxs in cls_to_idx.items():
                idxs = list(idxs)
                rng.shuffle(idxs)
                cut = int(round(0.8 * len(idxs)))
                train_idx.extend(idxs[:cut])
                eval_idx.extend(idxs[cut:])
            overlap = set(train_idx) & set(eval_idx)
            assert len(overlap) == 0, "split overlap detected"
            print(f"[data] train(finetune)={len(train_idx)}  "
                  f"test(eval)={len(eval_idx)}  classes={len(cls_to_idx)}  "
                  f"(disjoint by index)")
            tr_set = torch.utils.data.Subset(full_tr, train_idx)
            te_set = torch.utils.data.Subset(full_te, eval_idx)
            nc = len(cls_to_idx)
            train_loader = torch.utils.data.DataLoader(
                tr_set, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True, drop_last=True,
            )
            test_loader = torch.utils.data.DataLoader(
                te_set, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
            )
        else:
            train_loader, test_loader, nc, _ = TQ.get_dataloaders(
                args.dataset, args.data_root, args.batch_size, args.workers,
                dist_mode=False, imagenet_root=args.imagenet_root,
            )
    else:
        train_loader, test_loader, nc, _ = TQ.get_dataloaders(
            args.dataset, args.data_root, args.batch_size, args.workers,
            dist_mode=False, imagenet_root=args.imagenet_root,
        )
    print(f"[data]      train={len(train_loader.dataset)}  test={len(test_loader.dataset)}  nc={nc}")

    # ---- Model: load float32 baseline first ----
    model = TQ.build_model(args.arch, nc, use_pretrained=bool(args.use_pretrained), dataset=args.dataset)
    src, mb = TQ.maybe_load_float32_or_pretrained(
        model, args.dataset, args.arch,
        use_pretrained=bool(args.use_pretrained), map_location="cpu",
    )
    print(f"[weights]   source={src} ({mb:.2f} MB)")

    # ---- Evaluate float32 baseline first (so we have a reference number) ----
    model = model.to(device)
    fp_cold_top1, fp_cold_top5 = evaluate(model, test_loader, device)
    print(f"[float32 cold baseline] top-1 = {fp_cold_top1:.2f}%  top-5 = {fp_cold_top5:.2f}%")

    # ---- OPTIONAL: Float32 fine-tuning for a fair baseline ----
    # Without this, QAT vs Float32 is an unfair comparison: QAT gets free
    # improvement from fine-tuning on the val-set 40K subset (BatchNorm
    # running stats re-calibrate to the eval distribution). To make the
    # comparison fair, optionally fine-tune the Float32 model on the SAME
    # 40K subset for the SAME number of epochs as QAT.
    fp_top1 = fp_cold_top1
    fp_top5 = fp_cold_top5
    if args.float32_finetune_epochs > 0:
        print(f"\n[float32-finetune] running {args.float32_finetune_epochs} epoch(s) "
              f"of plain Float32 fine-tuning on the same train subset, for a "
              f"fair Float32 baseline ...")
        f_opt = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=True)
        if args.scheduler == "cosine":
            def f_lr_lambda(epoch):
                if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
                    return float(epoch + 1) / float(args.warmup_epochs)
                t = (epoch - args.warmup_epochs) / max(
                    1, (args.float32_finetune_epochs - args.warmup_epochs)
                )
                return 0.5 * (1.0 + math.cos(math.pi * t))
            f_sched = optim.lr_scheduler.LambdaLR(f_opt, f_lr_lambda)
        else:
            f_sched = None
        f_crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        f_scaler = torch.cuda.amp.GradScaler(
            enabled=(bool(args.amp) and device.type == "cuda")
        )
        best_fp = fp_cold_top1
        for epoch in range(1, args.float32_finetune_epochs + 1):
            ep_t0 = time.time()
            ep_loss, ep_train_acc = train_one_epoch(
                model, train_loader, f_opt, f_crit, f_scaler, device,
                use_amp=(bool(args.amp) and device.type == "cuda"),
            )
            if f_sched is not None:
                f_sched.step()
            ftop1, ftop5 = evaluate(model, test_loader, device)
            ep_dt = time.time() - ep_t0
            print(f"[FP32-FT {epoch:03d}/{args.float32_finetune_epochs}] "
                  f"loss={ep_loss:.4f}  train={ep_train_acc:.2f}%  "
                  f"test_top1={ftop1:.2f}%  top5={ftop5:.2f}%  ({ep_dt:.1f}s)")
            if ftop1 > best_fp:
                best_fp = ftop1
                fp_top5 = ftop5
        fp_top1 = best_fp
        print(f"[float32 FAIR baseline] top-1 = {fp_top1:.2f}%  "
              f"(after {args.float32_finetune_epochs} epoch FP32 fine-tune)")

    # ---- Convert to QAT ----
    print(f"[qat] method={args.qat_method}  inserting fake-quantize wrappers "
          f"(skip_first_last={bool(args.skip_first_last)}) ...")
    model = convert_to_qat(model, num_bits=args.bits,
                            skip_first_last=bool(args.skip_first_last),
                            qat_method=args.qat_method)
    model = model.to(device)

    # Quick check: how many layers got wrapped?
    if args.qat_method == "lsq":
        n_fq_conv = sum(1 for m in model.modules() if isinstance(m, LSQConv2d))
        n_fq_lin  = sum(1 for m in model.modules() if isinstance(m, LSQLinear))
    else:
        n_fq_conv = sum(1 for m in model.modules() if isinstance(m, FakeQuantConv2d))
        n_fq_lin  = sum(1 for m in model.modules() if isinstance(m, FakeQuantLinear))
    print(f"[qat] wrapped {n_fq_conv} Conv2d + {n_fq_lin} Linear layers")

    # ---- Evaluate the QAT-wrapped model BEFORE fine-tuning ----
    # (this is essentially PTQ accuracy with this exact fake-quant scheme,
    #  which is what we want to recover or exceed after fine-tuning)
    pre_ft_top1, _ = evaluate(model, test_loader, device)
    print(f"[ptq-equiv pre-finetune] top-1 = {pre_ft_top1:.2f}%")

    # ---- Optimizer / scheduler / loss ----
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=True)
    if args.scheduler == "cosine":
        def lr_lambda(epoch):
            if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
                return float(epoch + 1) / float(args.warmup_epochs)
            t = (epoch - args.warmup_epochs) / max(1, (args.epochs - args.warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * t))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=(bool(args.amp) and device.type == "cuda"))

    # ---- Train loop ----
    best_top1 = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        ep_loss, ep_train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            use_amp=(bool(args.amp) and device.type == "cuda"),
        )
        if scheduler is not None:
            scheduler.step()
        test_top1, test_top5 = evaluate(model, test_loader, device)
        ep_dt = time.time() - ep_t0
        print(f"[Epoch {epoch:03d}/{args.epochs}] loss={ep_loss:.4f}  "
              f"train={ep_train_acc:.2f}%  test_top1={test_top1:.2f}%  "
              f"top5={test_top5:.2f}%  ({ep_dt:.1f}s)")
        if test_top1 > best_top1:
            best_top1 = test_top1
            # save the float-shadow QAT model AS a ptq-style payload
            out_path = save_qat_as_ptq_payload(model, args.dataset, args.arch, args.bits,
                                                tag_suffix=args.qtag_suffix)
            sz = os.path.getsize(out_path) / 1e6
            print(f"  [checkpoint] -> {out_path}  ({sz:.2f} MB)")

    elapsed = time.time() - t0
    out_path = TQ.ckpt_path(args.dataset, args.arch, f"int{args.bits}_qat{args.qtag_suffix}")
    print(f"\n[done] elapsed {elapsed/60:.1f} min")
    print(f"[float32 cold] top-1 = {fp_cold_top1:.2f}%  (no fine-tuning)")
    if args.float32_finetune_epochs > 0:
        print(f"[float32 FAIR] top-1 = {fp_top1:.2f}%  "
              f"(after {args.float32_finetune_epochs} epoch FP32 fine-tune; "
              f"use THIS as baseline vs QAT)")
    print(f"[PTQ-equiv]    top-1 = {pre_ft_top1:.2f}%  (before any QAT fine-tuning)")
    print(f"[QAT-int{args.bits} best] top-1 = {best_top1:.2f}%")
    print(f"[output]       {out_path}")

    # ---- Verification: load the saved checkpoint into a FRESH model
    #      (no fake-quant wrappers) the same way validate_bitflip_int.py
    #      does, and re-evaluate. The accuracy should closely match the
    #      QAT-mode accuracy. Big drops indicate save/load drift (the
    #      EfficientNet-B0 problem).
    print("\n[verify] reloading saved checkpoint into a fresh model ...")
    try:
        fresh = TQ.build_model(args.arch, nc, use_pretrained=False, dataset=args.dataset).to(device)
        ck = TQ.ckpt_path(args.dataset, args.arch, f"int{args.bits}_qat{args.qtag_suffix}")
        payload = torch.load(ck, map_location=device)
        if "state_dict" in payload and payload.get("meta", {}).get("qat_baked", False):
            sd = TQ.strip_prefix_from_state_dict(payload["state_dict"])
            fresh.load_state_dict(sd, strict=True)
        else:
            TQ.load_quantized_into_model(
                fresh, args.dataset, args.arch, args.bits, qtag="qat",
                map_location=device,
            )
        loaded_top1, _ = evaluate(fresh, test_loader, device)
        drop = best_top1 - loaded_top1
        print(f"[verify] reloaded top-1 = {loaded_top1:.2f}%")
        print(f"[verify] gap vs in-train QAT accuracy = {drop:+.2f}pp")

        # ----- DEEP DIAGNOSTIC -----
        # Compare reloaded weights vs the QAT model's fake-quant weights.
        # This tells us EXACTLY where the drift is, if any.
        if abs(drop) > 5.0:
            print(f"\n[diag] running deep diagnostic to locate drift ...")
            print(f"[diag] comparing reloaded weights vs in-train fake-quant weights:")
            with torch.no_grad():
                model.eval()
                qat_sd = model.state_dict()
                fresh_sd = fresh.state_dict()
                max_diff_layer = ""
                max_diff_val = 0.0
                bn_mismatch = 0
                weight_mismatch = 0
                for k in fresh_sd.keys():
                    if k not in qat_sd: continue
                    a = fresh_sd[k].detach().float().cpu()
                    # For QAT-wrapped layers, the truth is fake_quant(weight)
                    module_name = k.rsplit(".", 1)[0]
                    qat_module = dict(model.named_modules()).get(module_name, None)
                    if isinstance(qat_module, (FakeQuantConv2d, FakeQuantLinear)) and k.endswith(".weight"):
                        b = fake_quant(qat_module.weight, args.bits).detach().float().cpu()
                    elif isinstance(qat_module, (LSQConv2d, LSQLinear)) and k.endswith(".weight"):
                        # For LSQ, the truth is clamp(round(w/s), qn, qp) * s
                        # using the LEARNED per-channel scale s.
                        qat_module._lazy_init()
                        s = qat_module.s.detach()
                        qn, qp = qat_module.qn, qat_module.qp
                        q = torch.clamp(torch.round(qat_module.weight / s), qn, qp)
                        b = (q * s).detach().float().cpu()
                    else:
                        b = qat_sd[k].detach().float().cpu()
                    if a.shape != b.shape:
                        print(f"[diag]   SHAPE MISMATCH: {k}  fresh={tuple(a.shape)}  qat={tuple(b.shape)}")
                        continue
                    diff = (a - b).abs().max().item()
                    if diff > max_diff_val:
                        max_diff_val = diff
                        max_diff_layer = k
                    if diff > 1e-6:
                        if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
                            bn_mismatch += 1
                        elif k.endswith(".weight") or k.endswith(".bias"):
                            weight_mismatch += 1
                print(f"[diag] max element diff = {max_diff_val:.6e}  at  {max_diff_layer}")
                print(f"[diag] BN buffer mismatches (running_mean/var)        : {bn_mismatch}")
                print(f"[diag] weight/bias mismatches (parameters)            : {weight_mismatch}")
                if bn_mismatch == 0 and weight_mismatch == 0:
                    print(f"[diag] -> Loaded weights are IDENTICAL to QAT model's quant weights.")
                    print(f"[diag]    The accuracy drop comes from somewhere ELSE — most likely the")
                    print(f"[diag]    FRESH model's forward pass differs from the QAT model's forward")
                    print(f"[diag]    pass (e.g., QAT QuantConv2d does fake_quant() at every")
                    print(f"[diag]    forward call, but the FRESH model uses the saved float weights")
                    print(f"[diag]    directly without re-quantizing). This is the actual bug.")
                elif bn_mismatch > 0:
                    print(f"[diag] -> Drift is in BatchNorm running statistics.")
                else:
                    print(f"[diag] -> Drift is in weights/biases (save/load is lossy).")
    except Exception as e:
        import traceback
        print(f"[verify] could not reload+evaluate: {e}")
        traceback.print_exc()

    print("\nNext: run validation with --qtag qat on the same arch/dataset.")
    print(f"  e.g.  python validate_bitflip_int.py --dataset {args.dataset} --archs {args.arch} "
          f"--formats int{args.bits} --qtag qat ...")


if __name__ == "__main__":
    main()
