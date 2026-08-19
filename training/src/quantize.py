"""Group-wise symmetric quantization primitives.

Used by quantization evaluation. Both functions are pure tensor math: no
checkpoints, no datasets, no model-specific naming.

The exporter does NOT call these. It needs the packed nibble layout as well as
the dequantized values, so it carries its own implementation of the same
arithmetic; the two must agree or the golden stops matching.

  quantize_groupwise(w, bits, group, fp16_scales) -> dequantized tensor
      Symmetric per-group scales along the last dim. With fp16_scales the
      scale is rounded to fp16 BEFORE dequantizing, so the result matches
      what a runtime reading fp16 scales computes.

  quantize_model(model, ...) -> number of parameters quantized
      Applies the above in place to every >=2D parameter. Norms and biases
      are left fp32.
"""

import torch


def _check(bits, group):
    # bits=1 leaves no magnitude range and produces NaN; group<=0 divides the
    # row into nothing. Both are silent numerical failures rather than errors.
    if not 2 <= bits <= 8:
        raise ValueError(f"bits must be 2..8, got {bits}")
    if group <= 0:
        raise ValueError(f"group must be positive, got {group}")


def quantize_groupwise(w, bits=4, group=64, fp16_scales=False):
    """Symmetric group-wise quant along the last dim; returns dequantized fp32.

    Simulates storage in `bits` with one fp scale per `group` consecutive weights
    of each row -- the standard edge format. qmax = 2^(bits-1)-1.
    """
    _check(bits, group)
    orig_shape = w.shape
    x = w.reshape(-1, orig_shape[-1]).float()
    out, cols = x.shape
    pad = (group - cols % group) % group
    if pad:
        x = torch.cat([x, torch.zeros(out, pad, device=x.device)], dim=1)
    x = x.reshape(out, -1, group)  # (out, n_groups, group)
    qmax = 2 ** (bits - 1) - 1  # 7 for 4-bit
    scale = x.abs().amax(dim=2, keepdim=True) / qmax
    scale = scale.clamp_min(1e-8)
    if fp16_scales:
        # Match the flash artifact: round stored group scales to IEEE fp16
        # before quantizing and dequantizing the weights.
        scale = scale.half().float()
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    dq = (q * scale).reshape(out, -1)[:, :cols]
    return dq.reshape(orig_shape).to(w.dtype)


def quantize_model(model, bits=4, group=64, quant_table=True, fp16_scales=False):
    _check(bits, group)
    n_q = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim < 2:
                continue  # norms, biases -> leave fp32
            if "table" in name and not quant_table:
                continue
            p.copy_(quantize_groupwise(p.data, bits, group, fp16_scales))
            n_q += p.numel()
    return n_q
