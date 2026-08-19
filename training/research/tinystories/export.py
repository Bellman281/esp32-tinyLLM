"""Export a trained PLE model to a flat binary the C inference can mmap, plus a
golden logits reference so the C port can be proven correct before it touches
hardware.

Format is deliberately dead-simple (the C reader is ~30 lines):
  [header: magic "PLE\0", version, header_bytes, flags, input/output vocab,
   then int32 config fields]
  header_bytes lets a later version append fields without breaking readers.
  flags bit 0 is TIED_HEAD: the output head IS the token embedding and no
  separate head tensor follows. An untied model sets output_vocab
  independently and appends the head as the final tensor.
  then, in a fixed order the C code hard-codes, each tensor as either
    QUANT: int4 codes packed 2-per-byte (group-wise, group=G along last dim)
           followed by fp16 scales, one per group
    FP32 : raw fp32 (norms only - tiny)

quant_pack below reimplements the symmetric group-wise int4 of src/quantize.py
because it also needs the packed byte layout, so the
golden logits are the *4-bit* model's logits: C-vs-PyTorch then isolates port
correctness from quantization error, which was measured separately.
"""

import argparse
import hashlib
import os
import shutil
import struct

import numpy as np
import torch

from tokenizers import Tokenizer

from model import Config, TinyLM

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = str(ROOT / "runs")
OUT = str(ROOT / "artifacts" / "tinystories")
MAGIC = 0x00454C50  # "PLE\0"
FORMAT_VERSION = 1
HEADER_BYTES = 56
FLAG_TIED_HEAD = 1 << 0
GROUP = 128  # uniform; fp16 scales + ragged packing keep the 28.9M model < 16MB


def quant_pack(w, group=GROUP):
    """Group-wise symmetric int4, ragged (no padding) with fp16 scales.

    Returns (packed_uint8[rows, ceil(cols/2)], scales_fp16[rows, n_groups], dq).
    The last group of each row may be shorter than `group`; the C reader derives
    n_groups=ceil(cols/group) and row_bytes=ceil(cols/2). Scales are rounded to
    fp16 *before* computing dq so the golden matches the C exactly. dq is the
    dequantized fp32 tensor the C reconstructs.
    """
    w = w.float()
    out_shape = w.shape
    x = w.reshape(-1, out_shape[-1])
    rows, cols = x.shape
    n_groups = (cols + group - 1) // group
    q = torch.zeros(rows, cols)
    dq = torch.zeros(rows, cols)
    scales = torch.zeros(rows, n_groups)
    for gi in range(n_groups):
        a, b = gi * group, min((gi + 1) * group, cols)
        seg = x[:, a:b]
        sc = (seg.abs().amax(dim=1, keepdim=True) / 7).clamp_min(1e-8)
        sc = sc.half().float()  # fp16-round the scale
        scales[:, gi] = sc.squeeze(1)
        qi = torch.clamp(torch.round(seg / sc), -7, 7)
        q[:, a:b] = qi
        dq[:, a:b] = qi * sc
    dq = dq.reshape(out_shape)

    codes = (q.to(torch.int16) + 8).to(torch.uint8).numpy()  # rows x cols, 0..15
    row_bytes = (cols + 1) // 2
    packed = np.zeros((rows, row_bytes), dtype=np.uint8)
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    packed[:, : lo.shape[1]] = lo
    packed[:, : hi.shape[1]] |= (hi << 4)
    scales16 = scales.numpy().astype(np.float16)
    return packed.reshape(-1), scales16.reshape(-1), dq


def verify_tokenizer(ck, tokenizer_path, allow_unverified):
    """Refuse a tokenizer that did not train this checkpoint.

    The size check alone is not enough: a smaller wrong tokenizer passes it and
    silently produces a model with the wrong output vocabulary.

    Checkpoints written before the hash was recorded cannot be checked. This
    produces a binary intended for distribution, so an unverifiable tokenizer is
    an error by default; --allow-unverified-tokenizer makes that exception
    explicit and visible in the command that made the artifact.
    """
    want = ck.get("tokenizer_sha256")
    got = hashlib.sha256(open(tokenizer_path, "rb").read()).hexdigest()
    if want is None:
        if not allow_unverified:
            raise SystemExit(
                f"checkpoint records no tokenizer hash, so {os.path.basename(tokenizer_path)} "
                f"cannot be verified as the one it was trained with. Re-train to "
                f"record it, or pass --allow-unverified-tokenizer to accept the risk.")
        print(f"WARNING: tokenizer unverified ({os.path.basename(tokenizer_path)}); "
              f"checkpoint predates tokenizer hashing")
        return
    if want != got:
        raise SystemExit(
            f"tokenizer mismatch: {tokenizer_path} is sha256 {got[:16]}... but "
            f"this checkpoint was trained with {want[:16]}...")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tag", nargs="?", default="ple-cleandeploy-s0",
                    help="checkpoint tag under runs/, without .pt")
    ap.add_argument("--tokenizer", required=True,
                    help="tokenizer this checkpoint was trained with; its size "
                         "becomes output_vocab")
    ap.add_argument("--allow-unverified-tokenizer", action="store_true",
                    help="export from a checkpoint that records no tokenizer hash")
    args = ap.parse_args()
    tag = args.tag
    os.makedirs(OUT, exist_ok=True)
    ck = torch.load(os.path.join(RUNS, f"{tag}.pt"), map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    # This exporter writes the tied-head PLE layout. Another arm has a
    # different tensor set, and the plan below would bind garbage.
    if cfg.arm != "ple":
        raise SystemExit(f"{tag} is arm={cfg.arm}; this exporter writes PLE models")
    verify_tokenizer(ck, args.tokenizer, args.allow_unverified_tokenizer)
    model = TinyLM(cfg)
    model.load_state_dict(ck["state"])
    model.eval()

    # Ordered list: (name, tensor, is_quant). C reads in exactly this order.
    plan = []
    sd = model.state_dict()

    def add(name, quant):
        plan.append((name, sd[name], quant))

    add("tok_emb.weight", True)           # tied: input embed + output head
    add("ple_model_proj.weight", True)
    add("ple_proj_norm.weight", False)
    add("ple_table.weight", True)         # the 25M-param table
    for i in range(cfg.n_layers):
        p = f"blocks.{i}."
        add(p + "attn_norm.weight", False)
        add(p + "attn.qkv.weight", True)
        add(p + "attn.proj.weight", True)
        add(p + "ffn_norm.weight", False)
        add(p + "ffn.gate.weight", True)
        add(p + "ffn.up.weight", True)
        add(p + "ffn.down.weight", True)
        add(p + "ple_gate.weight", True)
        add(p + "ple_proj.weight", True)
        add(p + "ple_norm.weight", False)
    add("out_norm.weight", False)

    # Reconstruct a dequantized state dict to drive the golden forward.
    dq_sd = {k: v.clone() for k, v in sd.items()}
    blobs = []
    for name, t, quant in plan:
        if quant:
            packed, scales, dq = quant_pack(t)
            dq_sd[name] = dq
            blobs.append(("Q", name, t.shape, packed, scales))
        else:
            blobs.append(("F", name, t.shape, t.contiguous().numpy().astype(np.float32), None))

    # The embedding and PLE table store cfg.vocab_size padded rows, but only the
    # tokenizer's entries can ever be produced or decoded. output_vocab is that
    # count, so the header describes what the runtime actually scores instead of
    # leaving the firmware to cap it from a generated header.
    out_vocab = Tokenizer.from_file(args.tokenizer).get_vocab_size()
    # The header tells the runtime how many logits to score, and the tied head is
    # the first out_vocab rows of the embedding. A tokenizer larger than the
    # checkpoint's embedding would make the runtime read past that tensor, so
    # this is a hard error rather than a warning: it cannot be caught later.
    if not 0 < out_vocab <= cfg.vocab_size:
        raise SystemExit(
            f"tokenizer/checkpoint mismatch: {args.tokenizer} has {out_vocab} "
            f"entries but the checkpoint embedding has {cfg.vocab_size} rows. "
            f"Pass --tokenizer for the one this run was trained with.")
    print(f"input_vocab={cfg.vocab_size}  output_vocab={out_vocab} "
          f"(from {os.path.basename(args.tokenizer)})")

    # Write binary. artifacts/ is gitignored, so it does not exist in a fresh
    # clone.
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "model.bin")
    with open(path, "wb") as f:
        flags = FLAG_TIED_HEAD  # this model ties the head to the embedding
        f.write(struct.pack("<IIII", MAGIC, FORMAT_VERSION, HEADER_BYTES, flags))
        f.write(struct.pack("<II", cfg.vocab_size, out_vocab))
        for v in [cfg.d_model, cfg.n_layers, cfg.n_heads,
                  cfg.ffn_hidden, cfg.ple_dim, cfg.seq_len, GROUP]:
            f.write(struct.pack("<i", v))
        f.write(struct.pack("<f", cfg.rope_theta))
        for entry in blobs:
            kind = entry[0]
            if kind == "Q":
                _, _, _, packed, scales = entry
                f.write(struct.pack("<i", GROUP))   # per-tensor group
                f.write(packed.tobytes())           # ragged int4 codes
                f.write(scales.tobytes())           # fp16 scales
            else:
                _, _, _, arr, _ = entry
                f.write(arr.tobytes())
    size = os.path.getsize(path)
    print(f"wrote {path}  ({size/1e6:.2f} MB)  {len(plan)} tensors")

    # Copy the tokenizer beside the weights so the artifact directory is a
    # complete, self-contained model: everything the firmware build and the
    # device need, and nothing that has to be looked up elsewhere.
    # Re-exporting from the artifact directory passes the destination back in as
    # the source. Compare the files rather than the paths: a symlink or a hard
    # link has a different path and is still the same file, and copyfile raises
    # on that rather than doing nothing.
    tokenizer_out = os.path.join(OUT, "tokenizer.json")
    same = os.path.exists(tokenizer_out) and os.path.samefile(args.tokenizer, tokenizer_out)
    if not same:
        shutil.copyfile(args.tokenizer, tokenizer_out)

    # Keep the tied output head == dequantized input embedding. state_dict lists
    # both keys for tied weights; without this the head silently stays fp32 and
    # the golden no longer matches the (fully-quantized) C port.
    if "head.weight" in dq_sd:
        dq_sd["head.weight"] = dq_sd["tok_emb.weight"]

    # Golden: load dequantized weights, forward a fixed prompt, dump last-pos
    # logits, truncated to out_vocab. The runtime scores exactly that many, so a
    # longer reference would only compare untrained padding rows the device
    # never evaluates.
    gold = TinyLM(cfg)
    gold.load_state_dict(dq_sd)
    gold.eval()
    prompt = [1, 500, 1000, 200, 42, 777, 13, 99]  # arbitrary fixed token ids
    ids = torch.tensor([prompt])
    with torch.no_grad():
        logits, _ = gold(ids)
    last = logits[0, -1].numpy().astype(np.float32)[:out_vocab]
    np.savez(os.path.join(OUT, "golden.npz"),
             prompt=np.array(prompt, dtype=np.int32), logits=last)
    # Text form for the C host verifier: plen, prompt ids, then V logits.
    with open(os.path.join(OUT, "golden.txt"), "w") as gf:
        gf.write(f"{len(prompt)}\n")
        gf.write(" ".join(str(t) for t in prompt) + "\n")
        gf.write("\n".join(f"{v:.6f}" for v in last) + "\n")
    top5 = last.argsort()[-5:][::-1]
    print(f"golden: prompt={prompt}")
    print(f"golden: last-pos top5 token ids = {top5.tolist()}")
    print(f"golden: logit range [{last.min():.3f}, {last.max():.3f}]")


if __name__ == "__main__":
    main()
