"""Generate text from a trained run, so the loss numbers have something behind them.

The tokenizer is required and is checked against the hash the run recorded.
Legacy checkpoints predate that hash and can only be warned about. Decoding a
32,768-vocab run with the 4,096 tokenizer produces plausible nonsense rather
than an error, which is why the check exists.
"""

import argparse
import hashlib
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM

from .train import get_device

def verify_tokenizer(ck, tokenizer_path):
    """Refuse a tokenizer that did not train this checkpoint.

    The size check alone is not enough: a smaller wrong tokenizer passes it and
    silently produces a model with the wrong output vocabulary. Checkpoints
    written before this was recorded carry no hash, so they can only be warned
    about. Sampling is exploratory, so a warning is enough here; the exporter
    refuses instead, because it produces a distributable binary.
    """
    want = ck.get("tokenizer_sha256")
    got = hashlib.sha256(open(tokenizer_path, "rb").read()).hexdigest()
    if want is None:
        print(f"WARNING: checkpoint records no tokenizer hash; cannot verify "
              f"{os.path.basename(tokenizer_path)} is the right one")
        return
    if want != got:
        raise SystemExit(
            f"tokenizer mismatch: {tokenizer_path} is sha256 {got[:16]}... but "
            f"this checkpoint was trained with {want[:16]}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to a runs/*.pt checkpoint")
    ap.add_argument("--tokenizer", required=True,
                    help="tokenizer this run was trained with, e.g. "
                         "data/tinystories/vocab-32768/tokenizer.json")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed; generation is stochastic at temperature > 0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    ck = torch.load(args.run, map_location=device, weights_only=False)
    model = TinyLM(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state"])
    model.eval()

    verify_tokenizer(ck, args.tokenizer)
    tok = Tokenizer.from_file(args.tokenizer)
    ids = torch.tensor([tok.encode(args.prompt).ids], device=device)

    b = model.param_budget()
    print(f"{Path(args.run).name}  core={b['core']:,} table={b['table']:,}\n")
    for i in range(args.samples):
        out = model.generate(ids, args.tokens, temperature=args.temperature)
        print(f"--- sample {i + 1} ---")
        print(tok.decode(out[0].tolist()).strip(), "\n")


if __name__ == "__main__":
    main()
