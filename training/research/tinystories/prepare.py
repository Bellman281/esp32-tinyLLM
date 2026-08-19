"""Download a slice of TinyStories, train a small BPE, emit uint16 token bins.

The bins are uint16, so the vocabulary cannot exceed 65,536: train.py and
quantize_eval.py memmap them as uint16 and a wider dtype would be read as
garbage rather than fail.

The vocab is deliberately fixed across every ablation arm: cross-entropy is only
comparable between models that share a tokenizer.
"""

import argparse
import os
import sys

import numpy as np
import requests
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Each vocabulary variant keeps its tokenizer beside the bins it produced.
DATA = ROOT / "data" / "tinystories"
URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
RAW = DATA / "raw" / "tinystories-train-first-300MiB.txt"
DEFAULT_VOCAB = 32768   # what ships; every other size has to be asked for
VOCAB_SIZE = DEFAULT_VOCAB  # set from --vocab before anything reads it
# ~300MB of stories is ~75M tokens: enough to overtrain a 1M-param core well past
# its compute-optimal point, which is the regime we actually care about.
SLICE_BYTES = 300 * 1024 * 1024
VAL_FRACTION = 0.005


def variant_dir(vocab_size):
    return DATA / f"vocab-{vocab_size}"


def download():
    # Exactly the requested size, not approximately: a truncated slice trains a
    # different tokenizer and different bins, silently.
    if os.path.exists(RAW):
        have = os.path.getsize(RAW)
        if have == SLICE_BYTES:
            print(f"already have {RAW}")
            return
        print(f"{RAW} is {have:,} bytes, expected exactly {SLICE_BYTES:,}; "
              f"re-downloading")
    print(f"downloading first {SLICE_BYTES / 1e6:.0f}MB of TinyStories...")
    got = 0
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(RAW, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                # Trim the last chunk: writing it whole overshoots the slice, and
                # a slice of a different length is a different corpus.
                if got + len(chunk) > SLICE_BYTES:
                    chunk = chunk[:SLICE_BYTES - got]
                f.write(chunk)
                got += len(chunk)
                if got >= SLICE_BYTES:
                    break
                if got % (25 << 20) < (1 << 20):
                    print(f"  {got / 1e6:.0f}MB", flush=True)
    final = os.path.getsize(RAW)
    if final != SLICE_BYTES:
        raise SystemExit(f"downloaded {final:,} bytes, expected exactly "
                         f"{SLICE_BYTES:,}; the source may be shorter than the "
                         f"requested slice")
    print(f"done, {got / 1e6:.0f}MB")


def train_tokenizer(text):
    path = variant_dir(VOCAB_SIZE) / "tokenizer.json"
    if os.path.exists(path):
        print(f"already have {path}")
        return Tokenizer.from_file(str(path))
    print(f"training BPE vocab={VOCAB_SIZE}...")
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    # 40MB is plenty to fit a merge table of the sizes used here; feeding the
    # full slice just burns time for the same merges.
    tok.train_from_iterator([text[: 40 * 1024 * 1024]], trainer=trainer)
    tok.save(str(path))
    return tok


def main():
    global VOCAB_SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    args = ap.parse_args()
    if not 0 < args.vocab <= 65536:
        raise SystemExit(f"--vocab must be 1..65536; the bins are uint16 and "
                         f"every reader memmaps them as uint16")
    VOCAB_SIZE = args.vocab
    RAW.parent.mkdir(parents=True, exist_ok=True)
    out = variant_dir(VOCAB_SIZE)
    out.mkdir(parents=True, exist_ok=True)

    download()
    with open(RAW, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # Drop the trailing partial story left by the byte-slice.
    text = text[: text.rfind("<|endoftext|>") + len("<|endoftext|>")]

    tok = train_tokenizer(text)
    eot = tok.token_to_id("<|endoftext|>")
    print(f"eot id = {eot}")

    print("encoding...")
    docs = text.split("<|endoftext|>")
    ids = []
    for i in range(0, len(docs), 20000):
        batch = [d for d in docs[i : i + 20000] if d.strip()]
        for enc in tok.encode_batch(batch):
            ids.extend(enc.ids)
            ids.append(eot)
        print(f"  {i + len(batch)}/{len(docs)} docs, {len(ids) / 1e6:.1f}M tokens", flush=True)

    arr = np.array(ids, dtype=np.uint16)
    assert arr.max() < VOCAB_SIZE
    n_val = int(len(arr) * VAL_FRACTION)
    arr[:-n_val].tofile(out / "train.bin")
    arr[-n_val:].tofile(out / "val.bin")
    print(f"train {len(arr) - n_val:,} tokens / val {n_val:,} tokens")
    print(f"compression: {len(text) / len(arr):.2f} bytes/token")


if __name__ == "__main__":
    sys.exit(main())
