#!/usr/bin/env python3
"""Interactive CLI for the ESP32-S3 TinyLM (exact chip code via ./gen_prompt).
Self-contained pure-Python tokenizer, no pip. Commands:
  :temp X   sampling temperature (0 = greedy like the chip; try 0.8)
  :n N      tokens to generate (default 100)
It writes simple stories; it cannot answer questions."""
import json, re, os, sys, subprocess
try:
    import readline  # arrow-key history + inline line editing
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
_tok = os.path.join(HERE, "bpe32768.json")
if not os.path.exists(_tok): _tok = os.path.join(HERE, "bpe4096.json")
_bpe = json.load(open(_tok))["model"]
_vocab = _bpe["vocab"]; _ranks = {tuple(m): i for i, m in enumerate(_bpe["merges"])}
def _b2u():
    bs = list(range(33,127)) + list(range(161,173)) + list(range(174,256)); cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}
_B = _b2u()
_PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")
def _merge(w):
    w = list(w)
    while len(w) > 1:
        best=None; br=1<<30
        for p in zip(w[:-1], w[1:]):
            r=_ranks.get(p)
            if r is not None and r<br: br=r; best=p
        if best is None: break
        a,b=best; nw=[]; i=0
        while i<len(w):
            if i<len(w)-1 and w[i]==a and w[i+1]==b: nw.append(a+b); i+=2
            else: nw.append(w[i]); i+=1
        w=nw
    return w
def encode(t):
    out=[]
    for m in _PAT.findall(t):
        for pc in _merge("".join(_B[c] for c in m.encode("utf-8"))):
            if pc in _vocab: out.append(_vocab[pc])
    return out
GEN = os.path.join(HERE, "gen_prompt")
MODEL = os.path.join(HERE, "..", "reference-c", "esp32-llm-lab", "model.bin")
def repl(n_tok=100, temp=0.8):
    print("=== ESP32-S3 TinyLM CLI ===")
    print("Type a prompt. Commands:  :temp 0.8  (randomness, 0=greedy)   :n 150  (length)   Ctrl-C quits")
    print(f"[temp={temp}  (varied). Use :temp 0 for exact greedy chip behavior.  n={n_tok}]\n")
    while True:
        try: p = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt): print(); break
        if not p: continue
        if p.startswith(":temp"): temp=float(p.split()[1]); print(f"[temp={temp}]\n"); continue
        if p.startswith(":n"): n_tok=int(p.split()[1]); print(f"[n={n_tok}]\n"); continue
        ids = encode(p)
        if not ids: print("(couldn't tokenize)\n"); continue
        subprocess.run([GEN, MODEL, str(n_tok), ",".join(map(str,ids)), str(temp)])
        print()

if __name__ == "__main__":
    import sys
    repl(int(sys.argv[1]) if len(sys.argv)>1 else 100, 0.8)
