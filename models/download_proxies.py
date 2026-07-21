#!/usr/bin/env python3
"""
download_proxies.py — fetch the proxy LLMs into the layout run_scoring.py expects.

The scorer resolves a proxy named `<proxy>` to a LOCAL directory:

    models/cache/local.<proxy>/          <-- a normal HF model folder (config.json, *.safetensors, tokenizer files)

(this is the `local.` convention hard-coded in score_functions/model.py:from_pretrained).
This script snapshot_downloads each repo straight into that folder, so nothing else
needs configuring.

USAGE
    pip install huggingface_hub
    huggingface-cli login            # or pass --token
    python models/download_proxies.py --token hf_xxx                 # all default proxies
    python models/download_proxies.py --token hf_xxx --only gemma3_4b,gemma3_12b
    python models/download_proxies.py --list

GATED MODELS: gemma-3-* and Llama-3.* require accepting the license on the HF model
page with the same account as your token. falcon / gpt-neo are open.

>>> CONFIRM THE REPO IDS BELOW <<<
These are the standard base (pretrained, non-instruct) repos and match how the other
proxies were scored. If your original sweep used a different revision, edit PROXY_REPOS
here — the folder name (local.<proxy>) is what matters and must stay exactly as keyed.
"""
import argparse, os, sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PKG, "models", "cache")

# proxy short-name  ->  HuggingFace repo id  (base / pretrained checkpoints, used for log-prob scoring)
PROXY_REPOS = {
    "gemma3_4b":    "google/gemma-3-4b-pt",
    "gemma3_12b":   "google/gemma-3-12b-pt",
    "gemma_2b":     "google/gemma-2b",
    "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
    "llama3.1_8b":  "meta-llama/Llama-3.1-8B",
    "llama3.2_3b":  "meta-llama/Llama-3.2-3B",
    "falcon_40b":   "tiiuae/falcon-40b",     # large: ~90GB / needs a big or multi GPU — optional
    # already scored in the project, kept for reference only (do NOT need to re-download):
    # "falcon_7b":  "tiiuae/falcon-7b",
    # mask model, ONLY needed if a perturbation cache is missing (all caches ship in data/):
    # "t5-3b":      "google-t5/t5-3b",
}

APPROX_SIZE = {"gemma3_4b": "~9GB", "gemma3_12b": "~24GB", "gemma_2b": "~5GB",
               "gpt-neo-2.7B": "~11GB", "llama3.1_8b": "~16GB", "llama3.2_3b": "~7GB",
               "falcon_40b": "~90GB"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=None, help="HF token (or run `huggingface-cli login`)")
    ap.add_argument("--only", default=None, help="comma list of proxy short-names")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    names = a.only.split(",") if a.only else list(PROXY_REPOS)
    if a.list:
        print(f"{'proxy':14s} {'repo':30s} {'size':8s} -> folder")
        for n in names:
            print(f"{n:14s} {PROXY_REPOS[n]:30s} {APPROX_SIZE.get(n,'?'):8s} -> models/cache/local.{n}/")
        return

    from huggingface_hub import snapshot_download, login
    if a.token:
        login(token=a.token)
    os.makedirs(CACHE, exist_ok=True)
    for i, n in enumerate(names, 1):
        repo = PROXY_REPOS[n]
        dest = os.path.join(CACHE, f"local.{n}")
        if os.path.isdir(dest) and any(f.endswith((".safetensors", ".bin")) for f in os.listdir(dest)):
            print(f"[{i}/{len(names)}] SKIP {n} (exists at {dest})"); continue
        print(f"[{i}/{len(names)}] {repo} ({APPROX_SIZE.get(n,'?')}) -> {dest}")
        snapshot_download(repo, local_dir=dest, token=a.token, resume_download=True)
        print(f"[{i}/{len(names)}] done {n}")
    print("\nAll requested proxies downloaded into models/cache/local.<proxy>/")


if __name__ == "__main__":
    sys.exit(main())
