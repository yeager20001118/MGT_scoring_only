#!/usr/bin/env python3
"""
run_scoring.py — self-contained proxy scorer for the TriBet DetectGPT/NPR gap-fill.

WHAT THIS DOES
==============
For each (dataset, source, proxy) cell it produces the two score files the main
project expects:

    scoring_results/<dataset>.<source>.<proxy>.perturbation_30.json   (DetectGPT)
    scoring_results/<dataset>.<source>.<proxy>.npr.json               (NPR)

The expensive T5-3b perturbation step is ALREADY DONE — the perturbation caches
ship in data/raw_data/ (…​.t5-3b.perturbation_30.raw_data.json). This script only
runs the cheap per-proxy log-prob / log-rank passes over those caches, so you do
NOT need the t5-3b mask model and you do NOT regenerate perturbations.

INPUTS  (all relative to the package root, resolved automatically)
    data/raw_data/<dataset>.<source>.raw_data.json                     (source text)
    data/raw_data/<dataset>.<source>.t5-3b.perturbation_30.raw_data.json (cached perturbations)
    models/cache/local.<proxy>/                                        (each proxy, HF format)

OUTPUT
    scoring_results/<dataset>.<source>.<proxy>.{perturbation_30,npr}.json

USAGE
    # everything, on GPU 0
    python code/run_scoring.py --gpu 0

    # shard across GPUs by proxy (run each line on its own GPU / in its own shell)
    python code/run_scoring.py --gpu 0 --proxies gemma3_4b,gemma3_12b
    python code/run_scoring.py --gpu 1 --proxies llama3.1_8b,llama3.2_3b
    python code/run_scoring.py --gpu 2 --proxies gemma_2b,gpt-neo-2.7B

    # memory-constrained GPU: load proxies in fp16 (see note below)
    python code/run_scoring.py --gpu 0 --proxies gemma3_12b --fp16

    # just show the plan / what is already done
    python code/run_scoring.py --list

NUMERICAL NOTE
    The existing falcon_7b cells were scored in the framework default dtype (fp32).
    For apples-to-apples AUROC, this script also defaults to fp32. --fp16 halves
    memory (12B fp32 ≈ 48 GB → fp16 ≈ 24 GB) but can shift log-probs slightly.
"""
from __future__ import annotations
import argparse, os, sys, json, time
from types import SimpleNamespace

# ----------------------------------------------------------------------------- paths
PKG   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # package root
CODE  = os.path.join(PKG, "code")
LIB   = os.path.join(CODE, "score_functions")
DATA  = os.path.join(PKG, "data", "raw_data")
RES   = os.path.join(PKG, "scoring_results")
CACHE = os.path.join(PKG, "models", "cache")                          # local.<proxy> live here
os.makedirs(RES, exist_ok=True)
sys.path.insert(0, LIB)

# ----------------------------------------------------------------------------- config
# The 5 source LLMs that only have falcon_7b so far (need the rest of the sweep).
SOURCES = ["claude_sonnet_4.6", "claude_opus_4.5", "claude_haiku_4.5",
           "gpt4o_mini", "gemini_3.1_flash"]
DATASETS = ["olympic", "xsum"]           # olympic = self/target, xsum = cross reference

# Proxies still missing for the sources above (falcon_7b is already scored → excluded).
# falcon_40b is large (needs a big GPU); drop it from the default set if you like.
PROXIES = ["gemma3_4b", "gemma3_12b", "gemma_2b", "gpt-neo-2.7B",
           "llama3.1_8b", "llama3.2_3b", "falcon_40b"]

N_PERTURB = 30
MASK = "t5-3b"


def _valid(path, kind):
    if not os.path.exists(path):
        return False
    try:
        d = json.load(open(path))
        return "predictions" in d and len(d["predictions"].get("real", [])) >= 50
    except Exception:
        return False


def make_args(dataset, source, proxy, device):
    return SimpleNamespace(
        dataset_file=os.path.join(DATA, f"{dataset}.{source}"),
        output_file=os.path.join(RES, f"{dataset}.{source}.{proxy}"),
        scoring_model_name=proxy,
        mask_filling_model_name=MASK,
        n_perturbations=N_PERTURB,
        pct_words_masked=0.3, span_length=2, mask_top_p=0.96,
        seed=0, device=device, dataset="xsum",         # 'dataset' only tweaks OPT tokenizer
        cache_dir=CACHE, hf_token="skip",
    )


def score_perturbation(args):
    import detect_gpt
    detect_gpt.experiment(args)          # reuses cached perturbation file; writes .perturbation_30.json


def score_npr(args):
    import numpy as np, torch, tqdm
    from data_builder import load_data
    from metrics import get_roc_metrics, get_precision_recall_metrics
    from model import load_tokenizer, load_model
    from detect_llm import get_npr

    out = f"{args.output_file}.npr.json"
    prefix = f"{args.dataset_file}.{args.mask_filling_model_name}.perturbation_{args.n_perturbations}"
    if not os.path.exists(prefix + ".raw_data.json"):
        print(f"[err] missing perturbation cache: {prefix}.raw_data.json", file=sys.stderr); return
    tok = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir, args.hf_token)
    mdl = load_model(args.scoring_model_name, args.device, args.cache_dir, args.hf_token)
    mdl.eval(); mdl.to(args.device)
    data = load_data(prefix)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rows = []
    for i in tqdm.tqdm(range(len(data)), desc="npr"):
        d = data[i]
        rows.append({
            "original": d["original"],
            "original_crit": get_npr(args, mdl, tok, d["original"], d["perturbed_original"]),
            "sampled": d["sampled"],
            "sampled_crit": get_npr(args, mdl, tok, d["sampled"], d["perturbed_sampled"]),
        })
    pred = {"real": [r["original_crit"] for r in rows], "samples": [r["sampled_crit"] for r in rows]}
    fpr, tpr, roc = get_roc_metrics(pred["real"], pred["samples"])
    p, r, pr = get_precision_recall_metrics(pred["real"], pred["samples"])
    json.dump({"name": "npr_threshold", "info": {"n_samples": len(rows)}, "predictions": pred,
               "raw_results": rows, "metrics": {"roc_auc": roc, "fpr": fpr, "tpr": tpr},
               "pr_metrics": {"pr_auc": pr, "precision": p, "recall": r}, "loss": 1 - pr}, open(out, "w"))
    print(f"[done] {out}  (NPR AUROC={roc:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--proxies", type=str, default=None, help="comma list; default = all in PROXIES")
    ap.add_argument("--sources", type=str, default=None, help="comma list; default = all 5")
    ap.add_argument("--datasets", type=str, default=None, help="comma list; default = olympic,xsum")
    ap.add_argument("--skip-npr", action="store_true")
    ap.add_argument("--fp16", action="store_true", help="load proxies in float16 to save memory")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    a = ap.parse_args()

    proxies  = a.proxies.split(",")  if a.proxies  else PROXIES
    sources  = a.sources.split(",")  if a.sources  else SOURCES
    datasets = a.datasets.split(",") if a.datasets else DATASETS
    device = f"cuda:{a.gpu}"

    if a.fp16:
        import model as _m
        for p in proxies:
            if p not in _m.float16_models:
                _m.float16_models.append(p)

    cells = [(d, s, p) for p in proxies for s in sources for d in datasets]
    print(f"Plan: {len(cells)} cells  (proxies={proxies})  device={device}")
    if a.list:
        for d, s, p in cells:
            pj = os.path.join(RES, f"{d}.{s}.{p}.perturbation_30.json")
            nj = os.path.join(RES, f"{d}.{s}.{p}.npr.json")
            print(f"  {d}.{s}.{p}: detgpt={'OK' if _valid(pj,'p') else '--'}  npr={'OK' if _valid(nj,'n') else '--'}")
        return

    for k, (d, s, p) in enumerate(cells, 1):
        args = make_args(d, s, p, device)
        pj, nj = f"{args.output_file}.perturbation_{N_PERTURB}.json", f"{args.output_file}.npr.json"
        print(f"\n[{k}/{len(cells)}] {d}.{s}.{p}")
        t0 = time.time()
        try:
            if _valid(pj, "p"):
                print(f"  [skip] {os.path.basename(pj)}")
            else:
                score_perturbation(args)
            if not a.skip_npr:
                if _valid(nj, "n"):
                    print(f"  [skip] {os.path.basename(nj)}")
                else:
                    score_npr(args)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [FAIL] {d}.{s}.{p}: {e}")
        print(f"  ({(time.time()-t0)/60:.1f} min)")

    print("\nAll done. Score files are in scoring_results/  → follow sync_back/SYNC_BACK.md")


if __name__ == "__main__":
    sys.exit(main())
