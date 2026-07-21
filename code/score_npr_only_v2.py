"""score_npr_only_v2.py — NPR-only retry helper.

When detect_llm.py crashes during NPR after LRR has completed, we can avoid
re-computing LRR by running just NPR. This calls only `get_npr` over the
cached perturbation file.

Usage (same args as score_detect_llm_v2.py):
  python score_npr_only_v2.py \
    --dataset_file ~/MGT/exp/raw_data/olympic.X \
    --output_file ~/MGT/exp/evalue/results/olympic.X.proxy \
    --scoring_model_name proxy --device cuda
"""
from __future__ import annotations
import argparse, os, sys, json, time
import numpy as np
import torch
import tqdm

WORK = os.path.expanduser("~/MGT/exp/evalue")
ONLINE = os.path.expanduser("~/MGT/baselines/Online_Detection/scripts/score_functions")
DEFAULT_CACHE = os.path.expanduser("~/MGT/cache_for_scoring")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_file", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--scoring_model_name", required=True)
    ap.add_argument("--mask_filling_model_name", default="t5-3b")
    ap.add_argument("--n_perturbations", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dataset", type=str, default="xsum")
    ap.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE)
    ap.add_argument("--hf_token", type=str, default="skip")
    args = ap.parse_args()

    sys.path.insert(0, ONLINE)
    from data_builder import load_data
    from metrics import get_roc_metrics, get_precision_recall_metrics
    from model import load_tokenizer, load_model
    from detect_llm import get_npr

    npr_path = f"{args.output_file}.npr.json"
    if os.path.exists(npr_path):
        try:
            d = json.load(open(npr_path))
            if "predictions" in d and len(d["predictions"].get("real", [])) >= 50:
                print(f"[skip] {npr_path} already valid")
                return 0
        except Exception:
            pass

    perturb_prefix = (f"{args.dataset_file}.{args.mask_filling_model_name}"
                      f".perturbation_{args.n_perturbations}")
    perturb_file = perturb_prefix + ".raw_data.json"
    if not os.path.exists(perturb_file):
        print(f"[err] missing perturbation file: {perturb_file}", file=sys.stderr)
        return 2

    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir, args.hf_token)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir, args.hf_token)
    scoring_model.eval()
    scoring_model.to(args.device)

    data = load_data(perturb_prefix)
    n_samples = len(data)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    eval_results = []
    for idx in tqdm.tqdm(range(n_samples), desc="Computing npr criterion"):
        original_text = data[idx]["original"]
        sampled_text = data[idx]["sampled"]
        perturbed_original = data[idx]["perturbed_original"]
        perturbed_sampled = data[idx]["perturbed_sampled"]
        original_crit = get_npr(args, scoring_model, scoring_tokenizer, original_text, perturbed_original)
        sampled_crit = get_npr(args, scoring_model, scoring_tokenizer, sampled_text, perturbed_sampled)
        eval_results.append({"original": original_text, "original_crit": original_crit,
                             "sampled": sampled_text, "sampled_crit": sampled_crit})

    predictions = {'real': [x["original_crit"] for x in eval_results],
                   'samples': [x["sampled_crit"] for x in eval_results]}
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"NPR ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    out = {"name": "npr_threshold",
           "info": {"n_samples": n_samples},
           "predictions": predictions,
           "raw_results": eval_results,
           "metrics": {"roc_auc": roc_auc, "fpr": fpr, "tpr": tpr},
           "pr_metrics": {"pr_auc": pr_auc, "precision": p, "recall": r},
           "loss": 1 - pr_auc}
    with open(npr_path, "w") as f:
        json.dump(out, f)
    print(f"[done] {npr_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
