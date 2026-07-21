"""score_perturbation_v2.py — DetectGPT (perturbation_N) scorer.

Thin wrapper around Online_Detection/scripts/score_functions/detect_gpt.py.
Generates T5-3b perturbations and scores them with the proxy LLM,
producing the standard `<output_prefix>.perturbation_<N>.json` file.

Idempotency: checks `--output_file.perturbation_<N>.json` first; skip if valid.
The perturbation raw_data file (~50–200 MB) is written next to the input
dataset file as `<dataset_file>.t5-3b.perturbation_<N>.raw_data.json`, so that
NPR/LRR (score_detect_llm_v2.py) can reuse it for free.

Usage:
  python score_perturbation_v2.py \
    --dataset_file ~/MGT/exp/raw_data/olympic.gpt4o \
    --output_file ~/MGT/exp/evalue/results/olympic.gpt4o.falcon_7b \
    --scoring_model_name falcon_7b \
    --device cuda:0
"""
from __future__ import annotations
import argparse, os, sys, json, time

WORK = os.path.expanduser("~/MGT/exp/evalue")
ONLINE = os.path.expanduser("~/MGT/baselines/Online_Detection/scripts/score_functions")
DEFAULT_CACHE = os.path.expanduser("~/MGT/cache_for_scoring")


def _has_valid_output(out_path: str, min_records: int = 50) -> bool:
    if not os.path.exists(out_path):
        return False
    try:
        d = json.load(open(out_path))
        if "predictions" not in d:
            return False
        return (len(d["predictions"].get("real", [])) >= min_records
                and len(d["predictions"].get("samples", [])) >= min_records)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_file", required=True,
                    help="prefix to .raw_data.json (e.g. ~/MGT/exp/raw_data/olympic.gpt4o)")
    ap.add_argument("--output_file", required=True,
                    help="prefix for results (e.g. ~/MGT/exp/evalue/results/olympic.gpt4o.falcon_7b)")
    ap.add_argument("--scoring_model_name", required=True,
                    help="proxy LLM, e.g. falcon_7b, gemma_2b, gpt-neo-2.7B")
    ap.add_argument("--mask_filling_model_name", default="t5-3b")
    ap.add_argument("--n_perturbations", type=int, default=30)
    ap.add_argument("--pct_words_masked", type=float, default=0.3)
    ap.add_argument("--span_length", type=int, default=2)
    ap.add_argument("--mask_top_p", type=float, default=0.96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dataset", type=str, default="xsum",  # only matters for OPT tokenizer
                    help="passed through to load_tokenizer; cosmetic for our proxies")
    ap.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE)
    ap.add_argument("--hf_token", type=str, default="skip")
    args = ap.parse_args()

    name = f"perturbation_{args.n_perturbations}"
    out_path = f"{args.output_file}.{name}.json"
    if _has_valid_output(out_path):
        print(f"[skip] {out_path} already valid")
        return 0

    if not os.path.exists(args.dataset_file + ".raw_data.json"):
        print(f"[err] missing input: {args.dataset_file}.raw_data.json", file=sys.stderr)
        return 2

    sys.path.insert(0, ONLINE)
    import detect_gpt  # noqa: E402

    t0 = time.time()
    print(f"[run ] perturbation_{args.n_perturbations} on {args.dataset_file} "
          f"with scoring={args.scoring_model_name} mask={args.mask_filling_model_name} "
          f"device={args.device}")
    detect_gpt.experiment(args)
    print(f"[done] {out_path} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
