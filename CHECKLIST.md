# CHECKLIST — proxy scoring run

Tick top to bottom. Full detail for any step is in `AGENT_TASK.md` / `README.md`.

## 0. Pre-flight
- [ ] Unpacked the tarball; `pwd` is the `MGT_scoring_only/` root.
- [ ] GPU visible: `nvidia-smi` (or `rocm-smi` on AMD) runs and shows ≥1 free GPU.
- [ ] `data/raw_data/` has **30** files: `ls data/raw_data | wc -l` → 30.
- [ ] Know the server's accelerator + version (CUDA 12.x? ROCm 6.x?) — needed for the torch wheel.

## 1. Environment
- [ ] `conda env create -f env/environment.yml && conda activate scoring`
- [ ] Installed the matching torch build (NOT the default): e.g.
      `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True`
- [ ] `python -c "import transformers; print(transformers.__version__)"` → **≥ 4.50**

## 2. Proxy models
- [ ] Reviewed repo IDs: `python models/download_proxies.py --list`
- [ ] **Confirmed** `PROXY_REPOS` matches the project's original sweep (esp. `gemma-3-*`).
- [ ] `huggingface-cli login` (gemma-3 & llama-3 are gated — license accepted on same account).
- [ ] Downloaded proxies: `python models/download_proxies.py --token hf_xxx`
      (or start small: `--only gemma3_4b,gemma3_12b`)
- [ ] Each lives at `models/cache/local.<proxy>/` and contains `config.json` + weights.

## 3. Dry run
- [ ] `python code/run_scoring.py --list` prints the plan; all cells show `detgpt=-- npr=--`.

## 4. Score
- [ ] Ran scoring (shard across GPUs if available):
      `python code/run_scoring.py --gpu 0 --proxies gemma3_4b,gemma3_12b`
      (add `--fp16` if the GPU is memory-tight; drop `falcon_40b` unless the GPU is huge)
- [ ] No `[FAIL]` lines left unresolved in the console/log output.

## 5. Verify
- [ ] `python code/run_scoring.py --list` → every intended cell shows `detgpt=OK npr=OK`.
- [ ] Spot-check one file is valid JSON with predictions:
      `python -c "import json,glob; d=json.load(open(glob.glob('scoring_results/*.perturbation_30.json')[0])); print(len(d['predictions']['real']))"`
      → a number ≥ 50 (≈500).

## 6. Hand back
- [ ] `tar czf scoring_results.tar.gz scoring_results/`
- [ ] Sent `scoring_results.tar.gz` back to the project owner.
- [ ] Reported: which proxies were run, any cells skipped/failed, and the torch/transformers
      versions + dtype used (fp32 default, or fp16 if `--fp16`).

---
### Guardrails
- Do **not** regenerate perturbations or download t5-3b — caches ship in `data/`.
- Do **not** rename output files; the project keys on `<dataset>.<source>.<proxy>.<detector>.json`.
- The run is idempotent — safe to re-run; finished cells are skipped.
- If a proxy's HF repo id is uncertain, STOP and ask rather than guessing a different model.
