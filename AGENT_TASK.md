# AGENT TASK — run proxy scoring, in 6 steps

You are on a GPU server. Goal: produce DetectGPT + NPR score files for 5 source LLMs
under several proxy models, then hand `scoring_results/` back. The expensive T5
perturbation step is already done and shipped in `data/` — you only run cheap per-proxy
passes. No internet is needed except to download the proxy models once.

1. **Env**
   ```bash
   conda env create -f env/environment.yml && conda activate scoring
   # then install the torch build matching this server's CUDA/ROCm (see env/environment.yml comment)
   ```

2. **Download proxy models** (into `models/cache/local.<proxy>/` — the layout the scorer expects)
   ```bash
   python models/download_proxies.py --list           # see repos + sizes
   huggingface-cli login                               # gemma-3 & llama-3 are gated
   python models/download_proxies.py --token hf_xxx    # or --only gemma3_4b,gemma3_12b to start small
   ```
   ⚠️ Confirm the repo IDs in `models/download_proxies.py` (PROXY_REPOS) match the project's
   original sweep before downloading gemma-3 / llama-3 variants.

3. **Dry-run the plan**
   ```bash
   python code/run_scoring.py --list
   ```

4. **Run scoring** (shard by proxy across GPUs if you have several)
   ```bash
   python code/run_scoring.py --gpu 0 --proxies gemma3_4b,gemma3_12b
   python code/run_scoring.py --gpu 1 --proxies llama3.1_8b,llama3.2_3b
   python code/run_scoring.py --gpu 2 --proxies gemma_2b,gpt-neo-2.7B
   # add --fp16 on memory-tight GPUs (see note in code/run_scoring.py)
   ```
   Outputs land in `scoring_results/` as `<dataset>.<source>.<proxy>.{perturbation_30,npr}.json`.
   The script is idempotent — safe to re-run; it skips finished cells.

5. **Verify** every cell finished:
   ```bash
   python code/run_scoring.py --list     # all should show detgpt=OK npr=OK
   ```

6. **Hand back**: `tar czf scoring_results.tar.gz scoring_results/` and send it to the
   project owner. They finish with `sync_back/SYNC_BACK.md`.

Full details, scope, and rationale: `README.md`.
