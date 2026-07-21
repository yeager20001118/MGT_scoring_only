# MGT_scoring_only — proxy scoring package for the TriBet DetectGPT/NPR gap-fill

Self-contained bundle to run on a **GPU server**. It produces the DetectGPT and NPR
score files that are currently missing for 5 source LLMs, so the paper's Tables 7 & 8
(DetectGPT / NPR robustness) can use a real *best-AUROC proxy* for every source instead
of the weak `falcon_7b`-only numbers.

If you are the agent running this on the remote server, read **`AGENT_TASK.md`** — it is
the 6-step checklist. This README is the full reference.

---

## Why this exists

In the paper, the DetectGPT/NPR tables report the *best-AUROC proxy per source*. Six
sources got the full 8-proxy sweep; **five sources only ever had `falcon_7b` scored**, which
badly understates their AUROC (e.g. gpt4o was 0.47 at falcon_7b vs **0.78** at gemma3_4b,
flipping a `–(–)` cell into 100 % power). This package runs the remaining proxies for those
five sources so their rows become consistent with the rest.

The five sources:

```
claude_sonnet_4.6   claude_opus_4.5   claude_haiku_4.5   gpt4o_mini   gemini_3.1_flash
```

Proxies to add (falcon_7b already done):

```
gemma3_4b   gemma3_12b   gemma_2b   gpt-neo-2.7B   llama3.1_8b   llama3.2_3b   [falcon_40b: optional/large]
```

Datasets: `olympic` (self / target) and `xsum` (cross-domain reference).

Work per cell = **5 sources × ≤7 proxies × 2 datasets × {DetectGPT, NPR}**.

---

## What is (and isn't) needed on the remote server

- ✅ **Included in this package:** all source text and — importantly — the **T5-3b
  perturbation caches** (`data/raw_data/*.t5-3b.perturbation_30.raw_data.json`). Perturbations
  are proxy-independent, so the expensive masking step is already done.
- ✅ **You download once:** the proxy LLMs (step 2).
- ❌ **NOT needed:** the t5-3b mask model, and no perturbation regeneration —
  `score_functions/detect_gpt.py` detects the cache and prints *"Use existing perturbation
  file"*. (Only if a cache were missing would t5-3b be required.)

This is why each cell is cheap: just a log-likelihood pass (DetectGPT) and a log-rank pass
(NPR) of the proxy over ~500 cached samples.

---

## Folder layout

```
MGT_scoring_only/
├── README.md                     ← you are here (full reference)
├── AGENT_TASK.md                 ← 6-step checklist for the remote agent
├── env/
│   ├── environment.yml           ← conda env (install torch matching server CUDA/ROCm)
│   └── requirements.txt          ← pip fallback
├── code/
│   ├── run_scoring.py            ← the driver you run (config at top; CLI: --gpu --proxies --list --fp16)
│   ├── score_perturbation_v2.py  ← reference wrapper (not required by run_scoring.py)
│   ├── score_npr_only_v2.py      ← reference wrapper
│   └── score_functions/          ← scoring library (detect_gpt, detect_llm, model, data_builder, metrics, …)
├── data/
│   └── raw_data/                 ← 5 sources × 2 datasets × {raw, perturbation cache, args}  (30 files, ~260 MB)
├── models/
│   ├── download_proxies.py       ← fetches proxies into models/cache/local.<proxy>/
│   └── cache/                     ← (created) local.<proxy>/ model dirs live here
├── scoring_results/              ← (output) <dataset>.<source>.<proxy>.{perturbation_30,npr}.json
├── logs/
└── sync_back/
    └── SYNC_BACK.md              ← how the project owner folds results back in + re-runs the audit
```

---

## Model path format (important)

The scorer resolves a proxy short-name to a **local directory**:

```
models/cache/local.<proxy>/        e.g.  models/cache/local.gemma3_4b/
```

This is the `local.` convention in `code/score_functions/model.py:from_pretrained` — if that
folder exists it is loaded directly (no hub call). `download_proxies.py` places each model
there automatically. The folder must be a normal HF checkpoint (`config.json`,
`*.safetensors`/`*.bin`, tokenizer files) and the suffix after `local.` must match the proxy
name in `code/run_scoring.py` exactly.

`models/download_proxies.py` holds the short-name → HF-repo map (`PROXY_REPOS`).
**Confirm those repo IDs** match the project's original sweep before downloading — they are
the standard base/pretrained repos and are my best reconstruction, but the folder name is
what the pipeline keys on, so if you use a different revision just keep the `local.<proxy>`
name identical.

---

## Environment notes

- `transformers >= 4.50` is required for gemma-3 (`gemma3_4b`, `gemma3_12b`).
- Install the **torch** build matching the server's accelerator (CUDA vs ROCm) — see the
  comment in `env/environment.yml`. Everything else is framework-agnostic.
- `sentencepiece` + `protobuf` are needed by the t5/gemma/llama tokenizers.
- Gemma-3 and Llama-3 repos are **gated**: accept the license on HF with the same account as
  your token.

### Memory / dtype
For numerical consistency with the existing `falcon_7b` cells, models load in the framework
default (**fp32**). Rough fp32 footprints: 3–4B ≈ 12–16 GB, 8B ≈ 32 GB, 12B ≈ 48 GB,
falcon-40b ≈ 160 GB (big/multi-GPU). If memory-limited, pass `--fp16` to `run_scoring.py`
(halves memory; may shift log-probs slightly vs the fp32 falcon_7b cells).

---

## Run it

See `AGENT_TASK.md` for the exact command sequence. Shortest path:

```bash
conda env create -f env/environment.yml && conda activate scoring
# install torch for your CUDA/ROCm
python models/download_proxies.py --token hf_xxx --only gemma3_4b,gemma3_12b   # start small
python code/run_scoring.py --list                                             # preview
python code/run_scoring.py --gpu 0 --proxies gemma3_4b,gemma3_12b             # score
```

The gemma family gave the biggest AUROC gains in the original sweep, so scoring
`gemma3_4b` + `gemma3_12b` first is enough to see which sources cross the 0.63 bridge before
committing GPU time to the rest.

---

## When finished

`tar czf scoring_results.tar.gz scoring_results/` → send back → project owner follows
**`sync_back/SYNC_BACK.md`** to copy the files into `~/MGT/exp/evalue/results/` and re-run the
CPU audit that rebuilds the tables.
