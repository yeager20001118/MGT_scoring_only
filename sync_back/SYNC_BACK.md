# Syncing scored results back into the TriBet project

When the remote server finishes, `scoring_results/` will contain files named:

```
<dataset>.<source>.<proxy>.perturbation_30.json      (DetectGPT)
<dataset>.<source>.<proxy>.npr.json                  (NPR)
```

e.g. `olympic.claude_opus_4.5.gemma3_4b.perturbation_30.json`,
     `xsum.claude_opus_4.5.gemma3_4b.npr.json`.

## 1. Bring the folder back to the machine that has the project

On the remote server, compress just the outputs (small — a few MB each):

```bash
cd ~/MGT_scoring_only
tar czf scoring_results.tar.gz scoring_results/
```

Copy `scoring_results.tar.gz` back to the project machine (scp/rsync/whatever you used
to send the package), then unpack somewhere temporary and inspect.

## 2. Drop the score files into the project results directory

The project reads score files from `~/MGT/exp/evalue/results/`. Copy them in:

```bash
# from wherever you unpacked scoring_results/
rsync -av --ignore-existing scoring_results/  ~/MGT/exp/evalue/results/
```

`--ignore-existing` protects the falcon_7b files already there. Nothing is overwritten;
you are only ADDING the new `<proxy>` variants.

Sanity check the count (5 sources × up-to-7 proxies × 2 datasets × 2 detectors):

```bash
ls ~/MGT/exp/evalue/results/ | grep -E 'claude_(sonnet_4.6|opus_4.5|haiku_4.5)|gpt4o_mini|gemini_3.1_flash' \
  | grep -E '\.(perturbation_30|npr)\.json$' | wc -l
```

## 3. Re-run the best-proxy audit so the tables update  (PROJECT-SIDE, CPU only)

This step happens back on the project machine, not the remote GPU server. The detection
step picks the best-AUROC proxy per source (from the score files now sitting in
`results/`) and recomputes power/FPR into `results/focused_audit_v2.json` — the same
file that already backs the 6 good rows in Tables 7 & 8.

`run_focused_audit_v2.py` hardcodes its source list (it imports `SOURCES` from
`run_focused_audit.py`, currently the 6 multi-proxy sources). To fold in the 5 newly
scored sources:

1. Edit `~/MGT/exp/evalue/run_focused_audit.py`: add the 5 sources to `SOURCES`, and
   make sure the proxy-candidate list used by `select_best_proxy_table` / `find_score`
   includes the proxies you just scored (gemma3_4b, gemma3_12b, gemma_2b, gpt-neo-2.7B,
   llama3.1_8b, llama3.2_3b, and falcon_40b if you ran it).
2. Run it:

   ```bash
   cd ~/MGT/exp/evalue
   python run_focused_audit_v2.py        # writes results/focused_audit_v2.json
   ```

3. Regenerate Tables 7 & 8 from `focused_audit_v2.json` exactly as the 6 already-good
   sources were built. Any source whose best-proxy held-out Olympic AUROC clears 0.63
   switches from `–(–)` to a real power/τ number; the rest stay FPR-only but with the
   corrected (higher) AUROC and the best-proxy FPR.

> If a source's best proxy still does not clear 0.63, that is a real (not missing) result —
> report it as FPR-only, same admissibility rule as the rest of the table.
>
> NOTE: steps here are the project-side follow-up. The remote server's job ends at
> producing `scoring_results/` (steps 1–2).
