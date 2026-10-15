# Instruction Tuning and Direct Preference Optimization on a Small Language Model

This repository contains the code, configurations, results, and report source for CS5326 Programming Assignment 2. The project post-trains a small pretrained language model in two stages: supervised fine-tuning (SFT) and Direct Preference Optimization (DPO). The training loops for SFT and DPO are written from scratch. A third experiment uses TRL `DPOTrainer` with LoRA on a small persona dataset. All models are compared with blinded pairwise judgments from a language model judge.

## 1. Overview

The work has four training runs and four evaluation comparisons.

| Run | Model | Data | Method |
|---|---|---|---|
| SFT | Qwen2.5-0.5B | 30,000 UltraChat examples | Full fine-tuning, own training loop |
| HH-DPO | SFT checkpoint | 15,000 HH-RLHF pairs | Full fine-tuning, own DPO loop |
| UF-DPO | SFT checkpoint | 15,000 UltraFeedback pairs | Full fine-tuning, own DPO loop |
| LoRA-DPO | SmolLM2-360M-Instruct | 198 persona pairs | TRL `DPOTrainer` with LoRA |

Each DPO run starts from the best SFT checkpoint, which is also its frozen reference model. All runs use seed 42.

## 2. Main Results

Pairwise results from the judge `google/gemini-3.6-flash`. The win rate ignores ties. The full results are in `outputs/tables/results_table.md`.

| Comparison | Prompts | Wins | Ties | Losses | Win rate | 95% CI |
|---|---|---|---|---|---|---|
| HH-DPO vs. SFT (safety set) | 300 | 152 | 61 | 87 | 0.636 | [0.573, 0.694] |
| UF-DPO vs. SFT (AlpacaEval) | 300 | 114 | 62 | 124 | 0.479 | [0.416, 0.542] |
| LoRA-DPO vs. base, style | 40 | 29 | 11 | 0 | 1.000 | [0.883, 1.000] |
| LoRA-DPO vs. base, quality | 40 | 10 | 5 | 25 | 0.286 | [0.163, 0.451] |

Training summary:

| Run | Selected step | Validation loss | Reward accuracy | Reward margin | Time |
|---|---|---|---|---|---|
| SFT | 938 | 1.3901 | - | - | 100.4 min |
| HH-DPO | 450 | 0.6305 | 0.657 | 0.248 | 71.4 min |
| UF-DPO | 200 | 0.5948 | 0.684 | 0.391 | 114.2 min |
| LoRA-DPO | 39 | 0.1171 | 1.000 | 2.242 | 1.85 min |

## 3. Repository Structure

```text
submission_adapter.py      Interface used by the grader; wraps the code in alignment/
alignment/                 Implementation package
    batching.py            Prompt/response tokenization, masking, padding
    logprobs.py            Response-only log-probabilities
    losses.py              SFT loss, DPO loss, reward metrics
    sft.py, dpo.py         From-scratch training loops
    lora.py                TRL DPOTrainer with LoRA
    data.py                Dataset download and filtering
    generation.py          Batched greedy generation
    pairs.py, judge.py     Pair files and pairwise judging
    plotting.py            Figures
    reporting.py, tables.py  Result tables
scripts/                   Command-line entry points
configs/                   YAML configuration for each run
notebooks/                 Kaggle notebooks (they only run the scripts)
tests/                     Public smoke tests and extra tests
eval_sets/                 Fixed evaluation prompts
data/persona_lora/         Persona preference pairs (provided)
data/processed/            Manifest of the prepared training subsets
outputs/                   Metrics, generations, judgments, figures, tables
paper/                     Report source: main.tex, neurips.sty, figures/
docs/                      Assignment manual
```

## 4. Setup

The experiments were run on a Kaggle notebook with two Tesla T4 GPUs (PyTorch 2.11.0+cu128, Transformers 4.57.6, Datasets 4.8.5). Kaggle already provides PyTorch, so it is not reinstalled.

```bash
python -m pip uninstall -y -q torchao || true
python -m pip install -q -r requirements.txt
python -m pip install -q -e . --no-deps
```

Check that two GPUs are visible and record the package versions:

```bash
python scripts/check_environment.py --require_two_gpus
```

## 5. Running the Pipeline

All commands are run from the repository root. The three notebooks in `notebooks/` run the same commands in order: `01_train` (setup, data, training, figures), `02_generate` (generation and pair files), and `03_evaluate` (judging, figures, tables).

**Data**

```bash
python scripts/prepare_data.py --task all
python scripts/make_alpaca_subset.py --size 300 --seed 42
```

**Training**

```bash
python scripts/train_sft.py --config configs/sft.yaml
python scripts/train_dpo.py --config configs/dpo_hh.yaml
python scripts/train_dpo.py --config configs/dpo_uf.yaml
python scripts/train_lora_dpo.py --config configs/lora_dpo.yaml
```

Every training script accepts `--skip_if_complete` (do nothing if `summary.json` exists), `--resume` (continue from the latest rolling checkpoint), and `--set key=value` to override a config value.

**Generation and pair files**

```bash
python scripts/generate.py --model_path outputs/sft/best \
    --prompts_path eval_sets/alpaca_eval_300_seed42.json \
    --output_path outputs/generations/sft_alpaca.json \
    --generator_name sft --dataset_name alpaca_eval

python scripts/build_pairs.py \
    --candidate_path outputs/generations/dpo_uf_alpaca.json \
    --baseline_path outputs/generations/sft_alpaca.json \
    --output_path outputs/eval/dpo_uf_vs_sft_pairs.jsonl \
    --candidate_name dpo_uf --baseline_name sft --seed 42
```

Use `--limit 3 --print_outputs` with `generate.py` for a quick sanity check. For the persona model, add `--prompt_format chat`.

**Judging** (needs a Kaggle notebook with benchmark model access)

```bash
python scripts/judge_pairwise_kaggle.py \
    --pairs_path outputs/eval/dpo_uf_vs_sft_pairs.jsonl \
    --output_path outputs/eval/dpo_uf_vs_sft_judged.jsonl \
    --judge_model google/gemini-3.6-flash --rubric general \
    --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments
```

The available rubrics are `general`, `safety`, `persona_style`, and `persona_quality`. The script `scripts/judge_pairwise_local.py` has the same interface and uses a local Hugging Face model as the judge. It was not used for the reported results.

**Figures and tables**

```bash
python scripts/make_figures.py --output_root outputs
python scripts/make_tables.py --output_root outputs
```

## 6. Implementation Notes

- **Prompt format.** `### Instruction:\n<instruction>\n\n### Response:\n` for SFT, both DPO runs, and evaluation. The persona model uses its own chat template.
- **Masking.** Prompt and padding positions have label `-100`. An EOS token is added to every response before truncation, so truncation can remove it. Sequences are cut at 512 tokens, and DPO prompts at 256 tokens (the last tokens are kept).
- **SFT loss.** The mean negative log-probability over all unmasked response tokens in the batch.
- **DPO.** Sequence log-probabilities are sums over response tokens only, with `beta = 0.1`. The reference model runs under `no_grad`. The policy is kept in FP32 on `cuda:0` with FP16 autocast and gradient scaling. The frozen FP16 reference is placed on `cuda:1` (or the same device if only one GPU is available).
- **Checkpoints.** SFT keeps the lowest validation loss. DPO keeps the highest validation reward accuracy, with the reward margin as the tie-breaker. The final step is also saved.
- **Judging.** The judge sees only the instruction and Responses A and B. The A/B order is randomized with seed 42, and the model identity is stored separately. Requests contain five pairs, wait five seconds between attempts, and allow three attempts. Results are written after every batch and can be resumed.

## 7. Outputs

```text
outputs/<run>/config.yaml        Configuration used (run = sft, dpo_hh, dpo_uf, lora_dpo)
outputs/<run>/metrics.jsonl      Training and validation metrics
outputs/<run>/summary.json       Selected step, final metrics, run time
outputs/generations/             Model outputs and their decoding settings (*.meta.json)
outputs/sanity/                  Short sanity-check generations
outputs/eval/                    Pair files, judgments, failure logs, summaries, manifest
outputs/figures/                 Training and evaluation figures
outputs/tables/                  Result tables as .csv, .md, and .tex
outputs/environment.json         Package versions and GPUs used
```

Large model files (`best/`, `final/`, `adapter/`, `merged/`, `trainer_state/`) are listed in `.gitignore`. The metrics, generations, judgments, figures, and tables are small and are kept.

## 8. Tests

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests
```

On Windows PowerShell, set the path first with `$env:PYTHONPATH = "."`.

The test suite has 50 tests. It contains the public adapter smoke tests and extra tests for truncation boundaries, row order, response-only scoring, the DPO loss formula, reference detachment, unusual tokenizers and model outputs, and the judging utilities. The tests run offline and do not need a GPU.

## 9. Notes and Limitations

- The 40-prompt persona evaluation set (`eval_sets/persona_style_eval_40.json`) was written for this assignment, because the manual describes it but does not supply one.
- All comparisons use a single judge model and a single seed. The LoRA evaluation has only 40 prompts and 22 validation pairs.
- The synthetic safety prompts repeat templates within each category. A win on this set shows a relative preference over SFT and does not measure absolute safety.
- The refusal-phrase, exclamation, and capitalization counts in `text_stats_table` are simple surface statistics.
- The entries in `examples_table` are taken in prompt-ID order for each outcome. They were not hand-picked.

## 10. Report

The report source is in `paper/` (`main.tex`, `neurips.sty`, and `figures/`). It uses the NeurIPS preprint style and can be compiled with `pdflatex` or on Overleaf. The values in the report come from `outputs/eval/*_summary.json` and `outputs/*/summary.json`.
