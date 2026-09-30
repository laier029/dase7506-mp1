# DASE7506 MP1 — Small Language Model Challenge

This repository contains the frozen code, experiment record, and report for submission **#59**. The final predictor is a four-block decoder-only Transformer with block-level Post-Norm residuals, rotary position embeddings (RoPE), SwiGLU feed-forward layers, width 512, eight attention heads, and head dimension 64.

The frozen full-test result is **1.6412035634500914 BPB** under protocol `7506-mp1-wt2-v2`. Model selection used validation only. The final test was evaluated after the method and checkpoint had been frozen.

## Frozen artifact

Download the matching checkpoint bundle from the [v1.0 release](../../releases/tag/v1.0). The checkpoint SHA-256 must be:

```text
b96fe6b42b76be4c4aa9e5922e09f3f66e22a15f93f2ac442a6e0d53ad4afad7
```

The matching implementation SHA-256 is:

```text
b96e3fab30eefdf00a3d465406232ee3a07cbc43183952351766abaf94ca0d14
```

Place `checkpoint.pt` anywhere accessible to the evaluator. Its configuration and implementation module name are embedded in the checkpoint.

## Installation

Use Python 3.12. On macOS, run on CPU:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

PyTorch 2.7.1 was used. For Linux CPU-only reproduction, the official CPU wheel may be installed before the remaining requirements:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.5.3 tokenizers==0.21.4
```

No network access, external text, API key, or pretrained weights are used by training or evaluation.

## Reproduce the frozen score

Verify the checkpoint and implementation hashes first:

```bash
shasum -a 256 /path/to/checkpoint.pt student.py
```

Then run the fixed FP32 CPU evaluator:

```bash
python evaluate.py \
  --checkpoint /path/to/checkpoint.pt \
  --device cpu \
  --precision fp32 \
  --threads 4 \
  --split test \
  --output reproduced_test_cpu_fp32.json
```

Expected complete-test fields:

```text
bpb:          1.6412035634500914
token_ppl:   30.90254119918741
targets:     428405
utf8_bytes:  1292013
```

Small runtime differences across CPUs are expected. Two local runs produced identical BPB and scoring times of 15.615 s and 15.951 s.

## Reproduce training

The submitted checkpoint was selected at step 2,100 from a 2,400-step search run. Validation was evaluated every 300 steps, and `checkpoint.pt` stores the lowest-validation-BPB state; `last_checkpoint.pt` stores step 2,400.

```bash
python train_exp15.py \
  --implementation student \
  --config configs/exp15_w512_h8_d4_2400.json \
  --run-dir runs/exp15_postnorm_rope_swiglu_w512_h8_d4_2400_s17 \
  --device cpu \
  --precision fp32 \
  --threads 4 \
  --seed 17 \
  --steps 2400 \
  --batch-size 32 \
  --eval-every 300
```

The full search processed 19,660,800 training targets. The selected step-2,100 checkpoint had processed 17,203,200 targets. Exact floating-point weights may depend on the PyTorch build and hardware; evaluation of the distributed frozen checkpoint is the authoritative reproduction route.

## Resource measurements

Local macOS FP32 CPU measurements for the frozen predictor were:

| Metric | Measured | Limit |
|---|---:|---:|
| Test scoring time | 15.615 s | at most 5x local baseline |
| Time ratio to local baseline | 3.74x | 5x |
| Peak test RSS | 2.287 GiB | 4 GiB |
| Uncompressed inference assets | 56.128 MiB | 64 MiB |

The inference-asset total is the checkpoint plus matching `student.py` and configuration. Timing is machine-dependent; BPB is deterministic for the supplied frozen files.

## Repository map

| Path | Purpose |
|---|---|
| `student.py` | Frozen final architecture. |
| `train_exp15.py` | Final 2,400-step training and best-checkpoint recipe. |
| `evaluate.py`, `common.py` | Fixed course evaluation pipeline. |
| `data/` | Supplied tokenizer and WikiText-2 splits with hashes. |
| `results/` | Full experiment log, validation curves, and frozen result JSON files. |
| `experiments/` | Retained implementation/config snapshots for reported ablations. |
| `output/pdf/report.pdf` | Final report, at most 10 pages. |

## Data and assistance disclosure

The supplied WikiText-2 data and tokenizer were not modified. WikiText-2 was introduced by Merity et al. and contains text by Wikipedia contributors; the dataset notices in the supplied course materials apply.

All architectural hypotheses, experiment directions, comparison choices, and final model-selection decisions were proposed and made by the student. OpenAI ChatGPT/Codex was used as a technical assistant to explain the starter model, help translate the student's proposed experiments into code and reproducible commands, review implementation details, organize recorded measurements, and assist with editing the report. The student launched all training runs, inspected the outputs, interpreted the experimental results, and reviewed the final implementation and analysis. No external training data or pretrained weights were used.
