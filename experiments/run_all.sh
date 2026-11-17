#!/usr/bin/env bash
# Every acceptance run behind the README, in order. Each run resumes where it stopped, and every
# model sees the same requests in the same order (the first --n of one seeded sample).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
$PY experiments/run_acceptance.py --model Qwen/Qwen3-0.6B --keep-last 5 --n 200
$PY experiments/run_acceptance.py --model Qwen/Qwen3-1.7B --keep-last 5 --n 100
$PY experiments/run_acceptance.py --model Qwen/Qwen3-0.6B --keep-last 10 --n 100
# Ablations on 0.6B L5: reply KV as later prompts render it; logits rounded to bf16.
$PY experiments/run_acceptance.py --model Qwen/Qwen3-0.6B --keep-last 5 --n 100 --reply-as-rendered
$PY experiments/run_acceptance.py --model Qwen/Qwen3-0.6B --keep-last 5 --n 60 --bf16-logits
$PY experiments/run_acceptance.py --model Qwen/Qwen3-1.7B --keep-last 10 --n 60
