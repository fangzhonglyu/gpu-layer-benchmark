#!/bin/bash
# Run every model benchmark across N_GPUS, ONE process per card at a time (a per-GPU
# queue), so energy/latency are measured without two jobs sharing a board. Each script
# is pinned to a card via BENCH_DEVICE; set_device binds the NVML energy handle to that
# same physical GPU by UUID, so per-card energy is correct even though all GPUs are visible.
#
# NOTE: 4 cards running hot at once share the chassis power/thermal budget and may boost to
# a different clock than a single-card run. For energy numbers comparable to isolated runs,
# lock the clock first:  sudo nvidia-smi -lgc <freq>   (and -lmc <freq>).
# Do NOT run the transfer-curve characterization (characterize_transfer_curve.py) at the
# same time as this -- it needs an idle link / quiet power baseline.
#
# Usage:  ./run_all_benchmarks.sh            # 4 GPUs (default)
#         N_GPUS=2 ./run_all_benchmarks.sh   # 2 GPUs
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # make CUDA and NVML agree on physical order
N_GPUS=${N_GPUS:-4}
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
# the model scripts do `from kernels import ...`; running them as files puts model_benchmarks/
# on sys.path, not the repo root -- add it so kernels.py / pipeline_benchmark.py import.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

SCRIPTS=(
  model_benchmarks/llama3_1_8b_benchmark.py
  model_benchmarks/llama3_1_70b_benchmark.py
  model_benchmarks/qwen3_30b_a3b_benchmark.py
  model_benchmarks/qwen3_235b_a22b_benchmark.py
  model_benchmarks/vit_benchmark.py
  model_benchmarks/mobilenet_benchmark.py
  model_benchmarks/replknet_benchmark.py
)

mkdir -p logs
declare -A GPU_PID                            # gpu -> pid of the job currently on it
i=0
for s in "${SCRIPTS[@]}"; do
  gpu=$(( i % N_GPUS ))
  # serialize per GPU: wait for this card's previous job before launching the next on it
  if [[ -n "${GPU_PID[$gpu]:-}" ]]; then wait "${GPU_PID[$gpu]}" || true; fi
  base="$(basename "$s" .py)"
  echo "[launch] gpu${gpu} <- ${base}"
  BENCH_DEVICE=$gpu python "$s" > "logs/${base}.gpu${gpu}.log" 2>&1 &
  GPU_PID[$gpu]=$!
  i=$(( i + 1 ))
done
wait
echo "[done] all benchmarks finished; merge with:  python merge_summaries.py"
