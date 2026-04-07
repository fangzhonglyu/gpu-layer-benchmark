#!/bin/bash
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --job-name=gpu-bench
#SBATCH --output=slurm-%j.out

# activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bench

# exit on error
set -e

# python3 -m model_benchmarks.mobilenet_benchmark
# python3 -m model_benchmarks.replknet_benchmark
python3 -m model_benchmarks.llama3_1_8b_benchmark
python3 -m model_benchmarks.llama3_1_70b_benchmark
python3 -m model_benchmarks.qwen3_30b_a3b_benchmark
python3 -m model_benchmarks.qwen3_235b_a22b_benchmark

mkdir -p "result_archive/$(nvidia-smi --query-gpu=name --format=csv,noheader)"
cp -r benchmarks/* "result_archive/$(nvidia-smi --query-gpu=name --format=csv,noheader)"