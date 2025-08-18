# exit on error
set -e

python3 -m model_benchmarks.resnet50_benchmark
python3 -m model_benchmarks.mobilenet_benchmark
python3 -m model_benchmarks.efficientnet_benchmark
python3 -m model_benchmarks.replknet_benchmark
# python3 -m model_benchmarks.vit_benchmark
python3 -m model_benchmarks.gpt_OPT_66B_bench
# python3 -m model_benchmarks.gpt_OPT_1_3B_bench

mkdir -p "result_archive/$(nvidia-smi --query-gpu=name --format=csv,noheader)"
cp -r benchmarks/* "result_archive/$(nvidia-smi --query-gpu=name --format=csv,noheader)"