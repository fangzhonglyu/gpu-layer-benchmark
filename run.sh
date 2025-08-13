# Use nvidia-smi to get GPU name
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader)

python3 -m model_benchmarks.resnet50_benchmark
python3 -m model_benchmarks.resnet18_benchmark
python3 -m model_benchmarks.mobilenet_benchmark
python3 -m model_benchmarks.efficientnet_benchmark
python3 -m model_benchmarks.replknet_benchmark

cp -r model_benchmarks/* result_archive/$gpu_name