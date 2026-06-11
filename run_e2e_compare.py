"""End-to-end pipeline comparison: eager (cuBLAS/cuDNN) vs Inductor max-autotune backend.

Runs two representative Llama-3.1-8B decoder-layer pipelines (one decode, one prefill)
through the REAL kernels.py path (CUDA-graph + NVML energy), then dumps the combine_dag
pipeline metrics + per-op latency/energy tagged by backend.

    python run_e2e_compare.py                 # eager baseline
    COMPILE_BACKEND=1 python run_e2e_compare.py   # Inductor max(cuBLAS,Triton) per op

Then `python run_e2e_compare.py --diff` prints the eager-vs-compiled table.
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "model_benchmarks"))

SCALARS = ["total_pipeline_latency", "bottleneck_latency", "total_pipeline_energy",
           "total_pipeline_energy_with_idle", "total_pipeline_energy_with_idle_and_p2p",
           "n_chiplets"]


def run():
    import kernels
    from pipeline_benchmark import pipeline_benchmark
    from llama3_1_8b_benchmark import llama3_1_8b_pipeline, PREFILL_ITERS, DECODE_ITERS

    backend = "compiled" if kernels.COMPILE_BACKEND else "eager"
    configs = [
        llama3_1_8b_pipeline(1, 1, 2048, DECODE_ITERS),       # decode, b=1, kv=2048
        llama3_1_8b_pipeline(1, 2048, 2048, PREFILL_ITERS),   # prefill, b=1, s=2048
    ]
    res = pipeline_benchmark(output_dir=os.path.join(HERE, "benchmarks", f"e2e_{backend}"),
                             pipelines=configs, device_index=kernels.device_from_env())

    out = []
    for r in res:
        rec = {k: r.get(k) for k in SCALARS}
        rec["name"] = r["name"]
        rec["ops"] = {o["name"]: {"ms": o["avg_latency_ms"], "J": o["avg_energy_J"]}
                      for o in r["results"]}
        out.append(rec)

    path = os.path.join(HERE, "benchmarks", f"e2e_{backend}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {path}")


def diff():
    b = os.path.join(HERE, "benchmarks")
    eager = {r["name"]: r for r in json.load(open(os.path.join(b, "e2e_eager.json")))}
    comp = {r["name"]: r for r in json.load(open(os.path.join(b, "e2e_compiled.json")))}
    for name in eager:
        e, c = eager[name], comp.get(name)
        if not c:
            continue
        print(f"\n=== {name} ===")
        print(f"{'metric':<42} {'eager':>12} {'compiled':>12} {'speedup/ratio':>14}")
        for k in ["total_pipeline_latency", "bottleneck_latency",
                  "total_pipeline_energy", "total_pipeline_energy_with_idle_and_p2p"]:
            ev, cv = e[k], c[k]
            r = ev / cv if cv else float("nan")
            print(f"{k:<42} {ev:12.4f} {cv:12.4f} {r:13.3f}x")
        print(f"  -- per-op latency (ms): eager -> compiled (speedup) --")
        for op in e["ops"]:
            em = e["ops"][op]["ms"]
            cm = c["ops"].get(op, {}).get("ms", float("nan"))
            print(f"    {op:<28} {em:9.4f} -> {cm:9.4f}  ({em/cm:5.2f}x)" if cm else f"    {op}")


if __name__ == "__main__":
    if "--diff" in sys.argv:
        diff()
    else:
        run()
