"""Where does prefill vs decode profiling time differ, given BOTH use a 1.5s window?
Run llama3.1-8b prefill (b=8,s=2048) and decode (b=8,kv=2048) op-by-op, decomposing each
op's wall into: pre-heat | measure-window | other-overhead(capture+alloc)."""
import time
import kernels
from kernels import set_device
import llama3_1_8b_benchmark as l8

set_device(0)

def run_pipeline_decomposed(tag, pipe):
    name, phases, *_ = pipe
    print(f"\n===== {tag}: {name} =====")
    print(f"{'op':22s} {'preheat_s':>9} {'window_s':>9} {'other_s':>8} {'wall_s':>7} {'lat':>10} {'P_W':>7}")
    tot_pre = tot_win = tot_wall = 0.0
    for opname, fn in phases:
        t0 = time.perf_counter()
        r = fn()
        wall = time.perf_counter() - t0
        pre = kernels.LAST_PREHEAT_S
        win = r['avg_latency_ms'] * r['iters'] / 1000.0
        other = max(0.0, wall - pre - win)
        lat = r['avg_latency_ms']
        latstr = f"{lat*1000:.1f}us" if lat < 1 else f"{lat:.2f}ms"
        print(f"{opname:22s} {pre:9.2f} {win:9.2f} {other:8.2f} {wall:7.2f} {latstr:>10} {r['avg_power_W']:7.1f}")
        tot_pre += pre; tot_win += win; tot_wall += wall
    print(f"{'TOTAL':22s} {tot_pre:9.2f} {tot_win:9.2f} {tot_wall-tot_pre-tot_win:8.2f} {tot_wall:7.2f}")
    return tot_wall

wd = run_pipeline_decomposed("DECODE  b8 kv2048", l8.llama3_1_8b_pipeline(8, 1, 2048, l8.DECODE_ITERS))
wp = run_pipeline_decomposed("PREFILL b8 s2048",  l8.llama3_1_8b_pipeline(8, 2048, 2048, l8.PREFILL_ITERS))
print(f"\ndecode pipeline  {wd:.1f}s | prefill pipeline {wp:.1f}s | prefill/decode = {wp/wd:.1f}x")
