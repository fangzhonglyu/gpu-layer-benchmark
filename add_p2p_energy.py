#!/usr/bin/env python3
"""
Post-process benchmark results to add estimated P2P layer-to-layer transfer energy.

Computes the activation tensor size between each consecutive layer pair,
then estimates the transfer energy using a configurable pJ/bit rate.

Usage:
    python add_p2p_energy.py                             # default 200 pJ/bit
    python add_p2p_energy.py --pj_per_bit 150            # custom energy per bit
    python add_p2p_energy.py --result_dir benchmarks     # process benchmarks/ instead
"""

import os
import re
import math
import argparse
import csv

DEFAULT_PJ_PER_BIT = 200.0
DTYPE_BYTES = 2  # float16

# ─── Model architecture configs ──────────────────────────────────────

LLAMA_8B = dict(HIDDEN=4096, HEADS=32, HEAD_DIM=128,
                Q_DIM=32 * 128, KV_DIM=8 * 128, INTER=14336)

LLAMA_70B = dict(HIDDEN=8192, HEADS=64, HEAD_DIM=128,
                 Q_DIM=64 * 128, KV_DIM=8 * 128, INTER=28672)

QWEN_235B = dict(HIDDEN=4096, HEADS=64, HEAD_DIM=128,
                 Q_DIM=64 * 128, KV_DIM=4 * 128,
                 NUM_EXPERTS=128, ACTIVE_EXPERTS=8, MOE_INTER=1536, NUM_DEVICES=8)

QWEN_30B = dict(HIDDEN=2048, HEADS=32, HEAD_DIM=128,
                Q_DIM=32 * 128, KV_DIM=4 * 128,
                NUM_EXPERTS=128, ACTIVE_EXPERTS=8, MOE_INTER=768, NUM_DEVICES=8)


# ─── Transfer size calculators ───────────────────────────────────────

def _expert_gemm_m(cfg, b, seq):
    total_pairs = b * seq * cfg['ACTIVE_EXPERTS']
    if total_pairs >= cfg['NUM_EXPERTS']:
        tpe = total_pairs // cfg['NUM_EXPERTS']
        num_active = cfg['NUM_EXPERTS']
    else:
        tpe = 1
        num_active = total_pairs
    return math.ceil(num_active / cfg['NUM_DEVICES']) * tpe


def _llama_transfer_bytes(cfg, b, seq, ctx):
    """Total bytes transferred across all layer-to-layer boundaries (N-1 transfers)."""
    p = b * seq
    H = cfg['HEADS']
    # Output element count of each layer
    outputs = [
        p * cfg['Q_DIM'],              # Q-proj
        p * cfg['KV_DIM'],             # K-proj
        p * cfg['KV_DIM'],             # V-proj
        H * b * seq * ctx,             # QKT
        H * b * seq * ctx,             # Softmax
        H * b * seq * cfg['HEAD_DIM'], # AV
        p * cfg['HIDDEN'],             # O-proj
        p * cfg['INTER'],              # Gate-proj
        p * cfg['INTER'],              # Up-proj
        p * cfg['HIDDEN'],             # Down-proj  (last layer, no transfer after it)
    ]
    return sum(o * DTYPE_BYTES for o in outputs[:-1])


def _qwen_transfer_bytes(cfg, b, seq, ctx):
    p = b * seq
    H = cfg['HEADS']
    gem_m = _expert_gemm_m(cfg, b, seq)
    outputs = [
        p * cfg['Q_DIM'],              # Q-proj
        p * cfg['KV_DIM'],             # K-proj
        p * cfg['KV_DIM'],             # V-proj
        H * b * seq * ctx,             # QKT
        H * b * seq * ctx,             # Softmax
        H * b * seq * cfg['HEAD_DIM'], # AV
        p * cfg['HIDDEN'],             # O-proj
        p * cfg['NUM_EXPERTS'],        # Router
        gem_m * cfg['MOE_INTER'],      # Gate-proj
        gem_m * cfg['MOE_INTER'],      # Up-proj
        gem_m * cfg['HIDDEN'],         # Down-proj  (last layer)
    ]
    return sum(o * DTYPE_BYTES for o in outputs[:-1])


def _mobilenet_transfer_bytes(batch):
    # Conv output: batch * out_channels * P * Q;  Matmul output: M * N
    outputs = [
        batch * 72 * 56 * 56,    # layer06_feat2_expand
        batch * 72 * 28 * 28,    # layer07_feat2_dw
        batch * 24 * 28 * 28,    # layer08_feat2_project
        batch * 240 * 14 * 14,   # layer17_feat5_expand
        batch * 240 * 14 * 14,   # layer18_feat5_dw
        64 * batch,              # layer19_feat5_se_fc1
        240 * batch,             # layer20_feat5_se_fc2
        batch * 40 * 14 * 14,   # layer21_feat5_project
        batch * 576 * 7 * 7,    # layer42_feat10_expand
        batch * 576 * 7 * 7,    # layer43_feat10_dw
        144 * batch,             # layer44_feat10_se_fc1
        576 * batch,             # layer45_feat10_se_fc2
        batch * 96 * 7 * 7,     # layer46_feat10_project  (last layer)
    ]
    return sum(o * DTYPE_BYTES for o in outputs[:-1])


def _replknet_transfer_bytes(batch):
    outputs = [
        batch * 128 * 56 * 56,   # layer01_s0b0_pw1
        batch * 128 * 56 * 56,   # layer02_s0b0_lk31
        batch * 128 * 56 * 56,   # layer03_s0b0_sk5
        batch * 128 * 56 * 56,   # layer04_s0b0_pw2
        batch * 512 * 56 * 56,   # layer05_s0b1_pw1
        batch * 128 * 56 * 56,   # layer06_s0b1_pw2
        batch * 256 * 28 * 28,   # layer13_s1b0_pw1
        batch * 256 * 28 * 28,   # layer14_s1b0_lk29
        batch * 256 * 28 * 28,   # layer15_s1b0_sk5
        batch * 256 * 28 * 28,   # layer16_s1b0_pw2
        batch * 1024 * 28 * 28,  # layer17_s1b1_pw1
        batch * 256 * 28 * 28,   # layer18_s1b1_pw2  (last layer)
    ]
    return sum(o * DTYPE_BYTES for o in outputs[:-1])


# ─── Pipeline name → transfer bytes ─────────────────────────────────

# Regex patterns for pipeline names
_LLAMA_RE = re.compile(
    r'llama3\.1_(\d+b)_(prefill|decode)_b(\d+)_(?:s|kv)(\d+)')
_QWEN_RE = re.compile(
    r'qwen3_(\d+b)_a\d+b_(prefill|decode)_b(\d+)_(?:s|kv)(\d+)')
_MOBILE_RE = re.compile(r'mobilenet_v3_small_b(\d+)')
_REPLK_RE = re.compile(r'replknet31b_b(\d+)')


def get_transfer_bytes(pipeline_name):
    """Return total layer-to-layer transfer bytes for a pipeline, or None if unrecognized."""

    m = _LLAMA_RE.match(pipeline_name)
    if m:
        size, mode, b, sv = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        cfg = LLAMA_8B if size == '8b' else LLAMA_70B
        seq = sv if mode == 'prefill' else 1
        ctx = sv
        return _llama_transfer_bytes(cfg, b, seq, ctx)

    m = _QWEN_RE.match(pipeline_name)
    if m:
        size, mode, b, sv = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        cfg = QWEN_30B if size == '30b' else QWEN_235B
        seq = sv if mode == 'prefill' else 1
        ctx = sv
        return _qwen_transfer_bytes(cfg, b, seq, ctx)

    m = _MOBILE_RE.match(pipeline_name)
    if m:
        return _mobilenet_transfer_bytes(int(m.group(1)))

    m = _REPLK_RE.match(pipeline_name)
    if m:
        return _replknet_transfer_bytes(int(m.group(1)))

    return None


def p2p_energy_J(transfer_bytes, pj_per_bit):
    return transfer_bytes * 8 * pj_per_bit * 1e-12


# ─── File processors ─────────────────────────────────────────────────

P2P_TAG = "P2P Transfer Energy"
TOTAL_P2P_TAG = "Total Pipeline Energy with Idle + P2P"


def process_txt(filepath, pj_per_bit):
    """Add P2P energy lines to a single .txt result file. Idempotent."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if not lines:
        return False

    # Extract pipeline name from first line: "Results for <name>:"
    first = lines[0].strip()
    m = re.match(r'Results for (.+):', first)
    if not m:
        return False
    pipeline_name = m.group(1)

    tbytes = get_transfer_bytes(pipeline_name)
    if tbytes is None:
        return False

    energy = p2p_energy_J(tbytes, pj_per_bit)

    # Parse existing "Total Pipeline Energy with Idle" value
    idle_energy = None
    for line in lines:
        if 'Total Pipeline Energy with Idle' in line and P2P_TAG not in line and TOTAL_P2P_TAG not in line:
            val = re.search(r':\s*([\d.eE+-]+)', line)
            if val:
                idle_energy = float(val.group(1))

    # Remove old P2P lines (for idempotency)
    cleaned = [l for l in lines if P2P_TAG not in l and TOTAL_P2P_TAG not in l]

    # Remove trailing blank lines
    while cleaned and cleaned[-1].strip() == '':
        cleaned.pop()

    # Append new lines
    cleaned.append(f"{P2P_TAG} (J): {energy:.6f}  [@ {pj_per_bit:.0f} pJ/bit]\n")
    if idle_energy is not None:
        total = idle_energy + energy
        cleaned.append(f"{TOTAL_P2P_TAG} (J): {total:.6f}\n")

    with open(filepath, 'w') as f:
        f.writelines(cleaned)
    return True


def _read_idle_energy_from_txt(csv_dir, pipeline_name):
    """Read 'Total Pipeline Energy with Idle' from the corresponding .txt file."""
    txt_path = os.path.join(csv_dir, f"{pipeline_name}.txt")
    if not os.path.exists(txt_path):
        return None
    with open(txt_path, 'r') as f:
        for line in f:
            if 'Total Pipeline Energy with Idle' in line and P2P_TAG not in line and TOTAL_P2P_TAG not in line:
                val = re.search(r':\s*([\d.eE+-]+)', line)
                if val:
                    return float(val.group(1))
    return None


def process_csv(filepath, pj_per_bit):
    """Add P2P energy columns and backfill pipeline_energy_with_idle_J to summary.csv."""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []

    if not rows or not fieldnames:
        return False

    csv_dir = os.path.dirname(filepath)

    # Ensure columns exist (in order)
    new_cols = ['pipeline_energy_with_idle_J',
                'p2p_transfer_energy_J',
                'pipeline_energy_with_idle_and_p2p_J']
    for col in new_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    # Remove old column if present
    if 'pipeline_energy_with_p2p_J' in fieldnames:
        fieldnames.remove('pipeline_energy_with_p2p_J')

    any_updated = False
    for row in rows:
        pipeline_name = row.get('pipeline', '')
        row.pop('pipeline_energy_with_p2p_J', None)

        # Backfill pipeline_energy_with_idle_J from TXT if missing
        idle_e = row.get('pipeline_energy_with_idle_J', '')
        if not idle_e:
            val = _read_idle_energy_from_txt(csv_dir, pipeline_name)
            if val is not None:
                idle_e = f'{val:.6f}'
                row['pipeline_energy_with_idle_J'] = idle_e

        tbytes = get_transfer_bytes(pipeline_name)
        if tbytes is None:
            row['p2p_transfer_energy_J'] = ''
            row['pipeline_energy_with_idle_and_p2p_J'] = ''
            continue

        energy = p2p_energy_J(tbytes, pj_per_bit)
        row['p2p_transfer_energy_J'] = f'{energy:.6f}'

        if idle_e:
            row['pipeline_energy_with_idle_and_p2p_J'] = f'{float(idle_e) + energy:.6f}'
        else:
            row['pipeline_energy_with_idle_and_p2p_J'] = ''
        any_updated = True

    if not any_updated:
        return False

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Add estimated P2P transfer energy to benchmark results")
    parser.add_argument('--pj_per_bit', type=float, default=DEFAULT_PJ_PER_BIT,
                        help=f'Energy per bit in pJ (default: {DEFAULT_PJ_PER_BIT})')
    parser.add_argument('--result_dir', type=str, default='result_archive',
                        help='Root directory containing benchmark results (default: result_archive)')
    args = parser.parse_args()

    txt_count = 0
    csv_count = 0
    skip_count = 0

    for root, dirs, files in os.walk(args.result_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            if fname == 'summary.csv':
                if process_csv(fpath, args.pj_per_bit):
                    csv_count += 1
                    print(f"  CSV  {fpath}")
            elif fname.endswith('.txt'):
                if process_txt(fpath, args.pj_per_bit):
                    txt_count += 1
                else:
                    skip_count += 1

    print(f"\nDone. Updated {txt_count} txt files, {csv_count} csv files. "
          f"Skipped {skip_count} unrecognized txt files.")
    print(f"P2P energy rate: {args.pj_per_bit:.0f} pJ/bit")


if __name__ == '__main__':
    main()