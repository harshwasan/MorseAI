"""
Fixed benchmark for MorseAI — tracks model quality over time.

Runs three suites:
  1. Synthetic  — known ground truth, covers WPM buckets & content types
  2. Real audio — decode + optional CER where a transcript is provided

Results are saved to benchmark/results/<timestamp>.json so every run is
comparable. A summary diff vs the previous run is printed at the end.

Usage:
    python benchmark/run_benchmark.py
    python benchmark/run_benchmark.py --checkpoint checkpoints/best_model.pt
    python benchmark/run_benchmark.py --no-real   # skip real audio files
"""
import os, sys, json, argparse, time
from datetime import datetime
from pathlib import Path

# Windows console may default to cp1252 — force UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from inference.transcribe import load_model, decode_audio, decode_file
from data.generate import synthesize_morse_audio, SAMPLE_RATE
from utils.morse_map import IDX_TO_CHAR
from utils.wpm import normalize_to_wpm

# ─────────────────────────────────────────────────────────────
# 1. SYNTHETIC TEST CASES
#    Each entry: (label, text, wpm, bucket, content_type)
#    bucket       : "slow" | "medium" | "fast"
#    content_type : "simple" | "callsign" | "digits" | "cw_proc" | "edge"
# ─────────────────────────────────────────────────────────────
SYNTHETIC_CASES = [
    # ── Slow (5-9 WPM) ────────────────────────────────────────
    ("slow_sos_5wpm",          "SOS",                   5,  "slow",   "cw_proc"),
    ("slow_cq_6wpm",           "CQ DE K1ABC",           6,  "slow",   "callsign"),
    ("slow_digits_5wpm",       "73",                    5,  "slow",   "digits"),
    ("slow_hello_8wpm",        "HELLO WORLD",           8,  "slow",   "simple"),
    ("slow_rst_7wpm",          "RST 599",               7,  "slow",   "digits"),

    # ── Medium (13-22 WPM) ────────────────────────────────────
    ("med_hello_20wpm",        "HELLO WORLD",           20, "medium", "simple"),
    ("med_fox_20wpm",          "THE QUICK BROWN FOX",   20, "medium", "simple"),
    ("med_cq_w1aw_20wpm",      "CQ CQ DE W1AW K",       20, "medium", "callsign"),
    ("med_rst_serial_18wpm",   "RST 599 NR 001",        18, "medium", "digits"),
    ("med_callsign_20wpm",     "DE K1ABC",              20, "medium", "callsign"),
    ("med_nato_15wpm",         "ALPHA BRAVO CHARLIE",   15, "medium", "simple"),
    ("med_numbers_20wpm",      "ONE TWO THREE",         20, "medium", "simple"),
    ("med_mixed_20wpm",        "K1ABC 599 5NN TU",      20, "medium", "digits"),
    ("med_sos_repeat_15wpm",   "SOS SOS SOS",           15, "medium", "cw_proc"),
    ("med_qrz_20wpm",          "QRZ DE W1AW",           20, "medium", "cw_proc"),

    # ── Fast (28-40 WPM) ──────────────────────────────────────
    ("fast_cq_35wpm",          "CQ CQ DE W1AW",         35, "fast",   "callsign"),
    ("fast_test_35wpm",        "TEST DE K1ABC K",        35, "fast",   "cw_proc"),
    ("fast_fox_30wpm",         "THE QUICK BROWN FOX",   30, "fast",   "simple"),
    ("fast_qso_30wpm",         "QSO QRN QRM QSB",       30, "fast",   "cw_proc"),
    ("fast_digits_30wpm",      "NR 1234",               30, "fast",   "digits"),

    # ── Edge cases ────────────────────────────────────────────
    ("edge_single_e_20wpm",    "E",                     20, "edge",   "edge"),
    ("edge_all_dits_20wpm",    "EEE III SSS",           20, "edge",   "edge"),
    ("edge_all_dahs_20wpm",    "TTT MMM OOO",           20, "edge",   "edge"),
    ("edge_digits_only_20wpm", "1 2 3 4 5 6 7 8 9 0",  20, "edge",   "digits"),
    ("edge_ar_sk_bt_20wpm",    "AR SK BT",              20, "edge",   "cw_proc"),
]

# ─────────────────────────────────────────────────────────────
# 2. REAL AUDIO CASES
#    transcript=None means we log the output for human review only.
#    Provide a transcript string to get CER.
# ─────────────────────────────────────────────────────────────
REAL_CASES = [
    {
        "label":      "real_sos",
        "path":       "test_audio/sos.mp3",
        "transcript": "SOS",
        "wpm_est":    20,
        "notes":      "Short SOS clip",
    },
    {
        "label":      "real_alphabet",
        "path":       "test_audio/alphabet.mp3",
        "transcript": None,   # long — human review only
        "wpm_est":    15,
        "notes":      "NATO phonetic alphabet A-Z",
    },
    {
        "label":      "real_arrl_5wpm",
        "path":       "test_audio/arrl_5wpm.mp3",
        "transcript": None,
        "wpm_est":    5,
        "notes":      "ARRL W1AW code practice 5 WPM",
    },
    {
        "label":      "real_arrl_15wpm",
        "path":       "test_audio/arrl_15wpm.mp3",
        "transcript": None,
        "wpm_est":    15,
        "notes":      "ARRL W1AW code practice 15 WPM",
    },
    {
        "label":      "real_arrl_20wpm",
        "path":       "test_audio/arrl_20wpm.mp3",
        "transcript": None,
        "wpm_est":    20,
        "notes":      "ARRL W1AW code practice 20 WPM",
    },
]

# Kaggle MLMv2 labeled subset — real ham radio QSO recordings
import csv as _csv, os as _os
_mlmv2_dir = "data/kaggle_mlmv2"
_mlmv2_csv = os.path.join(_mlmv2_dir, "SampleSubmission.csv") if False else "data/kaggle_mlmv2/SampleSubmission.csv"
_KAGGLE_REAL_CASES = []
if os.path.exists(_mlmv2_csv):
    with open(_mlmv2_csv) as _f:
        for _row in _csv.reader(_f):
            if _row[0] == "ID":
                continue
            if len(_row) > 1 and _row[1].strip():
                _path = f"data/kaggle_mlmv2/audio/cw{int(_row[0]):03d}.wav"
                _KAGGLE_REAL_CASES.append({
                    "label":      f"kaggle_cw{int(_row[0]):03d}",
                    "path":       _path,
                    "transcript": _row[1].strip(),
                    "wpm_est":    None,
                    "notes":      "Kaggle MLMv2 real ham QSO",
                })


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def cer(decoded: str, target: str) -> float:
    """Character Error Rate via Levenshtein distance."""
    a, b = decoded.upper().replace(" ", ""), target.upper().replace(" ", "")
    if not b:
        return 0.0 if not a else 1.0
    # DP edit distance
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        ndp = [i + 1]
        for j, cb in enumerate(b):
            ndp.append(min(dp[j] + (0 if ca == cb else 1),
                           dp[j + 1] + 1,
                           ndp[j] + 1))
        dp = ndp
    return dp[len(b)] / len(b)


def wer(decoded: str, target: str) -> float:
    """Word Error Rate."""
    a_words = decoded.upper().split()
    b_words = target.upper().split()
    if not b_words:
        return 0.0 if not a_words else 1.0
    dp = list(range(len(b_words) + 1))
    for wa in a_words:
        ndp = [dp[0] + 1]
        for j, wb in enumerate(b_words):
            ndp.append(min(dp[j] + (0 if wa == wb else 1),
                           dp[j + 1] + 1,
                           ndp[j] + 1))
        dp = ndp
    return dp[len(b_words)] / len(b_words)


def _bar(cer_val: float, width: int = 20) -> str:
    filled = int((1 - min(cer_val, 1.0)) * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def load_prev_run(results_dir: Path) -> dict | None:
    files = sorted(results_dir.glob("*.json"))
    if len(files) < 2:
        return None
    with open(files[-2]) as f:
        return json.load(f)


def delta_str(now: float | None, prev: float | None) -> str:
    if prev is None or now is None:
        return ""
    d = now - prev
    if abs(d) < 0.001:
        return "  (=)"
    sign = "+" if d > 0 else "-"
    return f"  ({sign}{abs(d):.3f})"


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_benchmark(checkpoint: str, run_real: bool, device: torch.device, normalize_wpm: bool = False):
    repo_root = Path(__file__).parent.parent
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MorseAI Benchmark")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  WPM norm   : {normalize_wpm}")
    print(f"  Device     : {device}")
    print(f"  Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    model = load_model(str(repo_root / checkpoint), device)

    run_record = {
        "timestamp":  datetime.now().isoformat(),
        "checkpoint": checkpoint,
        "synthetic":  {},
        "real":       {},
    }

    # ── SYNTHETIC ────────────────────────────────────────────
    print("── SYNTHETIC SUITE ─────────────────────────────────────\n")

    bucket_scores  = {"slow": [], "medium": [], "fast": [], "edge": []}
    content_scores = {}

    # Seed once for reproducible augmentation across benchmark runs
    import random as _random
    _random.seed(0)
    np.random.seed(0)

    for label, text, wpm, bucket, ctype in SYNTHETIC_CASES:
        # augment=True matches training distribution (QSB, QRM, bandpass, noise)
        audio = synthesize_morse_audio(text, wpm=wpm, augment=True)
        # Normalize with known WPM (same as training) to avoid auto-detect errors on noisy audio
        if normalize_wpm:
            audio = normalize_to_wpm(audio, SAMPLE_RATE, known_wpm=wpm)
        decoded = decode_audio(audio, model, device, normalize_wpm=False).strip()
        c       = cer(decoded, text)
        w       = wer(decoded, text)

        bucket_scores[bucket].append(c)
        content_scores.setdefault(ctype, []).append(c)

        status = "OK" if c < 0.05 else ("~" if c < 0.25 else "FAIL")
        print(f"  [{status:4s}] {label:<32s}  CER={c:.3f}  WPM={wpm:2.0f}")
        print(f"         target : {text}")
        print(f"         decoded: {decoded}\n")

        run_record["synthetic"][label] = {
            "target": text, "decoded": decoded,
            "wpm": wpm, "bucket": bucket, "content_type": ctype,
            "cer": round(c, 4), "wer": round(w, 4),
        }

    # Bucket summary
    print("── SYNTHETIC BUCKET SUMMARY ─────────────────────────────\n")
    bucket_avg = {}
    for b in ["slow", "medium", "fast", "edge"]:
        scores = bucket_scores[b]
        avg = sum(scores) / len(scores) if scores else 0.0
        bucket_avg[b] = avg
        print(f"  {b:8s}  {_bar(avg)}  avg CER = {avg:.3f}  ({len(scores)} cases)")

    synth_all = [v for vlist in bucket_scores.values() for v in vlist]
    overall_synth = sum(synth_all) / len(synth_all) if synth_all else 0.0
    print(f"\n  OVERALL   {_bar(overall_synth)}  avg CER = {overall_synth:.3f}\n")

    print("── CONTENT TYPE SUMMARY ─────────────────────────────────\n")
    for ctype, scores in sorted(content_scores.items()):
        avg = sum(scores) / len(scores)
        print(f"  {ctype:12s}  avg CER = {avg:.3f}  ({len(scores)} cases)")

    run_record["bucket_avg"]        = {k: round(v, 4) for k, v in bucket_avg.items()}
    run_record["overall_synth_cer"] = round(overall_synth, 4)

    # ── REAL AUDIO ───────────────────────────────────────────
    if run_real:
        print("\n── REAL AUDIO SUITE ─────────────────────────────────────\n")
        real_cers = []
        for case in REAL_CASES + _KAGGLE_REAL_CASES:
            path = repo_root / case["path"]
            if not path.exists():
                print(f"  [SKIP] {case['label']} — file not found: {path}\n")
                continue

            t0 = time.time()
            try:
                decoded = decode_file(str(path), model, device, normalize_wpm=normalize_wpm).strip()
            except Exception as e:
                print(f"  [ERR ] {case['label']} — {e}\n")
                continue
            elapsed = time.time() - t0

            transcript = case["transcript"]
            c = cer(decoded, transcript) if transcript else None

            print(f"  {case['label']}  (est. {case['wpm_est']} WPM, {elapsed:.1f}s)")
            print(f"  notes  : {case['notes']}")
            if transcript:
                print(f"  target : {transcript}")
                print(f"  CER    : {c:.3f}")
            print(f"  decoded: {decoded[:120]}{'...' if len(decoded) > 120 else ''}\n")

            entry = {
                "path": case["path"],
                "wpm_est": case["wpm_est"],
                "decoded": decoded,
                "elapsed_s": round(elapsed, 2),
            }
            if transcript:
                entry["transcript"] = transcript
                entry["cer"] = round(c, 4)
                real_cers.append(c)
            run_record["real"][case["label"]] = entry

        if real_cers:
            avg_real = sum(real_cers) / len(real_cers)
            run_record["overall_real_cer"] = round(avg_real, 4)
            print(f"  Real-audio CER (labelled only): {avg_real:.3f}\n")

    # ── SAVE & DIFF ──────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"{ts}.json"
    with open(out_path, "w") as f:
        json.dump(run_record, f, indent=2)
    print(f"\nResults saved → {out_path}")

    prev = load_prev_run(results_dir)
    if prev:
        print("\n── DELTA VS PREVIOUS RUN ────────────────────────────────\n")
        print(f"  Previous: {prev.get('timestamp', '?')[:19]}")
        for b in ["slow", "medium", "fast", "edge"]:
            now_v  = bucket_avg.get(b)
            prev_v = prev.get("bucket_avg", {}).get(b)
            print(f"  {b:8s}  CER {now_v:.3f}{delta_str(now_v, prev_v)}")
        now_o  = overall_synth
        prev_o = prev.get("overall_synth_cer")
        print(f"  overall   CER {now_o:.3f}{delta_str(now_o, prev_o)}")
        print()

    print(f"{'='*60}")
    print(f"  Benchmark complete.")
    print(f"{'='*60}\n")

    return run_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--no-real", action="store_true", help="Skip real audio files")
    parser.add_argument("--normalize-wpm", action="store_true", help="Normalize WPM before decoding (use for checkpoints trained with WPM normalization)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_benchmark(args.checkpoint, run_real=not args.no_real, device=device, normalize_wpm=args.normalize_wpm)


if __name__ == "__main__":
    main()
