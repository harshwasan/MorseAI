"""
Real-audio holdout evaluation.

Tests the model against data/test_holdout.json — the split that was
set aside before any training and never used for training or labeling.

  Kaggle holdout  : 10 real ham radio QSO clips (ground-truth labels)
  ARRL holdout    : 50 real W1AW broadcast chunks (ground-truth labels)

Both sets give a true measure of real-world decoding accuracy.

Usage:
    python benchmark/eval_holdout.py
    python benchmark/eval_holdout.py --checkpoint checkpoints/best_model.pt
    python benchmark/eval_holdout.py --normalize-wpm   # if model trained with WPM norm
"""
import os, sys, json, argparse, time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from inference.transcribe import load_model, decode_audio, decode_file
from utils.wpm import normalize_to_wpm
from data.generate import SAMPLE_RATE

HOLDOUT_FILE = "data/test_holdout.json"
DEFAULT_ARRL_MANIFEST = "benchmark/source_benchmark.json"


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def cer(pred: str, target: str) -> float:
    a = pred.upper().replace(" ", "")
    b = target.upper().replace(" ", "")
    if not b:
        return 0.0 if not a else 1.0
    dp = list(range(len(b) + 1))
    for ca in a:
        ndp = [dp[0] + 1]
        for j, cb in enumerate(b):
            ndp.append(min(dp[j] + (0 if ca == cb else 1),
                           dp[j + 1] + 1, ndp[j] + 1))
        dp = ndp
    return dp[len(b)] / len(b)


def wer(pred: str, target: str) -> float:
    a = pred.upper().split()
    b = target.upper().split()
    if not b:
        return 0.0 if not a else 1.0
    dp = list(range(len(b) + 1))
    for wa in a:
        ndp = [dp[0] + 1]
        for j, wb in enumerate(b):
            ndp.append(min(dp[j] + (0 if wa == wb else 1),
                           dp[j + 1] + 1, ndp[j] + 1))
        dp = ndp
    return dp[len(b)] / len(b)


def _bar(v: float, width: int = 20) -> str:
    filled = int((1 - min(v, 1.0)) * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _evaluate_arrl_items(model, device, items, normalize_wpm_flag: bool, beam_width: int,
                         repo_root: Path, heading: str, decoder: str):
    print(f"{heading}\n")
    arrl_cers, arrl_wers = [], []

    for item in items:
        npy_path = str(repo_root / item["npy"])
        text = item["text"].strip().upper()

        if not os.path.exists(npy_path):
            print(f"  [SKIP] {os.path.basename(npy_path)} — not found")
            continue

        try:
            audio = np.load(npy_path)
        except Exception as e:
            print(f"  [ERR ] {os.path.basename(npy_path)}: {e}")
            continue
        if getattr(audio, "size", 0) == 0:
            print(f"  [SKIP] {os.path.basename(npy_path)} — empty audio")
            continue

        if normalize_wpm_flag:
            audio = normalize_to_wpm(audio, SAMPLE_RATE)

        t0 = time.time()
        try:
            decoded = decode_audio(audio, model, device,
                                   normalize_wpm=False,
                                   beam_width=beam_width,
                                   decoder=decoder).strip()
        except Exception as e:
            print(f"  [ERR ] {os.path.basename(npy_path)}: {e}")
            continue
        elapsed = time.time() - t0

        c = cer(decoded, text)
        w = wer(decoded, text)
        arrl_cers.append(c)
        arrl_wers.append(w)

        status = "OK" if c < 0.10 else ("~" if c < 0.40 else "FAIL")
        print(f"  [{status:4s}] {os.path.basename(npy_path)}  CER={c:.3f}  ({elapsed:.1f}s)")
        print(f"         target : {text[:80]}")
        print(f"         decoded: {decoded[:80]}\n")

    if not arrl_cers:
        print("  No ARRL items evaluated.\n")
        return None, []

    avg_c = sum(arrl_cers) / len(arrl_cers)
    avg_w = sum(arrl_wers) / len(arrl_wers)
    print(f"  ARRL    {_bar(avg_c)}  avg CER={avg_c:.3f}  WER={avg_w:.3f}  ({len(arrl_cers)} chunks)\n")
    return {
        "cer": round(avg_c, 4),
        "wer": round(avg_w, 4),
        "num_items": len(arrl_cers),
    }, arrl_cers


def _load_arrl_eval_sets(repo_root: Path, arrl_manifest: str | None, arrl_fold: str):
    if arrl_manifest:
        manifest_path = repo_root / arrl_manifest
        if not manifest_path.exists():
            raise FileNotFoundError(f"ARRL manifest not found: {manifest_path}")
        with open(manifest_path) as f:
            payload = json.load(f)

        folds = payload.get("folds", [])
        if arrl_fold == "all":
            return [
                {
                    "name": fold["name"],
                    "title": f"── ARRL SOURCE BENCHMARK ({fold['name']}) ──────────────────",
                    "items": fold.get("items", []),
                    "sources": fold.get("sources", []),
                }
                for fold in folds
            ]

        for fold in folds:
            if fold.get("name") == arrl_fold:
                return [{
                    "name": fold["name"],
                    "title": f"── ARRL SOURCE BENCHMARK ({fold['name']}) ──────────────────",
                    "items": fold.get("items", []),
                    "sources": fold.get("sources", []),
                }]
        raise ValueError(f"Fold {arrl_fold} not found in {arrl_manifest}")

    with open(HOLDOUT_FILE) as f:
        holdout = json.load(f)
    return [{
        "name": "holdout",
        "title": "── ARRL HOLDOUT (W1AW real broadcasts) ──────────────────",
        "items": holdout.get("arrl_labeled", []),
        "sources": holdout.get("arrl_sources", []),
    }]


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_holdout(checkpoint: str, normalize_wpm_flag: bool, device: torch.device,
                beam_width: int = 0, arrl_manifest: str | None = None,
                arrl_fold: str = "all", decoder: str = "greedy"):
    repo_root = Path(__file__).parent.parent

    if not os.path.exists(HOLDOUT_FILE):
        print(f"Holdout file not found: {HOLDOUT_FILE}")
        print("Run:  python data/label_kaggle.py  to create it.")
        return

    with open(HOLDOUT_FILE) as f:
        holdout = json.load(f)

    print(f"\n{'='*60}")
    print(f"  MorseAI Real-Audio Holdout Evaluation")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  WPM norm   : {normalize_wpm_flag}")
    print(f"  Beam width : {beam_width if beam_width > 0 else 'greedy'}")
    print(f"  Decoder    : {decoder}")
    if arrl_manifest:
        print(f"  ARRL split : {arrl_manifest} ({arrl_fold})")
    print(f"  Device     : {device}")
    print(f"  Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    model = load_model(str(repo_root / checkpoint), device)

    results = {}

    # ── Kaggle holdout ────────────────────────────────────────
    print("── KAGGLE HOLDOUT (real ham radio QSOs) ─────────────────\n")
    kaggle_cers, kaggle_wers = [], []

    for item in holdout.get("kaggle_labeled", []):
        path       = str(repo_root / item["path"])
        transcript = item["transcript"].strip().upper()

        if not os.path.exists(path):
            print(f"  [SKIP] {os.path.basename(path)} — not found")
            continue

        t0 = time.time()
        try:
            decoded = decode_file(path, model, device,
                                  normalize_wpm=normalize_wpm_flag,
                                  beam_width=beam_width,
                                  decoder=decoder).strip()
        except Exception as e:
            print(f"  [ERR ] {os.path.basename(path)}: {e}")
            continue
        elapsed = time.time() - t0

        c = cer(decoded, transcript)
        w = wer(decoded, transcript)
        kaggle_cers.append(c)
        kaggle_wers.append(w)

        status = "OK" if c < 0.10 else ("~" if c < 0.40 else "FAIL")
        print(f"  [{status:4s}] {os.path.basename(path)}  CER={c:.3f}  WER={w:.3f}  ({elapsed:.1f}s)")
        print(f"         target : {transcript[:80]}")
        print(f"         decoded: {decoded[:80]}\n")

    if kaggle_cers:
        avg_c = sum(kaggle_cers) / len(kaggle_cers)
        avg_w = sum(kaggle_wers) / len(kaggle_wers)
        print(f"  Kaggle  {_bar(avg_c)}  avg CER={avg_c:.3f}  WER={avg_w:.3f}  ({len(kaggle_cers)} clips)\n")
        results["kaggle_cer"] = round(avg_c, 4)
        results["kaggle_wer"] = round(avg_w, 4)
    else:
        print("  No Kaggle holdout clips evaluated.\n")

    # ── ARRL evaluation ───────────────────────────────────────
    arrl_sets = _load_arrl_eval_sets(repo_root, arrl_manifest, arrl_fold)
    arrl_cers = []
    fold_results = []
    for arrl_set in arrl_sets:
        fold_result, fold_cers = _evaluate_arrl_items(
            model, device, arrl_set["items"], normalize_wpm_flag, beam_width,
            repo_root, arrl_set["title"], decoder
        )
        arrl_cers.extend(fold_cers)
        if fold_result:
            fold_result["name"] = arrl_set["name"]
            fold_result["num_sources"] = len(arrl_set.get("sources", []))
            fold_results.append(fold_result)

    if fold_results:
        if len(fold_results) > 1:
            print("  ARRL fold summary:")
            for fr in fold_results:
                print(
                    f"    {fr['name']}: CER={fr['cer']:.3f} "
                    f"WER={fr['wer']:.3f} ({fr['num_sources']} sources, {fr['num_items']} chunks)"
                )
            print()
        if len(fold_results) == 1:
            results["arrl_cer"] = fold_results[0]["cer"]
            results["arrl_wer"] = fold_results[0]["wer"]
        else:
            total_items = sum(fr["num_items"] for fr in fold_results)
            results["arrl_folds"] = fold_results
            results["arrl_cer"] = round(
                sum(fr["cer"] * fr["num_items"] for fr in fold_results) / total_items, 4
            )
            results["arrl_wer"] = round(
                sum(fr["wer"] * fr["num_items"] for fr in fold_results) / total_items, 4
            )

    # ── Kaggle v1 holdout (uncontaminated) ───────────────────
    print("── KAGGLE V1 HOLDOUT (morse-challenge, uncontaminated) ───\n")
    v1_cers, v1_wers = [], []
    v1_labels_path = "data/kaggle_v1/labels.json"

    if os.path.exists(v1_labels_path):
        import json as _json
        with open(v1_labels_path) as f:
            v1_labels = _json.load(f)

        for fname, transcript in sorted(v1_labels.items()):
            path = str(repo_root / "data" / "kaggle_v1" / fname)
            transcript = transcript.strip().upper()
            if not os.path.exists(path):
                continue

            t0 = time.time()
            try:
                decoded = decode_file(path, model, device,
                                      normalize_wpm=normalize_wpm_flag,
                                      beam_width=beam_width,
                                      decoder=decoder).strip()
            except Exception as e:
                print(f"  [ERR ] {fname}: {e}")
                continue
            elapsed = time.time() - t0

            c = cer(decoded, transcript)
            w = wer(decoded, transcript)
            v1_cers.append(c)
            v1_wers.append(w)

            status = "OK" if c < 0.10 else ("~" if c < 0.40 else "FAIL")
            print(f"  [{status:4s}] {fname}  CER={c:.3f}  WER={w:.3f}  ({elapsed:.1f}s)")
            print(f"         target : {transcript[:80]}")
            print(f"         decoded: {decoded[:80]}\n")

        if v1_cers:
            avg_c = sum(v1_cers) / len(v1_cers)
            avg_w = sum(v1_wers) / len(v1_wers)
            print(f"  Kaggle v1 {_bar(avg_c)}  avg CER={avg_c:.3f}  WER={avg_w:.3f}  ({len(v1_cers)} clips)\n")
            results["kaggle_v1_cer"] = round(avg_c, 4)
            results["kaggle_v1_wer"] = round(avg_w, 4)
    else:
        print("  No Kaggle v1 labels found (data/kaggle_v1/labels.json)\n")

    # ── Summary ───────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  SUMMARY")
    if "kaggle_cer" in results:
        print(f"  Kaggle real QSO  CER = {results['kaggle_cer']:.3f}  WER = {results['kaggle_wer']:.3f}")
    if "arrl_cer" in results:
        print(f"  ARRL broadcasts  CER = {results['arrl_cer']:.3f}  WER = {results['arrl_wer']:.3f}")
    if "kaggle_v1_cer" in results:
        print(f"  Kaggle v1 (clean)CER = {results['kaggle_v1_cer']:.3f}  WER = {results['kaggle_v1_wer']:.3f}")
    all_cers = kaggle_cers + arrl_cers + v1_cers
    if all_cers:
        overall = sum(all_cers) / len(all_cers)
        print(f"  Overall real     CER = {overall:.3f}  ({len(all_cers)} samples)")
        results["overall_cer"] = round(overall, 4)
    print(f"{'='*60}\n")

    # Save results alongside benchmark results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"holdout_{ts}.json"
    payload = {
        "timestamp":  datetime.now().isoformat(),
        "checkpoint": checkpoint,
        "normalize_wpm": normalize_wpm_flag,
        "arrl_manifest": arrl_manifest,
        "arrl_fold": arrl_fold,
        "decoder": decoder,
        **results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved -> {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    default="checkpoints/best_model.pt")
    parser.add_argument("--normalize-wpm", action="store_true")
    parser.add_argument("--beam",          type=int, default=0,
                        help="Beam width (0=greedy, 5=recommended)")
    parser.add_argument("--decoder",       default="greedy",
                        choices=["greedy", "beam", "lm"],
                        help="Decoder type")
    parser.add_argument("--arrl-manifest", default=None,
                        help="Optional ARRL benchmark manifest, e.g. benchmark/source_benchmark.json")
    parser.add_argument("--arrl-fold", default="all",
                        help="ARRL fold name from the manifest, or 'all'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = args.arrl_manifest
    if manifest is None and os.path.exists(DEFAULT_ARRL_MANIFEST):
        manifest = DEFAULT_ARRL_MANIFEST
    run_holdout(
        args.checkpoint, args.normalize_wpm, device,
        beam_width=args.beam, arrl_manifest=manifest, arrl_fold=args.arrl_fold,
        decoder=args.decoder
    )


if __name__ == "__main__":
    main()
