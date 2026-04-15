"""
Verify all training data in data/arrl_labeled/ before training.

For each chunk prefix group, samples N chunks and DSP-decodes them,
computing CER vs the ground-truth label. High DSP CER = likely mislabeled.

DSP is reliable at 15+ WPM on clean audio, so:
  - avg_cer < 0.30  → GOOD
  - avg_cer 0.30-0.50 → MARGINAL (boundary/noise issues)
  - avg_cer > 0.50  → BAD (likely labeling drift or mismatch)

Usage:
    python data/verify_training_data.py
    python data/verify_training_data.py --samples 50  # more samples per group
    python data/verify_training_data.py --remove-bad  # delete bad prefix groups
"""
import os, sys, json, random, argparse, re
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference.dsp_decode import decode_audio_dsp
from data.generate import SAMPLE_RATE

LABELED_DIR = 'data/arrl_labeled'
SAMPLES_PER_GROUP = 30


def cer(pred: str, target: str) -> float:
    a = pred.upper().replace(' ', '')
    b = target.upper().replace(' ', '')
    if not b: return 0.0 if not a else 1.0
    dp = list(range(len(b) + 1))
    for ca in a:
        ndp = [dp[0] + 1]
        for j, cb in enumerate(b):
            ndp.append(min(dp[j] + (0 if ca == cb else 1),
                           dp[j+1] + 1, ndp[j] + 1))
        dp = ndp
    return dp[len(b)] / len(b)


def get_prefix_groups(labeled_dir: str) -> dict:
    """Group chunks by episode prefix (everything before the last _NNNN)."""
    groups = {}
    for fname in os.listdir(labeled_dir):
        if not fname.endswith('.json'):
            continue
        m = re.match(r'^(.+)_(\d{4})\.json$', fname)
        if not m:
            continue
        prefix = m.group(1)
        groups.setdefault(prefix, []).append(fname)
    return groups


def audit_group(prefix: str, chunk_files: list, labeled_dir: str,
                n_samples: int, wpm: float) -> dict:
    """DSP-audit a random sample from one episode group."""
    sample = random.sample(chunk_files, min(n_samples, len(chunk_files)))
    cers = []
    for jf in sample:
        with open(os.path.join(labeled_dir, jf)) as f:
            d = json.load(f)
        npy = os.path.join(labeled_dir, jf.replace('.json', '.npy'))
        if not os.path.exists(npy):
            continue
        audio = np.load(npy)
        try:
            result = decode_audio_dsp(audio, SAMPLE_RATE, known_wpm=wpm)
            c = cer(result['text'], d['text'])
        except Exception:
            c = 1.0
        cers.append(c)

    if not cers:
        return {'avg_cer': 1.0, 'n': 0, 'bad_frac': 1.0}
    return {
        'avg_cer':   float(np.mean(cers)),
        'p50_cer':   float(np.median(cers)),
        'bad_frac':  sum(1 for c in cers if c > 0.5) / len(cers),
        'n':         len(cers),
    }


def extract_wpm(prefix: str) -> float:
    """Infer WPM from prefix string (e.g. '200101_15wpm' → 15.0)."""
    m = re.search(r'(\d+)wpm', prefix)
    return float(m.group(1)) if m else 20.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',    type=int, default=SAMPLES_PER_GROUP)
    parser.add_argument('--remove-bad', action='store_true',
                        help='Delete chunks from groups with avg_cer > 0.60')
    parser.add_argument('--threshold',  type=float, default=0.60,
                        help='avg_cer threshold for --remove-bad')
    args = parser.parse_args()

    random.seed(42)

    groups = get_prefix_groups(LABELED_DIR)
    print(f'Found {len(groups)} episode groups, {sum(len(v) for v in groups.values())} chunks total\n')

    # Group by source category for summary
    categories = {}
    results_by_prefix = {}

    for prefix in sorted(groups):
        chunk_files = groups[prefix]
        wpm = extract_wpm(prefix)

        # Skip 5wpm — DSP unreliable on Farnsworth
        if wpm <= 5:
            continue

        stats = audit_group(prefix, chunk_files, LABELED_DIR, args.samples, wpm)
        results_by_prefix[prefix] = stats

        # Categorize
        if prefix.startswith('ams_'):
            cat = f"ams_{int(wpm)}wpm"
        elif re.match(r'^\d{6}_\d+wpm$', prefix):
            # date-only prefixes: arrl_direct
            cat = f"arrl_direct_{int(wpm)}wpm"
        else:
            cat = 'other'

        categories.setdefault(cat, []).append(stats['avg_cer'])

        rating = 'GOOD' if stats['avg_cer'] < 0.30 else \
                 ('MARG' if stats['avg_cer'] < 0.50 else 'BAD ')
        print(f'[{rating}] {prefix:<35}  n={len(chunk_files):4d}  '
              f'avg_cer={stats["avg_cer"]:.3f}  bad%={stats["bad_frac"]*100:.0f}%')

    # ── Summary by category ──────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print(f'  SUMMARY BY CATEGORY')
    print(f'{"="*65}')
    for cat in sorted(categories):
        cers = categories[cat]
        bad = sum(1 for c in cers if c > 0.50)
        print(f'  {cat:<30}  episodes={len(cers):4d}  '
              f'avg_cer={np.mean(cers):.3f}  bad_eps={bad}/{len(cers)}')

    # ── Remove bad groups if requested ──────────────────────────────────────
    if args.remove_bad:
        removed_chunks = 0
        removed_eps = 0
        for prefix, stats in results_by_prefix.items():
            if stats['avg_cer'] > args.threshold:
                chunk_files = groups[prefix]
                for jf in chunk_files:
                    for ext in ['.json', '.npy']:
                        fp = os.path.join(LABELED_DIR, jf.replace('.json', ext))
                        if os.path.exists(fp):
                            os.remove(fp)
                removed_chunks += len(chunk_files)
                removed_eps += 1
        print(f'\nRemoved {removed_eps} bad episodes ({removed_chunks} chunks)')

    print(f'\nTotal chunks remaining: '
          f'{sum(len(v) for v in get_prefix_groups(LABELED_DIR).values())}')


if __name__ == '__main__':
    main()
