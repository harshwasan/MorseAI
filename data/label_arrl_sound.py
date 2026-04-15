"""
Label arrl_morse_sound data: real W1AW broadcasts where each MP3
has its own paired TXT file (1:1 correspondence, no drift possible).

Folder structure:
    data/arrl_morse_sound/
        10wpm/  130109.mp3  130109.txt  ...  (218 episodes)
        15wpm/  ...
        20wpm/  ...
        30wpm/  ...

Strategy: duration-based labeling — identical to label_sequential.py
but per-file instead of per-directory, so there is zero inter-file drift.

Usage:
    python data/label_arrl_sound.py
    python data/label_arrl_sound.py --force   # reprocess already-labeled files
    python data/label_arrl_sound.py --verbose
"""
import os, sys, re, json, argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.real_dataset import _load_audio
from data.generate import SAMPLE_RATE

CHUNK_SECONDS   = 8   # match MAX_MEL_FRAMES=2000 @ hop32/8kHz = 8s
MIN_WPM_USABLE  = 13.0   # below this → too few words per 8s mel window
MIN_WORDS       = 2
INPUT_ROOT      = 'data/arrl_morse_sound'
OUTPUT_DIR      = 'data/arrl_labeled'


def _clean_text(raw: str) -> str:
    """Normalize to model vocab, strip BOM and non-printable chars."""
    from utils.morse_map import CHAR_TO_IDX
    allowed = set(CHAR_TO_IDX.keys())
    text = raw.upper()
    cleaned = ''.join(c if c in allowed else ' ' for c in text)
    return re.sub(r'\s+', ' ', cleaned).strip()


def label_episode(mp3_path: str, txt_path: str, prefix: str,
                  out_dir: str, force: bool = False, verbose: bool = False) -> int:
    """Label one MP3+TXT pair into 10s chunks. Returns number of chunks saved."""
    with open(txt_path, encoding='utf-8', errors='replace') as f:
        gt_text = _clean_text(f.read())
    gt_words = gt_text.split()
    if len(gt_words) < MIN_WORDS:
        return 0

    try:
        audio = _load_audio(mp3_path)
    except Exception as e:
        if verbose:
            print(f'    Skip {os.path.basename(mp3_path)}: {e}')
        return 0

    duration = len(audio) / SAMPLE_RATE
    if duration < 1.0:
        return 0

    words_per_sec = len(gt_words) / duration
    real_wpm = words_per_sec * 60.0

    if real_wpm < MIN_WPM_USABLE:
        if verbose:
            print(f'    Skip {os.path.basename(mp3_path)}: {real_wpm:.1f} WPM < {MIN_WPM_USABLE}')
        return 0

    # Check if already labeled
    existing = [f for f in os.listdir(out_dir)
                if f.startswith(f'{prefix}_') and f.endswith('.npy')]
    if existing and not force:
        return len(existing)
    for f in existing:
        os.remove(os.path.join(out_dir, f))
        j = os.path.join(out_dir, f.replace('.npy', '.json'))
        if os.path.exists(j):
            os.remove(j)

    saved, chunk_idx, t = 0, 0, 0.0
    while t < duration:
        t_end = min(t + CHUNK_SECONDS, duration)
        w_start = min(int(round(t     * words_per_sec)), len(gt_words))
        w_end   = min(int(round(t_end * words_per_sec)), len(gt_words))

        if w_end - w_start >= MIN_WORDS:
            a0 = int(t     * SAMPLE_RATE)
            a1 = int(t_end * SAMPLE_RATE)
            chunk_audio = audio[a0:a1]
            text = ' '.join(gt_words[w_start:w_end])

            npy_path  = os.path.join(out_dir, f'{prefix}_{chunk_idx:04d}.npy')
            json_path = os.path.join(out_dir, f'{prefix}_{chunk_idx:04d}.json')
            np.save(npy_path, chunk_audio)
            with open(json_path, 'w') as jf:
                json.dump({'text': text, 'start_s': round(t, 3),
                           'end_s': round(t_end, 3), 'wpm': round(real_wpm, 1)}, jf)
            saved      += 1
            chunk_idx  += 1

        t = t_end

    return saved


def label_wpm_folder(wpm_dir: str, out_dir: str, force: bool, verbose: bool) -> int:
    speed_tag = os.path.basename(wpm_dir)   # e.g. '15wpm'
    mp3s = sorted(f for f in os.listdir(wpm_dir) if f.endswith('.mp3'))
    total = 0
    for fname in mp3s:
        mp3_path = os.path.join(wpm_dir, fname)
        txt_path = mp3_path.replace('.mp3', '.txt')
        if not os.path.exists(txt_path):
            continue
        date_tag = fname.replace('.mp3', '')          # e.g. '130109'
        prefix   = f'ams_{speed_tag}_{date_tag}'
        n = label_episode(mp3_path, txt_path, prefix, out_dir, force, verbose)
        total += n
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   default=INPUT_ROOT, help='Root with WPM subdirs')
    parser.add_argument('--out',     default=OUTPUT_DIR)
    parser.add_argument('--force',   action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    grand_total = 0
    for dname in sorted(os.listdir(args.input)):
        dpath = os.path.join(args.input, dname)
        if not os.path.isdir(dpath):
            continue
        print(f'\n{dname}', flush=True)
        n = label_wpm_folder(dpath, args.out, args.force, args.verbose)
        print(f'  -> {n} chunks', flush=True)
        grand_total += n

    print(f'\nTotal: {grand_total} chunks -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
