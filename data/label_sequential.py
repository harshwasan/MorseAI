"""
Label sequential Morse audio datasets where many numbered MP3 segments
share a single TXT file (books, jumbles, Koch training).

Strategy (per-segment alignment — drift-resistant):
  1. DSP-decode each MP3 segment individually
  2. Estimate expected GT word count from WPM × segment duration
  3. Align each segment's DSP words against its local GT slice only
     → errors in one segment cannot propagate to later segments
  4. GT position advances deterministically by WPM × duration each segment
  5. Chunk aligned words into ~10s pieces with ground-truth text labels

Usage:
    python data/label_sequential.py --input data/extra_morse/jumbles_16_20 --wpm 20
    python data/label_sequential.py --input data/extra_morse/book_starborn  --wpm 20
    python data/label_sequential.py --all        # process all extra_morse subdirs
    python data/label_sequential.py --all --force  # reprocess already-labeled dirs
"""
import os, sys, re, json, argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.real_dataset import _load_audio
from data.label_arrl import MIN_WORDS
from data.generate import SAMPLE_RATE

CHUNK_SECONDS = 8   # match MAX_MEL_FRAMES=2000 @ hop32/8kHz = 8s
OUTPUT_DIR    = 'data/arrl_labeled'   # same pool as ARRL chunks


# ─────────────────────────────────────────────────────────────────────────────
# Clean text for books / jumbles (same as ARRL but more permissive)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_book_text(raw: str) -> str:
    """Strip metadata header lines and normalize to model vocab."""
    from utils.morse_map import CHAR_TO_IDX
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        # Skip metadata lines: *WPM=16* *FARN=0* etc.
        if re.match(r'^\*[A-Z]+=', line):
            continue
        if not line:
            continue
        lines.append(line)
    text = ' '.join(lines).upper()
    allowed = set(CHAR_TO_IDX.keys())
    cleaned = ''.join(c if c in allowed else ' ' for c in text)
    return re.sub(r'\s+', ' ', cleaned).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Label one directory
# ─────────────────────────────────────────────────────────────────────────────

def label_directory(input_dir: str, wpm: float, out_dir: str = OUTPUT_DIR,
                    verbose: bool = False, force: bool = False) -> int:
    """
    Label a directory with numbered MP3 segments + one TXT file.
    Uses per-segment alignment so drift in one segment cannot affect others.
    Returns number of chunks saved.
    """
    # Find TXT
    txts = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    if not txts:
        print(f'  No TXT in {input_dir}', flush=True)
        return 0
    with open(os.path.join(input_dir, txts[0]), encoding='utf-8', errors='replace') as f:
        gt_text = _clean_book_text(f.read())
    gt_words = gt_text.split()
    if not gt_words:
        print(f'  Empty GT text in {input_dir}', flush=True)
        return 0

    # Find and sort MP3 segments
    mp3s = sorted(f for f in os.listdir(input_dir) if f.endswith('.mp3'))
    if not mp3s:
        print(f'  No MP3 in {input_dir}', flush=True)
        return 0

    base_name = os.path.basename(input_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Check if already done (unless --force)
    existing_chunks = [f for f in os.listdir(out_dir)
                       if f.startswith(f'seq_{base_name}_') and f.endswith('.npy')]
    if existing_chunks and not force:
        print(f'  Already labeled: {len(existing_chunks)} chunks (skipping — use --force to redo)',
              flush=True)
        return len(existing_chunks)
    elif existing_chunks and force:
        # Remove old chunks for this directory
        for f in existing_chunks:
            os.remove(os.path.join(out_dir, f))
            json_f = f.replace('.npy', '.json')
            json_path = os.path.join(out_dir, json_f)
            if os.path.exists(json_path):
                os.remove(json_path)
        print(f'  Removed {len(existing_chunks)} old chunks, re-labeling...', flush=True)

    # ── Pass 1: load all audio segments ──────────────────────────────────────
    segment_info = []   # (mp3_path, audio, seg_start_s)
    cumulative_s = 0.0

    for seg_idx, fname in enumerate(mp3s):
        mp3_path = os.path.join(input_dir, fname)
        try:
            audio = _load_audio(mp3_path)
        except Exception as e:
            print(f'    Skip {fname}: {e}', flush=True)
            continue
        segment_info.append((mp3_path, audio, cumulative_s))
        cumulative_s += len(audio) / SAMPLE_RATE
        if verbose and seg_idx % 50 == 0:
            print(f'    Loaded {seg_idx}/{len(mp3s)}', flush=True)

    if not segment_info:
        print(f'  No segments loaded', flush=True)
        return 0

    total_dur = cumulative_s
    # Real WPM from total duration and total GT words — no DSP needed
    words_per_sec = len(gt_words) / total_dur
    real_wpm = words_per_sec * 60.0
    print(f'  Total: {total_dur/60:.1f} min, {len(gt_words)} GT words, '
          f'real WPM={real_wpm:.1f} (declared={wpm})', flush=True)

    # Skip datasets too slow to produce useful chunks within the 8s mel window
    # (MAX_MEL_FRAMES=2000 @ 250 frames/s = 8s effective audio per chunk)
    MIN_WPM_USABLE = 15.0
    if real_wpm < MIN_WPM_USABLE:
        print(f'  Skipping: {real_wpm:.1f} WPM < {MIN_WPM_USABLE} WPM — '
              f'too slow for 8s mel window (< 2 words/chunk)', flush=True)
        return 0

    # Use a lower min-words threshold for seq data than ARRL
    MIN_WORDS_SEQ = 2

    # ── Pass 2: emit fixed 10s chunks, label by time→word-index mapping ──────
    # This is drift-free: text order matches audio order, speed is near-constant.
    saved            = 0
    global_chunk_idx = 0
    t                = 0.0

    while t < total_dur:
        t_end = min(t + CHUNK_SECONDS, total_dur)

        # Map time window to GT word range
        w_start = int(round(t * words_per_sec))
        w_end   = int(round(t_end * words_per_sec))
        w_start = min(w_start, len(gt_words))
        w_end   = min(w_end,   len(gt_words))

        if w_end - w_start < MIN_WORDS_SEQ:
            t = t_end
            continue

        text = ' '.join(gt_words[w_start:w_end])

        # Collect audio from segments spanning [t, t_end]
        audio_parts = []
        for _mp3, seg_audio, seg_s in segment_info:
            seg_end_s = seg_s + len(seg_audio) / SAMPLE_RATE
            if seg_end_s <= t or seg_s >= t_end:
                continue
            a0 = max(0, int((t     - seg_s) * SAMPLE_RATE))
            a1 = min(len(seg_audio), int((t_end - seg_s) * SAMPLE_RATE))
            if a1 > a0:
                audio_parts.append(seg_audio[a0:a1])

        if audio_parts:
            chunk_audio = np.concatenate(audio_parts)
            npy_path  = os.path.join(out_dir, f'seq_{base_name}_{global_chunk_idx:04d}.npy')
            json_path = os.path.join(out_dir, f'seq_{base_name}_{global_chunk_idx:04d}.json')
            np.save(npy_path, chunk_audio)
            with open(json_path, 'w') as jf:
                json.dump({'text': text, 'start_s': round(t, 3),
                           'end_s': round(t_end, 3), 'wpm': round(real_wpm, 1)}, jf)
            saved            += 1
            global_chunk_idx += 1

        t = t_end

    print(f'  Saved {saved} chunks -> {out_dir}', flush=True)
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# WPM detection from directory name or content
# ─────────────────────────────────────────────────────────────────────────────

def _detect_wpm(input_dir: str) -> float:
    """Try to infer WPM from directory name or TXT header metadata."""
    name = os.path.basename(input_dir)

    # Pattern: jumbles_16_20 -> midpoint = 18, book_death -> from name
    # Look for two numbers: take the larger (character speed)
    nums = [int(x) for x in re.findall(r'\d+', name)]
    if len(nums) >= 2:
        return float(max(nums))
    if nums:
        return float(nums[0])

    # Try reading metadata from TXT header
    txts = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    if txts:
        with open(os.path.join(input_dir, txts[0]), encoding='utf-8', errors='replace') as f:
            header = f.read(200)
        m = re.search(r'\*WPM=(\d+)\*', header)
        if m:
            return float(m.group(1))

    return 20.0   # fallback


def label_all_extra(extra_root: str = 'data/extra_morse',
                    out_dir: str = OUTPUT_DIR, verbose: bool = False,
                    force: bool = False):
    """Label all subdirectories under extra_root."""
    if not os.path.exists(extra_root):
        print(f'{extra_root} not found', flush=True)
        return

    total = 0
    for dname in sorted(os.listdir(extra_root)):
        dpath = os.path.join(extra_root, dname)
        if not os.path.isdir(dpath):
            continue
        wpm = _detect_wpm(dpath)
        print(f'\n{dname}  (WPM={wpm})', flush=True)
        n = label_directory(dpath, wpm=wpm, out_dir=out_dir, verbose=verbose,
                            force=force)
        total += n

    print(f'\nTotal extra chunks saved: {total} -> {out_dir}', flush=True)
    return total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   type=str, help='Directory with MP3s + TXT')
    parser.add_argument('--wpm',     type=float, default=None)
    parser.add_argument('--all',     action='store_true', help='Process all data/extra_morse dirs')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--force',   action='store_true',
                        help='Re-label even if output chunks already exist')
    args = parser.parse_args()

    if args.all:
        label_all_extra(verbose=args.verbose, force=args.force)
    elif args.input:
        wpm = args.wpm or _detect_wpm(args.input)
        label_directory(args.input, wpm=wpm, verbose=args.verbose, force=args.force)
    else:
        print('Use --input DIR or --all')
