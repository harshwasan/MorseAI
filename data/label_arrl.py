"""
ARRL Labeling Pipeline
======================
For each valid ARRL (MP3 + TXT) pair:
  1. DSP-decode the MP3 → words with start/end timestamps
  2. Clean the ground-truth TXT → list of words
  3. Align DSP words to ground-truth words (edit distance)
  4. Chunk into ~10s segments; each chunk gets exact ground-truth text
  5. Save: audio chunk as .npy + sidecar .json with text and timings

Output directory: data/arrl_labeled/
  <date>_<wpm>_<chunk_idx>.npy   — float32 audio at 8kHz
  <date>_<wpm>_<chunk_idx>.json  — {'text': '...', 'start_s': ..., 'end_s': ..., 'wpm': ...}

Usage:
    python data/label_arrl.py              # label all ARRL files
    python data/label_arrl.py --wpm 20     # only 20 WPM files
    python data/label_arrl.py --test       # dry-run on one file, print alignment
"""
import os, sys, re, json, argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference.dsp_decode import decode_audio_dsp, detect_carrier
from data.real_dataset import _load_audio, _clean_arrl_text, CHUNK_SECONDS
from data.generate import SAMPLE_RATE

ARRL_ROOT   = 'data/arrl'
OUTPUT_DIR  = 'data/arrl_labeled'
MIN_WORDS   = 3     # skip chunks with fewer words than this


# ─────────────────────────────────────────────────────────────
# Word-level alignment (edit distance on word sequences)
# ─────────────────────────────────────────────────────────────

def _word_edit_distance(a: list, b: list):
    """Standard edit distance between two word lists. Returns (dist, ops)."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def align_words(dsp_words: list, gt_words: list) -> list:
    """
    Align DSP decoded words (with timings) to ground-truth words.
    Uses a sliding window approach: for each GT word, find the best
    matching DSP word in a local window to get its timing.

    Returns list of dicts:
        [{'word': str, 'start_s': float, 'end_s': float}, ...]
        Length == len(gt_words). Words without a timing match get None.
    """
    result = []
    dsp_idx = 0  # pointer into dsp_words

    for gt_i, gt_word in enumerate(gt_words):
        # Search a window of dsp_words near current dsp_idx
        window_start = max(0, dsp_idx - 2)
        window_end   = min(len(dsp_words), dsp_idx + 5)
        window       = dsp_words[window_start:window_end]

        best_score = float('inf')
        best_local = 0

        for local_i, dw in enumerate(window):
            # Character-level edit distance for word matching
            a = list(gt_word)
            b = list(dw['word'].strip('=<>?'))
            d = _word_edit_distance(a, b)
            if d < best_score:
                best_score = d
                best_local = local_i

        matched_dsp = window[best_local] if window else None
        dsp_idx = window_start + best_local + 1

        if matched_dsp and best_score <= max(2, len(gt_word) // 2):
            result.append({
                'word':    gt_word,
                'start_s': matched_dsp['start_s'],
                'end_s':   matched_dsp['end_s'],
            })
        else:
            result.append({'word': gt_word, 'start_s': None, 'end_s': None})

    return result


def interpolate_missing(aligned: list) -> list:
    """Fill in None timings by linear interpolation from neighbours."""
    # Forward fill known anchors
    result = [dict(w) for w in aligned]
    n = len(result)

    # Find all known timing indices
    known = [i for i, w in enumerate(result) if w['start_s'] is not None]
    if not known:
        return result

    # Interpolate between known anchors
    for k in range(len(known) - 1):
        i0, i1 = known[k], known[k+1]
        if i1 - i0 <= 1:
            continue
        t0_s, t0_e = result[i0]['start_s'], result[i0]['end_s']
        t1_s = result[i1]['start_s']
        steps = i1 - i0
        for step in range(1, steps):
            frac = step / steps
            idx = i0 + step
            result[idx]['start_s'] = t0_e + frac * (t1_s - t0_e)
            result[idx]['end_s']   = result[idx]['start_s'] + (t0_e - t0_s)

    # Extrapolate edges
    if known[0] > 0:
        ref = result[known[0]]
        avg_dur = ref['end_s'] - ref['start_s']
        for i in range(known[0] - 1, -1, -1):
            result[i]['end_s']   = result[i+1]['start_s'] - 0.05
            result[i]['start_s'] = result[i]['end_s'] - avg_dur

    if known[-1] < n - 1:
        ref = result[known[-1]]
        avg_dur = ref['end_s'] - ref['start_s']
        for i in range(known[-1] + 1, n):
            result[i]['start_s'] = result[i-1]['end_s'] + 0.05
            result[i]['end_s']   = result[i]['start_s'] + avg_dur

    return result


# ─────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────

def chunk_aligned(aligned: list, audio: np.ndarray,
                  sample_rate: int, chunk_s: float = CHUNK_SECONDS) -> list:
    """
    Split aligned word list into ~chunk_s second chunks.
    Each chunk contains words whose start_s falls within the window.

    Returns list of dicts:
        [{'text': str, 'start_s': float, 'end_s': float,
          'audio': np.ndarray}, ...]
    """
    if not aligned:
        return []

    chunks = []
    audio_dur = len(audio) / sample_rate

    chunk_start = aligned[0]['start_s'] or 0.0
    chunk_words = []

    for w in aligned:
        ws = w['start_s'] or chunk_start

        if ws - chunk_start >= chunk_s and chunk_words:
            # Emit current chunk
            text = ' '.join(cw['word'] for cw in chunk_words)
            end_s = chunk_words[-1]['end_s'] or ws
            end_s = min(end_s + 0.5, audio_dur)   # small trailing silence
            a0 = int(chunk_start * sample_rate)
            a1 = int(end_s * sample_rate)
            chunks.append({
                'text':    text,
                'start_s': chunk_start,
                'end_s':   end_s,
                'audio':   audio[a0:a1],
            })
            chunk_start = ws
            chunk_words = []

        chunk_words.append(w)

    # Flush last chunk
    if chunk_words:
        text  = ' '.join(cw['word'] for cw in chunk_words)
        end_s = chunk_words[-1]['end_s'] or audio_dur
        end_s = min(end_s + 0.5, audio_dur)
        a0 = int(chunk_start * sample_rate)
        a1 = int(end_s * sample_rate)
        chunks.append({
            'text':    text,
            'start_s': chunk_start,
            'end_s':   end_s,
            'audio':   audio[a0:a1],
        })

    return chunks


# ─────────────────────────────────────────────────────────────
# Main labeling function
# ─────────────────────────────────────────────────────────────

def label_file(mp3_path: str, txt_path: str, wpm: float,
               out_dir: str, base_name: str, verbose: bool = False) -> int:
    """
    Label one ARRL MP3+TXT pair. Returns number of chunks saved.
    """
    # Load audio
    audio = _load_audio(mp3_path)

    # Load and clean ground truth
    with open(txt_path, encoding='utf-8', errors='replace') as f:
        gt_text = _clean_arrl_text(f.read())
    gt_words = gt_text.split()
    if not gt_words:
        return 0

    # DSP decode — pass known_wpm so dit duration is exact, not estimated.
    # 5 WPM W1AW uses Farnsworth timing: characters sent at ~15 WPM speed
    # with stretched gaps, so the dit duration matches 15 WPM, not 5 WPM.
    carrier = detect_carrier(audio, SAMPLE_RATE)
    char_wpm = 15.0 if wpm <= 5 else float(wpm)
    dsp_result = decode_audio_dsp(audio, SAMPLE_RATE, carrier_hz=carrier, known_wpm=char_wpm)
    dsp_words  = dsp_result['words']

    if verbose:
        print(f'  Carrier: {carrier:.0f} Hz  WPM: {dsp_result["wpm"]}')
        print(f'  DSP words: {len(dsp_words)}  GT words: {len(gt_words)}')
        print(f'  DSP text[:100]: {dsp_result["text"][:100]}')
        print(f'  GT  text[:100]: {gt_text[:100]}')

    # Align DSP timings to ground truth
    aligned = align_words(dsp_words, gt_words)
    aligned = interpolate_missing(aligned)

    # Chunk into ~10s segments
    chunks = chunk_aligned(aligned, audio, SAMPLE_RATE)

    # Save
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for i, chunk in enumerate(chunks):
        if len(chunk['text'].split()) < MIN_WORDS:
            continue
        npy_path  = os.path.join(out_dir, f'{base_name}_{i:04d}.npy')
        json_path = os.path.join(out_dir, f'{base_name}_{i:04d}.json')

        np.save(npy_path, chunk['audio'])
        meta = {
            'text':    chunk['text'],
            'start_s': round(chunk['start_s'], 3),
            'end_s':   round(chunk['end_s'],   3),
            'wpm':     wpm,
        }
        with open(json_path, 'w') as f:
            json.dump(meta, f)
        saved += 1

    return saved


def label_all(arrl_root: str = ARRL_ROOT, out_dir: str = OUTPUT_DIR,
              wpm_filter: int = None, verbose: bool = False):
    """Label all valid ARRL pairs found under arrl_root."""
    total_saved = 0

    for wpm_dir in sorted(os.listdir(arrl_root)):
        m = re.match(r'(\d+)wpm', wpm_dir)
        if not m:
            continue
        wpm = int(m.group(1))
        if wpm_filter and wpm != wpm_filter:
            continue

        dir_path = os.path.join(arrl_root, wpm_dir)
        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith('.mp3'):
                continue
            base     = fname[:-4]
            mp3_path = os.path.join(dir_path, fname)
            txt_path = os.path.join(dir_path, base + '.txt')
            if not os.path.exists(txt_path):
                continue

            base_name = f'{base}_{wpm}wpm'
            print(f'Labeling {wpm_dir}/{fname} ...', flush=True)
            try:
                n = label_file(mp3_path, txt_path, wpm,
                               out_dir, base_name, verbose=verbose)
                print(f'  -> {n} chunks saved', flush=True)
                total_saved += n
            except Exception as e:
                print(f'  ERROR: {e}', flush=True)

    print(f'\nTotal chunks saved: {total_saved}  ->  {out_dir}')
    return total_saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',    type=str, default=ARRL_ROOT,
                        help='Root directory with WPM subdirs (default: data/arrl)')
    parser.add_argument('--wpm',     type=int, default=None, help='Filter by WPM (5/15/20)')
    parser.add_argument('--test',    action='store_true',    help='Dry-run on first file')
    parser.add_argument('--verbose', action='store_true',    help='Print alignment details')
    args = parser.parse_args()

    if args.test:
        # Quick test on first available file
        for wpm_dir in ['20wpm', '15wpm', '5wpm']:
            d = os.path.join(args.root, wpm_dir)
            if not os.path.exists(d):
                continue
            for fname in sorted(os.listdir(d)):
                if fname.endswith('.mp3'):
                    base = fname[:-4]
                    mp3  = os.path.join(d, fname)
                    txt  = os.path.join(d, base + '.txt')
                    if os.path.exists(txt):
                        wpm = int(wpm_dir[:-3])
                        print(f'Test run: {mp3}')
                        n = label_file(mp3, txt, wpm, OUTPUT_DIR,
                                       f'test_{base}_{wpm}wpm', verbose=True)
                        print(f'Chunks saved: {n}')
                        break
            break
    else:
        label_all(arrl_root=args.root, wpm_filter=args.wpm, verbose=args.verbose)
