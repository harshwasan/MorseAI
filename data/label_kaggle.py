"""
DSP pseudo-label the 144 unlabeled Kaggle MLMv2 clips.
Also creates a test holdout split from both Kaggle and ARRL labeled data.

Outputs:
  data/kaggle_labeled/   — DSP-labeled Kaggle clips (.npy + .json)
  data/test_holdout.json — list of held-out files for evaluation

Holdout strategy (never used for training):
  - 10 random labeled Kaggle clips (out of 56)
  - 50 random ARRL labeled chunks (out of 583)

Usage:
    python data/label_kaggle.py
"""
import os, sys, json, random, csv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference.dsp_decode import decode_audio_dsp
from data.real_dataset import _load_audio
from data.generate import SAMPLE_RATE, audio_to_melspec, encode_label, MAX_MEL_FRAMES
import torch

KAGGLE_ROOT  = 'data/kaggle_mlmv2'
OUTPUT_DIR   = 'data/kaggle_labeled'
HOLDOUT_FILE = 'data/test_holdout.json'

# Clip any single chunk to this length before saving
CHUNK_SAMPLES = 10 * SAMPLE_RATE


def _arrl_source_key(path_or_name: str) -> str:
    """Collapse chunk filenames like 200212_15wpm_0014.npy to 200212_15wpm."""
    stem = os.path.splitext(os.path.basename(path_or_name))[0]
    parts = stem.split('_')
    return '_'.join(parts[:2]) if len(parts) >= 2 else stem


def _save_chunk(audio: np.ndarray, text: str, path_base: str):
    """Save audio (.npy) + metadata (.json) for one chunk."""
    np.save(path_base + '.npy', audio)
    with open(path_base + '.json', 'w') as f:
        json.dump({'text': text.upper()}, f)


def label_unlabeled_kaggle():
    """DSP-decode 144 unlabeled Kaggle clips and save as training chunks."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(KAGGLE_ROOT, 'SampleSubmission.csv')) as f:
        rows = list(csv.reader(f))

    unlabeled_ids = []
    for row in rows:
        if row[0] == 'ID':
            continue
        idx = int(row[0])
        has_label = len(row) > 1 and row[1].strip()
        path = os.path.join(KAGGLE_ROOT, 'audio', f'cw{idx:03d}.wav')
        if not has_label and os.path.exists(path):
            unlabeled_ids.append(idx)

    print(f'DSP-labeling {len(unlabeled_ids)} unlabeled Kaggle clips...', flush=True)
    saved = 0

    for idx in unlabeled_ids:
        path = os.path.join(KAGGLE_ROOT, 'audio', f'cw{idx:03d}.wav')
        out_base = os.path.join(OUTPUT_DIR, f'cw{idx:03d}')

        # Skip if already done
        if os.path.exists(out_base + '_0000.npy'):
            saved += 1
            continue

        try:
            audio  = _load_audio(path)
            result = decode_audio_dsp(audio, SAMPLE_RATE)
            text   = result['text'].strip()

            if not text or len(text) < 3:
                continue

            # Chunk long clips into CHUNK_SAMPLES windows
            if len(audio) <= CHUNK_SAMPLES:
                _save_chunk(audio, text, f'{out_base}_0000')
                saved += 1
            else:
                n_chunks = (len(audio) + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
                n_chars  = len(text)
                for k in range(n_chunks):
                    a_start = k * CHUNK_SAMPLES
                    a_end   = min(a_start + CHUNK_SAMPLES, len(audio))
                    chunk   = audio[a_start:a_end]

                    frac_s  = a_start / len(audio)
                    frac_e  = a_end   / len(audio)
                    t_start = int(frac_s * n_chars)
                    t_end   = int(frac_e * n_chars)
                    sub     = text[t_start:t_end].strip()
                    if not sub:
                        continue
                    _save_chunk(chunk, sub, f'{out_base}_{k:04d}')
                saved += 1

        except Exception as e:
            print(f'  SKIP cw{idx:03d}: {e}', flush=True)
            continue

        if saved % 20 == 0:
            print(f'  {saved}/{len(unlabeled_ids)} done', flush=True)

    print(f'Saved {saved} pseudo-labeled Kaggle clips -> {OUTPUT_DIR}', flush=True)
    return saved


def create_holdout():
    """
    Set aside test clips that will NEVER be used for training.
    Writes data/test_holdout.json listing held-out file paths.

    ARRL holdout is source-level, not chunk-level: once a source recording is
    selected, every chunk derived from that recording is excluded from training.
    """
    holdout = {'kaggle_labeled': [], 'arrl_sources': [], 'arrl_labeled': []}

    # Hold out 10 labeled Kaggle clips
    kaggle_labeled_ids = []
    with open(os.path.join(KAGGLE_ROOT, 'SampleSubmission.csv')) as f:
        for row in csv.reader(f):
            if row[0] == 'ID': continue
            if len(row) > 1 and row[1].strip():
                idx  = int(row[0])
                path = os.path.join(KAGGLE_ROOT, 'audio', f'cw{idx:03d}.wav')
                if os.path.exists(path):
                    kaggle_labeled_ids.append({
                        'path':       path,
                        'transcript': row[1].strip().upper(),
                    })

    random.seed(42)
    holdout['kaggle_labeled'] = random.sample(kaggle_labeled_ids,
                                              min(10, len(kaggle_labeled_ids)))

    # Hold out whole ARRL source recordings until we reach roughly 50 chunks.
    arrl_sources = {}
    arrl_dir = 'data/arrl_labeled'
    if os.path.exists(arrl_dir):
        for fname in sorted(os.listdir(arrl_dir)):
            if fname.endswith('.json') and not fname.startswith('test_'):
                json_path = os.path.join(arrl_dir, fname)
                npy_path  = json_path.replace('.json', '.npy')
                if os.path.exists(npy_path):
                    with open(json_path) as f:
                        meta = json.load(f)
                    source_key = _arrl_source_key(fname)
                    arrl_sources.setdefault(source_key, []).append({
                        'npy': npy_path,
                        'text': meta['text'],
                        'source_key': source_key,
                    })

    if arrl_sources:
        selected_sources = []
        selected_chunks = []
        remaining_sources = {
            source_key: list(items) for source_key, items in arrl_sources.items()
        }

        # Prefer a mixed-speed holdout with at least two source recordings.
        preferred_wpm_order = ['15wpm', '20wpm', '5wpm', '30wpm', '10wpm']
        target_chunks = 50

        def pick_best_source(candidates, remaining_target):
            shuffled = list(candidates)
            random.shuffle(shuffled)
            return min(
                shuffled,
                key=lambda key: (
                    abs(len(remaining_sources[key]) - remaining_target),
                    len(remaining_sources[key]),
                    key,
                ),
            )

        for wpm in preferred_wpm_order:
            bucket = [key for key in remaining_sources if key.endswith(f'_{wpm}')]
            if not bucket:
                continue
            remaining_target = max(1, target_chunks - len(selected_chunks))
            chosen = pick_best_source(bucket, remaining_target)
            selected_sources.append(chosen)
            selected_chunks.extend(remaining_sources.pop(chosen))
            if len(selected_chunks) >= target_chunks and len(selected_sources) >= 2:
                break

        while remaining_sources and len(selected_chunks) < target_chunks:
            remaining_target = max(1, target_chunks - len(selected_chunks))
            chosen = pick_best_source(list(remaining_sources.keys()), remaining_target)
            selected_sources.append(chosen)
            selected_chunks.extend(remaining_sources.pop(chosen))

        holdout['arrl_sources'] = selected_sources
        holdout['arrl_labeled'] = selected_chunks

    with open(HOLDOUT_FILE, 'w') as f:
        json.dump(holdout, f, indent=2)

    print(f'Holdout saved: {len(holdout["kaggle_labeled"])} Kaggle + '
          f'{len(holdout["arrl_labeled"])} ARRL chunks from '
          f'{len(holdout["arrl_sources"])} source recordings -> {HOLDOUT_FILE}', flush=True)
    return holdout


if __name__ == '__main__':
    random.seed(42)
    create_holdout()
    label_unlabeled_kaggle()
