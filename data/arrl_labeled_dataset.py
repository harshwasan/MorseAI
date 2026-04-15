"""
Dataset that loads DSP-labeled ARRL chunks from data/arrl_labeled/.

Each item is a (.npy audio, .json metadata) pair produced by label_arrl.py.
During training, 50% of samples get synthetic noise augmentation so the
neural net learns to handle real-world noise on top of real audio.
50% of samples also get WPM normalization (matching synthetic data behavior).
"""
import os, sys, json, random
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.generate import audio_to_melspec, encode_label, SAMPLE_RATE, MAX_MEL_FRAMES
from utils.wpm import normalize_to_wpm


def _source_key_from_name(path_or_name: str) -> str:
    """Collapse chunk names like 200212_15wpm_0014.npy to 200212_15wpm."""
    stem = os.path.splitext(os.path.basename(path_or_name))[0]
    parts = stem.split('_')
    return '_'.join(parts[:2]) if len(parts) >= 2 else stem


def _load_eval_exclusions():
    """Sources reserved for evaluation should not be used for training."""
    data_dir = os.path.dirname(__file__)
    manifest_paths = [
        os.path.join(data_dir, 'test_holdout.json'),
        os.path.join(data_dir, '..', 'benchmark', 'source_benchmark.json'),
    ]

    excluded_sources = set()
    excluded_npys = set()

    for manifest_path in manifest_paths:
        if not os.path.exists(manifest_path):
            continue
        with open(manifest_path) as f:
            payload = json.load(f)

        for source_key in payload.get('arrl_sources', []):
            excluded_sources.add(source_key)

        for item in payload.get('arrl_labeled', []):
            name = os.path.basename(item['npy'])
            excluded_npys.add(name)
            excluded_sources.add(_source_key_from_name(name))

        for fold in payload.get('folds', []):
            for source_key in fold.get('sources', []):
                excluded_sources.add(source_key)
            for item in fold.get('items', []):
                name = os.path.basename(item['npy'])
                excluded_npys.add(name)
                excluded_sources.add(_source_key_from_name(name))

    return excluded_sources, excluded_npys


# ─────────────────────────────────────────────────────────────
# Noise augmentation (applied to real audio chunks)
# ─────────────────────────────────────────────────────────────

def _pink_noise(n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    for _ in range(6):
        stride = max(1, n // (2 ** random.randint(1, 6)))
        vals   = np.random.randn(n // stride + 2).astype(np.float32)
        idx    = np.clip(np.arange(n) // stride, 0, len(vals) - 1)
        out   += vals[idx]
    peak = np.max(np.abs(out))
    return out / peak if peak > 0 else out


def _qsb_envelope(n: int, sample_rate: int) -> np.ndarray:
    fade_hz = random.uniform(0.2, 1.5)
    depth   = random.uniform(0.2, 0.6)
    t = np.arange(n) / sample_rate
    return (1.0 - depth * (0.5 - 0.5 * np.cos(2 * np.pi * fade_hz * t))).astype(np.float32)


def augment_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Apply random noise augmentations to a real audio chunk.
    Each augmentation is applied independently with its own probability.
    """
    audio = audio.copy()

    # QSB fading (50%)
    if random.random() < 0.5:
        audio *= _qsb_envelope(len(audio), sample_rate)

    # Pink noise (60%)
    if random.random() < 0.6:
        level = random.uniform(0.02, 0.15)
        audio += _pink_noise(len(audio)) * level

    # White noise (40%)
    if random.random() < 0.4:
        level = random.uniform(0.01, 0.08)
        audio += np.random.randn(len(audio)).astype(np.float32) * level

    # QRM: second carrier (30%)
    if random.random() < 0.3:
        qrm_freq = random.uniform(300, 1100)
        qrm_amp  = random.uniform(0.05, 0.25)
        t = np.arange(len(audio)) / sample_rate
        audio += (qrm_amp * np.sin(2 * np.pi * qrm_freq * t)).astype(np.float32)

    # Bandpass filter (40%) — simulate receiver selectivity
    if random.random() < 0.4:
        try:
            from scipy.signal import butter, sosfilt
            center = random.uniform(500, 900)
            bw     = random.uniform(200, 600)
            lo     = max(50.0, center - bw / 2)
            hi     = min(3900.0, center + bw / 2)
            sos    = butter(4, [lo, hi], btype='band', fs=sample_rate, output='sos')
            audio  = sosfilt(sos, audio).astype(np.float32)
        except Exception:
            pass

    # Re-normalize
    if len(audio) > 0:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio /= peak

    return audio


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class ARRLLabeledDataset(Dataset):
    """
    Loads DSP-labeled ARRL chunks produced by data/label_arrl.py.

    augment_prob: fraction of samples to apply noise augmentation (default 0.5)
    min_wpm: skip chunks below this WPM (default 13.0 — excludes bad 5/10 WPM)
    exclude_holdout: if True, skip files listed in data/test_holdout.json
    """

    def __init__(self, root: str = 'data/arrl_labeled', augment_prob: float = 0.5,
                 min_wpm: float = 13.0, exclude_holdout: bool = True):
        self.augment_prob = augment_prob
        self.items = []   # list of (npy_path, json_path)

        if not os.path.exists(root):
            print(f'ARRLLabeledDataset: {root} not found — 0 samples', flush=True)
            return

        # Build evaluation exclusion set
        excluded_sources, excluded_npys = set(), set()
        if exclude_holdout:
            excluded_sources, excluded_npys = _load_eval_exclusions()

        skipped_wpm = 0
        skipped_eval = 0
        for fname in sorted(os.listdir(root)):
            if not fname.endswith('.npy'):
                continue
            # Exclude evaluation files/sources
            if fname in excluded_npys or _source_key_from_name(fname) in excluded_sources:
                skipped_eval += 1
                continue
            json_path = os.path.join(root, fname[:-4] + '.json')
            if not os.path.exists(json_path):
                continue
            # Filter by WPM
            if min_wpm > 0:
                with open(json_path) as f:
                    meta = json.load(f)
                if meta.get('wpm', 0) < min_wpm:
                    skipped_wpm += 1
                    continue
            self.items.append((os.path.join(root, fname), json_path))

        print(f'ARRLLabeledDataset: {len(self.items)} chunks from {root}'
              f' (skipped {skipped_eval} eval, {skipped_wpm} low-WPM)', flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        npy_path, json_path = self.items[i]

        audio = np.load(npy_path)
        with open(json_path) as f:
            meta = json.load(f)
        text = meta['text'].upper()

        # Skip empty chunks — fall back to a neighbour
        if len(audio) == 0:
            return self[(i + 1) % len(self)]

        # 50% WPM normalization — match synthetic data behavior and eval
        if random.random() < 0.5:
            wpm = meta.get('wpm', 20.0)
            audio = normalize_to_wpm(audio, SAMPLE_RATE, known_wpm=wpm)

        # Noise augmentation
        if random.random() < self.augment_prob:
            audio = augment_audio(audio, SAMPLE_RATE)

        mel = audio_to_melspec(audio)        # [64, T]
        T   = mel.shape[1]
        if T > MAX_MEL_FRAMES:
            # Truncate at word boundary instead of mid-character
            kept_secs  = MAX_MEL_FRAMES * 32 / SAMPLE_RATE
            kept_frac  = kept_secs / (len(audio) / SAMPLE_RATE)
            kept_chars = max(1, int(len(text) * kept_frac))
            # Find last space before kept_chars to avoid splitting words
            trunc = text[:kept_chars]
            last_space = trunc.rfind(' ')
            if last_space > 0:
                text = trunc[:last_space]
            else:
                text = trunc.rstrip()
            mel  = mel[:, :MAX_MEL_FRAMES]

        label = encode_label(text)
        return mel, torch.tensor(label, dtype=torch.long), text
