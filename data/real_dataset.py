"""
Real audio datasets for mixed training.

KaggleDataset  — 56 labeled WAV clips from data/kaggle_mlmv2/audio/
ARRLDataset    — paired MP3+TXT from data/arrl/, chunked into ~10s segments
MixedDataset   — weighted mix: 50% synthetic + 25% Kaggle + 25% ARRL
"""
import os, sys, re, csv, random
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.generate import audio_to_melspec, MorseDataset, SAMPLE_RATE, encode_label, MAX_MEL_FRAMES
from utils.morse_map import CHAR_TO_IDX, CHAR_TO_MORSE

CHUNK_SECONDS = 8   # match MAX_MEL_FRAMES=2000 @ hop32/8kHz = 8s


# ─────────────────────────────────────────────────────────────
# Audio I/O
# ─────────────────────────────────────────────────────────────

def _load_audio(path: str) -> np.ndarray:
    """Load any audio file → float32 mono at SAMPLE_RATE via ffmpeg."""
    import subprocess, imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # Resolve to absolute path and normalise slashes for Windows
    abs_path = str(os.path.abspath(path))
    cmd = [
        ffmpeg, '-i', abs_path,
        '-f', 's16le', '-ac', '1', '-ar', str(SAMPLE_RATE),
        '-loglevel', 'error', 'pipe:1',
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _audio_to_item(audio: np.ndarray, text: str):
    """Convert raw audio + text → (mel, label_tensor, text) matching MorseDataset format."""
    mel = audio_to_melspec(audio)            # [64, T]
    T = mel.shape[1]
    if T > MAX_MEL_FRAMES:
        kept_secs = MAX_MEL_FRAMES * 32 / SAMPLE_RATE
        kept_frac = kept_secs / (len(audio) / SAMPLE_RATE)
        kept_chars = max(1, int(len(text) * kept_frac))
        # Truncate at word boundary
        trunc = text[:kept_chars]
        last_space = trunc.rfind(' ')
        text = trunc[:last_space] if last_space > 0 else trunc.rstrip()
        mel  = mel[:, :MAX_MEL_FRAMES]
    label = encode_label(text)
    return mel, torch.tensor(label, dtype=torch.long), text


def _word_timing_units(word: str) -> int:
    """
    Relative Morse duration for one word in timing units.
    This gives a better proxy for transcript timing than raw character count.
    """
    total = 0
    chars = [c for c in word.upper() if c in CHAR_TO_MORSE]
    for ci, char in enumerate(chars):
        morse = CHAR_TO_MORSE[char]
        for si, symbol in enumerate(morse):
            total += 1 if symbol == '.' else 3
            if si < len(morse) - 1:
                total += 1
        if ci < len(chars) - 1:
            total += 3
    return max(total, 1)


def _word_spans(text: str):
    words = text.strip().split()
    if not words:
        return []

    spans = []
    pos = 0.0
    for word in words:
        units = float(_word_timing_units(word))
        spans.append((pos, pos + units, word))
        pos += units + 7.0  # inter-word gap
    return spans


def _slice_transcript_by_fraction(text: str, frac_s: float, frac_e: float) -> str:
    """
    Slice transcript by approximate timing instead of raw character position.
    Uses word-level Morse timing as a proxy and selects words whose midpoint
    falls inside a slightly shrunken core interval to avoid partial edge words.
    """
    spans = _word_spans(text)
    if not spans:
        return text.strip().upper()

    total = spans[-1][1]
    start = max(0.0, min(frac_s, 1.0)) * total
    end = max(start, min(frac_e, 1.0)) * total
    window = end - start
    core_margin = min(window * 0.15, total * 0.05)
    core_start = start if frac_s <= 0.02 else min(end, start + core_margin)
    core_end = end if frac_e >= 0.98 else max(start, end - core_margin)

    selected = [
        word for w_start, w_end, word in spans
        if core_start <= (w_start + w_end) / 2 <= core_end
    ]
    if not selected:
        selected = [
            word for w_start, w_end, word in spans
            if w_end > start and w_start < end
        ]
    if not selected:
        # Fall back to nearest word around the window midpoint.
        midpoint = (start + end) / 2
        nearest = min(spans, key=lambda item: abs(((item[0] + item[1]) / 2) - midpoint))
        selected = [nearest[2]]

    return ' '.join(selected).strip().upper() or text.strip().upper()


# ─────────────────────────────────────────────────────────────
# ARRL text parsing
# ─────────────────────────────────────────────────────────────

def _clean_arrl_text(raw: str) -> str:
    """Strip ARRL header/footer markers and normalize to model vocabulary."""
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        # Skip marker lines: =  NOW X WPM  = , < QST DE W1AW =, = END ...
        if re.match(r'^[=<]', line):
            continue
        if 'QST DE W1AW' in line:
            continue
        lines.append(line)
    text = ' '.join(lines).upper()
    # Keep only chars that exist in the model vocabulary
    allowed = set(CHAR_TO_IDX.keys())
    cleaned = ''.join(c if c in allowed else ' ' for c in text)
    return re.sub(r'\s+', ' ', cleaned).strip()


def _chars_per_second(wpm: float) -> float:
    """Estimated characters per second at given WPM (PARIS standard: 5 chars/word)."""
    return wpm * 5.0 / 60.0


# ─────────────────────────────────────────────────────────────
# KaggleDataset
# ─────────────────────────────────────────────────────────────

class KaggleDataset(Dataset):
    """
    56 labeled WAV clips from data/kaggle_mlmv2.
    Clips shorter than CHUNK_SECONDS are used whole.
    Longer clips are randomly windowed to CHUNK_SECONDS with proportional text.
    """

    def __init__(self, root: str = 'data/kaggle_mlmv2', exclude_holdout: bool = True):
        self.samples = []   # list of (path, transcript)
        csv_path = os.path.join(root, 'SampleSubmission.csv')
        if not os.path.exists(csv_path):
            return

        # Build holdout exclusion set
        holdout_wavs = set()
        if exclude_holdout:
            import json
            holdout_path = os.path.join(os.path.dirname(__file__), 'test_holdout.json')
            if os.path.exists(holdout_path):
                with open(holdout_path) as f:
                    holdout = json.load(f)
                for item in holdout.get('kaggle_labeled', []):
                    holdout_wavs.add(os.path.basename(item['path']))

        audio_dir = os.path.join(root, 'audio')
        skipped = 0
        with open(csv_path) as f:
            for row in csv.reader(f):
                if row[0] == 'ID':
                    continue
                if len(row) > 1 and row[1].strip():
                    idx  = int(row[0])
                    fname = f'cw{idx:03d}.wav'
                    if fname in holdout_wavs:
                        skipped += 1
                        continue
                    path = os.path.join(audio_dir, fname)
                    if os.path.exists(path):
                        self.samples.append((path, row[1].strip().upper()))
        print(f'KaggleDataset: {len(self.samples)} labeled clips'
              f' (skipped {skipped} holdout)', flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, transcript = self.samples[i % len(self.samples)]
        audio = _load_audio(path)
        chunk_samples = CHUNK_SECONDS * SAMPLE_RATE

        if len(audio) > chunk_samples:
            # Random 8s window; assign transcript using approximate Morse timing.
            start = random.randint(0, len(audio) - chunk_samples)
            frac_s = start / len(audio)
            frac_e = (start + chunk_samples) / len(audio)
            sub = _slice_transcript_by_fraction(transcript, frac_s, frac_e)
            audio = audio[start: start + chunk_samples]
            transcript = sub

        return _audio_to_item(audio, transcript)


# ─────────────────────────────────────────────────────────────
# ARRLDataset
# ─────────────────────────────────────────────────────────────

class ARRLDataset(Dataset):
    """
    ARRL W1AW paired MP3+TXT files (data/arrl/5wpm/, 15wpm/, 20wpm/).
    Each ~16-minute file is sliced into CHUNK_SECONDS segments.
    Text is assigned proportionally based on estimated character rate at known WPM.
    Audio is cached in memory after first load (30 files × ~16MB each ≈ 480MB RAM).
    """

    def __init__(self, root: str = 'data/arrl'):
        self.chunks = []        # (path, wpm, frac_start, frac_end, text_slice)
        self._cache: dict = {}  # path → np.ndarray

        if not os.path.exists(root):
            return

        for wpm_dir in sorted(os.listdir(root)):
            m = re.match(r'(\d+)wpm', wpm_dir)
            if not m:
                continue
            wpm = float(m.group(1))
            dir_path = os.path.join(root, wpm_dir)

            for fname in sorted(os.listdir(dir_path)):
                if not fname.endswith('.mp3'):
                    continue
                base     = fname[:-4]
                mp3_path = os.path.join(dir_path, fname)
                txt_path = os.path.join(dir_path, base + '.txt')
                if not os.path.exists(txt_path):
                    continue

                with open(txt_path, encoding='utf-8', errors='replace') as f:
                    text = _clean_arrl_text(f.read())
                if not text:
                    continue

                # Build chunk list — each chunk = CHUNK_SECONDS of audio
                audio = _load_audio(mp3_path)
                audio_duration_s = len(audio) / SAMPLE_RATE
                n_chunks = max(1, int(np.ceil(audio_duration_s / CHUNK_SECONDS)))
                for k in range(n_chunks):
                    frac_s = k / n_chunks
                    frac_e = (k + 1) / n_chunks
                    sub = _slice_transcript_by_fraction(text, frac_s, frac_e)
                    if not sub:
                        continue
                    self.chunks.append((mp3_path, wpm, frac_s, frac_e, sub))

        print(f'ARRLDataset: {len(self.chunks)} chunks from {root}', flush=True)

    def __len__(self):
        return len(self.chunks)

    def _get_audio(self, path: str) -> np.ndarray:
        if path not in self._cache:
            self._cache[path] = _load_audio(path)
        return self._cache[path]

    def __getitem__(self, i):
        mp3_path, wpm, frac_s, frac_e, text = self.chunks[i % len(self.chunks)]
        audio = self._get_audio(mp3_path)

        start = int(frac_s * len(audio))
        end   = int(frac_e * len(audio))
        chunk = audio[start: start + CHUNK_SECONDS * SAMPLE_RATE]

        return _audio_to_item(chunk, text)


# ─────────────────────────────────────────────────────────────
# MixedDataset
# ─────────────────────────────────────────────────────────────

class MixedDataset(Dataset):
    """
    Weighted mix:
        40% synthetic (MorseDataset)
        20% Kaggle labeled real clips  (ground-truth labels)
        20% Kaggle pseudo-labeled clips (DSP labels, noise-augmented)
        20% ARRL DSP-labeled chunks    (ground-truth labels, noise-augmented)

    total_size sets the virtual epoch length.
    """

    def __init__(
        self,
        total_size:    int   = 30_000,
        ratio_synth:   float = 0.20,
        ratio_kaggle:  float = 0.15,
        ratio_kpseudo: float = 0.15,
        ratio_arrl:    float = 0.50,
    ):
        n_synth    = int(total_size * ratio_synth)
        n_kaggle   = int(total_size * ratio_kaggle)
        n_kpseudo  = int(total_size * ratio_kpseudo)
        n_arrl     = total_size - n_synth - n_kaggle - n_kpseudo

        from data.arrl_labeled_dataset import ARRLLabeledDataset
        self.synth    = MorseDataset(size=n_synth)
        self.kaggle   = KaggleDataset()
        self.kpseudo  = ARRLLabeledDataset(root='data/kaggle_labeled', augment_prob=0.6,
                                           min_wpm=0, exclude_holdout=False)
        self.arrl     = ARRLLabeledDataset(root='data/arrl_labeled',   augment_prob=0.5)

        # Build shuffled index: (source, local_idx)
        index = [('synth', i) for i in range(n_synth)]

        if len(self.kaggle) > 0:
            index += [('kaggle',  i % len(self.kaggle))  for i in range(n_kaggle)]
        else:
            index += [('synth', random.randint(0, n_synth - 1)) for _ in range(n_kaggle)]

        if len(self.kpseudo) > 0:
            index += [('kpseudo', i % len(self.kpseudo)) for i in range(n_kpseudo)]
        else:
            index += [('synth', random.randint(0, n_synth - 1)) for _ in range(n_kpseudo)]

        if len(self.arrl) > 0:
            index += [('arrl',    i % len(self.arrl))    for i in range(n_arrl)]
        else:
            index += [('synth', random.randint(0, n_synth - 1)) for _ in range(n_arrl)]

        random.shuffle(index)
        self.index = index

        print(
            f'MixedDataset: {n_synth} synth + {n_kaggle} kaggle-gt + '
            f'{n_kpseudo} kaggle-dsp + {n_arrl} arrl = {len(index)} total',
            flush=True,
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        source, idx = self.index[i]
        if source == 'synth':
            return self.synth[idx]
        elif source == 'kaggle':
            return self.kaggle[idx]
        elif source == 'kpseudo':
            return self.kpseudo[idx]
        else:
            return self.arrl[idx]
