"""
DSP-based Morse code decoder with word-level timing output.

Pipeline:
  1. FFT peak → detect carrier frequency automatically
  2. Short-time RMS envelope → signal strength over time
  3. Adaptive threshold → binary on/off signal
  4. Edge detection → list of (start_s, duration_s, is_tone) pulses
  5. Histogram of short pulses → estimate dit duration → WPM
  6. Classify pulses as dit/dah and gaps as element/char/word
  7. Build characters and words with start/end timestamps

Usage:
    from inference.dsp_decode import decode_audio_dsp
    words = decode_audio_dsp(audio, sample_rate=8000)
    # words = [{'word': 'HELLO', 'start_s': 1.2, 'end_s': 2.4}, ...]
"""
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.morse_map import MORSE_TO_CHAR


# ─────────────────────────────────────────────────────────────
# 1. Carrier frequency detection
# ─────────────────────────────────────────────────────────────

def detect_carrier(audio: np.ndarray, sample_rate: int,
                   f_min: float = 300.0, f_max: float = 1200.0) -> float:
    """Find dominant CW tone frequency via FFT magnitude peak."""
    # Use middle 10s of audio for stable estimate, avoid silences at edges
    mid = len(audio) // 2
    window = audio[mid - sample_rate * 5: mid + sample_rate * 5]
    if len(window) < sample_rate:
        window = audio

    fft_mag  = np.abs(np.fft.rfft(window))
    fft_freq = np.fft.rfftfreq(len(window), d=1.0 / sample_rate)

    mask = (fft_freq >= f_min) & (fft_freq <= f_max)
    if not np.any(mask):
        return 700.0

    peak_idx = np.argmax(fft_mag[mask])
    carrier  = fft_freq[mask][peak_idx]
    return float(carrier)


# ─────────────────────────────────────────────────────────────
# 2. Band-pass envelope extraction
# ─────────────────────────────────────────────────────────────

def _bandpass(audio: np.ndarray, center: float, bw: float,
              sample_rate: int) -> np.ndarray:
    """Simple IIR bandpass around the carrier."""
    from scipy.signal import butter, sosfilt
    lo = max(50.0, center - bw / 2)
    hi = min(sample_rate / 2 - 50.0, center + bw / 2)
    sos = butter(4, [lo, hi], btype='band', fs=sample_rate, output='sos')
    return sosfilt(sos, audio).astype(np.float32)


def envelope(audio: np.ndarray, sample_rate: int,
             carrier_hz: float, window_ms: float = 5.0) -> np.ndarray:
    """
    Short-time RMS envelope after bandpass filtering.
    Returns array of RMS values, one per window.
    """
    filtered   = _bandpass(audio, carrier_hz, bw=400.0, sample_rate=sample_rate)
    win_samples = max(1, int(window_ms / 1000.0 * sample_rate))

    n_frames = len(filtered) // win_samples
    frames   = filtered[:n_frames * win_samples].reshape(n_frames, win_samples)
    rms      = np.sqrt(np.mean(frames ** 2, axis=1))
    return rms


# ─────────────────────────────────────────────────────────────
# 3. Adaptive threshold → binary on/off
# ─────────────────────────────────────────────────────────────

def binarize(rms: np.ndarray, smooth_k: int = 3) -> np.ndarray:
    """
    Adaptive threshold: use rolling median with a percentile split.
    Returns bool array: True = tone on.
    """
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(rms.astype(np.float32), size=smooth_k)

    # Noise floor = 10th pct, signal peak = 90th pct
    lo = np.percentile(smoothed, 10)
    hi = np.percentile(smoothed, 90)
    threshold = lo + (hi - lo) * 0.35   # 35% of dynamic range

    return smoothed > threshold


# ─────────────────────────────────────────────────────────────
# 4. Edge detection → pulse list
# ─────────────────────────────────────────────────────────────

def get_pulses(on_off: np.ndarray, frame_duration_s: float) -> list:
    """
    Convert binary on/off array into list of pulses:
        [(start_s, duration_s, is_tone), ...]
    """
    pulses = []
    if len(on_off) == 0:
        return pulses

    current = bool(on_off[0])
    start   = 0
    for i in range(1, len(on_off)):
        if bool(on_off[i]) != current:
            dur = (i - start) * frame_duration_s
            pulses.append((start * frame_duration_s, dur, current))
            start   = i
            current = bool(on_off[i])
    # Last segment
    dur = (len(on_off) - start) * frame_duration_s
    pulses.append((start * frame_duration_s, dur, current))
    return pulses


# ─────────────────────────────────────────────────────────────
# 5. Dit length estimation from tone histogram
# ─────────────────────────────────────────────────────────────

def estimate_dit(pulses: list, min_ms: float = 10.0) -> float:
    """
    Histogram of tone durations → two clusters (dit, dah).
    The shorter cluster mean = dit duration in seconds.
    """
    tone_durs = [dur for (_, dur, is_tone) in pulses
                 if is_tone and dur >= min_ms / 1000.0]

    if not tone_durs:
        return 0.06  # fallback: 20 WPM

    tone_durs = np.array(tone_durs)

    # Simple approach: shortest 33% of tones are dits
    cutoff = np.percentile(tone_durs, 33)
    dits   = tone_durs[tone_durs <= cutoff * 1.5]
    return float(np.median(dits)) if len(dits) > 0 else float(np.median(tone_durs))


# ─────────────────────────────────────────────────────────────
# 6. Classify pulses and decode characters
# ─────────────────────────────────────────────────────────────

def classify_and_decode(pulses: list, dit_s: float) -> list:
    """
    Classify each pulse as dit/dah/element-gap/char-gap/word-gap.
    Returns list of word dicts:
        [{'word': str, 'start_s': float, 'end_s': float,
          'chars': [{'char': str, 'start_s': float, 'end_s': float}]}, ...]
    """
    # Thresholds (ratios of dit)
    # Tone: < 2 dit → dit, else → dah
    # Gap : < 2 dit → element gap, < 5 dit → char gap, else → word gap
    DAH_THRESH      = 2.0
    CHAR_GAP_THRESH = 2.0
    WORD_GAP_THRESH = 5.0

    words  = []
    cur_word_chars = []
    cur_pattern    = ''
    cur_char_start = None
    word_start     = None

    for start_s, dur_s, is_tone in pulses:
        ratio = dur_s / dit_s if dit_s > 0 else 1.0

        if is_tone:
            if cur_char_start is None:
                cur_char_start = start_s
            if word_start is None:
                word_start = start_s
            cur_pattern += '.' if ratio < DAH_THRESH else '-'

        else:  # gap
            if ratio < CHAR_GAP_THRESH:
                # Element gap — keep accumulating current character
                pass
            elif ratio < WORD_GAP_THRESH:
                # Character gap — emit current character
                if cur_pattern:
                    char = MORSE_TO_CHAR.get(cur_pattern, '?')
                    char_end = start_s
                    cur_word_chars.append({
                        'char':    char,
                        'pattern': cur_pattern,
                        'start_s': cur_char_start,
                        'end_s':   char_end,
                    })
                    cur_pattern    = ''
                    cur_char_start = None
            else:
                # Word gap — emit current character then current word
                if cur_pattern:
                    char = MORSE_TO_CHAR.get(cur_pattern, '?')
                    char_end = start_s
                    cur_word_chars.append({
                        'char':    char,
                        'pattern': cur_pattern,
                        'start_s': cur_char_start,
                        'end_s':   char_end,
                    })
                    cur_pattern    = ''
                    cur_char_start = None

                if cur_word_chars:
                    word_text = ''.join(c['char'] for c in cur_word_chars)
                    words.append({
                        'word':    word_text,
                        'start_s': word_start,
                        'end_s':   cur_word_chars[-1]['end_s'],
                        'chars':   cur_word_chars,
                    })
                    cur_word_chars = []
                    word_start     = None

    # Flush remaining character
    if cur_pattern:
        char = MORSE_TO_CHAR.get(cur_pattern, '?')
        cur_word_chars.append({
            'char':    char,
            'pattern': cur_pattern,
            'start_s': cur_char_start,
            'end_s':   cur_char_start + dit_s,
        })

    # Flush remaining word
    if cur_word_chars:
        word_text = ''.join(c['char'] for c in cur_word_chars)
        words.append({
            'word':    word_text,
            'start_s': word_start,
            'end_s':   cur_word_chars[-1]['end_s'],
            'chars':   cur_word_chars,
        })

    return words


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def decode_audio_dsp(audio: np.ndarray, sample_rate: int = 8000,
                     carrier_hz: float = None,
                     known_wpm: float = None) -> dict:
    """
    Full DSP decode of a Morse audio array.

    known_wpm: if provided, skip dit estimation and use the exact dit
               duration for the given WPM (1.2 / wpm seconds per dit).
               Always pass this when the WPM is known (e.g. ARRL files).

    Returns:
        {
          'words':      [{'word', 'start_s', 'end_s', 'chars'}, ...],
          'text':       'HELLO WORLD ...',
          'wpm':        float,
          'carrier_hz': float,
        }
    """
    if carrier_hz is None:
        carrier_hz = detect_carrier(audio, sample_rate)

    rms          = envelope(audio, sample_rate, carrier_hz, window_ms=5.0)
    frame_dur_s  = 5.0 / 1000.0
    on_off       = binarize(rms, smooth_k=3)
    pulses       = get_pulses(on_off, frame_dur_s)

    if known_wpm:
        dit_s = 1.2 / known_wpm   # exact: PARIS standard
        wpm   = known_wpm
    else:
        dit_s = estimate_dit(pulses)
        wpm   = 1.2 / dit_s if dit_s > 0 else 20.0

    words = classify_and_decode(pulses, dit_s)
    text  = ' '.join(w['word'] for w in words)

    return {
        'words':      words,
        'text':       text,
        'wpm':        round(wpm, 1),
        'carrier_hz': round(carrier_hz, 1),
        'dit_s':      round(dit_s, 4),
    }


def decode_file_dsp(path: str, sample_rate: int = 8000,
                    known_wpm: float = None) -> dict:
    """Load audio file and DSP-decode it."""
    import subprocess, imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, '-i', os.path.abspath(path),
        '-f', 's16le', '-ac', '1', '-ar', str(sample_rate),
        '-loglevel', 'error', 'pipe:1',
    ]
    raw   = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return decode_audio_dsp(audio, sample_rate, known_wpm=known_wpm)
