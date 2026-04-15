"""
WPM detection and time-normalization for Morse audio.

The core problem: a dit at 5 WPM is 8x longer than a dit at 40 WPM.
The model has to learn the same symbol at wildly different scales.
Solution: detect WPM from the audio, then time-stretch to a fixed target
WPM before computing the mel spectrogram.

detect_wpm(audio, sample_rate) -> float
normalize_to_wpm(audio, sample_rate, target_wpm) -> np.ndarray
"""
import numpy as np

TARGET_WPM   = 20.0   # everything gets stretched to this
WPM_MIN      = 3.0
WPM_MAX      = 80.0
WINDOW_MS    = 2      # energy envelope window — must be << 1 dit at max WPM
SMOOTH_K     = 2      # median filter half-width (windows) — keep < 1 dit at 40 WPM


def _energy_envelope(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Short-time RMS energy, one value per WINDOW_MS."""
    w = max(1, int(WINDOW_MS / 1000 * sample_rate))
    n = len(audio) // w
    if n == 0:
        return np.array([np.sqrt(np.mean(audio ** 2))])
    frames = audio[:n * w].reshape(n, w)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Simple 1-D median filter with edge padding."""
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - k)
        hi = min(len(x), i + k + 1)
        out[i] = np.median(x[lo:hi])
    return out


def detect_wpm(audio: np.ndarray, sample_rate: int) -> float:
    """
    Estimate WPM by finding the shortest 'on' pulse (= one dit).

    Returns a float in [WPM_MIN, WPM_MAX].
    Falls back to TARGET_WPM if detection is unreliable.
    """
    env = _energy_envelope(audio, sample_rate)
    env = _median_filter(env, SMOOTH_K)

    if env.max() == 0:
        return TARGET_WPM

    # Otsu-like threshold: use a percentile between noise floor and peak
    noise_floor = np.percentile(env, 20)
    peak        = np.percentile(env, 95)
    if peak <= noise_floor:
        return TARGET_WPM
    threshold = noise_floor + 0.35 * (peak - noise_floor)

    binary = (env > threshold).astype(np.int8)
    diff   = np.diff(binary, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]

    if len(starts) == 0 or len(ends) == 0:
        return TARGET_WPM

    # Pair up starts and ends
    pairs = list(zip(starts, ends[:len(starts)]))
    durations_sec = [(e - s) * WINDOW_MS / 1000 for s, e in pairs]

    if not durations_sec:
        return TARGET_WPM

    # Discard pulses shorter than minimum possible dit at WPM_MAX
    min_dit_sec = 1.2 / WPM_MAX
    durations_sec = [d for d in durations_sec if d >= min_dit_sec * 0.5]
    if not durations_sec:
        return TARGET_WPM

    # The shortest 10th percentile of on-pulse durations ≈ dit length
    # (robust: ignores dahs, which are 3x longer)
    dit_sec = float(np.percentile(durations_sec, 10))

    if dit_sec <= 0:
        return TARGET_WPM

    wpm = 1.2 / dit_sec
    return float(np.clip(wpm, WPM_MIN, WPM_MAX))


def time_stretch(audio: np.ndarray, factor: float) -> np.ndarray:
    """
    Resample audio by `factor` using linear interpolation.
    factor > 1 → audio gets longer (slowed down)
    factor < 1 → audio gets shorter (sped up)
    """
    if abs(factor - 1.0) < 0.01:
        return audio
    old_len = len(audio)
    new_len = max(1, int(round(old_len * factor)))
    return np.interp(
        np.linspace(0, old_len - 1, new_len),
        np.arange(old_len),
        audio,
    ).astype(np.float32)


def normalize_to_wpm(
    audio: np.ndarray,
    sample_rate: int,
    target_wpm: float = TARGET_WPM,
    known_wpm: float = None,
) -> np.ndarray:
    """
    Stretch audio so its effective WPM matches `target_wpm`.

    If `known_wpm` is provided (e.g. during synthetic training where we
    know the exact WPM), we skip detection and use that value directly.
    """
    src_wpm = known_wpm if known_wpm is not None else detect_wpm(audio, sample_rate)
    # Cap to ±2× to prevent catastrophic stretching when detection fails
    # (e.g. noisy clips where detect_wpm hits the 80 WPM cap → factor=4 → garbage)
    src_wpm = float(np.clip(src_wpm, target_wpm * 0.5, target_wpm * 2.0))
    # stretch factor: slower src → need to speed up (factor < 1)
    #                 faster src → need to slow down (factor > 1)
    factor = src_wpm / target_wpm
    return time_stretch(audio, factor)
