"""
Synthetic Morse audio data generator.
Produces (waveform, label) pairs for training.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.morse_map import text_to_morse_timing, CHAR_TO_IDX, VOCAB
from utils.wpm import normalize_to_wpm, TARGET_WPM

SAMPLE_RATE = 8000  # 8kHz is plenty for Morse tones

WORDLIST = [
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    "HELLO WORLD", "SOS", "CQ CQ DE W1AW",
    "MORSE CODE IS FUN", "PYTHON IS GREAT",
    "NEURAL NETWORK", "DEEP LEARNING",
    "ALPHA BRAVO CHARLIE DELTA ECHO FOXTROT",
    "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE ZERO",
]

# Extended word pool for random generation
WORDS = (
    # Common English words
    "THE AND FOR ARE BUT NOT YOU ALL CAN HER WAS ONE OUR OUT DAY GET HAS HIM HIS HOW ITS "
    "NEW NOW OLD SEE TWO WAY WHO BOY DID ITS LET PUT SAY SHE TOO USE "
    "ABOUT AFTER AGAIN ALSO BACK BEEN BEFORE BEING BETWEEN BOTH CAME COME COULD "
    "EACH EVEN FIRST FROM GOOD GREAT HAD HAVE INTO JUST KNOW LAST LIKE LONG LOOK "
    "MADE MAKE MANY MORE MOST MUCH MUST NAME NEVER NEXT ONLY OTHER OVER PART PLACE "
    "SAME STILL SUCH TAKE TELL THAN THAT THEM THEN THERE THESE THEY THIS TIME VERY "
    "WANT WELL WENT WERE WHAT WHEN WHERE WHICH WHILE WILL WITH WORD WORK WOULD YEAR "
    # Ham radio words
    "HELLO WORLD MORSE CODE RADIO SIGNAL TEST MESSAGE SEND RECEIVE TRANSMIT "
    "ANTENNA BAND FREQUENCY POWER STATION CONTACT OPERATOR RELAY CHANNEL COPY "
    "ROGER WILCO BREAK OVER CLEAR CONFIRM REPEAT CORRECTION NEGATIVE AFFIRMATIVE "
    "EQUIPMENT TRANSCEIVER AMPLIFIER OSCILLATOR MODULATION INTERFERENCE "
    # NATO phonetic alphabet
    "ALPHA BRAVO CHARLIE DELTA ECHO FOXTROT GOLF HOTEL INDIA JULIET KILO LIMA "
    "MIKE NOVEMBER OSCAR PAPA QUEBEC ROMEO SIERRA TANGO UNIFORM VICTOR WHISKEY XRAY YANKEE ZULU "
    # Callsign-style tokens (letters + digits)
    "W1AW K5ABC N3XYZ WA2BCD KE7QRS VE3ABC JA1XYZ G4ABC DL5XY F6ABC "
    "W2 K9 N1 W7 K4 N6 W3 K8 N5 W6 "
    # Signal reports and numbers
    "599 559 579 589 569 539 449 339 "
    "73 88 55 33 44 "
    "1 2 3 4 5 6 7 8 9 0 "
    "10 15 20 25 30 35 40 50 75 100 "
    "14 21 28 7 3 144 430 "
    # QSO shorthand
    "QTH QSL QRZ QRM QRN QSB QSY QRT QRV QRP QRO RST TNX TU "
    "CQ DE ES UR HR FB OM SK CL DX NR PSE AGN BK "
    # Common text content (ARRL bulletin style)
    "AMATEUR BULLETIN PRACTICE COMMISSION FEDERAL COMMUNICATIONS "
    "REGULATIONS ELECTRONIC TECHNICAL ENGINEERING MEMBERSHIP ASSOCIATION "
    "EMERGENCY OPERATIONS CENTER VOLUNTEER EXAMINATION COORDINATOR "
    "PROPAGATION IONOSPHERE SUNSPOT MAGNETIC SOLAR FORECAST "
).split()


MAX_MEL_FRAMES = 2000   # ~8s at 8kHz/hop32 — keeps attention memory manageable

def _random_callsign() -> str:
    """Generate a random ham radio callsign like W1ABC, KE5XYZ."""
    prefixes = ['W', 'K', 'N', 'WA', 'WB', 'KA', 'KB', 'KE', 'KD', 'VE', 'JA', 'G', 'DL', 'F']
    prefix = random.choice(prefixes)
    digit = str(random.randint(0, 9))
    suffix_len = random.randint(1, 3)
    suffix = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=suffix_len))
    return prefix + digit + suffix


def _random_digits() -> str:
    """Generate a random digit string (signal report, frequency, etc.)."""
    templates = [
        lambda: f"{random.randint(1,5)}{random.randint(1,9)}{random.randint(1,9)}",  # RST report
        lambda: str(random.randint(1, 999)),  # generic number
        lambda: f"{random.choice([3,7,14,21,28])}.{random.randint(0,999):03d}",  # frequency
        lambda: f"{random.randint(1,31)} {random.choice(['JANUARY','FEBRUARY','MARCH','APRIL','MAY','JUNE','JULY','AUGUST','SEPTEMBER','OCTOBER','NOVEMBER','DECEMBER'])}",
    ]
    return random.choice(templates)()


def random_sentence(min_words=2, max_words=5):
    n = random.randint(min_words, max_words)
    parts = []
    for _ in range(n):
        r = random.random()
        if r < 0.10:
            parts.append(_random_callsign())
        elif r < 0.15:
            parts.append(_random_digits())
        else:
            parts.append(random.choice(WORDS))
    return ' '.join(parts)


def _qsb_envelope(n: int, sample_rate: int) -> np.ndarray:
    """QSB (fading): slow sinusoidal amplitude modulation."""
    fade_hz = random.uniform(0.3, 2.0)   # fade rate in Hz
    depth   = random.uniform(0.3, 0.8)   # how deep the fade goes (0=no fade, 1=full)
    t = np.arange(n) / sample_rate
    env = 1.0 - depth * (0.5 - 0.5 * np.cos(2 * np.pi * fade_hz * t))
    return env.astype(np.float32)


def _qrm_interference(n: int, sample_rate: int, signal_peak: float) -> np.ndarray:
    """QRM: another carrier or CW signal bleeding in at a different frequency."""
    qrm_freq  = random.uniform(200, 1200)
    qrm_amp   = random.uniform(0.05, 0.4) * signal_peak
    t = np.arange(n) / sample_rate
    return (qrm_amp * np.sin(2 * np.pi * qrm_freq * t)).astype(np.float32)


def _bandpass_filter(audio: np.ndarray, freq: float, sample_rate: int) -> np.ndarray:
    """Simulate a narrow receiver bandpass filter centred on the signal."""
    from scipy.signal import butter, sosfilt
    bw   = random.uniform(150, 500)      # filter bandwidth in Hz
    low  = max(50,  freq - bw / 2)
    high = min(3900, freq + bw / 2)
    sos  = butter(4, [low, high], btype='band', fs=sample_rate, output='sos')
    return sosfilt(sos, audio).astype(np.float32)


def _pink_noise(n: int) -> np.ndarray:
    """Pink noise via fast Voss algorithm — realistic radio static, no FFT."""
    # Simple IIR approximation: sum of white noise at different sample rates
    octaves = 8
    out = np.zeros(n, dtype=np.float32)
    for _ in range(octaves):
        stride = max(1, n // (2 ** random.randint(1, octaves)))
        vals   = np.random.randn(n // stride + 2).astype(np.float32)
        idx    = np.arange(n) // stride
        idx    = np.clip(idx, 0, len(vals) - 1)
        out   += vals[idx]
    peak = np.max(np.abs(out))
    if peak > 0:
        out /= peak
    return out


def _add_harmonics(audio: np.ndarray, freq: float, sample_rate: int, n: int) -> np.ndarray:
    """Add harmonic distortion — real transmitters aren't pure sine waves."""
    t = np.arange(n) / sample_rate
    result = audio.copy()
    # 2nd harmonic (strongest), 3rd harmonic (weaker)
    result[:n] += random.uniform(0.05, 0.15) * np.sin(2 * np.pi * 2 * freq * t).astype(np.float32)
    result[:n] += random.uniform(0.01, 0.05) * np.sin(2 * np.pi * 3 * freq * t).astype(np.float32)
    return result


def _soft_clip(audio: np.ndarray) -> np.ndarray:
    """Soft clipping — simulates AGC/compression in radio receivers."""
    threshold = random.uniform(0.5, 0.85)
    return np.tanh(audio / threshold).astype(np.float32)


def _chirp_on_keydown(audio: np.ndarray, events: list, freq: float,
                      sample_rate: int) -> np.ndarray:
    """Simulate transmitter chirp: frequency shifts on key-down and settles."""
    audio = audio.copy()
    chirp_hz = random.uniform(5, 30)       # how far off-freq the TX starts
    settle_ms = random.uniform(3, 15)      # how fast it settles
    settle_samples = int(settle_ms / 1000 * sample_rate)

    pos = 0
    for duration, is_tone in events:
        n = int(duration * sample_rate)
        if is_tone and n > 0 and settle_samples > 0:
            t = np.arange(min(settle_samples, n)) / sample_rate
            # Exponential chirp decay — starts off-freq, settles to nominal
            chirp_env = chirp_hz * np.exp(-t / (settle_ms / 3000))
            chirp_phase = 2 * np.pi * chirp_env * t
            # Subtract the clean tone and add chirped version for the settle period
            s = min(settle_samples, n)
            t_full = np.arange(s) / sample_rate
            clean = np.sin(2 * np.pi * freq * t_full).astype(np.float32)
            chirped = np.sin(2 * np.pi * freq * t_full + chirp_phase[:s]).astype(np.float32)
            end = min(pos + s, len(audio))
            actual = end - pos
            audio[pos:end] += (chirped[:actual] - clean[:actual]) * 0.5
        pos += n
    return audio


def _multipath_echo(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Simulate ionospheric multipath: delayed copy of signal mixed in."""
    delay_ms = random.uniform(0.5, 5.0)    # path delay in ms
    delay_samples = int(delay_ms / 1000 * sample_rate)
    attenuation = random.uniform(0.1, 0.5)  # echo is weaker
    # Slight frequency offset on the echo (different ionospheric path)
    freq_shift = random.uniform(-3, 3)

    result = audio.copy()
    if delay_samples < len(audio):
        echo = audio[:-delay_samples] * attenuation
        # Apply tiny frequency shift to echo
        if abs(freq_shift) > 0.1:
            t = np.arange(len(echo)) / sample_rate
            echo = echo * np.cos(2 * np.pi * freq_shift * t).astype(np.float32)
        result[delay_samples:] += echo
    return result


def _agc_pumping(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Simulate AGC: gain increases during silence, drops when signal appears."""
    # Compute signal envelope via RMS in short windows
    win = int(0.01 * sample_rate)  # 10ms windows
    n = len(audio)
    envelope = np.zeros(n, dtype=np.float32)
    for i in range(0, n, win):
        chunk = audio[i:i+win]
        envelope[i:i+win] = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0

    # Smooth envelope
    from scipy.ndimage import uniform_filter1d
    envelope = uniform_filter1d(envelope, size=int(0.05 * sample_rate))

    # AGC: invert envelope to boost quiet parts
    agc_speed = random.uniform(0.3, 0.8)
    noise_boost = random.uniform(0.1, 0.4)
    gain = 1.0 - agc_speed * np.clip(envelope / (np.max(envelope) + 1e-8), 0, 1)
    gain = np.clip(gain, noise_boost, 1.0)

    # Add noise that scales with gain (AGC boosts noise floor during gaps)
    agc_noise = np.random.randn(n).astype(np.float32) * 0.05 * gain
    return (audio * gain + agc_noise).astype(np.float32)


def _power_hum(n: int, sample_rate: int) -> np.ndarray:
    """50/60Hz power line hum with harmonics."""
    hum_freq = random.choice([50.0, 60.0])
    amp = random.uniform(0.02, 0.10)
    t = np.arange(n) / sample_rate
    hum = amp * np.sin(2 * np.pi * hum_freq * t)
    # Add 2nd and 3rd harmonics
    hum += amp * 0.3 * np.sin(2 * np.pi * 2 * hum_freq * t)
    hum += amp * 0.1 * np.sin(2 * np.pi * 3 * hum_freq * t)
    return hum.astype(np.float32)


def _relay_clicks(n: int, sample_rate: int, events: list) -> np.ndarray:
    """Simulate T/R relay clicks at start/end of transmission."""
    clicks = np.zeros(n, dtype=np.float32)
    click_amp = random.uniform(0.05, 0.20)
    click_dur = int(0.002 * sample_rate)  # 2ms click

    pos = 0
    prev_tone = False
    for duration, is_tone in events:
        samples = int(duration * sample_rate)
        # Click on tone→silence and silence→tone transitions
        if is_tone != prev_tone and random.random() < 0.3:
            click_start = max(0, min(pos, n - click_dur))
            click_end = min(click_start + click_dur, n)
            clicks[click_start:click_end] += click_amp * np.random.randn(
                click_end - click_start).astype(np.float32)
        prev_tone = is_tone
        pos += samples
    return clicks


def _ionospheric_flutter(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Rapid amplitude and phase fluctuations from ionospheric turbulence."""
    n = len(audio)
    t = np.arange(n) / sample_rate
    # Random sum of fast modulation frequencies (5-25 Hz)
    flutter = np.ones(n, dtype=np.float32)
    n_components = random.randint(2, 5)
    for _ in range(n_components):
        f = random.uniform(5, 25)
        depth = random.uniform(0.05, 0.20)
        phase = random.uniform(0, 2 * np.pi)
        flutter += depth * np.sin(2 * np.pi * f * t + phase).astype(np.float32)
    flutter = np.clip(flutter, 0.1, 2.0)
    return (audio * flutter).astype(np.float32)


def _apply_timing_jitter(events: list, jitter: float = 0.15) -> list:
    """
    Simulate human operator keying by adding per-element timing variation.

    jitter: std-dev of relative timing error (0.15 = ±15% per element).
    Also adds slow WPM drift across the transmission and a 'fist' bias
    (operators tend to send dahs slightly long and gaps slightly short).
    """
    # Slow global drift: WPM drifts ±8% over the transmission
    n = len(events)
    drift = np.linspace(1.0, random.uniform(0.92, 1.08), n)

    # 'Fist' bias: tendency to send dahs ~10% long, gaps ~5% short
    fist_dah  = random.uniform(1.0, 1.12)
    fist_gap  = random.uniform(0.90, 1.0)

    result = []
    for i, (dur, is_tone) in enumerate(events):
        # Per-element multiplicative jitter
        scale = np.random.normal(1.0, jitter)
        scale = float(np.clip(scale, 0.6, 1.6))  # prevent extreme outliers

        # Apply drift and fist bias
        scale *= drift[i]
        if is_tone and dur > 0.1:  # dahs (dahs are > 2x dits, roughly)
            scale *= fist_dah
        elif not is_tone:
            scale *= fist_gap

        result.append((max(dur * scale, 0.001), is_tone))
    return result


def synthesize_morse_audio(
    text: str,
    sample_rate: int = SAMPLE_RATE,
    wpm: float = None,
    freq: float = None,
    noise_level: float = None,
    rise_ms: float = None,
    augment: bool = True,
) -> np.ndarray:
    """Render Morse code text as a numpy audio waveform with real-world augmentations."""
    wpm         = wpm         or random.uniform(5, 40)
    freq        = freq        or random.uniform(400, 900)
    noise_level = noise_level if noise_level is not None else random.uniform(0.0, 0.6)
    rise_ms     = rise_ms     or random.uniform(3, 10)

    events = text_to_morse_timing(text, wpm=wpm)

    # Timing jitter: simulate human operator (70% of augmented samples)
    if augment and random.random() < 0.70:
        jitter_amount = random.uniform(0.05, 0.20)  # mild to heavy hand-keying
        events = _apply_timing_jitter(events, jitter=jitter_amount)
    if not events:
        return np.zeros(sample_rate, dtype=np.float32)

    total_samples = int(sum(d for d, _ in events) * sample_rate) + sample_rate // 4
    audio = np.zeros(total_samples, dtype=np.float32)

    rise_samples = int(rise_ms / 1000 * sample_rate)
    pos = 0
    for duration, is_tone in events:
        n = int(duration * sample_rate)
        if is_tone and n > 0:
            t = np.arange(n) / sample_rate
            tone = np.sin(2 * np.pi * freq * t).astype(np.float32)
            # Cosine rise/fall to eliminate clicks
            env = np.ones(n, dtype=np.float32)
            r = min(rise_samples, n // 2)
            if r > 0:
                ramp = (1 - np.cos(np.pi * np.arange(r) / r)) / 2
                env[:r]  = ramp
                env[-r:] = ramp[::-1]
            tone *= env
            end = min(pos + n, total_samples)
            audio[pos:end] += tone[:end - pos]
        pos += n

    # ── Normalize ────────────────────────────────────────────────────────────
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak

    if augment:
        # ── Chirp on key-down (40% chance) — TX frequency shift on keying ────
        if random.random() < 0.4:
            audio = _chirp_on_keydown(audio, events, freq, sample_rate)

        # ── Harmonic distortion (40% chance) — real transmitters aren't pure sine
        if random.random() < 0.4:
            n_samples = len(audio)
            audio = _add_harmonics(audio, freq, sample_rate, n_samples)

        # ── QSB fading (60% chance) ──────────────────────────────────────────
        if random.random() < 0.6:
            audio *= _qsb_envelope(len(audio), sample_rate)

        # ── Ionospheric flutter (30% chance) — rapid amplitude fluctuations ──
        if random.random() < 0.3:
            audio = _ionospheric_flutter(audio, sample_rate)

        # ── Multipath echo (35% chance) — ionospheric signal reflection ──────
        if random.random() < 0.35:
            audio = _multipath_echo(audio, sample_rate)

        # ── Bandpass filter (60% chance) — simulates receiver selectivity ────
        if random.random() < 0.6:
            try:
                audio = _bandpass_filter(audio, freq, sample_rate)
            except Exception:
                pass

        # ── QRM interference (40% chance) ────────────────────────────────────
        if random.random() < 0.4:
            audio += _qrm_interference(len(audio), sample_rate, signal_peak=1.0)

        # ── Multi-signal QRM (25% chance) — 2-3 overlapping CW signals ──────
        if random.random() < 0.25:
            n_interferers = random.randint(1, 3)
            for _ in range(n_interferers):
                audio += _qrm_interference(len(audio), sample_rate,
                                           signal_peak=random.uniform(0.3, 0.8))

        # ── Constant noise floor (70% chance) — real radio always has noise ──
        if random.random() < 0.7:
            floor_level = random.uniform(0.02, 0.15)
            audio += np.random.randn(len(audio)).astype(np.float32) * floor_level

        # ── Pink noise (realistic radio static) ──────────────────────────────
        if noise_level > 0:
            if random.random() < 0.5:
                audio += _pink_noise(len(audio)) * noise_level
            else:
                audio += np.random.randn(len(audio)).astype(np.float32) * noise_level

        # ── Power line hum (20% chance) ──────────────────────────────────────
        if random.random() < 0.2:
            audio += _power_hum(len(audio), sample_rate)

        # ── Relay clicks (15% chance) ────────────────────────────────────────
        if random.random() < 0.15:
            audio += _relay_clicks(len(audio), sample_rate, events)

        # ── AGC pumping (30% chance) — gain shifts between signal and gaps ───
        if random.random() < 0.3:
            try:
                audio = _agc_pumping(audio, sample_rate)
            except Exception:
                pass

        # ── Soft clipping / AGC compression (30% chance) ─────────────────────
        if random.random() < 0.3:
            audio = _soft_clip(audio)

        # ── Pitch drift (50% chance, slightly wider range) ────────────────────
        if random.random() < 0.5:
            drift = random.uniform(-0.01, 0.01)   # ±1% speed variation
            old_len = len(audio)
            new_len = int(old_len * (1 + drift))
            audio = np.interp(
                np.linspace(0, old_len - 1, new_len),
                np.arange(old_len),
                audio,
            ).astype(np.float32)

        # Re-normalise after augmentation
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio /= peak
    else:
        if noise_level > 0:
            audio += np.random.randn(len(audio)).astype(np.float32) * noise_level

    return audio.astype(np.float32)


def audio_to_melspec(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Convert waveform to log-mel spectrogram tensor [n_mels, time]."""
    import torchaudio.transforms as T
    wav = torch.from_numpy(audio).unsqueeze(0)  # [1, samples]
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=256,
        hop_length=32,
        n_mels=64,
        f_min=200,
        f_max=4000,
    )
    mel = mel_transform(wav)          # [1, n_mels, time]
    mel = torch.log1p(mel).squeeze(0) # [n_mels, time]
    return mel


def encode_label(text: str) -> list[int]:
    """Encode text string as list of vocab indices (skip unknowns)."""
    indices = []
    for ch in text.upper():
        if ch in CHAR_TO_IDX:
            indices.append(CHAR_TO_IDX[ch])
    return indices


class MorseDataset(Dataset):
    def __init__(self, size: int = 50_000, fixed_sentences: bool = False):
        self.size = size
        self.fixed_sentences = fixed_sentences
        if fixed_sentences:
            self.sentences = [random_sentence() for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        text = self.sentences[idx] if self.fixed_sentences else random_sentence()
        # At very slow speeds a long sentence won't fit in MAX_MEL_FRAMES — use fewer words
        wpm  = random.uniform(5, 40)
        if wpm < 10:
            text = random_sentence(min_words=1, max_words=3)
        audio = synthesize_morse_audio(text, wpm=wpm)
        # 50% of samples: normalize to TARGET_WPM (speed-invariant training)
        # 50% of samples: keep raw speed (model learns to handle variable speed)
        if random.random() < 0.5:
            audio = normalize_to_wpm(audio, SAMPLE_RATE, known_wpm=wpm)
        mel   = audio_to_melspec(audio)          # [64, T]
        # Trim to max length to avoid OOM — also trim label to only include
        # characters whose audio fits within the truncated mel frames.
        if mel.shape[1] > MAX_MEL_FRAMES:
            kept_secs = MAX_MEL_FRAMES * 32 / SAMPLE_RATE   # hop_length=32
            kept_frac = kept_secs / (len(audio) / SAMPLE_RATE)
            kept_chars = max(1, int(len(text) * kept_frac))
            # Truncate at word boundary
            trunc = text[:kept_chars]
            last_space = trunc.rfind(' ')
            text = trunc[:last_space] if last_space > 0 else trunc.rstrip()
            mel   = mel[:, :MAX_MEL_FRAMES]
        label = encode_label(text)
        return mel, torch.tensor(label, dtype=torch.long), text


def collate_fn(batch):
    mels, labels, texts = zip(*batch)

    # Hard cap: prevent OOM from any oversized chunk slipping through
    mels = [m[:, :MAX_MEL_FRAMES] for m in mels]

    # Pad mels to max time length in batch
    max_t = max(m.shape[1] for m in mels)
    padded = torch.zeros(len(mels), mels[0].shape[0], max_t)
    input_lengths = []
    for i, m in enumerate(mels):
        t = m.shape[1]
        padded[i, :, :t] = m
        input_lengths.append(t)

    # Flatten labels for CTC
    label_lengths = [len(l) for l in labels]
    flat_labels   = torch.cat(labels)

    return (
        padded,                                    # [B, n_mels, T]
        flat_labels,                               # [sum of label lens]
        torch.tensor(input_lengths, dtype=torch.long),
        torch.tensor(label_lengths, dtype=torch.long),
        texts,
    )
