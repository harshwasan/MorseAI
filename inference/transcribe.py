"""
Inference: decode Morse audio file or live microphone → text.
Usage:
    python inference/transcribe.py --file path/to/audio.wav
    python inference/transcribe.py --mic          # live mic input
"""
import os, sys, argparse
from functools import lru_cache
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from models.model import MorseModel
from data.generate import audio_to_melspec, SAMPLE_RATE, MAX_MEL_FRAMES
from utils.morse_map import IDX_TO_CHAR
from utils.wpm import normalize_to_wpm


BLANK_IDX = 0


def load_model(checkpoint_path: str, device: torch.device) -> MorseModel:
    ckpt  = torch.load(checkpoint_path, map_location=device)
    args  = ckpt.get('args', {})
    model = MorseModel(
        n_mels=64,
        d_model=args.get('d_model', 256),
        nhead=args.get('nhead', 8),
        num_layers=args.get('num_layers', 4),
        dropout=0.0,
    ).to(device)
    state = ckpt['model_state']
    state.pop('pos_enc.pe', None)   # PE is deterministic — drop & let model recompute
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def greedy_decode(log_probs: torch.Tensor) -> str:
    """log_probs: [T, vocab_size]"""
    indices = log_probs.argmax(dim=-1).tolist()
    chars, prev = [], None
    for idx in indices:
        if idx != BLANK_IDX and idx != prev:
            chars.append(IDX_TO_CHAR.get(idx, ''))
        prev = idx
    return ''.join(chars)


def _ctc_collapse(indices: list) -> str:
    """Collapse CTC output: remove blanks and repeated chars."""
    chars, prev = [], None
    for idx in indices:
        if idx != BLANK_IDX and idx != prev:
            chars.append(IDX_TO_CHAR.get(idx, ''))
        prev = idx
    return ''.join(chars)


def beam_decode(log_probs: torch.Tensor, beam_width: int = 5) -> str:
    """
    CTC beam search using only acoustic scores (no LM).
    log_probs: [T, vocab_size] — log probabilities from the model.
    """
    T, V = log_probs.shape

    # Each beam: (log_prob, sequence_of_indices)
    beams = [(0.0, [])]

    for t in range(T):
        candidates = []
        probs_t = log_probs[t]  # [vocab_size]

        for score, seq in beams:
            # Top-k tokens at this timestep (prune early for speed)
            topk = min(beam_width, V)
            top_vals, top_idxs = probs_t.topk(topk)

            for i in range(topk):
                idx = top_idxs[i].item()
                new_score = score + top_vals[i].item()
                candidates.append((new_score, seq + [idx]))

        # Keep top beam_width candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

    # Return best beam, collapsed via CTC rules
    best_seq = beams[0][1]
    return _ctc_collapse(best_seq)


@lru_cache(maxsize=4)
def _get_lm_decoder(n: int = 3, alpha: float = 0.35):
    from inference.beam_search import build_lm
    return build_lm(n=n, alpha=alpha)


def lm_beam_decode(log_probs: torch.Tensor, beam_width: int = 16,
                   lm_order: int = 3, lm_alpha: float = 0.35) -> str:
    """
    CTC beam search with a lightweight Morse-domain character LM.
    """
    from inference.beam_search import ctc_beam_search
    lm = _get_lm_decoder(n=lm_order, alpha=lm_alpha)
    text = ctc_beam_search(log_probs, lm=lm, beam_width=beam_width)
    return ' '.join(text.split())


HOP_LENGTH = 32
# Chunk size derived from training limit so inference always matches training
CHUNK_SECONDS   = MAX_MEL_FRAMES * HOP_LENGTH / SAMPLE_RATE   # 2000*32/8000 = 8.0s
OVERLAP_SECONDS = 1     # 1s overlap to avoid cutting symbols at chunk boundaries

def decode_audio(audio: np.ndarray, model: MorseModel, device: torch.device,
                 normalize_wpm: bool = False, beam_width: int = 0,
                 decoder: str = 'greedy') -> str:
    """
    Decode audio to text.
    decoder:
      - greedy : argmax CTC decode
      - beam   : beam search with acoustic scores only
      - lm     : beam search with a lightweight character LM
    """
    if normalize_wpm:
        audio = normalize_to_wpm(audio, SAMPLE_RATE)

    chunk_samples   = int(CHUNK_SECONDS  * SAMPLE_RATE)
    overlap_samples = int(OVERLAP_SECONDS * SAMPLE_RATE)

    def _run_chunk(chunk_audio):
        mel = audio_to_melspec(chunk_audio).unsqueeze(0)
        mel = mel[:, :, :MAX_MEL_FRAMES].to(device)
        with torch.no_grad():
            log_probs = model(mel)
        lp = log_probs.squeeze(1).cpu()
        if decoder == 'lm':
            width = beam_width if beam_width > 0 else 16
            return lm_beam_decode(lp, beam_width=width)
        if decoder == 'beam' or beam_width > 0:
            width = beam_width if beam_width > 0 else 5
            return beam_decode(lp, beam_width=width)
        if decoder != 'greedy':
            raise ValueError(f"Unknown decoder: {decoder}")
        return greedy_decode(lp)

    if len(audio) <= chunk_samples:
        return _run_chunk(audio)

    parts, pos = [], 0
    while pos < len(audio):
        chunk = audio[pos: pos + chunk_samples]
        parts.append(_run_chunk(chunk).strip())
        pos += chunk_samples - overlap_samples

    return ' '.join(p for p in parts if p)


def decode_file(path: str, model: MorseModel, device: torch.device,
                normalize_wpm: bool = False, beam_width: int = 0,
                decoder: str = 'greedy') -> str:
    import subprocess, imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # Decode any audio format → raw 16-bit PCM mono at SAMPLE_RATE via ffmpeg pipe
    cmd = [
        ffmpeg, '-i', path,
        '-f', 's16le', '-ac', '1', '-ar', str(SAMPLE_RATE),
        '-loglevel', 'error', 'pipe:1',
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return decode_audio(
        audio, model, device,
        normalize_wpm=normalize_wpm, beam_width=beam_width, decoder=decoder
    )


def decode_mic(model: MorseModel, device: torch.device, duration: float = 10.0):
    """Record from microphone and decode."""
    import sounddevice as sd
    print(f"Recording {duration}s from microphone... (Ctrl+C to stop early)")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
    )
    sd.wait()
    audio = audio.flatten()
    result = decode_audio(audio, model, device)
    print(f"Decoded: {result}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--file',       type=str, help='Path to .wav file')
    parser.add_argument('--mic',        action='store_true', help='Use microphone')
    parser.add_argument('--duration',   type=float, default=10.0, help='Mic recording seconds')
    parser.add_argument('--text',       type=str,   help='Synthesize and decode this text (test)')
    parser.add_argument('--wpm',        type=float, default=20.0, help='WPM for --text synthesis')
    parser.add_argument('--beam',       type=int,   default=0,    help='Beam width (0=greedy, 5=recommended)')
    parser.add_argument('--decoder',    type=str,   default='greedy',
                        choices=['greedy', 'beam', 'lm'],
                        help='Decoder type')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}")

    if args.text:
        from data.generate import synthesize_morse_audio
        print(f"Synthesizing: '{args.text}'")
        audio  = synthesize_morse_audio(args.text, wpm=args.wpm, augment=False)
        result_greedy = decode_audio(audio, model, device)
        print(f"Target : {args.text.upper()}")
        print(f"Greedy : {result_greedy}")
        if args.decoder != 'greedy' or args.beam > 0:
            result_alt = decode_audio(audio, model, device, beam_width=args.beam, decoder=args.decoder)
            label = f"{args.decoder.upper()}-{args.beam}" if args.beam > 0 else args.decoder.upper()
            print(f"{label} : {result_alt}")

    elif args.file:
        result_greedy = decode_file(args.file, model, device)
        print(f"Greedy : {result_greedy}")
        if args.decoder != 'greedy' or args.beam > 0:
            result_alt = decode_file(args.file, model, device, beam_width=args.beam, decoder=args.decoder)
            label = f"{args.decoder.upper()}-{args.beam}" if args.beam > 0 else args.decoder.upper()
            print(f"{label} : {result_alt}")

    elif args.mic:
        decode_mic(model, device, duration=args.duration)

    else:
        print("Provide --file, --mic, or --text. Use --help for options.")


if __name__ == '__main__':
    main()
