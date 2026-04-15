"""
Gradio web demo for real-time Morse audio decoding.
Usage: python inference/demo.py
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import gradio as gr
from data.generate import audio_to_melspec, synthesize_morse_audio, SAMPLE_RATE
from inference.transcribe import load_model, decode_audio, greedy_decode
from utils.morse_map import CHAR_TO_MORSE

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.environ.get('MODEL_PATH', 'checkpoints/best_model.pt')

model = None
def get_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH, device)
    return model


def decode_uploaded(audio_tuple):
    """Handle audio uploaded via Gradio (sr, numpy array)."""
    if audio_tuple is None:
        return "No audio provided."
    sr, data = audio_tuple
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    if data.ndim > 1:
        data = data.mean(axis=1)
    import torchaudio.functional as F
    import torch
    wav = torch.from_numpy(data).unsqueeze(0)
    wav = F.resample(wav, sr, SAMPLE_RATE)
    audio = wav.squeeze().numpy()
    result = decode_audio(audio, get_model(), device)
    return result or "(nothing decoded)"


def decode_text_morse(text: str) -> tuple[str, tuple]:
    """Synthesize Morse audio from text and decode it back."""
    if not text.strip():
        return "", None
    audio = synthesize_morse_audio(text.upper())
    result = decode_audio(audio, get_model(), device)
    # Return text result + audio for playback
    return result, (SAMPLE_RATE, audio)


def text_to_morse_symbols(text: str) -> str:
    parts = []
    for ch in text.upper():
        if ch == ' ':
            parts.append('/')
        elif ch in CHAR_TO_MORSE:
            parts.append(CHAR_TO_MORSE[ch])
    return ' '.join(parts)


with gr.Blocks(title="MorseAI Decoder") as demo:
    gr.Markdown("# MorseAI — Morse Code Audio Decoder")
    gr.Markdown("Upload a Morse audio file **or** type text to synthesize & decode.")

    with gr.Tab("Upload Audio"):
        audio_in  = gr.Audio(label="Morse Audio Input", type="numpy")
        decode_btn = gr.Button("Decode", variant="primary")
        decoded_out = gr.Textbox(label="Decoded Text")
        decode_btn.click(decode_uploaded, inputs=audio_in, outputs=decoded_out)

    with gr.Tab("Text → Morse → Decoded"):
        text_in   = gr.Textbox(label="Input Text", placeholder="HELLO WORLD")
        synth_btn = gr.Button("Synthesize & Decode", variant="primary")
        morse_sym = gr.Textbox(label="Morse Symbols")
        audio_out = gr.Audio(label="Synthesized Audio", type="numpy")
        text_out  = gr.Textbox(label="Model Output")

        def full_pipeline(text):
            symbols = text_to_morse_symbols(text)
            decoded, audio = decode_text_morse(text)
            return symbols, audio, decoded

        synth_btn.click(full_pipeline, inputs=text_in, outputs=[morse_sym, audio_out, text_out])

    gr.Markdown(f"Running on `{device}` | Model: `{MODEL_PATH}`")

if __name__ == '__main__':
    demo.launch(share=False, server_port=7860)
