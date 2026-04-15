# MorseAI

MorseAI is an end-to-end Morse code audio transcription project built around a `CNN -> Transformer -> CTC` model, a DSP auto-labeling pipeline, and a mixed real/synthetic training workflow.

This is best presented as an ML engineering and research pipeline, not as a production-ready Morse decoder.

## What It Does

- trains a neural transcription model on 64-band log-mel spectrograms at 8 kHz
- bootstraps real-data labels with a DSP decoder and alignment pipeline
- mixes synthetic CW, ARRL broadcasts, and Kaggle Morse datasets
- evaluates on fixed holdout and source-level benchmark splits excluded from training
- provides command-line and demo inference paths

## What Problem It Tries To Solve

The project targets a hard problem: transcribing noisy Morse audio where signal conditions, sending style, spacing, and recording quality vary materially across sources.

The engineering challenge is not just model training. It also includes:

- getting usable labels for messy real-world data
- avoiding train/test contamination
- combining synthetic and real audio without misleading benchmark results
- measuring performance honestly across unseen sources

## Current Honest Status

Latest saved benchmark result from `benchmark/results/holdout_20260325_172015.json`, using the fixed 5-fold ARRL source benchmark in `benchmark/source_benchmark.json`:

| Set | CER | WER |
| --- | ---: | ---: |
| Kaggle real QSO | 0.4732 | 0.8675 |
| ARRL source benchmark | 0.2796 | 0.4055 |
| Kaggle v1 clean holdout | 0.4534 | 2.5172 |
| Overall | 0.3005 | - |

ARRL fold variance on the fixed benchmark:

- `fold_1`: CER `0.2708`
- `fold_2`: CER `0.2559`
- `fold_3`: CER `0.2278`
- `fold_4`: CER `0.2712`
- `fold_5`: CER `0.3753`

Takeaway:

- the model learned Morse structure
- it performs materially better on cleaner or more in-distribution sources
- real ham-radio/QSO decoding is still weak
- source variance remains high enough that this should be treated as a research system, not a dependable decoder

## Main Problems And Limitations

- real-data transcription quality is still not strong enough for serious operational use
- benchmark results vary significantly by source
- synthetic-to-real transfer is still incomplete
- auto-labeling helps bootstrap data but can also introduce error
- the project depends on careful dataset hygiene to avoid misleading performance claims


## AI-Assisted Development Note

This project was built with AI assistance.

Part of the point of the project was to test a practical question:

> can a developer with strong general engineering skills, but limited starting domain expertise in Morse decoding, use AI effectively enough to build a useful research pipeline and get non-trivial results?

My answer from this project is:

- AI can accelerate exploration, implementation speed, and iteration
- AI can help bridge gaps in unfamiliar domains
- useful results are possible without deep initial domain expertise
- but domain understanding still becomes necessary to evaluate data quality, interpret benchmark results, catch contamination issues, and make sound decisions about what the model is actually learning


## Architecture

- Input: 64-band log-mel spectrogram
- Sample rate: 8 kHz
- Acoustic model: CNN front-end + Transformer encoder
- Loss: CTC
- Decoder: greedy CTC, optional beam search without an external language model

Core files:

- `models/model.py`
- `training/finetune.py`
- `data/build_source_benchmark.py`
- `data/label_kaggle.py`
- `data/arrl_labeled_dataset.py`
- `benchmark/eval_holdout.py`
- `inference/transcribe.py`

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run transcription on a file:

```bash
python inference/transcribe.py --checkpoint checkpoints/best_model.pt --file path/to/audio.wav
```

Run the holdout evaluator:

```bash
python benchmark/eval_holdout.py --checkpoint checkpoints/best_model.pt --normalize-wpm
```

Rebuild the fixed ARRL benchmark manifest:

```bash
python data/build_source_benchmark.py
```

Launch the demo:

```bash
python inference/demo.py
```

