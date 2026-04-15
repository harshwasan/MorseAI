# MorseAI Development Log

## Project Goal
Build a CNN+Transformer+CTC neural network to decode real Morse code audio from ham radio QSOs, ARRL W1AW broadcasts, and live radio — with real-world noise robustness.

---

## Architecture

- **Model**: CNN feature extractor → Transformer encoder → CTC head
- **Input**: 64-band log-mel spectrogram (8kHz, hop=32, n_fft=256)
- **Output**: CTC over character vocabulary (A-Z, 0-9, punctuation, space)
- **Loss**: CTC (Connectionist Temporal Classification)
- **MAX_MEL_FRAMES**: 2000 (~8s max input) — reduced from 3000 to prevent attention OOM
- **SAMPLE_RATE**: 8000 Hz

---

## What We Tried & Learned

### Phase 1 — Synthetic-Only Training
- Trained on `MorseDataset`: random sentences synthesized at 5–40 WPM with augmentation (QSB, QRM, bandpass, pink noise, harmonic distortion)
- **Problem**: Model output "OO OO" on all real audio — complete domain gap
- **Root cause**: Clean synthetic tones look nothing like real radio audio in mel space

### Phase 2 — DSP Decoder for Auto-Labeling
Built `inference/dsp_decode.py`:
1. FFT peak → detect carrier frequency (300–1200 Hz)
2. Short-time RMS envelope after bandpass filter
3. Adaptive threshold (35% of 10th–90th percentile range)
4. Edge detection → pulse list
5. Histogram of tone durations → estimate dit length → WPM
6. Classify pulses as dit/dah/gaps → characters → words with timestamps

**Key fix — Farnsworth timing**: W1AW 5 WPM broadcasts send characters at 15 WPM speed with stretched gaps. Passing `known_wpm=5` made DSP decode fail (dit_s = 0.24s too long). Fixed by using `char_wpm = 15.0 if wpm <= 5 else float(wpm)`.

DSP validation results:
- 15 WPM: CER = 0.011
- 20 WPM: CER = 0.10
- 5 WPM:  CER = 0.11 (after Farnsworth fix)

### Phase 3 — ARRL Labeling Pipeline
Built `data/label_arrl.py`:
- Loads ARRL MP3+TXT pairs
- DSP-decodes audio → word-level timestamps
- Edit-distance alignment of DSP words to ground-truth text
- Linear interpolation for unmatched words
- Chunks into ~10s segments with exact ground-truth text
- Saves `.npy` audio + `.json` metadata per chunk

Downloaded ARRL W1AW directly:
- `data/arrl/5wpm/`: 41 pairs (2015–2025)
- `data/arrl/15wpm/`: 41 pairs (2015–2025)
- `data/arrl/20wpm/`: 4 pairs (2025)
- Result: **1,136 labeled chunks**

### Phase 4 — Mixed Training (Round 1)
Built `data/real_dataset.py` with `MixedDataset`:
- 40% synthetic + 20% Kaggle GT + 20% Kaggle DSP + 20% ARRL
- Added `ARRLLabeledDataset` with noise augmentation (QSB, pink/white noise, QRM, bandpass)
- Added Kaggle pseudo-labeling (`data/label_kaggle.py`): DSP-labeled 144 unlabeled Kaggle clips → 414 chunks

Results after 20 epochs:
- Training val CER: **0.1697** (on synthetic normalized)
- ARRL holdout: **0.613** (improved from ~1.3 baseline)
- Kaggle holdout: **0.953** (still mostly failing)

**Problem discovered**: Catastrophic forgetting — model got worse at synthetic (0.753 → 0.875 benchmark CER).

### Phase 5 — Benchmark Fixes
**Root cause of misleading metrics**:
1. Training val CER (0.1697) measured on WPM-normalized synthetic — easy
2. Benchmark used `augment=False` (clean pure sine waves) — model trained on augmented audio, so clean tones looked "foreign"
3. Benchmark used auto-detect WPM normalization on noisy audio — detection was wrong

**Fixes applied to `benchmark/run_benchmark.py`**:
- Changed synthetic cases to `augment=True` (matches training distribution)
- Pass `known_wpm` to normalization for synthetic cases (bypasses unreliable auto-detect)
- New honest baseline: **0.623 CER** (was 0.875 — the old number was wrong)

### Phase 6 — Massive Data Expansion
Downloaded `bg4xsd/morse-sound` from Kaggle (2.55 GB):
- **ARRL 2013–2022**: 872 paired MP3+TXT files (10/15/20/30 WPM)
- **Books in Morse**: 4 Project Gutenberg books (Death of a Spaceman, Sjambak, Star Born, Dunwich Horror) — 722 numbered segments with full text
- **Jumbles**: Random words at 16–30 WPM (92 MP3+1 TXT each)
- **realQSO**: 3 real QSO WAV clips

Extraction + labeling:
- ARRL 2013–2022 → `data/arrl_morse_sound/` → labeled via existing `label_arrl.py` → **23,384 new chunks**
- Books + Jumbles → `data/extra_morse/` → labeled via new `data/label_sequential.py` → **~4,000 more chunks**
- Total in `data/arrl_labeled/`: **28,548 chunks** (was 1,136 — 25× increase)

Built `data/label_sequential.py` for datasets with one TXT + many numbered MP3 segments:
- DSP-decodes all segments with cumulative timestamps
- Aligns full DSP sequence against book/jumble text
- Extracts audio across segment boundaries for each chunk

### Phase 7 — Training Run 2 (crashed × 2 — OOM)
Configuration attempted:
- 50,000 samples/epoch, 30 epochs, LR=3e-5, batch=32, workers=0
- **20% synthetic + 10% Kaggle GT + 10% Kaggle DSP + 60% ARRL**
- Starting from checkpoint: epoch ft20, CER=0.1697

**Bug 1 — Oversized chunks → CTC OOM (128 GiB)**
Some labeled chunks were ~25s (over the 10s target), producing mel frames that caused CTC to attempt 128 GiB GPU allocation → GPU driver crash (TDR) → black screen.
Fix: Added hard cap in `collate_fn` in `data/generate.py`:
```python
mels = [m[:, :MAX_MEL_FRAMES] for m in mels]
```

**Bug 2 — Transformer self-attention OOM (85 GiB) — real root cause**
Even with the collate cap, the second run OOMed again. Root cause identified:
- CNN output: 1500 frames (from MAX_MEL_FRAMES=3000 ÷ 2 CNN stride)
- Transformer attention per layer: `[batch=32, heads=8, T=1500, T=1500]` × 4 bytes = **2.88 GB/layer**
- 4 layers forward + backward = **~46 GB** — exceeds the 24 GB RTX 5090
- CTC then attempts to allocate on top of an already-full GPU → OOM

Fix: Reduced `MAX_MEL_FRAMES = 3000 → 2000` (8s instead of 12s):
- CNN output T = 1000 frames
- Attention: `[16, 8, 1000, 1000]` = 512 MB/layer × 4 = ~2 GB total
- All datasets import `MAX_MEL_FRAMES` from `generate.py` — change propagates automatically

Also reduced batch size: 32 → 16 for extra headroom.

Real-audio holdout baseline (before any run 2 training completed):
| Set | CER | WER |
|-----|-----|-----|
| Kaggle QSOs | 0.928 | 0.990 |
| ARRL broadcasts | 0.651 | 0.953 |
| **Overall** | **0.697** | — |

---

## Changes Made Today (2026-03-19)

### 1. Benchmark corrected
- `benchmark/run_benchmark.py`: `augment=True`, `known_wpm` passed to normalizer
- Synthetic baseline now **0.623 CER** (honest)

### 2. Real-audio holdout evaluator
- Created `benchmark/eval_holdout.py`
- Tests against `data/test_holdout.json` (10 Kaggle clips + 50 ARRL chunks)
- Saved results to `benchmark/results/holdout_<timestamp>.json`

### 3. Timing jitter in synthetic generator (`data/generate.py`)
Added `_apply_timing_jitter()` — simulates human operator keying:
- Per-element ±5–20% random duration variation
- Slow WPM drift across transmission (±8%)
- "Fist" bias: dahs sent slightly long, gaps slightly short
- Applied to 70% of augmented synthetic samples

### 4. Variable-speed training (`data/generate.py` — `MorseDataset`)
- 50% of synthetic samples skip WPM normalization
- Model now sees raw variable-speed audio in addition to normalized audio
- Helps generalize to real QSOs where speed changes mid-transmission

### 5. Kaggle QSO ratio increased (`data/real_dataset.py`)
Updated `MixedDataset` defaults:
- Before: 40% synth / 20% kaggle-GT / 20% kaggle-DSP / 20% ARRL
- After:  **20% synth / 15% kaggle-GT / 15% kaggle-DSP / 50% ARRL**
- More real human-keyed QSO audio in training

### 6. Training CLI ratio args (`training/finetune.py`)
Added `--ratio-synth`, `--ratio-kaggle`, `--ratio-kpseudo` flags so ratios can be set per run without editing source.

---

## Current State (2026-03-21)

### Run 5 (completed — 30 epochs from scratch, all fixes applied)
- Best synthetic val CER: **0.0554** (epoch 24)
- Training: 80% real (21,442 ARRL + 46 Kaggle GT + 414 Kaggle pseudo) / 20% synthetic
- All pipeline fixes applied: 8s chunks, holdout exclusion, WPM normalization, digits, word-boundary truncation

### Performance trajectory (honest numbers only)
| Checkpoint | Kaggle v1 CER | Kaggle QSO CER | ARRL CER | Synth CER | Notes |
|------------|---------------|----------------|----------|-----------|-------|
| Pre-training | — | 0.928 | 0.651 | 0.623 | Baseline |
| Run 4 (contaminated) | 0.470 | — | 0.295 (FAKE) | 0.077 | Holdout was in training data |
| **Run 5 (clean)** | **0.453** | **0.473** | **0.759** | **0.055** | From scratch, all fixes |

**Key insight**: ARRL holdout CER went from 0.295 → 0.759. The old 0.295 was fake (model memorized holdout). Kaggle v1 improved slightly (0.470 → 0.453). Synthetic improved (0.077 → 0.055). The domain gap between synthetic (0.055) and real (0.453) is the main bottleneck.

### Beam search
Added beam search decoding to `inference/transcribe.py` and `benchmark/eval_holdout.py`. Beam-5 produced identical results to greedy on run 5 — model probabilities too peaky for beam to help. May help more as model improves.

---

## Pipeline Audit (2026-03-20)

### CRITICAL — Holdout 100% contaminated (FIXED)
ALL holdout samples were in training data:
- ARRL holdout (44 chunks): same .npy files in `arrl_labeled/` used for training
- Kaggle holdout (10 clips): `KaggleDataset` reads same WAVs from `kaggle_mlmv2/audio/`
- Every "holdout" CER (0.295, 0.331) was meaningless — model was trained on its test data

**Fix**: `ARRLLabeledDataset` and `KaggleDataset` now read `test_holdout.json` and exclude those files.
Kaggle v1 (58 clips from morse-challenge) added to `eval_holdout.py` as uncontaminated holdout.

### CRITICAL — 98.7% of training chunks too long (FIXED)
ARRL chunks were ~10s but MAX_MEL_FRAMES=2000 = 8s. During training:
- Mel truncated to 2000 frames → last ~2s of audio discarded
- Label truncated by char count → split words mid-character ("APPROXIMATELY" → "APPROX")

**Fix**: CHUNK_SECONDS changed from 10 → 8 in all labeling scripts. Label truncation now uses word boundaries (finds last space). Data needs re-labeling.

### CRITICAL — Synthetic data had ZERO digits (FIXED)
WORDS pool had 73 words, all letters. Digits only appeared in ARRL real data (2.97%).

**Fix**: WORDS pool expanded to 285 words including callsigns (W1AW, K5ABC), signal reports (599, 559), digit strings, frequencies. `random_sentence()` now generates callsigns (10%) and digit strings (5%).

### MAJOR — WPM normalization mismatch (FIXED)
| Data source | Training normalization | Eval normalization |
|-------------|----------------------|-------------------|
| Synthetic | 50% yes | N/A |
| ARRL labeled | **Never** | **Always** |
| Kaggle GT | **Never** | **Always** |

**Fix**: `ARRLLabeledDataset.__getitem__` now applies 50% WPM normalization (same as synthetic), using `known_wpm` from chunk metadata.

### MAJOR — 5/10 WPM data polluting training (FIXED)
- 179 chunks at 5 WPM (95% bad labels — Farnsworth timing breaks DSP)
- 2,693 chunks at 10 WPM (DSP unreliable, model can't decode these speeds in 8s window)

**Fix**: `ARRLLabeledDataset` now has `min_wpm=13.0` parameter, skips 2,872 low-WPM chunks.

### MODERATE — Model comment said T//4, actual is T//2 (FIXED)
CNN only has temporal MaxPool in Block 2. Comments in `model.py` said T//4 everywhere. Fixed to T//2.

### MODERATE — Inference truncation bug (FIXED earlier)
`inference/transcribe.py` used hardcoded CHUNK_SECONDS=10 (2500 mel frames) but model only trained on 2000. Fixed: chunk size now derived from MAX_MEL_FRAMES with hard mel clamp.

---

## Changes Made (2026-03-19, data audit session)

### 1. WPM normalization cap (`utils/wpm.py`)
- Added `np.clip(src_wpm, target_wpm * 0.5, target_wpm * 2.0)` before stretch
- Prevents 4× audio stretching when `detect_wpm` hits 80 WPM cap on noisy clips
- Improved Kaggle QSO CER: 0.736 → 0.635

### 2. Holdout cleaned (`data/test_holdout.json`)
Removed 6 mislabeled ARRL chunks (DSP vs label completely mismatched):
- `260107_15wpm_0001` — audio was "5 WPM" speed announcement, label was unrelated content
- `250903_5wpm_0001`, `200729_5wpm_0002` — preamble audio with wrong labels
- `250917_15wpm_0006`, `260107_15wpm_0012`, `250903_15wpm_0015` — alignment drift
Holdout now: 10 Kaggle + 44 ARRL = 54 samples

### 3. Training data audit
Checked all 28,548 arrl_labeled chunks for mislabeling (DSP CER > 0.7):
- `ms_arrl` (23,384 chunks): **6.5% bad** — acceptable
- `arrl_direct` 15 WPM (840 chunks): **13.8% bad** — moderate
- `arrl_direct` 5 WPM (160 chunks): **95% bad** — Farnsworth timing, DSP fails
- `arrl_direct` 20 WPM (105 chunks): **43.8% bad** — alignment drift
- `seq_book` (3,336 chunks): **97% bad** — label_sequential.py global alignment drifts over long books
- `seq_jumble` (692 chunks): **100% bad** — TXT has 3.6× more words than transmitted audio

### 4. seq_* data moved out (`data/seq_excluded/`)
Root causes found:
- **Books** (dunwich 18.6 WPM, starborn 18.9 WPM): 8% WPM estimation error causes 70+ word label drift by chunk 900. Even per-segment alignment can't fix this without DSP anchor points every few minutes.
- **Jumbles** (`jumbles_16_20`, `jumbles_24_30`): recorded at ~18.5 WPM Farnsworth character speed but TXT file lists 3.6× more words than transmitted — duration-based mapping is structurally wrong.
- **Slow books** (death 11.3 WPM, sjambak 12.1 WPM): too slow for 8s mel window (<2 words/chunk).
All 32,935 seq chunks moved to `data/seq_excluded/` (kept, not deleted).

### 5. `data/label_sequential.py` rewritten
New approach: duration-based labeling (no DSP alignment).
- Computes real WPM = total_words / total_duration
- Maps time → word index directly (drift-free for uniform-speed recordings)
- Skips datasets with real_wpm < 15 (too slow for 8s mel window)
- Added `--force` flag to reprocess already-labeled directories
- Still fails on jumbles (TXT mismatch) — seq labeling remains a TODO

---

## Changes Made (2026-03-20, comprehensive pipeline fix)

### 1. arrl_morse_sound data labeled and verified
- `label_arrl_sound.py` (duration-based) produced 28,294 chunks — **70% CER, unusable** (prosign timing drift)
- Re-labeled via `label_arrl.py` (DSP-aligned): 5,276 (15wpm) + 7,445 (20wpm) + 7,864 (30wpm) = **20,585 good chunks**
- `verify_training_data.py` removed 107 bad episodes (1,703 chunks)
- Final: **42,665 total chunks** in `arrl_labeled/`

### 2. Kaggle v1 holdout established
- Extracted 58 labeled WAVs from `morse-challenge.zip` → `data/kaggle_v1/`
- Labels from `sampleSubmission.csv` saved to `data/kaggle_v1/labels.json`
- Random alphanumeric targets (12–80 WPM, variable SNR)
- **Only uncontaminated holdout** — never used in any training pipeline

### 3. Inference truncation fix (`inference/transcribe.py`)
- CHUNK_SECONDS derived from MAX_MEL_FRAMES: `2000 * 32 / 8000 = 8.0s`
- Hard mel clamp: `mel[:, :, :MAX_MEL_FRAMES]`
- OVERLAP_SECONDS = 1 for long-audio chunking

### 4. Holdout contamination fix (`arrl_labeled_dataset.py`, `real_dataset.py`)
- `ARRLLabeledDataset(exclude_holdout=True)`: reads `test_holdout.json`, skips 44 ARRL .npy files
- `KaggleDataset(exclude_holdout=True)`: skips 10 Kaggle WAVs
- Training dataset: 39,752 ARRL chunks (42,665 - 44 holdout - 2,869 low-WPM)

### 5. Low-WPM filter (`arrl_labeled_dataset.py`)
- `ARRLLabeledDataset(min_wpm=13.0)`: skips 179 (5wpm) + 2,693 (10wpm) = 2,872 unreliable chunks

### 6. WPM normalization for real data (`arrl_labeled_dataset.py`)
- 50% of ARRL chunks normalized to TARGET_WPM during training
- Uses `known_wpm` from chunk metadata — matches synthetic data behavior and eval

### 7. Word-boundary truncation (all dataset files)
- `generate.py`, `real_dataset.py`, `arrl_labeled_dataset.py`: truncation now finds last space before char limit
- Prevents mid-word label corruption ("APPROXIMATELY" → "APPROX" no longer happens)

### 8. CHUNK_SECONDS 10 → 8 (all labeling scripts)
- `label_arrl_sound.py`, `label_sequential.py`, `real_dataset.py`: chunk duration matches MAX_MEL_FRAMES
- Existing data needs re-labeling with `--force` to produce 8s chunks

### 9. Synthetic data expanded (`data/generate.py`)
- WORDS pool: 73 → **285 words** (callsigns, signal reports, NATO alphabet, ham radio terms, common English)
- **60 words contain digits** (was 0)
- `random_sentence()` generates callsigns (10%), digit strings (5%) probabilistically
- Includes `_random_callsign()` and `_random_digits()` generators

### 10. Kaggle v1 in eval (`benchmark/eval_holdout.py`)
- Added Kaggle v1 section: evaluates all 58 clips from `data/kaggle_v1/labels.json`
- Summary now shows Kaggle QSO + ARRL + Kaggle v1 CER

### 11. Model comment fix (`models/model.py`)
- All T//4 references corrected to T//2 (CNN only has temporal MaxPool in Block 2)

---

## Data Inventory

| Dataset | Location | Chunks | Notes |
|---------|----------|--------|-------|
| ARRL direct (5/15/20 WPM) | `data/arrl/` | 86 pairs | 2015–2025 source MP3+TXT |
| ARRL labeled (all, 8s) | `data/arrl_labeled/` | **22,841** | Re-labeled with 8s chunks, verified |
| ARRL labeled (training) | — | **21,442** | After holdout (27) + low-WPM (1,372) exclusion |
| Seq excluded | `data/seq_excluded/` | **32,935** | Books + jumbles — bad labels, set aside |
| Kaggle MLMv2 GT | `data/kaggle_mlmv2/` | 56 clips (46 training) | 10 in holdout |
| Kaggle DSP pseudo | `data/kaggle_labeled/` | 414 chunks | DSP-labeled .npy |
| Kaggle v1 (holdout) | `data/kaggle_v1/` | **58 WAVs** | Uncontaminated holdout |
| ARRL morse-sound | `data/arrl_morse_sound/` | 872 pairs | 10/15/20/30 WPM, 2013–2022 |
| Extra sequential | `data/extra_morse/` | 6 dirs | Books + Jumbles source audio |
| Holdout (legacy) | `data/test_holdout.json` | 10 Kaggle + 44 ARRL | Now excluded from training |

---

## Key Files

| File | Purpose |
|------|---------|
| `models/model.py` | CNN+Transformer+CTC model (3.57M params) |
| `data/generate.py` | Synthetic audio generation + MorseDataset (285 words, digits) |
| `data/real_dataset.py` | KaggleDataset (holdout-aware), ARRLDataset, MixedDataset |
| `data/arrl_labeled_dataset.py` | ARRLLabeledDataset (holdout-aware, WPM filter, WPM normalization) |
| `data/label_arrl.py` | ARRL MP3+TXT → labeled .npy chunks (DSP-aligned) |
| `data/label_arrl_sound.py` | arrl_morse_sound labeling (duration-based, 8s chunks) |
| `data/label_sequential.py` | Books/jumbles (many MP3 + 1 TXT) → labeled chunks |
| `data/verify_training_data.py` | DSP-audit training data quality by episode group |
| `data/label_kaggle.py` | Kaggle DSP pseudo-labeling + holdout split |
| `inference/dsp_decode.py` | DSP Morse decoder (FFT → envelope → CTC-free decode) |
| `inference/transcribe.py` | Neural model inference (8s chunks, MAX_MEL_FRAMES-derived) |
| `training/finetune.py` | Fine-tuning script with mixed dataset support |
| `benchmark/run_benchmark.py` | Synthetic benchmark suite |
| `benchmark/eval_holdout.py` | Real-audio holdout (Kaggle QSO + ARRL + Kaggle v1) |
| `utils/wpm.py` | WPM detection + time-stretch normalization (capped ±2×) |
| `utils/morse_map.py` | Morse code tables + timing generation |

---

## Known Issues / TODO

### Completed
- [x] ~~Holdout contamination~~ — fixed: holdout files excluded from training datasets
- [x] ~~Chunk length mismatch (10s vs 8s)~~ — fixed: CHUNK_SECONDS=8, word-boundary truncation
- [x] ~~Zero digits in synthetic data~~ — fixed: 285 words, 60 with digits
- [x] ~~WPM normalization mismatch~~ — fixed: 50% normalization on real data during training
- [x] ~~5/10 WPM bad chunks in training~~ — fixed: min_wpm=13.0 filter
- [x] ~~Model comment T//4~~ — fixed: T//2
- [x] ~~Re-label all ARRL data with 8s chunks~~ — done: 22,841 verified chunks
- [x] ~~Train run 5 from scratch~~ — done: best synth CER 0.055, Kaggle v1 CER 0.453
- [x] ~~Beam search decoding~~ — added to inference + eval, no improvement on run 5 (model too peaky)
- [x] ~~`--from-scratch` training flag~~ — added to finetune.py

### Priority: Close the domain gap (synth 0.055 vs real 0.453)
- [ ] **Stronger synthetic augmentation** — biggest lever for improving real-audio CER:
  - [ ] Higher noise floor: real radio always has background noise, synthetic gaps are too clean
  - [ ] SNR variation: real clips range from clean to barely audible (current noise 0.0–0.3, real can be 0.5–1.0+)
  - [ ] Chirp: real transmitters shift frequency on key-down (tone starts off-freq and settles)
  - [ ] Multipath/echo: HF signals bounce off ionosphere, arrive delayed, cause ghosting
  - [ ] AGC pumping: real receivers auto-adjust gain, noise floor jumps up when signal stops
  - [ ] Multi-signal QRM: real bands have 2–3 overlapping CW signals, not just one tone
  - [ ] 50/60Hz hum and relay clicks
  - [ ] Ionospheric flutter: rapid amplitude/phase fluctuations on HF
- [ ] Hybrid LLM decode: show raw + LLM-corrected output, skip callsign-shaped tokens
- [ ] More diverse real data: real QSO recordings from different operators (not just W1AW)
- [ ] realQSO clips from morse-sound zip (3 WAVs) not yet used

### Future
- [ ] seq_book / seq_jumble labeling: needs forced-alignment tool or human anchors
- [ ] No streaming/real-time decoder yet (Phase 4 in roadmap)
- [ ] Consider chunked/local attention for longer sequences
- [ ] Consider gradient checkpointing for memory efficiency
