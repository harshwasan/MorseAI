"""
Training script for MorseModel.
Usage:  python training/train.py
"""
import os, sys, time, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Disable cuDNN — RTX 5090 (Blackwell) has cuDNN compat issues with LSTM+CTC.
# Native CUDA kernels are used instead; still very fast on the 5090.
torch.backends.cudnn.enabled = False
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from data.generate import MorseDataset, collate_fn
from models.model import MorseModel
from utils.morse_map import IDX_TO_CHAR


# ── Greedy CTC decoder ────────────────────────────────────────────────────────
def greedy_decode(log_probs, idx_to_char=IDX_TO_CHAR, blank_idx=0):
    """log_probs: [T, B, vocab] → list of decoded strings."""
    preds = log_probs.argmax(dim=-1).permute(1, 0)  # [B, T]
    results = []
    for seq in preds:
        chars, prev = [], None
        for idx in seq.tolist():
            if idx != blank_idx and idx != prev:
                chars.append(idx_to_char.get(idx, ''))
            prev = idx
        results.append(''.join(chars))
    return results


def cer(pred: str, target: str) -> float:
    """Character error rate."""
    if not target:
        return 0.0 if not pred else 1.0
    import editdistance
    return editdistance.eval(pred, target) / len(target)


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Data
    train_ds = MorseDataset(size=args.train_size)
    val_ds   = MorseDataset(size=args.val_size, fixed_sentences=True)

    print("Building DataLoaders...", flush=True)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=False,
        persistent_workers=args.workers > 0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=False,
        persistent_workers=args.workers > 0,
    )
    print(f"DataLoaders ready — {len(train_dl)} train batches, {len(val_dl)} val batches", flush=True)

    # Model
    model = MorseModel(
        n_mels=64,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"Parameters: {model.count_params():,}", flush=True)

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_dl),
        epochs=args.epochs,
        pct_start=0.1,
    )

    # Train in FP32 — 5090 has 24GB VRAM, no need for mixed precision.
    # cuBLAS mixed-precision GEMM has issues on Blackwell in PyTorch 2.10.
    use_amp = False

    os.makedirs(args.save_dir, exist_ok=True)
    best_cer = float('inf')

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total_loss, steps = 0.0, 0
        t0 = time.time()

        log_every = max(1, len(train_dl) // 10)  # log ~10x per epoch
        for batch_idx, (mels, labels, input_lens, label_lens, _) in enumerate(train_dl):
            mels      = mels.to(device, non_blocking=True)
            labels    = labels.to(device, non_blocking=True)
            input_lens = input_lens.to(device)
            label_lens = label_lens.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                log_probs   = model(mels)                   # [T_out, B, V]
                # Use actual input lengths (not padded) to compute output lengths.
                # CNN reduces time by 2x (one MaxPool2d with stride 2 on time axis).
                output_lens = (input_lens // 2).clamp(min=1)
                loss = ctc_loss(log_probs, labels, output_lens, label_lens)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            steps      += 1

            if (batch_idx + 1) % log_every == 0:
                elapsed_so_far = time.time() - t0
                print(
                    f"  Epoch {epoch} | batch {batch_idx+1}/{len(train_dl)} | "
                    f"loss {total_loss/steps:.4f} | {elapsed_so_far:.0f}s elapsed",
                    flush=True,
                )

        avg_loss = total_loss / steps
        elapsed  = time.time() - t0

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        cers, samples = [], 0
        with torch.no_grad():
            for mels, labels, input_lens, label_lens, texts in val_dl:
                mels = mels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    log_probs = model(mels)
                preds = greedy_decode(log_probs.cpu())
                for pred, target in zip(preds, texts):
                    cers.append(cer(pred.upper(), target.upper()))
                    samples += 1
                if samples >= 500:   # fast validation cap
                    break

        avg_cer = sum(cers) / len(cers)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | CER: {avg_cer:.4f} | "
            f"Time: {elapsed:.1f}s",
            flush=True,
        )

        # Show a few examples every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample_mels, _, _, _, sample_texts = next(iter(val_dl))
                sample_mels = sample_mels[:4].to(device)
                lp = model(sample_mels)
                decoded = greedy_decode(lp.cpu())
            for t, p in zip(sample_texts[:4], decoded[:4]):
                print(f"  Target : {t}")
                print(f"  Decoded: {p}")
                print()

        # Save best
        if avg_cer < best_cer:
            best_cer = avg_cer
            path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'cer':         avg_cer,
                'args':        vars(args),
            }, path)
            print(f"  [best] Saved model (CER={best_cer:.4f})", flush=True)

    print(f"\nTraining complete. Best CER: {best_cer:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch-size', type=int,   default=64)
    parser.add_argument('--train-size', type=int,   default=50_000)
    parser.add_argument('--val-size',   type=int,   default=2_000)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--d-model',    type=int,   default=256)
    parser.add_argument('--nhead',      type=int,   default=8)
    parser.add_argument('--num-layers', type=int,   default=4)
    parser.add_argument('--dropout',    type=float, default=0.1)
    parser.add_argument('--workers',    type=int,   default=4)
    parser.add_argument('--save-dir',   type=str,   default='checkpoints')
    args = parser.parse_args()
    train(args)
