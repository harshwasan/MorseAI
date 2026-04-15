"""
Fine-tuning script: loads best_model.pt and continues training
with more aggressive real-world augmentations to close the domain gap.
Usage: python training/finetune.py
"""
import os, sys, time, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = False

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.generate import MorseDataset, collate_fn
from data.real_dataset import MixedDataset
from models.model import MorseModel
from utils.morse_map import IDX_TO_CHAR

import editdistance


def greedy_decode(log_probs, idx_to_char=IDX_TO_CHAR, blank_idx=0):
    preds = log_probs.argmax(dim=-1).permute(1, 0)
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
    if not target:
        return 0.0 if not pred else 1.0
    return editdistance.eval(pred, target) / len(target)


def finetune(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    ckpt_args = {'d_model': 256, 'nhead': 8, 'num_layers': 4}

    if args.from_scratch:
        model = MorseModel(
            n_mels=64, d_model=256, nhead=8, num_layers=4, dropout=0.1,
        ).to(device)
        best_cer_init = float('inf')
        print(f"Training from scratch ({model.count_params():,} params)", flush=True)
    else:
        ckpt  = torch.load(args.checkpoint, map_location=device)
        ckpt_args = ckpt.get('args', ckpt_args)
        model = MorseModel(
            n_mels=64,
            d_model=ckpt_args.get('d_model', 256),
            nhead=ckpt_args.get('nhead', 8),
            num_layers=ckpt_args.get('num_layers', 4),
            dropout=0.1,
        ).to(device)
        state = ckpt['model_state']
        state.pop('pos_enc.pe', None)
        model.load_state_dict(state, strict=False)
        best_cer_init = ckpt.get('cer', float('inf'))
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}, CER={best_cer_init:.4f})", flush=True)

    # Data
    if args.mixed:
        train_ds = MixedDataset(
            total_size=args.train_size,
            ratio_synth=args.ratio_synth,
            ratio_kaggle=args.ratio_kaggle,
            ratio_kpseudo=args.ratio_kpseudo,
            ratio_arrl=1.0 - args.ratio_synth - args.ratio_kaggle - args.ratio_kpseudo,
        )
    else:
        train_ds = MorseDataset(size=args.train_size)
    val_ds = MorseDataset(size=args.val_size, fixed_sentences=True)

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
    print("DataLoaders ready", flush=True)

    print("Building optimizer...", flush=True)
    ctc_loss  = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine annealing — gentler than OneCycle for fine-tuning
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 20)

    os.makedirs(args.save_dir, exist_ok=True)
    best_cer = best_cer_init
    log_path = os.path.join('training', 'finetune_log.txt')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        t0 = time.time()

        log_every  = max(1, len(train_dl) // 10)   # log ~10x per epoch
        save_every = 100                            # mid-epoch checkpoint every 100 batches
        batch_times = []
        batch_t0 = time.time()
        for batch_idx, (mels, labels, input_lens, label_lens, _) in enumerate(train_dl):
            mels       = mels.to(device, non_blocking=True)
            labels     = labels.to(device, non_blocking=True)
            input_lens = input_lens.to(device)
            label_lens = label_lens.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=False):
                log_probs   = model(mels)
                output_lens = (input_lens // 2).clamp(min=1)
                loss = ctc_loss(log_probs, labels, output_lens, label_lens)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            steps      += 1
            batch_times.append(time.time() - batch_t0)
            batch_t0 = time.time()

            if (batch_idx + 1) % log_every == 0:
                avg_ms = sum(batch_times[-log_every:]) / len(batch_times[-log_every:]) * 1000
                eta_s  = avg_ms / 1000 * (len(train_dl) - batch_idx - 1)
                from datetime import datetime
                print(
                    f"  [{datetime.now().strftime('%H:%M:%S')}] "
                    f"Epoch {epoch} | batch {batch_idx+1}/{len(train_dl)} | "
                    f"loss {total_loss/steps:.4f} | "
                    f"{avg_ms:.0f}ms/batch | "
                    f"ETA {eta_s/60:.1f}min",
                    flush=True,
                )

            # Mid-epoch safety checkpoint — survives GPU crashes
            if (batch_idx + 1) % save_every == 0:
                torch.save({
                    'epoch':       f'ft{epoch}_b{batch_idx+1}',
                    'model_state': model.state_dict(),
                    'cer':         best_cer,
                    'args':        ckpt_args,
                }, os.path.join(args.save_dir, 'recovery.pt'))

        scheduler.step()
        avg_loss = total_loss / steps
        elapsed  = time.time() - t0

        # Validate
        model.eval()
        cers, samples = [], 0
        with torch.no_grad():
            for mels, labels, input_lens, label_lens, texts in val_dl:
                mels = mels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=False):
                    log_probs = model(mels)
                preds = greedy_decode(log_probs.cpu())
                for pred, target in zip(preds, texts):
                    cers.append(cer(pred.upper(), target.upper()))
                    samples += 1
                if samples >= 500:
                    break

        avg_cer = sum(cers) / len(cers)
        from datetime import datetime
        avg_batch_ms = sum(batch_times) / len(batch_times) * 1000 if batch_times else 0
        msg = (f"[{datetime.now().strftime('%H:%M:%S')}] "
               f"Epoch {epoch:3d}/{args.epochs} | "
               f"Loss: {avg_loss:.4f} | CER: {avg_cer:.4f} | "
               f"Time: {elapsed:.1f}s | {avg_batch_ms:.0f}ms/batch")
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

        # Show examples every 5 epochs
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

        if avg_cer < best_cer:
            best_cer = avg_cer
            path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch':       f'ft{epoch}',
                'model_state': model.state_dict(),
                'cer':         avg_cer,
                'args':        ckpt_args,
            }, path)
            msg2 = f"  [best] Saved model (CER={best_cer:.4f})"
            print(msg2, flush=True)
            with open(log_path, 'a') as f:
                f.write(msg2 + '\n')

    print(f"\nFine-tuning complete. Best CER: {best_cer:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,   default='checkpoints/best_model.pt')
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch-size', type=int,   default=64)
    parser.add_argument('--train-size', type=int,   default=30_000)
    parser.add_argument('--val-size',   type=int,   default=2_000)
    parser.add_argument('--lr',         type=float, default=5e-5)
    parser.add_argument('--workers',    type=int,   default=4)
    parser.add_argument('--save-dir',   type=str,   default='checkpoints')
    parser.add_argument('--ratio-synth',   type=float, default=0.20)
    parser.add_argument('--ratio-kaggle',  type=float, default=0.10)
    parser.add_argument('--ratio-kpseudo', type=float, default=0.10)
    # ratio_arrl = 1 - synth - kaggle - kpseudo (default 0.60)
    parser.add_argument('--mixed',        action='store_true', help='Use mixed real+synthetic dataset')
    parser.add_argument('--from-scratch', action='store_true', help='Train from random init instead of checkpoint')
    args = parser.parse_args()
    finetune(args)
