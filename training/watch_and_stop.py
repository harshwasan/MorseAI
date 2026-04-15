"""
Watches train_log.txt and kills the training process when CER
stays below target for N consecutive epochs.
Usage: python training/watch_and_stop.py
"""
import time, re, os, signal, subprocess

TARGET_CER    = 0.015   # stop at 1.5% CER
PATIENCE      = 3       # must hold for 3 consecutive epochs
CHECK_EVERY   = 60      # check every 60 seconds
LOG_PATH      = 'training/train_log.txt'

def get_epochs(log):
    return [(int(m[0]), float(m[1]))
            for m in re.findall(r'Epoch\s+(\d+)/\d+.*?CER:\s+([\d.]+)', log)]

def find_training_pid():
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if 'train.py' in line and 'watch' not in line:
            return int(line.split()[0])
    return None

print(f"Watching for CER < {TARGET_CER*100:.1f}% for {PATIENCE} epochs...")
consecutive = 0

while True:
    try:
        with open(LOG_PATH) as f:
            log = f.read()
        epochs = get_epochs(log)
        if epochs:
            last_epoch, last_cer = epochs[-1]
            print(f"  Epoch {last_epoch}: CER={last_cer:.4f} | streak={consecutive}/{PATIENCE}")
            if last_cer < TARGET_CER:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= PATIENCE:
                print(f"\nTarget reached! CER={last_cer:.4f} for {PATIENCE} epochs.")
                pid = find_training_pid()
                if pid:
                    os.kill(pid, signal.SIGTERM)
                    print(f"Sent SIGTERM to training process (PID {pid}). Done.")
                else:
                    print("Training process not found — may have already finished.")
                break
    except FileNotFoundError:
        print("Log not found yet...")
    time.sleep(CHECK_EVERY)
