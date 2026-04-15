#!/bin/bash
echo "Watching for training completion..."

while true; do
    if grep -q "Epoch  50/50" D:/projects/MorseAI/training/train_log.txt 2>/dev/null; then
        echo "Training complete! Starting Phase 1 tests..." | tee -a D:/projects/MorseAI/training/test_results.txt
        echo "======================================" >> D:/projects/MorseAI/training/test_results.txt
        echo "Test run: $(date)" >> D:/projects/MorseAI/training/test_results.txt
        echo "======================================" >> D:/projects/MorseAI/training/test_results.txt

        cd D:/projects/MorseAI

        for file in test_audio/sos.mp3 test_audio/alphabet.mp3 test_audio/arrl_5wpm.mp3 test_audio/arrl_15wpm.mp3 test_audio/arrl_20wpm.mp3; do
            echo "" | tee -a training/test_results.txt
            echo "--- Testing: $file ---" | tee -a training/test_results.txt
            python inference/transcribe.py --file "$file" 2>&1 | tee -a training/test_results.txt
        done

        echo "" >> training/test_results.txt
        echo "--- Synthetic text test (20 WPM) ---" | tee -a training/test_results.txt
        python inference/transcribe.py --text "HELLO WORLD" --wpm 20 2>&1 | tee -a training/test_results.txt

        echo "" >> training/test_results.txt
        echo "--- Synthetic text test (5 WPM) ---" | tee -a training/test_results.txt
        python inference/transcribe.py --text "SOS" --wpm 5 2>&1 | tee -a training/test_results.txt

        echo "" >> training/test_results.txt
        echo "--- Synthetic text test (35 WPM) ---" | tee -a training/test_results.txt
        python inference/transcribe.py --text "CQ CQ DE W1AW" --wpm 35 2>&1 | tee -a training/test_results.txt

        echo "" | tee -a training/test_results.txt
        echo "Phase 1 complete. Results saved to training/test_results.txt" | tee -a training/test_results.txt
        break
    fi
    sleep 60
done
