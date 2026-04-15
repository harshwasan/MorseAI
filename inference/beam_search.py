"""
CTC beam search decoder with character n-gram language model.

Replaces greedy_decode in transcribe.py for better accuracy.

How it works:
  - Greedy decode: at each time step pick the most likely character. Fast
    but makes no use of context — a single noisy frame can corrupt a character.
  - Beam search: keep the top-K candidate sequences alive at each step,
    scoring them by acoustic probability (from the model) + language model
    probability (how likely this character sequence is in English/CW text).
  - The LM is a character n-gram model built from a large Morse-relevant
    corpus (CW QSO phrases, NATO alphabet, callsigns, etc.). It nudges the
    beam toward real words and away from random character strings.

Usage:
    from inference.beam_search import build_lm, ctc_beam_search
    lm = build_lm()                          # build once, reuse
    text = ctc_beam_search(log_probs, lm)    # replaces greedy_decode
"""
import os, sys, math, collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.morse_map import IDX_TO_CHAR, CHAR_TO_IDX, VOCAB_SIZE

BLANK_IDX = 0
SPACE_IDX = CHAR_TO_IDX.get(' ', None)


# ─────────────────────────────────────────────────────────────
# Character n-gram Language Model
# ─────────────────────────────────────────────────────────────

# Corpus: realistic CW text covering the vocabulary the model needs to handle
_LM_CORPUS = """
THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG
CQ CQ CQ DE W1AW W1AW W1AW K
RST 599 NR 001 BK
QRZ DE K1ABC K
QSL TNX FER QSO 73 ES DX
HELLO WORLD MORSE CODE RADIO SIGNAL
ALPHA BRAVO CHARLIE DELTA ECHO FOXTROT GOLF HOTEL INDIA JULIET
KILO LIMA MIKE NOVEMBER OSCAR PAPA QUEBEC ROMEO SIERRA TANGO
UNIFORM VICTOR WHISKEY XRAY YANKEE ZULU
ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE ZERO
THE AND FOR ARE BUT NOT YOU ALL CAN HER WAS ONE OUR OUT DAY
GET HAS HIM HIS HOW ITS NEW NOW OLD SEE TWO WAY WHO BOY DID
AR SK BT KN QRN QRM QSB QRO QRP PSE AGN
DE K9ABC ES R RST 579 HR IN QTH CHICAGO IL
UR SIG QSB DN NW CPI QRN BAD AGN PSE
WX HR SUNNY AND WARM ES TEMP 72 F
ANT IS 3 EL YAGI AT 40 FT
RIG HR IS ICOM IC7300 ES 100 W
NAME HR IS JOHN ES QTH IS TEXAS
HW CPY OM ES TNX FER NICE QSO
CUL ES 73 DE W5ABC SK
THE VOLTAGE TRANSFORMER SYNTHESIZER OUTPUT LEVEL
COMPONENTS RELATED CRYSTAL FILTER BANDWIDTH
SIGNAL BEING REPAIRED BOUGHT FIRST
SLAMMING INTO THE GULF STATES OR A MAJOR EARTHQUAKE
WHO IS ONE OF THE OWNERS I THINK IS IN A GROUP
WONT KEEP YOU MANY THANKS FOR QSO AND BEACON YOU AGAIN SOON
IK QSL CARL AND UR SIG UP NOW 599 HI 73
THEN I BOUGHT ONE OF THE FIRST KENWOOD
WHILE IT WAS BEING REPAIRED BY KENWOOD I BOUGHT A KIT
AROUND HERE LONG TIME AND FB FELLOW
NEED TO GIVE HIM A CALL
NOT SURE WHAT TO CHASE
GOOD CHRISTMAS YEAR AND QUIET NEW YEAR
"""

def build_lm(n: int = 3, alpha: float = 0.1) -> dict:
    """
    Build a character n-gram LM from the CW corpus.

    Returns a dict with:
      'ngrams'  : {context -> {char -> count}}
      'n'       : order
      'alpha'   : LM weight (how strongly to prefer LM-likely sequences)
      'vocab'   : set of known characters
    """
    text = _LM_CORPUS.upper()
    ngrams = collections.defaultdict(lambda: collections.defaultdict(int))
    vocab  = set()

    for i in range(len(text) - n):
        context = text[i:i + n - 1]
        char    = text[i + n - 1]
        vocab.add(char)
        ngrams[context][char] += 1

    # Convert counts to log-probs with Laplace smoothing
    lm = {}
    all_chars = sorted(vocab)
    for context, char_counts in ngrams.items():
        total = sum(char_counts.values()) + len(all_chars)  # Laplace
        lm[context] = {
            c: math.log((char_counts.get(c, 0) + 1) / total)
            for c in all_chars
        }

    return {'ngrams': lm, 'n': n, 'alpha': alpha, 'vocab': all_chars}


def _lm_score(lm: dict, context: str, char: str) -> float:
    """Log-prob of `char` given `context` under the n-gram LM."""
    n = lm['n']
    ctx = context[-(n - 1):]   # last n-1 chars
    ngrams = lm['ngrams']
    if ctx in ngrams and char in ngrams[ctx]:
        return ngrams[ctx][char]
    # Back-off: try shorter context
    if len(ctx) > 1:
        return _lm_score({'ngrams': ngrams, 'n': n - 1,
                          'alpha': lm['alpha'], 'vocab': lm['vocab']},
                         ctx[1:], char)
    # Uniform fallback
    return math.log(1.0 / max(len(lm['vocab']), 1))


# ─────────────────────────────────────────────────────────────
# CTC Beam Search
# ─────────────────────────────────────────────────────────────

def ctc_beam_search(
    log_probs,          # torch.Tensor [T, vocab_size]  or  numpy [T, vocab_size]
    lm: dict = None,
    beam_width: int = 16,
) -> str:
    """
    CTC beam search with optional character n-gram LM.

    log_probs : acoustic log-probabilities from the model
    lm        : language model from build_lm() — pass None for pure beam search
    beam_width: number of beams to keep (higher = better quality, slower)

    Returns the best decoded string.
    """
    try:
        T, V = log_probs.shape
    except Exception:
        return ""

    # Convert to plain Python floats (works for both torch and numpy)
    try:
        lp = log_probs.cpu().numpy()
    except AttributeError:
        lp = log_probs  # already numpy

    alpha = lm['alpha'] if lm else 0.0

    # Each beam: (prefix_text, last_char_idx, prob_blank, prob_no_blank)
    # prob_blank     = log-prob of ending this prefix with a blank
    # prob_no_blank  = log-prob of ending with a real character
    NEG_INF = float('-inf')

    # Initial state: empty prefix
    # beams: dict mapping (prefix_str, last_idx) -> (log_pb, log_pnb)
    beams = {('', BLANK_IDX): (0.0, NEG_INF)}

    for t in range(T):
        frame = lp[t]            # [vocab_size]
        new_beams = collections.defaultdict(lambda: [NEG_INF, NEG_INF])

        # Only consider top-k characters per frame for speed
        top_k = min(V, beam_width * 2)
        top_indices = sorted(range(V), key=lambda i: frame[i], reverse=True)[:top_k]

        for (prefix, last_idx), (log_pb, log_pnb) in beams.items():
            # Total log-prob for this beam
            log_total = _log_sum_exp(log_pb, log_pnb)

            for c in top_indices:
                ac_score = frame[c]   # acoustic log-prob

                if c == BLANK_IDX:
                    # Emit blank — prefix unchanged
                    key = (prefix, last_idx)
                    nb = new_beams[key]
                    nb[0] = _log_sum_exp(nb[0], log_total + ac_score)

                elif c == last_idx:
                    # Same char as last — only extends if previous ended with blank
                    key = (prefix, c)
                    nb = new_beams[key]
                    nb[1] = _log_sum_exp(nb[1], log_pb + ac_score)
                    # no-blank path stays same prefix
                    key2 = (prefix, c)
                    nb[1] = _log_sum_exp(nb[1], log_pnb + ac_score)

                else:
                    # New character
                    new_char = IDX_TO_CHAR.get(c, '')
                    new_prefix = prefix + new_char

                    # LM score
                    lm_score = 0.0
                    if lm and new_char and new_char != '<blank>':
                        lm_score = alpha * _lm_score(lm, prefix, new_char)

                    key = (new_prefix, c)
                    nb = new_beams[key]
                    nb[1] = _log_sum_exp(nb[1], log_total + ac_score + lm_score)

        # Prune to beam_width
        beams = dict(
            sorted(
                new_beams.items(),
                key=lambda x: _log_sum_exp(x[1][0], x[1][1]),
                reverse=True,
            )[:beam_width]
        )

    # Best beam
    best = max(beams.items(), key=lambda x: _log_sum_exp(x[1][0], x[1][1]))
    return best[0][0]


def _log_sum_exp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import torch
    from data.generate import synthesize_morse_audio, audio_to_melspec, SAMPLE_RATE
    from inference.transcribe import load_model, greedy_decode

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model('checkpoints/best_model.pt', device)
    lm     = build_lm(n=3, alpha=0.3)

    tests = [
        ('HELLO WORLD', 20),
        ('CQ CQ DE W1AW', 20),
        ('THE QUICK BROWN FOX', 20),
        ('RST 599 NR 001', 18),
    ]

    print(f"{'Text':<25} {'Greedy':<25} {'Beam+LM':<25}")
    print('-' * 75)
    for text, wpm in tests:
        audio    = synthesize_morse_audio(text, wpm=wpm, augment=False)
        mel      = audio_to_melspec(audio).unsqueeze(0).to(device)
        with torch.no_grad():
            lp = model(mel)                    # [T, 1, V]
        lp_2d = lp.squeeze(1).cpu()           # [T, V]
        greedy  = greedy_decode(lp_2d)
        beam    = ctc_beam_search(lp_2d, lm=lm, beam_width=16)
        print(f"  {text:<23} {greedy:<25} {beam:<25}")
