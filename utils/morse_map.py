MORSE_TO_CHAR = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '.-.-.-': '.', '--..--': ',', '..--..': '?', '.----.': "'",
    '-.-.--': '!', '-..-.': '/', '-.--.': '(', '-.--.-': ')',
    '.-...': '&', '---...': ':', '-.-.-.': ';', '-...-': '=',
    '.-.-.': '+', '-....-': '-', '..--.-': '_', '.-..-.': '"',
    '...-..-': '$', '.--.-.': '@', '...---...': 'SOS',
}

CHAR_TO_MORSE = {v: k for k, v in MORSE_TO_CHAR.items()}
CHAR_TO_MORSE[' '] = '/'  # word separator

# All characters the model can output (vocabulary)
VOCAB = ['<blank>'] + sorted(set(MORSE_TO_CHAR.values())) + [' ']
VOCAB_SIZE = len(VOCAB)
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}


def text_to_morse_timing(text: str, wpm: float = 20) -> list[tuple[float, bool]]:
    """
    Convert text to a list of (duration_seconds, is_tone) tuples.
    Standard PARIS timing: dit = 1 unit, dah = 3 units, intra-char gap = 1,
    inter-char gap = 3, inter-word gap = 7.
    """
    dit = 1.2 / wpm  # seconds per dit at given WPM
    events = []

    words = text.upper().split()
    for wi, word in enumerate(words):
        for ci, char in enumerate(word):
            morse = CHAR_TO_MORSE.get(char)
            if not morse:
                continue
            for si, symbol in enumerate(morse):
                duration = dit if symbol == '.' else dit * 3
                events.append((duration, True))   # tone on
                if si < len(morse) - 1:
                    events.append((dit, False))    # intra-char gap
            if ci < len(word) - 1:
                events.append((dit * 3, False))   # inter-char gap
        if wi < len(words) - 1:
            events.append((dit * 7, False))        # inter-word gap

    return events
