import re
from collections import Counter

def check_pauses(segments, pause_threshold=0.5):
    pauses = []
    commas = []
    for i in range(1, len(segments)):
        gap = segments[i]['start'] - segments[i-1]['end']
        if gap > pause_threshold:
            pauses.append((segments[i-1]['end'], segments[i]['start'], gap))
    for segment in segments:
        if ',' in segment['text']:
            commas.append((segment['text'].strip(), segment['start']))
    return pauses, commas

def check_fillers(text):
    fillers = ["um", "uh", "like", "kinda", "sort of", "you know", "i mean", "er", "ah", "so", "well", "actually"]
    found_fillers = []
    text_lower = text.lower()
    for filler in fillers:
        if filler in text_lower:
            count = text_lower.count(filler)
            found_fillers.append((filler, count))
    return found_fillers

def check_repeats(text, phrase_len=2):
    words = re.findall(r'\w+', text.lower())
    word_counts = Counter(words)
    repeated_words = {w: c for w, c in word_counts.items() if c > 1}
    phrases = [' '.join(words[i:i+phrase_len]) for i in range(len(words)-phrase_len+1)]
    phrase_counts = Counter(phrases)
    repeated_phrases = {p: c for p, c in phrase_counts.items() if c > 1}
    return repeated_words, repeated_phrases
