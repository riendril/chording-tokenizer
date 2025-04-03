import re
from collections import Counter
from typing import Dict, List, Tuple
from functools import reduce, partial

# ----------------------
# Keyboard Layout Models
# ----------------------

def create_keyboard_layout():
    """Create a mapping of keys to their positions and comfort values."""
    # Define the keyboard layout positions (row, column)
    keyboard_positions = {
        'j': (0, 0), 'y': (0, 1), 'o': (0, 2), 'u': (0, 3), '-': (0, 4),
        'q': (0, 6), 'g': (0, 7), 'n': (0, 8), 'w': (0, 9), 'x': (0, 10),
        'h': (1, 0), 'i': (1, 1), 'e': (1, 2), 'a': (1, 3), '.': (1, 4),
        'p': (1, 6), 'd': (1, 7), 'r': (1, 8), 's': (1, 9), 'l': (1, 10), 'z': (1, 11),
        'k': (2, 0), "'": (2, 1), '/': (2, 2), ',': (2, 3), ';': (2, 4),
        'b': (2, 6), 'c': (2, 7), 'm': (2, 8), 'f': (2, 9), 'v': (2, 10),
        ' ': (3, 5), 't': (3, 6)
    }
    
    # Comfort matrix provided (lower is better)
    comfort_values = {
        'j': 8, 'y': 4, 'o': 1, 'u': 3, '-': 7,
        'q': 7, 'g': 3, 'n': 1, 'w': 4, 'x': 8,
        'h': 2, 'i': 1, 'e': 0, 'a': 0, '.': 6,
        'p': 6, 'd': 0, 'r': 0, 's': 1, 'l': 2, 'z': 9,
        'k': 5, "'": 6, '/': 3, ',': 4, ';': 8,
        'b': 8, 'c': 4, 'm': 3, 'f': 6, 'v': 5,
        ' ': 0, 't': 0
    }
    
    # Finger assignments (0-7, left pinky to right pinky, 8-9 left and right thumb)
    finger_map = {
        'j': 0, 'y': 1, 'o': 2, 'u': 3, '-': 3,
        'q': 4, 'g': 4, 'n': 5, 'w': 6, 'x': 7,
        'h': 0, 'i': 1, 'e': 2, 'a': 3, '.': 3,
        'p': 4, 'd': 4, 'r': 5, 's': 6, 'l': 7, 'z': 7,
        'k': 0, "'": 1, '/': 2, ',': 3, ';': 3,
        'b': 4, 'c': 4, 'm': 5, 'f': 6, 'v': 7,
        ' ': 8, 't': 9  # Space and t with thumbs
    }
    
    return {
        'positions': keyboard_positions,
        'comfort': comfort_values,
        'fingers': finger_map
    }

# -----------------
# Text Preprocessing
# -----------------

def preprocess_text(text: str) -> str:
    """Clean and normalize text for token extraction."""
    return re.sub(r'\s+', ' ', text.lower()).strip()

# -----------------
# Token Extraction 
# -----------------

def character_tokens(text: str) -> Dict[str, int]:
    """Extract individual characters and their frequencies."""
    return Counter(text)

def character_ngrams(text: str, n: int) -> Dict[str, int]:
    """Extract character n-grams and their frequencies."""
    return Counter(text[i:i+n] for i in range(len(text) - n + 1))

def words(text: str) -> Dict[str, int]:
    """Extract words (no spaces) and their frequencies."""
    return Counter(re.findall(r'\b[\w\']+\b', text))

def words_with_space(text: str) -> Dict[str, int]:
    """Extract words with trailing space and their frequencies."""
    return Counter(re.findall(r'\b[\w\']+\s', text))

def word_ngrams(text: str, n: int) -> Dict[str, int]:
    """Extract word n-grams and their frequencies."""
    word_list = re.findall(r'\b[\w\']+\b', text)
    return Counter(' '.join(word_list[i:i+n]) for i in range(len(word_list) - n + 1))

def punctuation_patterns(text: str) -> Dict[str, int]:
    """Extract common punctuation patterns."""
    patterns = [
        r'\.{2,}',          # multiple periods
        r'[,\.;:][\'\"]',   # punctuation followed by quotes
        r'\w+\.\w+',        # domain-like patterns
        r'[!?]{2,}'         # multiple ! or ?
    ]
    
    extract_pattern = lambda pattern: re.findall(pattern, text)
    all_matches = [extract_pattern(pattern) for pattern in patterns]
    flattened = [item for sublist in all_matches for item in sublist]
    
    return Counter(flattened)

def merge_counters(counters: List[Counter]) -> Counter:
    """Merge multiple counters into one."""
    return reduce(lambda x, y: x + y, counters, Counter())

def extract_tokens(text: str) -> Dict[str, int]:
    """Extract all types of tokens from text using functional composition."""
    processed_text = preprocess_text(text)
    
    # Create extractors with partial application
    char_ngram_extractors = [partial(character_ngrams, processed_text, n) for n in range(2, 5)]
    word_ngram_extractors = [partial(word_ngrams, processed_text, n) for n in range(2, 4)]
    
    # List of all extractors
    extractors = [
        partial(character_tokens, processed_text),
        *char_ngram_extractors,
        partial(words, processed_text),
        partial(words_with_space, processed_text),
        *word_ngram_extractors,
        partial(punctuation_patterns, processed_text)
    ]
    
    # Apply all extractors and merge results
    return merge_counters([extractor() for extractor in extractors])

# ------------------
# Difficulty Scoring
# ------------------

def key_difficulty(char: str, layout: Dict) -> float:
    """Get difficulty value for a single key."""
    return layout['comfort'].get(char, 10)  # Default to 10 for unknown keys

def base_difficulty(token: str, layout: Dict) -> float:
    """Calculate cumulative base difficulty of all keys in token."""
    return sum(key_difficulty(char, layout) for char in token)

def same_finger(a: str, b: str, layout: Dict) -> bool:
    """Check if two characters are typed with the same finger."""
    fingers = layout['fingers']
    return a in fingers and b in fingers and fingers[a] == fingers[b]

def key_distance(a: str, b: str, layout: Dict) -> float:
    """Calculate physical distance between two keys ASSUMING AN ORTHO LAYOUT."""
    positions = layout['positions']
    if a not in positions or b not in positions:
        return 0
    
    a_pos, b_pos = positions[a], positions[b]
    return ((a_pos[0] - b_pos[0])**2 + (a_pos[1] - b_pos[1])**2)**0.5

def transition_difficulty(token: str, layout: Dict) -> float:
    """Calculate difficulty of transitions between adjacent keys."""
    if len(token) <= 1:
        return 0
    
    pairs = zip(token[:-1], token[1:])
    
    def pair_difficulty(pair):
        a, b = pair
        # High penalty for same-finger transitions
        if same_finger(a, b, layout):
            return 5
        return key_distance(a, b, layout) / 2
    
    return sum(map(pair_difficulty, pairs))

def typing_difficulty(token: str, layout: Dict) -> float:
    """Calculate overall typing difficulty score."""
    return base_difficulty(token, layout) + transition_difficulty(token, layout)

# ------------------
# Token Scoring
# ------------------

def score_token(token_freq_tuple: Tuple[str, int], layout: Dict) -> Dict:
    """Score a token based on frequency and typing difficulty."""
    token, frequency = token_freq_tuple
    difficulty = typing_difficulty(token, layout)
    length = len(token)
    
    # Length benefit is non-linear
    length_benefit = length ** 1.5
    
    # Final score: higher makes it more attractive for chording
    score = frequency * length_benefit * (difficulty + 1)
    
    return {
        'token': token,
        'frequency': frequency,
        'length': length,
        'difficulty': difficulty,
        'score': score
    }

def sort_by_score(tokens: List[Dict], reverse: bool = True) -> List[Dict]:
    """Sort tokens by their score."""
    return sorted(tokens, key=lambda x: x['score'], reverse=reverse)

def take_top_n(tokens: List[Dict], n: int) -> List[Dict]:
    """Take top n tokens from sorted list."""
    return tokens[:n]

# ------------------
# Program Composition
# ------------------

def analyze_corpus(corpus: str, top_n: int = 200) -> List[Dict]:
    """Analyze corpus and return top n tokens ranked by score."""
    # Create keyboard layout
    layout = create_keyboard_layout()
    
    # Extract tokens and compose scoring functions
    tokens_with_freq = extract_tokens(preprocess_text(corpus)).items()
    
    # Score each token
    score_with_layout = partial(score_token, layout=layout)
    scored_tokens = list(map(score_with_layout, tokens_with_freq))
    
    # Sort and take top n
    return take_top_n(sort_by_score(scored_tokens), top_n)

# ------------------
# Utility Functions
# ------------------

def format_token_result(index: int, token_data: Dict) -> str:
    """Format token result for display."""
    return (f"{index+1}. Token: '{token_data['token']}' | "
            f"Score: {token_data['score']:.2f} | "
            f"Freq: {token_data['frequency']} | "
            f"Difficulty: {token_data['difficulty']:.2f}")

def print_results(results: List[Dict]) -> None:
    """Print formatted token results."""
    for i, result in enumerate(results):
        print(format_token_result(i, result))

# ------------------
# Example Usage
# ------------------

def read_corpus_from_file(file_path: str) -> str:
    """Read corpus data from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def main():
    import sys
    
    # Use command line argument for corpus file if provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        try:
            corpus = read_corpus_from_file(file_path)
            print(f"Analyzing corpus from {file_path}...")
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    else:
        # Example usage with a small corpus
        print("No file provided, using sample text...")
        corpus = """
        The quick brown fox jumps over the lazy dog. The dog was not very happy about this.
        What the fox was thinking, nobody knows. It wasn't the first time this had happened,
        and it probably wouldn't be the last time either. The quick brown fox jumps over
        the lazy dog again. What the heck is going on with these animals?
        """
    
    # Get number of tokens from command line if provided
    top_n = 20
    if len(sys.argv) > 2:
        try:
            top_n = int(sys.argv[2])
        except ValueError:
            print(f"Invalid number for top_n: {sys.argv[2]}, using default of 20")
    
    # Analyze corpus and print results
    top_tokens = analyze_corpus(corpus, top_n=top_n)
    print(f"\nTop {top_n} tokens by score:")
    print("-" * 60)
    print_results(top_tokens)
    
    # Output to a file if requested
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, token in enumerate(top_tokens):
                    f.write(f"{format_token_result(i, token)}\n")
            print(f"\nResults saved to {output_file}")
        except Exception as e:
            print(f"Error writing to output file: {e}")

if __name__ == "__main__":
    main()
