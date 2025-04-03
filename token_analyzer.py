import multiprocessing
import re
from collections import Counter
from functools import lru_cache, partial, reduce
from typing import Any, Callable, Dict, List, Optional, Tuple

# ----------------------
# Memoization Decorator
# ----------------------


def memoize(func):
    """Simple memoization decorator for caching function results."""
    cache = {}

    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


# ----------------------
# Keyboard Layout Models
# ----------------------


def create_keyboard_layout():
    """Create a mapping of keys to their positions and comfort values."""
    # Define the keyboard layout positions (row, column)
    keyboard_positions = {
        "j": (0, 0),
        "y": (0, 1),
        "o": (0, 2),
        "u": (0, 3),
        "-": (0, 4),
        "q": (0, 6),
        "g": (0, 7),
        "n": (0, 8),
        "w": (0, 9),
        "x": (0, 10),
        "h": (1, 0),
        "i": (1, 1),
        "e": (1, 2),
        "a": (1, 3),
        ".": (1, 4),
        "p": (1, 6),
        "d": (1, 7),
        "r": (1, 8),
        "s": (1, 9),
        "l": (1, 10),
        "z": (1, 11),
        "k": (2, 0),
        "'": (2, 1),
        "/": (2, 2),
        ",": (2, 3),
        ";": (2, 4),
        "b": (2, 6),
        "c": (2, 7),
        "m": (2, 8),
        "f": (2, 9),
        "v": (2, 10),
        " ": (3, 5),
        "t": (3, 6),
    }

    # Comfort matrix provided (lower is better)
    comfort_values = {
        "j": 8,
        "y": 4,
        "o": 1,
        "u": 3,
        "-": 7,
        "q": 7,
        "g": 3,
        "n": 1,
        "w": 4,
        "x": 8,
        "h": 2,
        "i": 1,
        "e": 0,
        "a": 0,
        ".": 6,
        "p": 6,
        "d": 0,
        "r": 0,
        "s": 1,
        "l": 2,
        "z": 9,
        "k": 5,
        "'": 6,
        "/": 3,
        ",": 4,
        ";": 8,
        "b": 8,
        "c": 4,
        "m": 3,
        "f": 6,
        "v": 5,
        " ": 0,
        "t": 0,
    }

    # Finger assignments (0-7, left pinky to right pinky, 8-9 left and right thumb)
    finger_map = {
        "j": 0,
        "y": 1,
        "o": 2,
        "u": 3,
        "-": 3,
        "q": 4,
        "g": 4,
        "n": 5,
        "w": 6,
        "x": 7,
        "h": 0,
        "i": 1,
        "e": 2,
        "a": 3,
        ".": 3,
        "p": 4,
        "d": 4,
        "r": 5,
        "s": 6,
        "l": 7,
        "z": 7,
        "k": 0,
        "'": 1,
        "/": 2,
        ",": 3,
        ";": 3,
        "b": 4,
        "c": 4,
        "m": 5,
        "f": 6,
        "v": 7,
        " ": 8,
        "t": 9,  # Space with left thumb, t with right thumb
    }

    return {
        "positions": keyboard_positions,
        "comfort": comfort_values,
        "fingers": finger_map,
    }


# -----------------
# Text Preprocessing
# -----------------


def preprocess_text(text: str) -> str:
    """Clean and normalize text for token extraction."""
    return re.sub(r"\s+", " ", text.lower()).strip()


# -----------------
# Token Extraction
# -----------------


def character_tokens(text: str) -> Dict[str, int]:
    """Extract individual characters and their frequencies."""
    return Counter(text)


def character_ngrams(text: str, n: int) -> Dict[str, int]:
    """Extract character n-grams and their frequencies."""
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def words(text: str) -> Dict[str, int]:
    """Extract words (no spaces) and their frequencies."""
    return Counter(re.findall(r"\b[\w\']+\b", text))


def words_with_space(text: str) -> Dict[str, int]:
    """Extract words with trailing space and their frequencies."""
    return Counter(re.findall(r"\b[\w\']+\s", text))


def word_ngrams(text: str, n: int) -> Dict[str, int]:
    """Extract word n-grams and their frequencies."""
    word_list = re.findall(r"\b[\w\']+\b", text)
    return Counter(
        " ".join(word_list[i : i + n]) for i in range(len(word_list) - n + 1)
    )


def punctuation_patterns(text: str) -> Dict[str, int]:
    """Extract common punctuation patterns."""
    patterns = [
        r"\.{2,}",  # multiple periods
        r"[,\.;:][\'\"]",  # punctuation followed by quotes
        r"\w+\.\w+",  # domain-like patterns
        r"[!?]{2,}",  # multiple ! or ?
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
    char_ngram_extractors = [
        partial(character_ngrams, processed_text, n) for n in range(2, 5)
    ]
    word_ngram_extractors = [
        partial(word_ngrams, processed_text, n) for n in range(2, 4)
    ]

    # List of all extractors
    extractors = [
        partial(character_tokens, processed_text),
        *char_ngram_extractors,
        partial(words, processed_text),
        partial(words_with_space, processed_text),
        *word_ngram_extractors,
        partial(punctuation_patterns, processed_text),
    ]

    # Apply all extractors and merge results
    return merge_counters([extractor() for extractor in extractors])


# ------------------
# Difficulty Scoring
# ------------------


@memoize
def key_difficulty(char: str, layout: Dict) -> float:
    """Get difficulty value for a single key."""
    return layout["comfort"].get(char, 10)  # Default to 10 for unknown keys


def base_difficulty(token: str, layout: Dict) -> float:
    """Calculate cumulative base difficulty of all keys in token."""
    return sum(key_difficulty(char, layout) for char in token)


@memoize
def get_finger(char: str, layout: Dict) -> Optional[int]:
    """Get the finger used for a character."""
    return layout["fingers"].get(char)


@memoize
def same_finger(a: str, b: str, layout: Dict) -> bool:
    """Check if two characters are typed with the same finger."""
    finger_a = get_finger(a, layout)
    finger_b = get_finger(b, layout)
    return finger_a is not None and finger_b is not None and finger_a == finger_b


@memoize
def key_distance(a: str, b: str, layout: Dict) -> float:
    """Calculate physical distance between two keys ASSUMING AN ORTHO LAYOUT."""
    positions = layout["positions"]
    if a not in positions or b not in positions:
        return 0

    a_pos, b_pos = positions[a], positions[b]
    return ((a_pos[0] - b_pos[0]) ** 2 + (a_pos[1] - b_pos[1]) ** 2) ** 0.5


def transition_difficulty(
    token: str, layout: Dict, prev_token: str = "", next_token: str = ""
) -> float:
    """
    Calculate difficulty of transitions between adjacent keys, only considering
    difficulty when the same finger is used consecutively.

    This includes transitions from the previous token's last character to this token's first,
    and from this token's last character to the next token's first character.
    """
    if not token:
        return 0

    difficulty = 0

    # Check transition from previous token's last character to this token's first
    if prev_token and token:
        prev_last = prev_token[-1]
        curr_first = token[0]

        if same_finger(prev_last, curr_first, layout):
            difficulty += key_distance(prev_last, curr_first, layout)

    # Check transitions within the current token
    for i in range(len(token) - 1):
        a, b = token[i], token[i + 1]
        if same_finger(a, b, layout):
            difficulty += key_distance(a, b, layout)

    # Check transition from this token's last character to next token's first
    if token and next_token:
        curr_last = token[-1]
        next_first = next_token[0]

        if same_finger(curr_last, next_first, layout):
            difficulty += key_distance(curr_last, next_first, layout)

    return difficulty


def typing_difficulty(
    token: str, layout: Dict, prev_token: str = "", next_token: str = ""
) -> float:
    """Calculate overall typing difficulty score."""
    return base_difficulty(token, layout) + transition_difficulty(
        token, layout, prev_token, next_token
    )


# ------------------
# Token Scoring
# ------------------


def score_token(
    token_freq_tuple: Tuple[str, int],
    layout: Dict,
    context_tokens: Dict[str, str] = None,
) -> Dict:
    """
    Score a token based on frequency and typing difficulty.

    Args:
        token_freq_tuple: A tuple of (token, frequency)
        layout: The keyboard layout information
        context_tokens: A dict with 'prev' and 'next' tokens for context
    """
    token, frequency = token_freq_tuple

    # Get context tokens if available
    prev_token = context_tokens.get("prev", "") if context_tokens else ""
    next_token = context_tokens.get("next", "") if context_tokens else ""

    # Calculate difficulty with context if available
    difficulty = typing_difficulty(token, layout, prev_token, next_token)
    length = len(token)

    # Length benefit is non-linear
    length_benefit = length**1.5

    # Final score: higher makes it more attractive for chording
    # Using the formula from your updated code
    score = frequency * length_benefit * (difficulty + 1)

    return {
        "token": token,
        "frequency": frequency,
        "length": length,
        "difficulty": difficulty,
        "score": score,
    }


def sort_by_score(tokens: List[Dict], reverse: bool = True) -> List[Dict]:
    """Sort tokens by their score."""
    return sorted(tokens, key=lambda x: x["score"], reverse=reverse)


def take_top_n(tokens: List[Dict], n: int) -> List[Dict]:
    """Take top n tokens from sorted list."""
    return tokens[:n]


# ------------------
# Program Composition
# ------------------


def analyze_corpus(
    corpus: str, top_n: int = 200, parallel: bool = True, show_progress: bool = True
) -> List[Dict]:
    """Analyze corpus and return top n tokens ranked by score."""
    # Create keyboard layout
    layout = create_keyboard_layout()

    if show_progress:
        print("Extracting tokens from corpus...")

    # Extract tokens - no need to preprocess here as it's done in extract_tokens
    tokens_with_freq = extract_tokens(corpus).items()
    tokens_list = list(tokens_with_freq)

    if show_progress:
        total_tokens = len(tokens_list)
        print(f"Found {total_tokens} unique tokens. Scoring tokens...")

    # Score tokens (in parallel if requested)
    score_with_layout = partial(score_token, layout=layout)

    if parallel and total_tokens > 1000:  # Only parallelize for large token sets
        try:
            # Determine optimal chunk size based on number of cores
            num_cores = multiprocessing.cpu_count()
            chunk_size = max(1, total_tokens // (num_cores * 4))

            if show_progress:
                print(f"Parallel processing with {num_cores} cores...")

            # Create a pool and process tokens in parallel
            with multiprocessing.Pool(processes=num_cores) as pool:
                scored_tokens = pool.map(score_with_layout, tokens_list, chunk_size)
        except Exception as e:
            if show_progress:
                print(
                    f"Parallel processing failed: {e}. Falling back to sequential processing."
                )
            # Fall back to sequential processing
            scored_tokens = process_tokens_sequentially(
                tokens_list, score_with_layout, show_progress
            )
    else:
        # Process sequentially with progress updates
        scored_tokens = process_tokens_sequentially(
            tokens_list, score_with_layout, show_progress
        )

    if show_progress:
        print("Sorting tokens by score...")

    # Sort and take top n
    return take_top_n(sort_by_score(scored_tokens), top_n)


def process_tokens_sequentially(
    tokens_list: List[Tuple[str, int]],
    scoring_func: Callable,
    show_progress: bool = True,
) -> List[Dict]:
    """Process tokens sequentially with optional progress tracking."""
    total = len(tokens_list)
    results = []

    for i, token_freq in enumerate(tokens_list):
        # Update progress every 100 tokens
        if show_progress and i % max(1, total // 100) == 0:
            progress = (i / total) * 100
            print(
                f"  Progress: {progress:.1f}% ({i}/{total} tokens processed)", end="\r"
            )

        # Score the token
        results.append(scoring_func(token_freq))

    if show_progress:
        print("  Progress: 100.0% ({}/{} tokens processed)".format(total, total))

    return results


# ------------------
# Advanced Analysis
# ------------------


def analyze_token_context(
    tokens: List[Dict], corpus: str, layout: Dict, show_progress: bool = True
) -> List[Dict]:
    """
    Analyze tokens with context from the corpus.
    This will update difficulty scores based on transitions between tokens.
    """
    if show_progress:
        print("Performing advanced context analysis...")
        print("Processing corpus for context...")

    processed_corpus = preprocess_text(corpus)

    # Extract token contexts
    token_contexts = {}
    total_tokens = len(tokens)

    if show_progress:
        print(f"Analyzing context for {total_tokens} tokens...")

    for i, token_data in enumerate(tokens):
        token = token_data["token"]

        # Show progress update
        if show_progress and i % max(1, total_tokens // 20) == 0:
            progress = (i / total_tokens) * 100
            print(
                f"  Context analysis: {progress:.1f}% ({i}/{total_tokens} tokens)",
                end="\r",
            )

        # Find all occurrences of this token in the corpus
        contexts = []
        start = 0
        while True:
            pos = processed_corpus.find(token, start)
            if pos == -1:
                break

            # Get context (previous and next characters/tokens)
            prev_char = processed_corpus[pos - 1] if pos > 0 else ""
            next_pos = pos + len(token)
            next_char = (
                processed_corpus[next_pos] if next_pos < len(processed_corpus) else ""
            )

            contexts.append({"prev": prev_char, "next": next_char})

            start = pos + 1

        # Aggregate contexts for this token
        if contexts:
            prev_tokens = [c["prev"] for c in contexts]
            next_tokens = [c["next"] for c in contexts]
            most_common_prev = (
                Counter(prev_tokens).most_common(1)[0][0] if prev_tokens else ""
            )
            most_common_next = (
                Counter(next_tokens).most_common(1)[0][0] if next_tokens else ""
            )

            token_contexts[token] = {"prev": most_common_prev, "next": most_common_next}

    if show_progress:
        print(f"  Context analysis: 100.0% ({total_tokens}/{total_tokens} tokens)")
        print("Re-scoring tokens with context...")

    # Re-score tokens with context
    rescored_tokens = []
    for token_data in tokens:
        token = token_data["token"]
        context = token_contexts.get(token, {"prev": "", "next": ""})

        # Recalculate difficulty with context
        difficulty = typing_difficulty(
            token, layout, prev_token=context["prev"], next_token=context["next"]
        )

        # Update score
        frequency = token_data["frequency"]
        length = token_data["length"]
        length_benefit = length**1.5
        score = frequency * length_benefit * (difficulty + 1)

        rescored_tokens.append(
            {
                "token": token,
                "frequency": frequency,
                "length": length,
                "difficulty": difficulty,
                "score": score,
                "context": context,
            }
        )

    return sort_by_score(rescored_tokens)


# ------------------
# Utility Functions
# ------------------


def format_token_result(index: int, token_data: Dict) -> str:
    """Format token result for display."""
    return (
        f"{index+1}. Token: '{token_data['token']}' | "
        f"Score: {token_data['score']:.2f} | "
        f"Freq: {token_data['frequency']} | "
        f"Difficulty: {token_data['difficulty']:.2f}"
    )


def print_results(results: List[Dict]) -> None:
    """Print formatted token results."""
    for i, result in enumerate(results):
        print(format_token_result(i, result))


def read_corpus_from_file(file_path: str) -> str:
    """Read corpus data from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def print_help():
    """Print detailed help information about the program."""
    help_text = """
Token Analyzer for Chording Keyboard Layout Optimization
--------------------------------------------------------

DESCRIPTION:
    This program analyzes text to identify tokens (characters, character n-grams, 
    words, word n-grams) that would benefit most from chording on a custom keyboard layout.
    It scores tokens based on typing difficulty, length, and frequency in the input text.

USAGE:
    python script.py [corpus_file] [options]

ARGUMENTS:
    corpus_file            Path to a text file to analyze (optional, uses sample text if omitted)

OPTIONS:
    -n, --top_n N         Number of top tokens to display (default: 20)
    -a, --advanced        Perform advanced context analysis (default: True)
    -p, --parallel        Use parallel processing for large token sets (default: True)
    -o, --output FILE     Save results to specified file
    -q, --quiet           Suppress progress output
    -h, --help            Display this help message and exit

EXAMPLES:
    # Analyze a text file and show top 50 tokens
    python script.py mytext.txt -n 50

    # Analyze with advanced context and save to file
    python script.py mytext.txt --advanced --output results.txt

    # Run quietly with only basic analysis
    python script.py mytext.txt --quiet --advanced=False

OUTPUT FORMAT:
    For each token, the program displays:
    - Token: The actual text string
    - Score: Combined metric of frequency and typing efficiency
    - Freq: Number of occurrences in the corpus
    - Difficulty: Typing difficulty score (lower is easier)

ABOUT SCORING:
    The score is calculated based on:
    - Token frequency (more frequent tokens score higher)
    - Token length (longer tokens get a non-linear bonus)
    - Typing difficulty (based on key positions and finger usage)
    - Context transitions (in advanced mode)
    
    Higher scores indicate tokens that would benefit more from chording.
"""
    print(help_text)


# ------------------
# Main
# ------------------


def main():
    import argparse
    import sys

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Analyze tokens from a text corpus for chording efficiency",
        add_help=False,  # Disable default help to use our custom help
    )
    parser.add_argument("corpus_file", nargs="?", help="Path to the corpus file")
    parser.add_argument(
        "-n", "--top_n", type=int, default=20, help="Number of top tokens to return"
    )
    parser.add_argument(
        "-a",
        "--advanced",
        action="store_true",
        default=True,
        help="Perform advanced context analysis (default: True)",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        default=True,
        help="Use parallel processing for large token sets (default: True)",
    )
    parser.add_argument("-o", "--output", help="Output file to save results")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress output"
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show detailed help information"
    )

    # Check for help flag first
    if "-h" in sys.argv or "--help" in sys.argv:
        print_help()
        sys.exit(0)

    # Parse arguments
    if len(sys.argv) > 1:
        args = parser.parse_args()
        show_progress = not args.quiet

        if args.corpus_file:
            try:
                corpus = read_corpus_from_file(args.corpus_file)
                if show_progress:
                    print(f"Analyzing corpus from {args.corpus_file}...")
            except Exception as e:
                print(f"Error reading file: {e}")
                sys.exit(1)
        else:
            # Example usage with a small corpus
            if show_progress:
                print("No file provided, using sample text...")
            corpus = """
            The quick brown fox jumps over the lazy dog. The dog was not very happy about this.
            What the fox was thinking, nobody knows. It wasn't the first time this had happened,
            and it probably wouldn't be the last time either. The quick brown fox jumps over
            the lazy dog again. What the heck is going on with these animals?
            """
    else:
        # No arguments provided, use defaults
        args = parser.parse_args([])
        show_progress = True
        print("No arguments provided, using sample text and default settings...")
        corpus = """
        The quick brown fox jumps over the lazy dog. The dog was not very happy about this.
        What the fox was thinking, nobody knows. It wasn't the first time this had happened,
        and it probably wouldn't be the last time either. The quick brown fox jumps over
        the lazy dog again. What the heck is going on with these animals?
        """

    # Basic analysis
    top_tokens = analyze_corpus(
        corpus, top_n=args.top_n, parallel=args.parallel, show_progress=show_progress
    )

    # Optionally perform advanced context analysis
    if args.advanced:
        layout = create_keyboard_layout()
        top_tokens = analyze_token_context(
            top_tokens, corpus, layout, show_progress=show_progress
        )

    # Print results
    if show_progress:
        print(f"\nTop {args.top_n} tokens by score:")
        print("-" * 60)
    print_results(top_tokens)

    # Output to a file if requested
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                for i, token in enumerate(top_tokens):
                    f.write(f"{format_token_result(i, token)}\n")
            if show_progress:
                print(f"\nResults saved to {args.output}")
        except Exception as e:
            print(f"Error writing to output file: {e}")


if __name__ == "__main__":
    main()
