"""
PassGuard — Local Password Hash Cracking Lab

Controlled local demonstration only.

The engine:
- uses a target hash
- generates bounded local candidates
- hashes each candidate
- compares hashes
- reports progress through a callback

It does not connect to external systems.
"""

from pathlib import Path
import hashlib
import pickle
import re
import time


# ============================================================
# Paths / configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

WORDLIST_PATH = (
    BASE_DIR
    / "wordlists"
    / "common_passwords.txt"
)

MODEL_PATH = BASE_DIR / "cracking_model.pkl"

FEATURES_PATH = (
    BASE_DIR
    / "cracking_features.pkl"
)

MAX_ATTEMPTS = 5000


# ============================================================
# Load model
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Cracking model not found: {MODEL_PATH}"
    )

if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Cracking features not found: {FEATURES_PATH}"
    )


with open(MODEL_PATH, "rb") as file:
    cracking_model = pickle.load(file)


with open(FEATURES_PATH, "rb") as file:
    CRACKING_FEATURE_NAMES = pickle.load(file)


# ============================================================
# Strategy labels
# ============================================================

STRATEGY_LABELS = {
    0: "Dictionary",
    1: "Word + Number",
    2: "Word + Year",
    3: "Word + Symbol",
    4: "Word + Number + Symbol",
    5: "Multiple Words",
    6: "Multiple Words + Number",
    7: "Keyboard Pattern",
}


# ============================================================
# Local strategy data
# ============================================================

YEARS = tuple(
    str(year)
    for year in range(1980, 2027)
)

NUMBERS = (
    "0",
    "1",
    "7",
    "12",
    "21",
    "42",
    "69",
    "99",
    "123",
    "321",
    "007",
    "111",
    "222",
    "1234",
    "2024",
    "2025",
    "2026",
)

SYMBOLS = (
    "!",
    "@",
    "#",
    "$",
    "%",
    "&",
    "*",
    "?",
)

KEYBOARD_PATTERNS = (
    "qwerty",
    "qwertyui",
    "asdfgh",
    "asdfghj",
    "zxcvbn",
    "123456",
    "12345678",
    "qazwsx",
    "1qaz2wsx",
)


# ============================================================
# Hashing
# ============================================================

def hash_password(password: str) -> str:
    """
    SHA-256 demonstration hash.

    This is for the local lab only.
    It is not recommended for storing real passwords.
    """

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


# ============================================================
# Wordlist
# ============================================================

def load_wordlist(
    path: Path = WORDLIST_PATH,
) -> list:
    """Load unique non-empty candidates."""

    if not path.exists():
        raise FileNotFoundError(
            f"Wordlist not found: {path}"
        )

    candidates = []
    seen = set()

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        candidate = line.strip()

        if (
            candidate
            and candidate not in seen
        ):
            candidates.append(candidate)
            seen.add(candidate)

    return candidates


# ============================================================
# Feature extraction
# ============================================================

def extract_cracking_features(
    password: str,
) -> list:

    password = str(password)

    length = len(password)

    upper = sum(
        char.isupper()
        for char in password
    )

    lower = sum(
        char.islower()
        for char in password
    )

    digits = sum(
        char.isdigit()
        for char in password
    )

    symbols = sum(
        not char.isalnum()
        for char in password
    )

    unique_chars = len(
        set(password)
    )

    unique_ratio = (
        unique_chars / length
        if length
        else 0
    )

    digit_ratio = (
        digits / length
        if length
        else 0
    )

    symbol_ratio = (
        symbols / length
        if length
        else 0
    )

    has_year = int(
        bool(
            re.search(
                r"(19\d{2}|20\d{2})",
                password,
            )
        )
    )

    has_number_suffix = int(
        bool(
            re.search(
                r"\d+$",
                password,
            )
        )
    )

    has_symbol_suffix = int(
        bool(
            re.search(
                r"[^a-zA-Z0-9]$",
                password,
            )
        )
    )

    has_keyboard_pattern = int(
        any(
            pattern in password.lower()
            for pattern in KEYBOARD_PATTERNS
        )
    )

    repeated_chars = int(
        bool(
            re.search(
                r"(.)\1{2,}",
                password,
            )
        )
    )

    transitions = 0
    previous_type = None

    for char in password:

        if char.islower():
            current_type = "L"

        elif char.isupper():
            current_type = "U"

        elif char.isdigit():
            current_type = "D"

        else:
            current_type = "S"

        if (
            previous_type is not None
            and current_type != previous_type
        ):
            transitions += 1

        previous_type = current_type

    return [
        length,
        upper,
        lower,
        digits,
        symbols,
        unique_chars,
        unique_ratio,
        digit_ratio,
        symbol_ratio,
        has_year,
        has_number_suffix,
        has_symbol_suffix,
        has_keyboard_pattern,
        repeated_chars,
        transitions,
    ]


# ============================================================
# Strategy prediction
# ============================================================

def predict_strategy(
    password: str,
) -> dict:

    features = extract_cracking_features(
        password
    )

    prediction = int(
        cracking_model.predict(
            [features]
        )[0]
    )

    probabilities = (
        cracking_model.predict_proba(
            [features]
        )[0]
    )

    confidence = (
        float(
            probabilities[prediction]
        ) * 100
    )

    return {
        "strategy_id": prediction,
        "strategy": STRATEGY_LABELS.get(
            prediction,
            f"Unknown ({prediction})",
        ),
        "confidence": round(
            confidence,
            2,
        ),
    }


# ============================================================
# Candidate generation helpers
# ============================================================

def _case_variants(word: str) -> list:

    values = [
        word.lower(),
        word.capitalize(),
        word.upper(),
    ]

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(values))


def _dictionary_candidates(
    words: list,
) -> list:

    for word in words:

        yield word

        for variant in _case_variants(word):

            if variant != word:
                yield variant


def _word_number_candidates(
    words: list,
) -> list:

    for word in words:

        for variant in _case_variants(word):

            for number in NUMBERS:
                yield variant + number


def _word_year_candidates(
    words: list,
) -> list:

    for word in words:

        for variant in _case_variants(word):

            for year in YEARS:
                yield variant + year


def _word_symbol_candidates(
    words: list,
) -> list:

    for word in words:

        for variant in _case_variants(word):

            for symbol in SYMBOLS:
                yield variant + symbol


def _word_number_symbol_candidates(
    words: list,
) -> list:

    for word in words:

        for variant in _case_variants(word):

            for number in NUMBERS:

                for symbol in SYMBOLS:

                    yield (
                        variant
                        + number
                        + symbol
                    )


def _multiple_word_candidates(
    words: list,
) -> list:

    for first in words:

        for second in words:

            if first == second:
                continue

            first_variants = (
                _case_variants(first)
            )

            second_variants = (
                _case_variants(second)
            )

            for left in first_variants:

                for right in second_variants:

                    yield left + right
                    yield left + "-" + right
                    yield left + "_" + right


def _words_number_candidates(
    words: list,
) -> list:

    for candidate in _multiple_word_candidates(
        words
    ):

        for number in NUMBERS:

            yield candidate + number


def _keyboard_candidates() -> list:
    return list(
        KEYBOARD_PATTERNS
    )


# ============================================================
# Candidate generation
# ============================================================

def build_candidates(
    max_attempts: int = MAX_ATTEMPTS,
    preferred_strategy=None,
) -> list:
    """
    Build a bounded candidate list.

    The ML-selected strategy gets priority.
    """

    if max_attempts <= 0:
        return []

    words = load_wordlist()

    strategies = {
        0: lambda: _dictionary_candidates(words),
        1: lambda: _word_number_candidates(words),
        2: lambda: _word_year_candidates(words),
        3: lambda: _word_symbol_candidates(words),
        4: lambda: _word_number_symbol_candidates(words),
        5: lambda: _multiple_word_candidates(words),
        6: lambda: _words_number_candidates(words),
        7: lambda: _keyboard_candidates(),
    }

    order = list(
        strategies.keys()
    )

    if preferred_strategy in strategies:
        order.remove(
            preferred_strategy
        )
        order.insert(
            0,
            preferred_strategy
        )

    candidates = []
    seen = set()

    # Give the preferred ML strategy priority,
    # but reserve part of the candidate budget for
    # other strategies so a wrong prediction does
    # not prevent useful fallback strategies.

    if preferred_strategy in strategies:
        preferred_budget = max(1, int(max_attempts * 0.40))
    else:
        preferred_budget = 0

    for position, strategy_id in enumerate(order):

        generator = strategies[strategy_id]()

        if position == 0 and preferred_strategy in strategies:
            strategy_limit = preferred_budget
        else:
            remaining_strategies = len(order) - position
            remaining_budget = max_attempts - len(candidates)

            strategy_limit = max(
                1,
                remaining_budget // remaining_strategies
            )

        generated_for_strategy = 0

        for candidate in generator:

            if not candidate:
                continue

            if candidate in seen:
                continue

            candidates.append(candidate)
            seen.add(candidate)

            generated_for_strategy += 1

            if (
                generated_for_strategy >= strategy_limit
                or len(candidates) >= max_attempts
            ):
                break

        if len(candidates) >= max_attempts:
            break

    return candidates


# ============================================================
# Cracking
# ============================================================

def crack(
    target_hash: str,
    preferred_strategy=None,
    max_attempts: int = MAX_ATTEMPTS,
    progress_callback=None,
) -> dict:
    """
    Run a bounded local hash-recovery demonstration.

    progress_callback receives a dictionary containing
    the current status.
    """

    if not target_hash:
        raise ValueError(
            "Target hash cannot be empty."
        )

    candidates = build_candidates(
        max_attempts=max_attempts,
        preferred_strategy=preferred_strategy,
    )

    start_time = time.perf_counter()

    total = len(candidates)
    print(f"[debug] Actual candidate count: {total}")

    if progress_callback:
        progress_callback({
            "event": "start",
            "total_candidates": total,
        })

    for number, candidate in enumerate(
        candidates,
        start=1,
    ):

        candidate_hash = hash_password(
            candidate
        )

        matched = (
            candidate_hash == target_hash
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        speed = (
            number / elapsed
            if elapsed > 0
            else 0.0
        )

        if progress_callback:
            progress_callback({
                "event": "attempt",
                "number": number,
                "total_candidates": total,
                "candidate": candidate,
                "match": matched,
                "elapsed_seconds": elapsed,
                "candidates_per_second": speed,
            })

        if matched:

            return {
                "found": True,
                "attempts": number,
                "elapsed_seconds": round(
                    elapsed,
                    6,
                ),
                "candidates_per_second": round(
                    speed,
                    2,
                ),
                "password": candidate,
            }

    elapsed = (
        time.perf_counter()
        - start_time
    )

    speed = (
        total / elapsed
        if elapsed > 0
        else 0.0
    )

    return {
        "found": False,
        "attempts": total,
        "elapsed_seconds": round(
            elapsed,
            6,
        ),
        "candidates_per_second": round(
            speed,
            2,
        ),
    }


# ============================================================
# Local demonstration
# ============================================================

if __name__ == "__main__":

    test_password = input(
        "Enter a test password: "
    )

    print(
        "\n[model] Predicting password strategy..."
    )

    strategy = predict_strategy(
        test_password
    )

    print(
        f"[model] Strategy: "
        f"{strategy['strategy']}"
    )

    print(
        f"[model] Confidence: "
        f"{strategy['confidence']}%"
    )

    target_hash = hash_password(
        test_password
    )

    print(
        "\nTarget hash:"
    )

    print(target_hash)

    print(
        "\nStarting local cracking test..."
    )

    last_print = 0

    def show_progress(info):
        global last_print

        if info["event"] != "attempt":
            return

        number = info["number"]

        # Print every 10 attempts, plus the final match.
        if (
            number - last_print >= 10
            or info["match"]
        ):
            last_print = number

            print(
                "[attack] "
                f"{number}/{info['total_candidates']} "
                f"| "
                f"{info['candidates_per_second']:,.0f} "
                "candidates/sec"
            )

            if info["match"]:
                print(
                    "[attack] MATCH FOUND"
                )

    result = crack(
        target_hash,
        preferred_strategy=(
            strategy["strategy_id"]
        ),
        progress_callback=show_progress,
    )

    print("\nResult:")
    print(result)