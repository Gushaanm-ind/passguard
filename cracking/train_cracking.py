"""
PassGuard — Cracking Strategy Model Training

This model learns password-construction patterns so the
local PassGuard lab can prioritize controlled candidate
generation strategies.

The dataset is generated locally from transparent rules.
It is not a leaked-password database.
"""

from pathlib import Path
import pickle
import random
import re

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42
DATASET_SIZE = 250_000

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "cracking_model.pkl"
FEATURES_PATH = BASE_DIR / "cracking_features.pkl"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# Vocabulary
# ============================================================

COMMON_WORDS = [
    "password", "admin", "welcome", "hello", "football",
    "monkey", "dragon", "qwerty", "letmein", "baseball",
    "master", "summer", "winter", "computer", "shadow",
    "princess", "login", "secret", "orange", "testing",
    "flower", "sunshine", "freedom", "whatever", "access",
    "manager", "internet", "server", "security", "college",
    "student", "india", "cricket", "india123", "welcome",
    "friend", "love", "family", "school", "office",
    "welcomehome", "football", "soccer", "hockey", "cookie",
    "purple", "silver", "yellow", "green", "summer",
    "winter", "spring", "autumn", "computer", "laptop",
    "mobile", "camera", "gaming", "minecraft", "pokemon",
    "starwars", "superman", "batman", "matrix", "secret",
    "coffee", "chocolate", "apple", "banana", "orange",
]

YEARS = [
    str(year)
    for year in range(1980, 2027)
]

NUMBERS = [
    "0", "1", "7", "12", "21", "42", "69", "99",
    "123", "321", "007", "111", "222", "1234",
    "2024", "2025", "2026",
]

SYMBOLS = [
    "!", "@", "#", "$", "%", "&", "*", "?",
]

KEYBOARD_PATTERNS = [
    "qwerty",
    "qwertyui",
    "asdfgh",
    "asdfghj",
    "zxcvbn",
    "123456",
    "12345678",
    "qazwsx",
    "1qaz2wsx",
]


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
# Password generation helpers
# ============================================================

def random_case(word: str) -> str:
    choice = random.randint(0, 4)

    if choice == 0:
        return word.lower()

    if choice == 1:
        return word.capitalize()

    if choice == 2:
        return word.upper()

    if choice == 3:
        return word.swapcase()

    return "".join(
        char.upper() if random.random() < 0.5 else char.lower()
        for char in word
    )


def make_password() -> tuple:
    """
    Generate a password and its construction strategy.
    """

    word = random.choice(COMMON_WORDS)
    word2 = random.choice(COMMON_WORDS)
    number = random.choice(NUMBERS)
    year = random.choice(YEARS)
    symbol = random.choice(SYMBOLS)

    strategy = random.randint(0, 7)

    # --------------------------------------------------------
    # 0 — Dictionary
    # --------------------------------------------------------
    if strategy == 0:
        password = random_case(word)

    # --------------------------------------------------------
    # 1 — Word + Number
    # --------------------------------------------------------
    elif strategy == 1:
        password = (
            random_case(word)
            + number
        )

    # --------------------------------------------------------
    # 2 — Word + Year
    # --------------------------------------------------------
    elif strategy == 2:
        password = (
            random_case(word)
            + year
        )

    # --------------------------------------------------------
    # 3 — Word + Symbol
    # --------------------------------------------------------
    elif strategy == 3:
        password = (
            random_case(word)
            + symbol
        )

    # --------------------------------------------------------
    # 4 — Word + Number + Symbol
    # --------------------------------------------------------
    elif strategy == 4:
        password = (
            random_case(word)
            + number
            + symbol
        )

    # --------------------------------------------------------
    # 5 — Multiple Words
    # --------------------------------------------------------
    elif strategy == 5:

        separator = random.choice([
            "",
            "",
            "",
            "-",
            "_",
            ".",
        ])

        password = (
            random_case(word)
            + separator
            + random_case(word2)
        )

    # --------------------------------------------------------
    # 6 — Words + Number
    # --------------------------------------------------------
    elif strategy == 6:

        password = (
            random_case(word)
            + random_case(word2)
            + number
        )

    # --------------------------------------------------------
    # 7 — Keyboard Pattern
    # --------------------------------------------------------
    else:
        password = random.choice(KEYBOARD_PATTERNS)

    return password, strategy


# ============================================================
# Dataset creation
# ============================================================

def create_dataset(size: int) -> pd.DataFrame:

    rows = []

    for _ in range(size):

        password, strategy = make_password()

        rows.append({
            "password": password,
            "strategy": strategy,
            "strategy_name": STRATEGY_LABELS[strategy],
        })

    return pd.DataFrame(rows)


# ============================================================
# Feature extraction
# ============================================================

def extract_cracking_features(password: str) -> list:

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

    unique_chars = len(set(password))

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


FEATURE_NAMES = [
    "length",
    "upper",
    "lower",
    "digits",
    "symbols",
    "unique_chars",
    "unique_ratio",
    "digit_ratio",
    "symbol_ratio",
    "has_year",
    "has_number_suffix",
    "has_symbol_suffix",
    "has_keyboard_pattern",
    "repeated_chars",
    "transitions",
]


# ============================================================
# Train
# ============================================================

def train_model():

    print("=" * 65)
    print("PassGuard — Cracking Strategy Model Training")
    print("=" * 65)

    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------

    print("\n[1/6] Creating training dataset...")

    df = create_dataset(DATASET_SIZE)

    print(
        f"Dataset size: {len(df):,} passwords"
    )

    print("\nStrategy distribution:")

    print(
        df["strategy_name"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    print("\n[2/6] Extracting features...")

    X = np.array([
        extract_cracking_features(password)
        for password in df["password"]
    ])

    y = df["strategy"].to_numpy()

    print(
        f"Feature matrix: {X.shape}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    print("\n[3/6] Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(
        f"Training samples: {len(X_train):,}"
    )

    print(
        f"Testing samples:  {len(X_test):,}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n[4/6] Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=18,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Training complete.")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print("\n[5/6] Evaluating model...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=sorted(STRATEGY_LABELS),
            target_names=[
                STRATEGY_LABELS[i]
                for i in sorted(STRATEGY_LABELS)
            ],
            zero_division=0,
        )
    )

    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=sorted(STRATEGY_LABELS),
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\n[6/6] Saving model files...")

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    with open(FEATURES_PATH, "wb") as file:
        pickle.dump(FEATURE_NAMES, file)

    print(
        f"\nModel saved to:\n{MODEL_PATH}"
    )

    print(
        f"\nFeatures saved to:\n{FEATURES_PATH}"
    )

    print("\nTraining finished successfully.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    train_model()