
import math
import re
import string
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. Load PWLDS
# ============================================================

BASE_URL = "https://raw.githubusercontent.com/Infinitode/PWLDS/main/"

files = {
    0: "pwlds_very_weak.csv",
    1: "pwlds_weak.csv",
    2: "pwlds_average.csv",
    3: "pwlds_strong.csv",
    4: "pwlds_very_strong.csv"
}

samples = []

for label, filename in files.items():
    url = BASE_URL + filename

    class_df = pd.read_csv(url, nrows=20000)

    class_df = class_df[["Password"]].copy()
    class_df["Strength_Level"] = label

    samples.append(class_df)

    print(f"Loaded class {label}: {len(class_df)}")


df = pd.concat(samples, ignore_index=True)

# Remove missing and duplicate passwords
df = df.dropna(subset=["Password"])
df["Password"] = df["Password"].astype(str)
df = df.drop_duplicates(subset=["Password"])
df = df.reset_index(drop=True)

print("\nTotal passwords:", len(df))


# ============================================================
# 2. Feature Engineering
# ============================================================

COMMON_PASSWORDS = {
    "123456",
    "password",
    "123456789",
    "12345678",
    "12345",
    "1234567",
    "qwerty",
    "abc123",
    "football",
    "monkey",
    "letmein",
    "dragon",
    "111111",
    "baseball",
    "iloveyou",
    "admin",
    "welcome",
    "login",
    "hello",
    "password1",
    "qwerty123",
}


KEYBOARD_PATTERNS = [
    "qwerty",
    "asdf",
    "zxcv",
    "qaz",
    "wsx",
    "edc",
    "1234",
    "2345",
    "3456",
    "4567",
    "5678",
    "6789",
]


def extract_features(password):

    password = str(password)

    length = len(password)

    upper = sum(c.isupper() for c in password)
    lower = sum(c.islower() for c in password)
    digits = sum(c.isdigit() for c in password)
    symbols = sum(not c.isalnum() for c in password)

    unique_chars = len(set(password))

    unique_ratio = (
        unique_chars / length
        if length
        else 0
    )

    charset = 0

    if upper:
        charset += 26

    if lower:
        charset += 26

    if digits:
        charset += 10

    if symbols:
        charset += len(string.punctuation)

    entropy = (
        length * math.log2(charset)
        if charset > 0
        else 0
    )

    sequential = int(
        bool(
            re.search(
                r"(012|123|234|345|456|567|678|789|890|"
                r"abc|bcd|cde|def|efg|fgh|ghi|qwe|wer|"
                r"ert|rty|tyu|yui|uio|iop|asdf|sdfg)",
                password.lower()
            )
        )
    )

    repeated_chars = int(
        bool(re.search(r"(.)\1{2,}", password))
    )

    repeated_pattern = int(
        bool(re.search(r"(.{2,})\1", password))
    )

    is_common = int(
        password.lower() in COMMON_PASSWORDS
    )

    has_year = int(
        bool(re.search(r"(19\d{2}|20\d{2})", password))
    )

    lower_password = password.lower()

    keyboard_pattern = int(
        any(
            pattern in lower_password
            for pattern in KEYBOARD_PATTERNS
        )
    )

    def char_type(c):

        if c.islower():
            return "L"

        if c.isupper():
            return "U"

        if c.isdigit():
            return "D"

        return "S"

    types = [char_type(c) for c in password]

    transitions = sum(
        1
        for i in range(1, len(types))
        if types[i] != types[i - 1]
    )

    longest_run = 0

    if types:

        current_run = 1

        for i in range(1, len(types)):

            if types[i] == types[i - 1]:

                current_run += 1

            else:

                longest_run = max(
                    longest_run,
                    current_run
                )

                current_run = 1

        longest_run = max(
            longest_run,
            current_run
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

    has_letters = int(
        (upper + lower) > 0
    )

    has_digits = int(
        digits > 0
    )

    has_symbols = int(
        symbols > 0
    )

    single_type = int(
        sum([
            upper > 0,
            lower > 0,
            digits > 0,
            symbols > 0
        ]) == 1
    )

    return [
        length,
        upper,
        lower,
        digits,
        symbols,
        unique_chars,
        unique_ratio,
        entropy,
        sequential,
        repeated_chars,
        repeated_pattern,
        is_common,
        has_year,
        keyboard_pattern,
        transitions,
        longest_run,
        digit_ratio,
        symbol_ratio,
        has_letters,
        has_digits,
        has_symbols,
        single_type
    ]


FEATURE_NAMES = [
    "length",
    "upper",
    "lower",
    "digits",
    "symbols",
    "unique_chars",
    "unique_ratio",
    "entropy",
    "sequential",
    "repeated_chars",
    "repeated_pattern",
    "is_common",
    "has_year",
    "keyboard_pattern",
    "transitions",
    "longest_run",
    "digit_ratio",
    "symbol_ratio",
    "has_letters",
    "has_digits",
    "has_symbols",
    "single_type"
]


features = df["Password"].apply(extract_features)

X = pd.DataFrame(
    features.tolist(),
    columns=FEATURE_NAMES
)

y = df["Strength_Level"]


# ============================================================
# 3. Train / Validation / Test Split
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# ============================================================
# 4. Train Random Forest
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)


# ============================================================
# 5. Validation
# ============================================================

val_predictions = model.predict(X_val)

print(
    "\nValidation Accuracy:",
    accuracy_score(y_val, val_predictions)
)

print(
    classification_report(
        y_val,
        val_predictions,
        target_names=[
            "Very Weak",
            "Weak",
            "Average",
            "Strong",
            "Very Strong"
        ]
    )
)


# ============================================================
# 6. Final Test
# ============================================================

test_predictions = model.predict(X_test)

print(
    "Final Test Accuracy:",
    accuracy_score(y_test, test_predictions)
)

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Very Weak",
            "Weak",
            "Average",
            "Strong",
            "Very Strong"
        ]
    )
)


# ============================================================
# 7. Save Model
# ============================================================

with open("passguard_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("passguard_features.pkl", "wb") as f:
    pickle.dump(FEATURE_NAMES, f)

print("\n✅ passguard_model.pkl saved")
print("✅ passguard_features.pkl saved")
