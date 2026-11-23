"""
PassGuard — Local Password Security Lab
Backend: FastAPI (fully offline, no external calls)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import os
import pickle
import math
import re
import string

from cracking.cracker import (
    hash_password,
    predict_strategy,
    crack,
)

# ============================================================
# FastAPI setup
# ============================================================

app = FastAPI(title="PassGuard", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


# ============================================================
# Load trained model
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "passguard_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "passguard_features.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found. "
        "Download it from Google Colab first."
    )

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(FEATURES_PATH, "rb") as f:
        FEATURE_NAMES = pickle.load(f)

except Exception as e:
    raise RuntimeError(
        f"Failed to load PassGuard model files.\n"
        f"Model: {MODEL_PATH}\n"
        f"Features: {FEATURES_PATH}\n"
        f"Error: {e}"
    ) from e


# ============================================================
# Labels / colors
# ============================================================

LABELS = ["Very Weak", "Weak", "Average", "Strong", "Very Strong"]
COLORS = ["#ef4444", "#f97316", "#eab308", "#3b82f6", "#22c55e"]


# ============================================================
# Common passwords
# ============================================================

COMMON_PASSWORDS = {
    "123456", "password", "123456789", "12345678", "12345",
    "1234567", "qwerty", "abc123", "football", "monkey",
    "letmein", "dragon", "111111", "baseball", "iloveyou",
    "admin", "welcome", "login", "hello", "password1", "qwerty123",
}

KEYBOARD_PATTERNS = [
    "qwerty", "asdf", "zxcv", "qaz", "wsx", "edc",
    "1234", "2345", "3456", "4567", "5678", "6789",
]


# ============================================================
# Request model
# ============================================================

class PasswordRequest(BaseModel):
    password: str


# ============================================================
# Feature extraction — must match training code exactly
# ============================================================

def extract_features(password: str) -> list:
    password = str(password)
    length   = len(password)
    upper    = sum(c.isupper() for c in password)
    lower    = sum(c.islower() for c in password)
    digits   = sum(c.isdigit() for c in password)
    symbols  = sum(not c.isalnum() for c in password)

    unique_chars = len(set(password))
    unique_ratio = unique_chars / length if length else 0

    charset = 0
    if upper:   charset += 26
    if lower:   charset += 26
    if digits:  charset += 10
    if symbols: charset += len(string.punctuation)

    entropy = length * math.log2(charset) if charset > 0 else 0

    sequential = int(bool(re.search(
        r"(012|123|234|345|456|567|678|789|890|"
        r"abc|bcd|cde|def|efg|fgh|ghi|qwe|wer|"
        r"ert|rty|tyu|yui|uio|iop|asdf|sdfg)",
        password.lower()
    )))

    repeated_chars   = int(bool(re.search(r"(.)\1{2,}", password)))
    repeated_pattern = int(bool(re.search(r"(.{2,})\1", password)))
    is_common        = int(password.lower() in COMMON_PASSWORDS)
    has_year         = int(bool(re.search(r"(19\d{2}|20\d{2})", password)))
    keyboard_pattern = int(any(p in password.lower() for p in KEYBOARD_PATTERNS))

    def char_type(c):
        if c.islower(): return "L"
        if c.isupper(): return "U"
        if c.isdigit(): return "D"
        return "S"

    types       = [char_type(c) for c in password]
    transitions = sum(1 for i in range(1, len(types)) if types[i] != types[i - 1])

    longest_run, current_run = 0, 1
    for i in range(1, len(types)):
        if types[i] == types[i - 1]:
            current_run += 1
        else:
            longest_run  = max(longest_run, current_run)
            current_run  = 1
    longest_run = max(longest_run, current_run) if types else 0

    digit_ratio  = digits  / length if length else 0
    symbol_ratio = symbols / length if length else 0
    has_letters  = int((upper + lower) > 0)
    has_digits   = int(digits > 0)
    has_symbols  = int(symbols > 0)
    single_type  = int(sum([upper > 0, lower > 0, digits > 0, symbols > 0]) == 1)

    return [
        length, upper, lower, digits, symbols,
        unique_chars, unique_ratio, entropy,
        sequential, repeated_chars, repeated_pattern,
        is_common, has_year, keyboard_pattern,
        transitions, longest_run,
        digit_ratio, symbol_ratio,
        has_letters, has_digits, has_symbols, single_type,
    ]

# ============================================================
# Tips
# ============================================================

def get_tips(password: str) -> list:
    tips = []
    if len(password) < 12:
        tips.append("Use at least 12 characters")
    if not re.search(r"[A-Z]", password):
        tips.append("Add uppercase letters")
    if not re.search(r"[a-z]", password):
        tips.append("Add lowercase letters")
    if not re.search(r"\d", password):
        tips.append("Add numbers")
    if not re.search(r"[^a-zA-Z0-9]", password):
        tips.append("Add symbols (!@#$...)")
    if re.search(r"(012|123|234|345|456|567|678|789|abc|bcd|qwe|wer|ert)", password.lower()):
        tips.append("Avoid sequential patterns (123, abc, qwe)")
    if re.search(r"(.)\1{2,}", password):
        tips.append("Avoid repeated characters (aaa, 111)")
    if re.search(r"(.{2,})\1", password):
        tips.append("Avoid repeated patterns")
    if password.lower() in COMMON_PASSWORDS:
        tips.append("This is a commonly used password — change it immediately")
    if re.search(r"(19\d{2}|20\d{2})", password):
        tips.append("Avoid predictable year patterns")
    if not tips:
        tips.append("Looks good! Strong password.")
    return tips


# ============================================================
# Local password-guessing simulation
# ============================================================

def build_simulation_candidates(password: str, max_attempts: int = 5000) -> list:
    """
    Build a controlled list of local password guesses.

    This is an educational simulation only.
    It does not connect to or attack any external system.
    """

    candidates = []

    def add(candidate: str):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    # --------------------------------------------------------
    # 1. Common passwords
    # --------------------------------------------------------
    for candidate in sorted(COMMON_PASSWORDS):
        add(candidate)

        if len(candidates) >= max_attempts:
            return candidates

    # --------------------------------------------------------
    # 2. Simple predictable variations
    # --------------------------------------------------------
    base_words = [
        "password",
        "admin",
        "welcome",
        "hello",
        "qwerty",
        "letmein",
        "football",
        "monkey",
        "dragon",
    ]

    suffixes = [
        "",
        "1",
        "12",
        "123",
        "1234",
        "2024",
        "2025",
        "2026",
    ]

    for word in base_words:
        variations = [
            word,
            word.capitalize(),
            word.upper(),
        ]

        for variant in variations:
            for suffix in suffixes:
                add(variant + suffix)

                if len(candidates) >= max_attempts:
                    return candidates

    # --------------------------------------------------------
    # 3. Very small brute-force demonstration
    # --------------------------------------------------------
    # Deliberately limited so the application cannot generate
    # an enormous search space.
    charset = "abcdefghijklmnopqrstuvwxyz"

    for first in charset:
        add(first)

        if len(candidates) >= max_attempts:
            return candidates

    return candidates

def run_simulation(password: str, max_attempts: int = 5000) -> dict:
    candidates = build_simulation_candidates(
        password,
        max_attempts=max_attempts
    )

    attempts = []

    for index, candidate in enumerate(candidates, start=1):
        attempts.append({
            "number": index,
            "candidate": candidate,
            "match": candidate == password,
        })

        if candidate == password:
            return {
                "found": True,
                "attempts": index,
                "total_candidates": len(candidates),
                "attempt_log": attempts,
            }

    return {
        "found": False,
        "attempts": len(candidates),
        "total_candidates": len(candidates),
        "attempt_log": attempts,
    }


# ============================================================
# Analyze endpoint
# ============================================================

@app.post("/analyze")
def analyze(req: PasswordRequest):
    password = req.password
    if not password:
        return {"error": "Empty password"}

    features     = extract_features(password)
    label_idx    = int(model.predict([features])[0])
    probabilities = model.predict_proba([features])[0]
    confidence   = round(float(probabilities[label_idx]) * 100, 1)
    entropy      = round(features[7], 2)

    return {
        "label":      LABELS[label_idx],
        "color":      COLORS[label_idx],
        "confidence": confidence,
        "entropy":    entropy,
        "is_common":   bool(password.lower() in COMMON_PASSWORDS),
        "length":     len(password),
        "has_upper":  bool(re.search(r"[A-Z]", password)),
        "has_lower":  bool(re.search(r"[a-z]", password)),
        "has_digit":  bool(re.search(r"\d", password)),
        "has_symbol": bool(re.search(r"[^a-zA-Z0-9]", password)),
        "tips":       get_tips(password),
        "score":      label_idx,
        "class_probabilities": {
            LABELS[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(LABELS))
        },
    }


# ============================================================
# Local password cracking lab
# ============================================================

@app.post("/crack")
def crack_password(req: PasswordRequest):
    password = req.password

    if not password:
        return {"error": "Empty password"}

    # --------------------------------------------------------
    # 1. Use the cracking ML model to prioritize a strategy
    # --------------------------------------------------------

    strategy = predict_strategy(password)

    # --------------------------------------------------------
    # 2. Create the target hash locally
    #
    # The cracking engine works against the hash rather than
    # comparing the plaintext directly.
    # --------------------------------------------------------

    target_hash = hash_password(password)

    # --------------------------------------------------------
    # 3. Run bounded local cracking
    # --------------------------------------------------------

    result = crack(
        target_hash=target_hash,
        preferred_strategy=strategy["strategy_id"],
        max_attempts=5000,
    )

    # --------------------------------------------------------
    # 4. Return only the useful lab information
    # --------------------------------------------------------

    return {
        "found": result["found"],
        "strategy": strategy["strategy"],
        "strategy_id": strategy["strategy_id"],
        "confidence": strategy["confidence"],
        "attempts": result["attempts"],
        "elapsed_seconds": result["elapsed_seconds"],
        "candidates_per_second": result["candidates_per_second"],
    }


# ============================================================
# Simulation endpoint
# ============================================================

@app.post("/simulate")
def simulate(req: PasswordRequest):
    password = req.password

    if not password:
        return {"error": "Empty password"}

    return run_simulation(password)


# ============================================================
# Serve index.html locally (no FileResponse dependency on disk)
# ============================================================

@app.get("/", response_class=HTMLResponse)
def root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)