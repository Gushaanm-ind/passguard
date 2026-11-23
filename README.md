# 🔐 PassGuard

Ever wondered if your password is actually good, or just feels good? PassGuard tells you, then tries to crack it so you can see how it holds up.

It's a local, offline password strength analyzer and cracking simulator. Type a password into the web UI, and a FastAPI backend scores it with a trained machine-learning model and simulates how fast an attacker could break it.

**Your passwords never leave your machine.** The server binds to `127.0.0.1:8000` and the UI only talks to that origin. No accounts, no database, no telemetry, no external network calls at runtime.

## What It Does

- **Strength analysis:** label, confidence, entropy, a 0–100 score, and per-class probabilities from a trained model
- **Requirement checklist and tips** so you know what to fix
- **Cracking simulation:** a SHA-256 based attack with 8 strategies, capped at 5,000 attempts
- **Smart strategy pick:** a classifier chooses the attack strategy it thinks is most likely to work and gives it 40% of the attempt budget
- **A single-page UI** with show/hide input, an animated strength meter, a class-probability breakdown, an attack-simulation terminal, and a password generator
- **Two ways to run:** in your browser or in a native 1200×800 desktop window (pywebview)

## Quick Start

You'll need Python 3.9+.

```bash
cd Passguard
pip install -r requirements.txt
```

Run it in your browser (it opens automatically):

```bash
python3 launcher.py
```

Or as a desktop window:

```bash
python3 app.py
```

Heads up: the four `.pkl` model files and `cracking/wordlists/common_passwords.txt` need to be present, or it won't start.

### Retraining the Models

Want to train your own? Go for it:

```bash
python3 train.py
python3 cracking/train_cracking.py
```

`train.py` downloads the PWLDS dataset from the public [Infinitode/PWLDS](https://github.com/Infinitode/PWLDS) repo, so it needs internet access when you run it. Everything else stays offline.

## API

| Method | Endpoint    | What it does |
|--------|-------------|--------------|
| GET    | `/`         | Serves the UI |
| POST   | `/analyze`  | Strength label, confidence, entropy, 0–100 score, class probabilities, requirement flags, tips |
| POST   | `/crack`    | Runs a simulated attack |
| POST   | `/simulate` | Simulation endpoint |

CORS is restricted to `127.0.0.1:8000` and `localhost:8000`.

## Project Structure

```
PassGuard/
├── Main.py                  # FastAPI backend, loads models at startup
├── Index.html               # Single-page UI (vanilla HTML/CSS/JS)
├── launcher.py              # Checks deps and models, starts uvicorn, opens browser
├── app.py                   # Server in a thread + pywebview desktop window
├── train.py                 # Trains the strength model
├── passguard_model.pkl      # Strength model
├── passguard_features.pkl   # Feature definitions
├── LICENSE
└── cracking/
    ├── cracker.py           # Bounded SHA-256 crack engine
    ├── train_cracking.py    # Trains the strategy classifier
    ├── *.pkl                # Cracking models
    └── wordlists/
        └── common_passwords.txt
```

## Under the Hood

- **Strength model:** RandomForest with 300 estimators on 22 engineered features, trained on PWLDS with 20,000 samples per class
- **Strategy classifier:** 8-class model trained on 250k synthetic passwords
- **Launcher:** checks dependencies and model files, starts uvicorn on `127.0.0.1:8000`, polls until the server responds, then opens your default browser

## Tech Stack

Python 3.9+, FastAPI, Uvicorn, scikit-learn, pandas, pywebview, vanilla HTML/CSS/JS

## License

Released under the [MIT License](LICENSE).

---

Built as a learning project. If your password scores badly, that's the tool working as intended. 😄
