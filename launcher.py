"""
PassGuard — Cross-platform local launcher
Works on Windows, macOS, Linux.

Usage:
    python launcher.py

This starts the FastAPI backend and opens the browser automatically.
"""

import os
import sys
import time
import signal
import subprocess
import webbrowser
import threading
import platform
import urllib.request


HOST = "127.0.0.1"
PORT = 8000
URL  = f"http://{HOST}:{PORT}"


def check_dependencies():
    missing = []
    for pkg in ["fastapi", "uvicorn", "pydantic", "sklearn", "pandas"]:
        try:
            if pkg == "sklearn":
                __import__("sklearn")
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg if pkg != "sklearn" else "scikit-learn")
    return missing


def install_dependencies(missing):
    print("[ERROR] Missing required packages:")
    for pkg in missing:
        print(f"  - {pkg}")

    print("\nPassGuard is designed to run offline.")
    print("Install the required packages once before using offline mode.")
    print("Example:")
    print("  pip install fastapi uvicorn pydantic scikit-learn")

    sys.exit(1)


def open_browser_when_ready(url, timeout=30):
    """Wait until the server actually answers, then open the browser."""
    def _open():
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1):
                    break
            except Exception:
                time.sleep(0.2)
        else:
            print(f"[passguard] Server did not start in time. Open {url} manually.")
            return

        print(f"[passguard] Opening browser at {url}")
        webbrowser.open(url)

    t = threading.Thread(target=_open, daemon=True)
    t.start()


def main():
    print("=" * 52)
    print("  PassGuard — Local Password Security Lab")
    print("=" * 52)
    print(f"  Platform : {platform.system()} {platform.release()}")
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  URL      : {URL}")
    print("=" * 52)

    # Check model files exist
    for f in ["passguard_model.pkl", "passguard_features.pkl"]:
        if not os.path.exists(f):
            print(f"\n[ERROR] Missing file: {f}")
            print("  Train the model first in Google Colab,")
            print("  then download both .pkl files here.")
            sys.exit(1)

    # Check / install dependencies
    missing = check_dependencies()
    if missing:
        install_dependencies(missing)

    print("\n[passguard] Starting server... (Ctrl+C to stop)\n")

    # Build uvicorn command
    cmd = [
        sys.executable, "-m", "uvicorn",
        "Main:app",
        "--host", HOST,
        "--port", str(PORT),
        "--reload"
    ]

    try:
        proc = subprocess.Popen(cmd)

        # Open the browser only once the server is actually up
        open_browser_when_ready(URL)

        proc.wait()
    except KeyboardInterrupt:
        print("\n[passguard] Shutting down...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[passguard] Stopped.")
    except FileNotFoundError:
        print("[ERROR] uvicorn not found. Run: pip install uvicorn")
        sys.exit(1)


if __name__ == "__main__":
    main()