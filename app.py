import os
import sys
import time
import threading
import urllib.request

import uvicorn
import webview

from Main import app as fastapi_app


# ============================================================
# Configuration
# ============================================================

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


# ============================================================
# Start FastAPI in the background
# ============================================================

def start_server():
    uvicorn.run(
        fastapi_app,
        host=HOST,
        port=PORT,
        log_level="warning",
    )


def wait_for_server(url, timeout=10):
    start = time.time()

    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return True
        except Exception:
            time.sleep(0.2)

    return False


# ============================================================
# Main application
# ============================================================

def main():
    print("=" * 55)
    print("  PassGuard — Local Security Lab")
    print("=" * 55)
    print(f"  Local server: {URL}")
    print("  Starting desktop application...")
    print("=" * 55)

    # Start FastAPI in a background thread
    server_thread = threading.Thread(
        target=start_server,
        daemon=True,
    )
    server_thread.start()

    # Wait until FastAPI is actually available
    if not wait_for_server(URL):
        print("\n[ERROR] PassGuard backend failed to start.")
        sys.exit(1)

    print("[passguard] Backend ready.")
    print("[passguard] Opening desktop window...")

    # Open the existing web interface inside a desktop window
    window = webview.create_window(
        "PassGuard — Local Security Lab",
        URL,
        width=1200,
        height=800,
        min_size=(900, 600),
    )

    webview.start()

    print("[passguard] Application closed.")


if __name__ == "__main__":
    main()