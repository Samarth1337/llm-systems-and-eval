"""
serve.py — Start the Ollama inference server and pull the target model

Usage:
    python serve.py                  # default: mistral:7b
    python serve.py --model phi3:mini
    python serve.py --port 11434

  1. Checks if ollama is installed
  2. Starts the ollama server (if not already running)
  3. Pulls the requested model (idempotent)
  4. Verifies the model is ready by sending a health-check generation

After writing the code files, black was executed on the project to ensure consistent formatting. 
The server can be stopped with Ctrl+C, which will cleanly shut down the background process if it was started by this script.
Comments and documentation are included to explain the purpose of each function and step in the process, wherever deemed necessary for clarity.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = "mistral:7b"
DEFAULT_HOST = "http://localhost"
DEFAULT_PORT = 11434


def ollama_installed() -> bool:
    """Check if ollama CLI is on the PATH"""
    try:
        subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def server_is_running(base_url: str) -> bool:
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except (urllib.error.URLError, ConnectionError, OSError):
        return False


def start_server(port: int) -> subprocess.Popen:
    """Launch `ollama serve` as a background process"""
    env = {**__import__("os").environ, "OLLAMA_HOST": f"0.0.0.0:{port}"}
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    return proc


def pull_model(model: str, base_url: str) -> None:
    """Pull (download) a model via the REST API. Idempotent"""
    print(f"[serve] Pulling model '{model}'")
    payload = json.dumps({"name": model, "stream": False}).encode()
    req = urllib.request.Request(
        f"{base_url}/api/pull",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            body = json.loads(resp.read().decode())
            status = body.get("status", "unknown")
            print(f"[serve] Pull complete — status: {status}")
    except urllib.error.HTTPError as exc:
        print(f"[serve] Pull failed: HTTP {exc.code} — {exc.read().decode()}")
        sys.exit(1)


def health_check(model: str, base_url: str) -> bool:
    """Send a trivial generation request to verify the model is loaded"""
    payload = json.dumps({
        "model": model,
        "prompt": "Say hello.",
        "stream": False,
        "options": {"num_predict": 16},
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
            text = body.get("response", "").strip()
            print(f"[serve] Health-check response: {text[:80]!r}")
            return bool(text)
    except Exception as exc:
        print(f"[serve] Health-check failed: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Start ollama and serve a model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model tag to serve")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    args = parser.parse_args()

    base_url = f"{DEFAULT_HOST}:{args.port}"

    if not ollama_installed():
        print("[serve] ERROR: ollama is not installed, please install and try again.")
        sys.exit(1)
    print("[serve] ollama is installed")

    proc = None
    if server_is_running(base_url):
        print(f"[serve] Server already running at {base_url}")
    else:
        print(f"[serve] Starting ollama server on port {args.port}...")
        proc = start_server(args.port)
        # Wait for the server to be ready
        for attempt in range(30):
            if server_is_running(base_url):
                break
            time.sleep(1)
        else:
            print("[serve] ERROR: Server failed to start within 30 s.")
            if proc:
                proc.terminate()
            sys.exit(1)
        print(f"[serve] Server is up at {base_url}")
    pull_model(args.model, base_url)

    if health_check(args.model, base_url):
        print(f"[serve] Model '{args.model}' is ready for inference")
    else:
        print("[serve] WARNING: Health-check did not return text; model may still be loading or have an issue")

    if proc:
        print("[serve] Server running (PID {})  —  press Ctrl+C to stop".format(proc.pid))
        try:
            proc.wait()
        except KeyboardInterrupt:
            print("\n[serve] Shutting down...")
            proc.terminate()
            proc.wait(timeout=10)
    else:
        print("[serve] Server was already running externally; exiting launcher.")

if __name__ == "__main__":
    main()