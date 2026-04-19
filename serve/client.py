"""
client.py — Sample client that exercises the Ollama inference endpoint

Usage:
    python client.py
    python client.py --model mistral:7b --base-url http://localhost:11434

  • Single-turn generation
  • Chat-style multi-turn conversation
  • Streaming token output
  • Parameter overrides for given constraints (temperature, top_p, seed)

After writing the code files, black was executed on the project to ensure consistent formatting. 
Comments and documentation are included to explain the purpose of each function and step in the process, wherever deemed necessary for clarity.
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

DEFAULT_MODEL = "mistral:7b"
DEFAULT_BASE_URL = "http://localhost:11434"

# Low-level helpers (functions meant to be used internally by the high-level API wrappers below)
def _post(url: str, payload: dict, timeout: int = 120) -> dict:
    """Send a non-streaming POST and return the parsed JSON body."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _post_stream(url: str, payload: dict, timeout: int = 120):
    """Yield parsed JSON objects from a streaming (NDJSON) POST."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if line:
                yield json.loads(line)

# High-level API wrappers
def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    *,
    stream: bool = False,
    **options,
) -> str:
    """Run a single /api/generate call. Returns the full response text."""
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": options if options else {},
    }
    if not stream:
        body = _post(url, payload)
        return body.get("response", "")
    else:
        tokens = []
        for chunk in _post_stream(url, {**payload, "stream": True}):
            token = chunk.get("response", "")
            tokens.append(token)
            print(token, end="", flush=True)
        print("\n\n")  # newline after streaming
        return "".join(tokens)


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    **options,
) -> str:
    """Send a multi-turn chat via /api/chat. Returns assistant reply."""
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options if options else {},
    }
    body = _post(url, payload)
    return body.get("message", {}).get("content", "")

# Demo scenarios
def demo_single_generation(model: str, base_url: str) -> None:
    """Simple single-turn prompt."""
    print("=" * 60)
    print("Demo 1: Single-turn generation")
    print("=" * 60)
    prompt = "Explain the difference between compiled and interpreted languages in two sentences."
    print(f"Prompt : {prompt}\n")
    t0 = time.perf_counter()
    response = generate(prompt, model=model, base_url=base_url)
    elapsed = time.perf_counter() - t0
    print(f"Response: {response.strip()}")
    print(f"[{elapsed:.2f} s]\n")


def demo_streaming(model: str, base_url: str) -> None:
    """Streaming token-by-token output."""
    print("=" * 60)
    print("Demo 2: Streaming generation")
    print("=" * 60)
    prompt = "Write a haiku about machine learning."
    print(f"Prompt : {prompt}\n")
    t0 = time.perf_counter()
    generate(prompt, model=model, base_url=base_url, stream=True)
    elapsed = time.perf_counter() - t0
    print(f"[{elapsed:.2f} s]\n")


def demo_chat(model: str, base_url: str) -> None:
    """Multi-turn chat conversation."""
    print("=" * 60)
    print("Demo 3: Multi-turn chat")
    print("=" * 60)
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    print(f"User   : {messages[0]['content']}")
    reply1 = chat(messages, model=model, base_url=base_url)
    print(f"Assist : {reply1.strip()}\n")

    messages.append({"role": "assistant", "content": reply1})
    messages.append({"role": "user", "content": "What is its population?"})
    print(f"User   : {messages[-1]['content']}")
    reply2 = chat(messages, model=model, base_url=base_url)
    print(f"Assist : {reply2.strip()}")
    print()


def demo_deterministic(model: str, base_url: str) -> None:
    """Show deterministic output with fixed seed and temperature=0."""
    print("=" * 60)
    print("Demo 4: Deterministic mode (seed=42, temperature=0)")
    print("=" * 60)
    prompt = "What is 12 * 15?"
    opts = {"seed": 42, "temperature": 0, "top_p": 1.0}
    results = []
    for i in range(3):
        resp = generate(prompt, model=model, base_url=base_url, **opts)
        results.append(resp.strip())
        print(f"  Run {i+1}: {results[-1][:120]}")
    if len(set(results)) == 1:
        print("  → All runs identical")
    else:
        print("  → WARNING: Non-determinism detected across runs")
    print()


def demo_parameter_sweep(model: str, base_url: str) -> None:
    """Compare different temperature settings."""
    print("=" * 60)
    print("Demo 5: Temperature sweep")
    print("=" * 60)
    prompt = "Describe the color blue in one sentence."
    for temp in [0.0, 0.5, 1.0, 1.5]:
        resp = generate(prompt, model=model, base_url=base_url, temperature=temp)
        print(f"  temp={temp:.1f} → {resp.strip()[:120]}")
    print()

def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama client demo")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    print(f"\nUsing model={args.model}  base_url={args.base_url}\n")

    try:
        demo_single_generation(args.model, args.base_url)
        demo_streaming(args.model, args.base_url)
        demo_chat(args.model, args.base_url)
        demo_deterministic(args.model, args.base_url)
        demo_parameter_sweep(args.model, args.base_url)
    except urllib.error.URLError:
        print("ERROR: Cannot reach the ollama server. Run `python serve.py` first.")
        sys.exit(1)

    print("All demos complete")


if __name__ == "__main__":
    main()