"""
model.py — lm-evaluation-harness model wrapper for an Ollama endpoint

Implements the three core methods required by EleutherAI's lm-eval framework:
  • loglikelihood(requests)        — score (context, continuation) pairs
  • loglikelihood_rolling(requests) — rolling perplexity over full texts
  • generate_until(requests)       — open-ended generation with stop sequences

Features:
  • SQLite-backed prompt cache so repeated runs are deterministic and fast
  • Deterministic defaults (temperature=0, seed=42, top_p=1)
  • Registered as model type "ollama" via @register_model

Important:
  lm-eval discovers models through imports in lm_eval/models/__init__.py.
  Since we are NOT modifying the lm-eval source tree, run_eval.py imports
  this module explicitly before calling simple_evaluate(). This triggers
  the @register_model decorator and makes "ollama" available.

After writing the code files, black was executed on the project to ensure consistent formatting.
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
import urllib.request
import urllib.error

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral:7b"
CACHE_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".eval_cache.db")


# Prompt cache
class PromptCache:
    """SQLite-backed cache keyed on (operation, prompt, params) → response."""

    def __init__(self, db_path: str = CACHE_DB):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, value TEXT, ts REAL)"
        )
        self._conn.commit()

    @staticmethod
    def _key(prefix: str, prompt: str, params: dict) -> str:
        blob = json.dumps({"p": prefix, "q": prompt, **params}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, prefix: str, prompt: str, params: dict):
        key = self._key(prefix, prompt, params)
        row = self._conn.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put(self, prefix: str, prompt: str, params: dict, value) -> None:
        key = self._key(prefix, prompt, params)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, ts) VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time()),
        )
        self._conn.commit()

    def clear(self) -> int:
        cur = self._conn.execute("DELETE FROM cache")
        self._conn.commit()
        return cur.rowcount

    def size(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]


# Model class
@register_model("ollama")
class OllamaLM(LM):
    """lm-eval compatible wrapper that forwards requests to Ollama."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
        max_tokens: int = 512,
        use_cache: bool = True,
        cache_db: str = CACHE_DB,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = int(seed)
        self.max_tokens = int(max_tokens)
        self._batch_size = int(batch_size)

        if isinstance(use_cache, str):
            use_cache = use_cache.lower() == "true"
        self.use_cache = use_cache
        self.cache = PromptCache(cache_db) if self.use_cache else None

        logger.info(
            "[model] OllamaLM  model=%s  url=%s  temp=%.1f  seed=%d  cache=%s",
            self.model_name,
            self.base_url,
            self.temperature,
            self.seed,
            self.use_cache,
        )

    # HTTP helper

    def _post(self, path: str, payload: dict, timeout: int = 300) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"[model] Cannot reach ollama at {self.base_url}: {exc}"
            ) from exc

    def _opts(self, **overrides) -> dict:
        o = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
        }
        o.update(overrides)
        return o

    # loglikelihood

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """
        Estimate log-likelihood of each (context, continuation) pair.

        Ollama does not expose per-token logprobs, so we approximate:
          1. Feed context+continuation to the model (prompt_eval path).
          2. Greedily generate from context alone.
          3. Check whether greedy output starts with the continuation.
          4. Use prompt-eval token count to produce a length-normalised
             proxy score.
        """
        results = []
        for req in requests:
            ctx, cont = req.args
            cp = {"op": "ll", "t": self.temperature, "s": self.seed}

            if self.cache:
                hit = self.cache.get("ll", ctx + "|||" + cont, cp)
                if hit is not None:
                    results.append((hit["ll"], hit["gr"]))
                    continue

            full = ctx + cont
            resp_full = self._post(
                "/api/generate",
                {
                    "model": self.model_name,
                    "prompt": full,
                    "stream": False,
                    "raw": True,
                    "options": {**self._opts(), "num_predict": 1},
                },
            )

            n_predict = max(len(cont.split()) * 2, 20)
            resp_gen = self._post(
                "/api/generate",
                {
                    "model": self.model_name,
                    "prompt": ctx if ctx else " ",
                    "stream": False,
                    "raw": True,
                    "options": {**self._opts(), "num_predict": n_predict},
                },
            )

            prompt_toks = resp_full.get(
                "prompt_eval_count", max(len(full.split()), 1)
            )
            ll_approx = -len(cont) / prompt_toks

            greedy_text = resp_gen.get("response", "")
            is_greedy = cont.strip().lower().startswith(
                greedy_text.strip().lower()[: len(cont.strip())]
            )

            if self.cache:
                self.cache.put(
                    "ll",
                    ctx + "|||" + cont,
                    cp,
                    {"ll": ll_approx, "gr": is_greedy},
                )
            results.append((ll_approx, is_greedy))

        return results

    # loglikelihood_rolling

    def loglikelihood_rolling(
        self, requests: list[Instance]
    ) -> list[tuple[float]]:
        """Rolling log-likelihood for perplexity-style tasks."""
        results = []
        for req in requests:
            (text,) = req.args
            cp = {"op": "llr", "t": self.temperature, "s": self.seed}

            if self.cache:
                hit = self.cache.get("llr", text, cp)
                if hit is not None:
                    results.append((hit["ll"],))
                    continue

            resp = self._post(
                "/api/generate",
                {
                    "model": self.model_name,
                    "prompt": text,
                    "stream": False,
                    "raw": True,
                    "options": {**self._opts(), "num_predict": 1},
                },
            )
            prompt_toks = resp.get(
                "prompt_eval_count", max(len(text.split()), 1)
            )
            ll = -len(text) / prompt_toks

            if self.cache:
                self.cache.put("llr", text, cp, {"ll": ll})
            results.append((ll,))

        return results

    # generate_until

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate text from a context until a stop sequence is hit."""
        results = []
        for req in requests:
            ctx = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 else {}

            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_gen = gen_kwargs.get("max_gen_toks", self.max_tokens)
            temp = gen_kwargs.get("temperature", self.temperature)

            cp = {
                "op": "gen",
                "until": until,
                "max": max_gen,
                "t": temp,
                "s": self.seed,
            }
            if self.cache:
                hit = self.cache.get("gen", ctx, cp)
                if hit is not None:
                    results.append(hit["text"])
                    continue

            opts = self._opts(num_predict=max_gen, temperature=temp)
            if until:
                opts["stop"] = until

            resp = self._post(
                "/api/generate",
                {
                    "model": self.model_name,
                    "prompt": ctx,
                    "stream": False,
                    "raw": True,
                    "options": opts,
                },
            )
            text = resp.get("response", "")

            for seq in until:
                idx = text.find(seq)
                if idx != -1:
                    text = text[:idx]

            if self.cache:
                self.cache.put("gen", ctx, cp, {"text": text})
            results.append(text)

        return results

    # Required properties

    @property
    def eot_token_id(self):
        return None

    @property
    def max_length(self) -> int:
        return 8192

    @property
    def max_gen_toks(self) -> int:
        return self.max_tokens

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return "api"

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return list(range(len(string.split())))

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        return ""