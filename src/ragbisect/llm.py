"""One provider-agnostic LLM wrapper. Everything that costs money goes through here.

* Providers: ``openai`` (also any OpenAI-compatible base URL: Ollama, vLLM,
  OpenRouter, ...) and ``anthropic``. Selected by a ``provider:model`` string.
* Every completion and embedding is cached in SQLite, keyed by the full
  request. Re-running an eval costs nothing unless something changed.
* Token usage is tracked so the CLI can print what a run spent.
* Zero third-party dependencies: plain ``urllib``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "openai:gpt-5.4-mini"
DEFAULT_EMBED_MODEL = "openai:text-embedding-3-small"

# USD per 1M tokens: (input, output). Only models we are confident about;
# anything else reports tokens without a dollar figure.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}


class LLMError(Exception):
    pass


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embed_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0
    _by_model: dict[str, tuple[int, int]] = field(default_factory=dict)

    def add(self, model: str, prompt: int, completion: int, *, embed: bool = False) -> None:
        self.calls += 1
        if embed:
            self.embed_tokens += prompt
        else:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
        a, b = self._by_model.get(model, (0, 0))
        self._by_model[model] = (a + prompt, b + completion)

    def cost_usd(self) -> float | None:
        total = 0.0
        for model, (inp, out) in self._by_model.items():
            price = PRICES.get(model)
            if price is None:
                return None
            total += inp * price[0] / 1e6 + out * price[1] / 1e6
        return total

    def summary(self) -> str:
        parts = [
            f"{self.calls} API calls",
            f"{self.cache_hits} cache hits",
            f"{self.prompt_tokens + self.completion_tokens:,} completion tokens "
            f"({self.prompt_tokens:,} in / {self.completion_tokens:,} out)",
            f"{self.embed_tokens:,} embedding tokens",
        ]
        cost = self.cost_usd()
        parts.append(f"≈ ${cost:.4f}" if cost is not None else "cost: unknown model pricing")
        return "; ".join(parts)


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._conn.commit()

    @staticmethod
    def key(*parts: object) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value))
            self._conn.commit()


def parse_model(spec: str) -> tuple[str, str]:
    if ":" in spec:
        provider, model = spec.split(":", 1)
        return provider.lower(), model
    if spec.startswith("claude"):
        return "anthropic", spec
    return "openai", spec


class LLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        cache_path: str | Path = ".ragbisect/cache.sqlite",
        temperature: float = 0.0,
        reasoning_effort: str | None = "low",
        timeout: float = 120.0,
    ):
        self.provider, self.model = parse_model(model)
        self.embed_provider, self.embed_model = parse_model(embed_model)
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.usage = Usage()
        self._cache = _Cache(Path(cache_path))
        self._usage_lock = threading.Lock()

    # ------------------------------------------------------------------ http
    def _post(self, url: str, headers: dict[str, str], body: dict) -> dict:
        data = json.dumps(body).encode()
        delay = 1.0
        for attempt in range(6):
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")
                if e.code in (408, 409, 429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                raise LLMError(f"HTTP {e.code} from {url}: {text[:500]}") from e
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                if attempt < 5:
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                raise LLMError(f"network error calling {url}: {e}") from e
        raise LLMError("unreachable")

    @staticmethod
    def _env(name: str) -> str:
        val = os.environ.get(name)
        if not val:
            raise LLMError(f"{name} is not set")
        return val

    # ------------------------------------------------------------ completion
    def complete(self, prompt: str, *, system: str | None = None, json_mode: bool = True) -> str:
        key = self._cache.key("complete", self.provider, self.model, system, prompt, json_mode, self.temperature, self.reasoning_effort)
        cached = self._cache.get(key)
        if cached is not None:
            with self._usage_lock:
                self.usage.cache_hits += 1
            return cached

        if self.provider == "openai":
            text, inp, out = self._openai_complete(prompt, system, json_mode)
        elif self.provider == "anthropic":
            text, inp, out = self._anthropic_complete(prompt, system, json_mode)
        else:
            raise LLMError(f"unknown provider {self.provider!r} (use openai:<model> or anthropic:<model>)")

        with self._usage_lock:
            self.usage.add(self.model, inp, out)
        self._cache.put(key, text)
        return text

    def _openai_complete(self, prompt: str, system: str | None, json_mode: bool) -> tuple[str, int, int]:
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body: dict = {"model": self.model, "messages": messages}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        reasoning = self.model.startswith(("gpt-5", "o1", "o3", "o4"))
        if reasoning and self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if not reasoning:
            # Reasoning models only accept the default temperature.
            body["temperature"] = self.temperature
        headers = {"Authorization": f"Bearer {self._env('OPENAI_API_KEY')}"}
        try:
            resp = self._post(f"{base}/chat/completions", headers, body)
        except LLMError as e:
            msg = str(e)
            # Unknown model families: retry once without the parameter the server rejected.
            retry = False
            for param in ("temperature", "reasoning_effort", "response_format"):
                if "HTTP 400" in msg and param in body and f"'{param}'" in msg or (param == "response_format" and "response_format" in msg and param in body):
                    body.pop(param)
                    retry = True
            if not retry:
                raise
            resp = self._post(f"{base}/chat/completions", headers, body)
        usage = resp.get("usage", {})
        return resp["choices"][0]["message"]["content"], usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    def _anthropic_complete(self, prompt: str, system: str | None, json_mode: bool) -> tuple[str, int, int]:
        sys_text = (system or "").strip()
        if json_mode:
            sys_text = (sys_text + "\n\nRespond with a single JSON object and nothing else.").strip()
        body: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if sys_text:
            body["system"] = sys_text
        resp = self._post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self._env("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"},
            body,
        )
        text = "".join(block.get("text", "") for block in resp.get("content", []))
        if json_mode:
            text = _strip_fences(text)
        usage = resp.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    def complete_json(self, prompt: str, *, system: str | None = None) -> dict:
        text = self.complete(prompt, system=system, json_mode=True)
        try:
            obj = json.loads(_strip_fences(text))
        except json.JSONDecodeError as e:
            raise LLMError(f"model returned invalid JSON: {text[:200]!r}") from e
        if not isinstance(obj, dict):
            raise LLMError(f"model returned JSON that is not an object: {text[:200]!r}")
        return obj

    # ------------------------------------------------------------- embedding
    def embed(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        if self.embed_provider != "openai":
            raise LLMError(
                f"embeddings need an OpenAI-compatible endpoint; got provider {self.embed_provider!r}. "
                "Set --embed-model openai:<model> (OPENAI_BASE_URL may point at any compatible server)."
            )
        out: list[list[float] | None] = [None] * len(texts)
        keys = [self._cache.key("embed", self.embed_provider, self.embed_model, t) for t in texts]
        missing: list[int] = []
        for i, k in enumerate(keys):
            cached = self._cache.get(k)
            if cached is not None:
                out[i] = json.loads(cached)
                with self._usage_lock:
                    self.usage.cache_hits += 1
            else:
                missing.append(i)

        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        for start in range(0, len(missing), batch_size):
            idx = missing[start : start + batch_size]
            body = {"model": self.embed_model, "input": [texts[i] for i in idx]}
            resp = self._post(f"{base}/embeddings", {"Authorization": f"Bearer {self._env('OPENAI_API_KEY')}"}, body)
            vectors = sorted(resp["data"], key=lambda d: d["index"])
            if len(vectors) != len(idx):
                raise LLMError("embedding response length mismatch")
            for i, vec in zip(idx, vectors):
                out[i] = vec["embedding"]
                self._cache.put(keys[i], json.dumps(vec["embedding"]))
            with self._usage_lock:
                self.usage.add(self.embed_model, resp.get("usage", {}).get("prompt_tokens", 0), 0, embed=True)
        return [v for v in out if v is not None]


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()
