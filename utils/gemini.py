"""
utils/gemini.py
---------------
Thin, defensive wrapper around Google Generative AI (Gemini).

Design rules:
  * The API key is read from the environment (.env via python-dotenv) — never hardcoded.
  * Nothing in the app crashes when the key is missing, the package is not
    installed, the network is down or the API returns an error. Callers always
    receive a `GeminiResponse` and decide their own fallback.
  * `generate_json` additionally tolerates fenced/《chatty》 model output.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:  # python-dotenv is a hard requirement, but never crash if absent
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
    load_dotenv()  # also honour a .env in the current working directory
except Exception:  # pragma: no cover - defensive only
    pass

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
DEFAULT_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "20"))
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


@dataclass
class GeminiResponse:
    """Uniform Gemini result. `ok=False` means: use your fallback."""

    ok: bool
    text: str = ""
    error: Optional[str] = None
    model: Optional[str] = None

    def __bool__(self) -> bool:  # allows `if response:`
        return self.ok


class GeminiClient:
    """
    Lazy Gemini client.

    The SDK import and model construction happen on first use so importing this
    module is always safe (tests, CI, machines without the package).
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_MODEL):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self._model = None
        self._init_error: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ setup
    def has_key(self) -> bool:
        key = (self.api_key or "").strip()
        return bool(key) and key.lower() not in {"your_api_key_here", "changeme", "none"}

    def _ensure_model(self) -> bool:
        """Build the model once. Returns False and records why on failure."""
        if self._model is not None:
            return True
        if self._init_error is not None:
            return False
        with self._lock:
            if self._model is not None:
                return True
            if not self.has_key():
                self._init_error = (
                    "GEMINI_API_KEY is not set. Add it to your .env file to enable "
                    "AI-written answers (the app keeps working without it)."
                )
                return False
            try:
                import google.generativeai as genai  # imported lazily on purpose

                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model_name)
                return True
            except ImportError:
                self._init_error = (
                    "google-generativeai is not installed. Run: "
                    "python -m pip install -r requirements.txt"
                )
            except Exception as exc:  # bad key, quota, transport, ...
                self._init_error = f"Gemini could not be initialised: {exc}"
            return False

    def is_available(self) -> bool:
        """True when a real Gemini call can be attempted."""
        return self._ensure_model()

    def status(self) -> Dict[str, Any]:
        """Human-readable status used by the Streamlit sidebar."""
        available = self.is_available()
        return {
            "available": available,
            "model": self.model_name if available else None,
            "reason": None if available else (self._init_error or "unavailable"),
        }

    # ---------------------------------------------------------------- calling
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> GeminiResponse:
        """Send a prompt to Gemini and return clean text (never raises)."""
        if not prompt or not str(prompt).strip():
            return GeminiResponse(False, error="Empty prompt.")
        if not self._ensure_model():
            return GeminiResponse(False, error=self._init_error or "Gemini unavailable.")
        try:
            result = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                },
                request_options={"timeout": timeout},
            )
            text = self._extract_text(result)
            if not text:
                return GeminiResponse(False, error="Gemini returned an empty response.")
            return GeminiResponse(True, text=text, model=self.model_name)
        except Exception as exc:  # network, quota, safety block, SDK change...
            return GeminiResponse(False, error=f"Gemini request failed: {exc}")

    def generate_json(self, prompt: str, **kwargs: Any) -> GeminiResponse:
        """
        Ask for JSON and hand back only the JSON text.

        The model sometimes wraps JSON in ```json fences or prose; both are
        stripped here so callers can `json.loads(response.text)` safely.
        """
        response = self.generate(prompt, **kwargs)
        if not response.ok:
            return response
        payload = extract_json_block(response.text)
        if payload is None:
            return GeminiResponse(False, error="Gemini did not return valid JSON.")
        return GeminiResponse(True, text=payload, model=response.model)

    # --------------------------------------------------------------- internal
    @staticmethod
    def _extract_text(result: Any) -> str:
        """Pull text out of a Gemini result across SDK shapes."""
        text = getattr(result, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        try:
            parts = []
            for candidate in getattr(result, "candidates", []) or []:
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", []) or []:
                    piece = getattr(part, "text", None)
                    if piece:
                        parts.append(piece)
            return "\n".join(parts).strip()
        except Exception:
            return ""


# ------------------------------------------------------------------ utilities


def extract_json_block(text: str) -> Optional[str]:
    """Return the first JSON object/array found in `text`, or None."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(text.strip())
    brace = re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            continue
    return None


def load_prompt(name: str, **variables: Any) -> str:
    """
    Load `prompts/<name>.txt` and substitute {placeholders}.

    Missing files degrade to an empty string so a deleted prompt can never take
    the app down — the caller's template fallback takes over.
    """
    path = PROMPTS_DIR / (name if name.endswith(".txt") else f"{name}.txt")
    try:
        template = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    for key, value in variables.items():
        template = template.replace("{" + key + "}", str(value))
    return template


_default_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Process-wide singleton (cheap: construction does no I/O)."""
    global _default_client
    if _default_client is None:
        _default_client = GeminiClient()
    return _default_client


def ask_gemini(prompt: str, **kwargs: Any) -> GeminiResponse:
    """Convenience one-liner used by the agents."""
    return get_gemini_client().generate(prompt, **kwargs)
