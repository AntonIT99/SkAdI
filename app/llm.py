import logging

import ollama
from ollama import ResponseError

from app.config import LLM_MODEL

logger = logging.getLogger(__name__)


class LlmServiceUnavailable(RuntimeError):
    pass


class LlmService:
    def diagnostics(self, default_model: str = LLM_MODEL) -> dict:
        try:
            models = self.available_models()
        except LlmServiceUnavailable as exc:
            return {
                "status": "error",
                "default_model": default_model,
                "available_models": [],
                "error": str(exc),
            }

        return {
            "status": "ok" if default_model in models else "missing_model",
            "default_model": default_model,
            "available_models": models,
            "default_model_available": default_model in models,
        }

    def available_models(self) -> list[str]:
        try:
            response = ollama.list()
        except Exception as exc:
            raise LlmServiceUnavailable(f"Ollama is not reachable: {exc}") from exc

        models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
        names = []

        for model in models:
            if isinstance(model, dict):
                name = model.get("name") or model.get("model")
            else:
                name = getattr(model, "name", None) or getattr(model, "model", None)

            if name:
                names.append(str(name))

        return sorted(names)

    def generate(self, question: str, sources: list[dict], model: str = LLM_MODEL) -> str:
        context = "\n\n".join(
            f"[Quelle {i+1}] {s['book']}, Seite {s['page']}:\n{s['text']}"
            for i, s in enumerate(sources)
        )

        prompt = f"""
Du bist ein quellenbasierter philosophischer und weltanschaulicher Assistent.

Regeln:
- Antworte in der Sprache der Nutzerfrage.
- Nutze nur die bereitgestellten Quellen.
- Wenn du zitierst, zitiere wortwörtlich aus den Quellen.
- Wenn keine ausreichende Grundlage vorhanden ist, sage das ehrlich.
- Trenne Erklärung, direkte Zitate und Quellen.

Quellen:
{context}

Frage:
{question}

Antwort:
"""

        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        except ResponseError as exc:
            if exc.status_code == 404:
                raise LlmServiceUnavailable(self._missing_model_message(model)) from exc

            logger.exception("[LLM] Ollama request failed")
            raise LlmServiceUnavailable(f"Ollama request failed: {exc}") from exc
        except Exception as exc:
            logger.exception("[LLM] Ollama is unavailable")
            raise LlmServiceUnavailable(f"Ollama is unavailable: {exc}") from exc

        return response["message"]["content"]

    def _missing_model_message(self, model: str) -> str:
        try:
            available_models = self.available_models()
        except LlmServiceUnavailable:
            available_models = []

        if available_models:
            return (
                f"Ollama model '{model}' is not installed. "
                f"Install it with 'ollama pull {model}' or set LLM_MODEL to one of: "
                f"{', '.join(available_models)}"
            )

        return (
            f"Ollama model '{model}' is not installed and no local Ollama models are available. "
            f"Install it with 'ollama pull {model}' or set LLM_MODEL to an installed model."
        )
