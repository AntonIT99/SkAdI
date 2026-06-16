import ollama
from app.config import LLM_MODEL


class LlmService:
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

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]