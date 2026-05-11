from openai import OpenAI

from RAG_system.core.config_core import settings

SYSTEM_PROMPT = """
You are a job search assistant. Answer using ONLY the provided job listings and metadata.

Rules:
- Do not hallucinate or assume missing details.
- If the context is insufficient, say so.
- Combine relevant chunks from the same job when needed.
- Focus on skills, requirements, responsibilities, salary, location, and experience.
- Ignore boilerplate like EEO statements, benefits, and company filler text unless relevant.
- Be concise, clear, and recruiter-style.
- Use bullet points when helpful.
"""


class RAGService:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API,
        )
        self.model = "openai/gpt-oss-20b:free"

    def generate(self, query: str, contexts: list[str]) -> str:
        context = "\n\n---\n\n".join(contexts)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
            max_tokens=600,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
