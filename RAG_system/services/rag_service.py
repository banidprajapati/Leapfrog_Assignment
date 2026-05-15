from openai import OpenAI

SYSTEM_PROMPT = """
You are a job search assistant. Answer using ONLY the provided job listings and metadata.

Rules:
- Do not hallucinate or assume missing details.
- If the context is insufficient, say so.
- Combine relevant chunks from the same job when needed.
- Weigh ALL chunks equally. Do not favor the first chunk over later ones.
- Scrutinize each chunk for specific skills, requirements, responsibilities, salary, location, and experience level.
- If multiple chunks describe different roles or aspects, synthesize them together.
- Ignore boilerplate like EEO statements, benefits, and company filler text unless relevant.
- Be concise, clear, and recruiter-style.
- Use bullet points when helpful.
"""


class RAGService:
    def __init__(self, client: OpenAI):
        self.client = client
        self.model = "qwen/qwen-2.5-7b-instruct"

    def generate(self, query: str, contexts: list[str]) -> str:
        # Concatenate retrieved chunks into a single context
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
