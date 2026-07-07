SYSTEM_PROMPT = """You are Nya AI, an enterprise knowledge agent built by WSUITSINDUSTRIES.

You are a helpful chatbot that can answer general questions AND questions about the organization's documents.

Guidelines:
- For general questions (greetings, opinions, chit-chat, hobbies, definitions, world knowledge), answer naturally like a friendly assistant — use your own knowledge
- For questions that reference the organization's documents or uploaded knowledge, base your answer on the provided context
- When you use information from provided documents, cite the source using [Source N] notation
- If the provided context doesn't contain the answer, say so and offer to help search uploaded documents
- Be concise, accurate, and professional
- Use plain language that everyone can understand"""


def build_prompt(question: str, context_chunks: list[str], sources: list[dict] | None = None) -> str:
    context_parts = []
    if context_chunks:
        for i, chunk in enumerate(context_chunks):
            context_parts.append(f"[Source {i+1}]\n{chunk}")
    context_str = "\n\n".join(context_parts) if context_parts else "(No relevant documents found for this question.)"

    source_list = ""
    if sources:
        source_list = "\n\nReferenced documents:\n" + "\n".join(
            f"- {s.get('title', 'Unknown')}" + (f" (Page {s['page']})" if s.get("page") else "")
            for s in sources
        )

    prompt = f"""{SYSTEM_PROMPT}

{"=" * 60}
CONTEXT FROM COMPANY DOCUMENTS:
{context_str}
{"=" * 60}
{source_list}

Question: {question}

Answer (if you used any context documents, cite them using [Source N] notation):"""
    return prompt