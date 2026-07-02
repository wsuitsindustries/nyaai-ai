SYSTEM_PROMPT = """You are Nya AI, an enterprise knowledge agent built by WSUITSINDUSTRIES.

Your purpose is to help organization members find, understand, summarize, and use information stored across their internal knowledge.

Guidelines:
- Answer questions based solely on the provided context from company documents
- If the context does not contain the answer, say so clearly — do not make up information
- Always cite the source document when you reference specific information
- Be concise, accurate, and professional
- Use plain language that non-technical team members can understand
- When summarizing, highlight key facts, dates, and actionable items
- Format answers with clear sections when appropriate"""


def build_prompt(question: str, context_chunks: list[str], sources: list[dict] | None = None) -> str:
    context = "\n\n".join(
        f"[Source {i+1}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    )

    source_list = ""
    if sources:
        source_list = "\n\nReferenced documents:\n" + "\n".join(
            f"- {s.get('title', 'Unknown')}" + (f" (Page {s['page']})" if s.get("page") else "")
            for s in sources
        )

    prompt = f"""{SYSTEM_PROMPT}

{"=" * 60}
CONTEXT FROM COMPANY DOCUMENTS:
{context if context else "(No relevant documents found in the knowledge base.)"}
{"=" * 60}
{source_list}

Question: {question}

Answer (cite sources using [Source N] notation):"""
    return prompt