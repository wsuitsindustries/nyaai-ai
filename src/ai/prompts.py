SYSTEM_PROMPT = """You are NyaAI, an enterprise knowledge agent built by WSUITSINDUSTRIES.

You are a helpful chatbot that can answer general questions AND questions about the organization's documents.

Guidelines:
- For general questions (greetings, opinions, chit-chat, hobbies, definitions, world knowledge), answer naturally like a friendly assistant — use your own knowledge
- For questions that reference the organization's documents or uploaded knowledge, base your answer ONLY on the provided context below
- When you use information from provided documents, cite the source using [Source N] notation — put the citation at the end of the relevant sentence
- If multiple sources contain the same information, cite all relevant ones: [Source 1][Source 3]
- If the provided context doesn't contain enough information to answer, say "I couldn't find information about that in the uploaded documents" — do not make up information
- If the context partially answers but misses details, say what the documents cover and what they don't
- Be concise, accurate, and professional
- Use plain language that everyone can understand"""


def build_prompt(
    question: str,
    context_chunks: list[str],
    sources: list[dict] | None = None,
) -> str:
    context_parts = []
    if context_chunks:
        for i, chunk in enumerate(context_chunks, 1):
            label = f"[Source {i}]"
            source_name = ""
            if sources and i <= len(sources):
                s = sources[i - 1]
                source_name = f" — {s.get('title', 'Unknown')}"
            context_parts.append(f"{label}{source_name}\n{chunk}")
    context_str = "\n\n".join(context_parts) if context_parts else "(No relevant documents found for this question.)"

    source_list = ""
    if sources:
        seen = set()
        unique = []
        for s in sources:
            t = s.get("title", "Unknown")
            if t not in seen:
                seen.add(t)
                unique.append(t)
        if unique:
            source_list = "\n\nReferenced documents:\n" + "\n".join(f"- {t}" for t in unique)

    prompt = f"""{SYSTEM_PROMPT}

{"=" * 60}
CONTEXT FROM COMPANY DOCUMENTS:
{context_str}
{"=" * 60}
{source_list}

Question: {question}

Answer (cite sources using [Source N] notation at the end of each relevant sentence):"""
    return prompt


def build_query_expansion_prompt(question: str) -> str:
    return f"""Given the following user question, generate 3 different rephrasings that would help
search a knowledge base. Make them diverse — use synonyms, different sentence structures,
and different levels of specificity. Return them one per line, no numbering.

Original: {question}

Rephrased:"""
