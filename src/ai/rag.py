from ai.retrieval import retrieve_texts
from ai.llm import complete
from ai.prompts import build_prompt

GENERAL_PROMPT = """You are Nya AI, an enterprise knowledge agent built by WSUITSINDUSTRIES.

You help organization members answer questions using both general knowledge and internal documents.
- Answer concisely, accurately, and helpfully.
- If asked about the organization's specific documents, note that no documents have been uploaded yet.
- Be friendly and professional.

Question: {question}

Answer:"""


async def answer_with_rag(
    question: str,
    chunks: list[str],
    sources: list[dict] | None = None,
    use_llm: bool = False,
) -> tuple[str, list[str]]:
    relevant = retrieve_texts(question, chunks)

    if use_llm:
        if relevant:
            prompt = build_prompt(question, relevant, sources)
        else:
            prompt = GENERAL_PROMPT.format(question=question)
        try:
            answer = await complete([
                {"role": "user", "content": prompt},
            ])
            return answer, relevant
        except Exception:
            pass

    if relevant:
        context = "\n\n".join(
            f"Based on the retrieved documents:\n\n{chunk}"
            for i, chunk in enumerate(relevant)
        )
        return context, relevant

    return "I couldn't find relevant information in your organization's knowledge base. Try uploading documents or rephrasing your question.", []


async def stream_rag(
    question: str,
    chunks: list[str],
    sources: list[dict] | None = None,
):
    relevant = retrieve_texts(question, chunks)
    if not relevant:
        yield "I couldn't find relevant information in your organization's knowledge base."
        return

    prompt = build_prompt(question, relevant, sources)

    try:
        from ai.llm import complete_stream
        async for token in complete_stream([
            {"role": "user", "content": prompt},
        ]):
            yield token
    except Exception:
        yield "I found relevant documents but couldn't generate a complete answer. Please try again."
        return
