from ai.retrieval import retrieve_texts
from ai.llm import complete
from ai.prompts import build_prompt


async def answer_with_rag(
    question: str,
    chunks: list[str],
    sources: list[dict] | None = None,
    use_llm: bool = False,
) -> tuple[str, list[str]]:
    relevant = retrieve_texts(question, chunks)

    if use_llm and relevant:
        prompt = build_prompt(question, relevant, sources)
        try:
            answer = await complete([
                {"role": "user", "content": prompt},
            ])
            return answer, relevant
        except Exception:
            pass

    # Fallback: return formatted context
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