from ai.retrieval import retrieve_with_embeddings, retrieve_texts
from ai.llm import complete
from ai.prompts import build_prompt

CONVERSATION_PROMPT = """You are Nya AI, an enterprise knowledge agent built by WSUITSINDUSTRIES.

You are a helpful chatbot that can answer general questions AND questions about the organization's documents.

For general questions (greetings, chit-chat, definitions, world knowledge), answer naturally like a friendly assistant.
For questions about the organization's documents, base your answer on the provided context.
Be concise, accurate, and helpful.

Question: {question}

Answer:"""


async def answer_with_rag(
    question: str,
    chunks: list[str],
    chunk_embeddings: list[list[float]] | None = None,
    sources: list[dict] | None = None,
    use_llm: bool = False,
) -> tuple[str, list[str]]:
    relevant = retrieve_with_embeddings(question, chunks, chunk_embeddings)

    if use_llm:
        try:
            if relevant:
                prompt = build_prompt(question, relevant, sources)
            else:
                prompt = CONVERSATION_PROMPT.format(question=question)
            answer = await complete([
                {"role": "user", "content": prompt},
            ])
            return answer, relevant
        except Exception:
            if relevant:
                context = "\n\n".join(
                    f"Based on the retrieved documents:\n\n{chunk}"
                    for i, chunk in enumerate(relevant)
                )
                answer = f"""Based on the information I found in your organization's documents:

{context}

I found this information in your knowledge base. For a more detailed answer with natural language, please set up an LLM API key in the environment configuration."""
                return answer, relevant
            return "I couldn't find relevant information in your organization's knowledge base. Try uploading documents or rephrasing your question.", []

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
    chunk_embeddings: list[list[float]] | None = None,
    sources: list[dict] | None = None,
):
    relevant = retrieve_with_embeddings(question, chunks, chunk_embeddings)

    if not relevant:
        try:
            prompt = CONVERSATION_PROMPT.format(question=question)
            from ai.llm import complete_stream
            async for token in complete_stream([
                {"role": "user", "content": prompt},
            ]):
                yield token
        except Exception:
            yield "I couldn't find relevant information in your organization's knowledge base. Try uploading documents or rephrasing your question."
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
