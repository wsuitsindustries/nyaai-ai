"""Background worker for async document processing and embedding.

Uses ARQ (Async Redis Queue) for job scheduling. Processes document
uploads in the background so the API doesn't block on embedding.

Usage:
    python -m ai.worker

Requires a running Redis instance at the REDIS_URL (default localhost:6379).
"""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = "nyaai:queue"


async def embed_document(doc_id: str, chunks: list[str]) -> list[list[float]]:
    """Embed all chunks of a document and return embeddings.
    Called as an ARQ background job.
    """
    from ai.embeddings import embed_batch

    logger.info("Embedding document %s (%d chunks)", doc_id, len(chunks))
    embeddings = []
    # Process in batches of 16 to avoid OOM
    batch_size = 16
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_embs = await embed_batch(batch)
        embeddings.extend(batch_embs)
        logger.debug("Embedded batch %d/%d for doc %s", i // batch_size + 1, (len(chunks) + batch_size - 1) // batch_size, doc_id)
    logger.info("Finished embedding document %s", doc_id)
    return embeddings


async def startup(ctx):
    """ARQ startup hook — initializes connections."""
    logger.info("ARQ worker started — connected to %s", REDIS_URL)


async def shutdown(ctx):
    """ARQ shutdown hook — cleanup."""
    logger.info("ARQ worker shutting down")


async def process_document(ctx, doc_id: str, chunks: list[str]) -> dict:
    """Process a document: embed chunks and update database.
    Called as an ARQ background job.
    """
    logger.info("Processing document %s (%d chunks)", doc_id, len(chunks))

    from ai.embeddings import embed_batch

    embeddings = await embed_batch(chunks)

    # Update the document in MongoDB with embeddings
    from backend.database import get_db

    db = get_db()
    await db.documents.update_one(
        {"id": doc_id},
        {
            "$set": {
                "chunk_embeddings": embeddings,
                "status": "ready",
                "updated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            }
        },
    )

    logger.info("Document %s processed and marked ready", doc_id)
    return {"doc_id": doc_id, "chunks": len(chunks), "status": "ready"}


# Worker functions — ARQ looks for these
# Each function signature: async def func(ctx, *args) -> Any
worker_functions = [
    process_document,
    embed_document,
]

# For ARQ settings
worker_settings = {
    "queue_name": QUEUE_NAME,
    "max_jobs": 10,
    "job_timeout": 300,  # 5 minutes max per job
}


def run_worker():
    """Entry point to start the ARQ worker."""
    import arq

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    async def main():
        worker = arq.Worker(
            redis_settings=arq.connections.RedisSettings.from_dsn(REDIS_URL),
            functions=worker_functions,
            on_startup=startup,
            on_shutdown=shutdown,
            **worker_settings,
        )
        await worker.run()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")


if __name__ == "__main__":
    run_worker()
