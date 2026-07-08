"""Pre-download FastEmbed model and cache it for faster first startup.

Usage:
    python setup.py                    # Download gte-tiny
    python setup.py --model all        # Download all supported models
    python setup.py --model <name>     # Download specific model

This should be run during Docker build (or first deploy) so the model
is cached in the HuggingFace cache directory, avoiding slow downloads
on every container restart.
"""

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default model — lightweight 22MB ONNX model
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Alternative models (larger, more accurate)
SUPPORTED_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "arctic-embed-xs": "snowflake/snowflake-arctic-embed-xs",
}

# Hugging Face cache directory
HF_CACHE = os.path.expanduser("~/.cache/huggingface")


def download_model(model_name: str) -> None:
    """Download and cache a FastEmbed model."""
    logger.info("Downloading embedding model: %s", model_name)
    logger.info("Cache directory: %s", HF_CACHE)

    try:
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name=model_name, max_length=512)
        # Trigger actual download by embedding a test sentence
        list(model.embed(["Nya AI setup test — warming model cache."]))
        logger.info("Successfully downloaded and cached: %s", model_name)
    except ImportError:
        logger.error("fastembed not installed. Run: pip install fastembed")
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to download model %s: %s", model_name, e)
        sys.exit(1)


def get_cache_size() -> str:
    """Return human-readable size of the HF cache directory."""
    total = 0
    try:
        for dirpath, _, filenames in os.walk(HF_CACHE):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    except OSError:
        return "unknown"
    if total < 1024:
        return f"{total} B"
    elif total < 1024**2:
        return f"{total / 1024:.1f} KB"
    elif total < 1024**3:
        return f"{total / 1024**2:.1f} MB"
    else:
        return f"{total / 1024**3:.1f} GB"


def main():
    parser = argparse.ArgumentParser(description="Pre-download FastEmbed models")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to download (default: {DEFAULT_MODEL}). "
        f"Supported: {', '.join(SUPPORTED_MODELS.keys())}. Use 'all' for all models.",
    )
    args = parser.parse_args()

    os.makedirs(HF_CACHE, exist_ok=True)
    logger.info("Current HF cache size: %s", get_cache_size())

    if args.model == "all":
        for name in SUPPORTED_MODELS.values():
            download_model(name)
    elif args.model in SUPPORTED_MODELS:
        download_model(SUPPORTED_MODELS[args.model])
    else:
        # Allow arbitrary model names for custom models
        download_model(args.model)

    logger.info("Setup complete. HF cache size: %s", get_cache_size())
    logger.info(
        "Tip: To speed up container startup, mount %s as a persistent volume.",
        HF_CACHE,
    )


if __name__ == "__main__":
    main()
