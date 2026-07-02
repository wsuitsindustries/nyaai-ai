# Nya AI — AI Service

Enterprise AI service powering Nya AI's RAG pipeline. Handles chunking, embeddings, retrieval, and LLM communication.

## Directory Structure

```
src/ai/          Source code
  __init__.py
  chunking.py    Text splitting & chunking strategies
  embeddings.py  Vector embeddings (hash-based placeholder)
  llm.py         LLM provider interface (OpenAI-compatible)
  prompts.py     System prompt templates
  retrieval.py   Similarity search & context retrieval
tests/           Tests
```

## LLM Providers

Configure via environment variables:

| Variable          | Default                  | Description                  |
|-------------------|--------------------------|------------------------------|
| `LLM_PROVIDER`    | `openai`                 | Provider name                |
| `LLM_API_KEY`     | `""`                     | API key                      |
| `LLM_BASE_URL`    | `https://api.openai.com/v1` | API base URL              |
| `LLM_MODEL`       | `gpt-4o-mini`            | Model name                   |
| `LLM_MAX_TOKENS`  | `1024`                   | Max response tokens          |
| `LLM_TEMPERATURE` | `0.7`                    | Response temperature         |

### Free Provider Quick Start

Set these env vars for free LLM options:

```bash
# Groq (free tier, fast)
LLM_PROVIDER=groq LLM_BASE_URL=https://api.groq.com/openai/v1 LLM_MODEL=llama-3.3-70b-versatile

# Ollama (local, fully free)
LLM_PROVIDER=ollama LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.2 LLM_API_KEY=ollama

# OpenRouter (free tier)
LLM_PROVIDER=openrouter LLM_BASE_URL=https://openrouter.ai/api/v1 LLM_MODEL=meta-llama/llama-3.3-70b-instruct

# GitHub Models (free with GH account)
LLM_PROVIDER=github LLM_BASE_URL=https://models.inference.ai.azure.com LLM_MODEL=gpt-4o-mini
```
