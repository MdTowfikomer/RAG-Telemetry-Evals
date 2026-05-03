# Modular RAG with Telemetry and Evals

This project is a developer-focused RAG workbench built around FastAPI, Qdrant, OpenRouter, and Arize Phoenix.

## Backend Architecture (Issue #8)

The backend now follows a **Ports & Adapters** design so orchestration can be tested independently from external services.

### Core Ports

Defined in `backend/core/interfaces.py`:

- `Retriever`
- `Reranker`
- `Generator`

Shared core model:

- `Document` in `backend/core/models.py`

### Orchestrator

`RAGPipeline` in `backend/core/pipeline.py` owns the request flow:

- `prepare_context(query, k)`
- `execute(query, k)`
- `stream(query, k)`

This keeps sequence/telemetry/error boundaries in one place.

### Production Adapters

Defined in `backend/adapters/`:

- `QdrantRetriever`
- `PassThroughReranker`
- `OpenRouterGenerator`

`backend/app.py` composes these adapters at startup and injects them into `RAGPipeline`.

### Why this helps

- Easier experimentation (swap one stage without rewriting route logic)
- Isolated unit tests for orchestration without network I/O
- Cleaner boundaries between core logic and external dependencies

## Testing

### Pipeline unit tests

Run:

`python -m unittest backend.tests.test_rag_pipeline`

Covers:

- stage sequencing
- context-only preparation
- execute error bubbling
- stream token flow and stream error bubbling

### Adapter unit tests

Run:

`python -m unittest backend.tests.test_qdrant_retriever`

Covers:

- mapping vectorstore docs into core `Document`
- fallback behavior for missing fields

### Existing API/eval tests

Run:

`python -m unittest backend.tests.test_api_with_mock backend.tests.test_ragas_evaluator`
