# PRD: Modular RAG with Telemetry and Evals

## Problem Statement
Developing and debugging Retrieval-Augmented Generation (RAG) systems is often a "black box" process. Developers struggle to understand why an LLM provided a specific answer, whether the retrieval phase was successful, or if the generated response is faithful to the source material. Manual evaluation is slow and inconsistent, and without granular tracing, identifying bottlenecks in the RAG chain (Retrieval vs. Generation) is difficult.

## Solution
A modular RAG system that prioritizes observability and automated quality measurement. The system will provide:
1.  **Visual Tracing**: Nested step-by-step visualization of the RAG pipeline (Query -> Retrieval -> Context Construction -> Generation).
2.  **Automated Evals**: Background computation of "RAGAS" metrics (Faithfulness, Answer Relevancy, Answer Correctness) for every query.
3.  **Modular Service Layer**: Clean separation between retrieval, reranking, and generation logic to allow for easy component swapping.

## User Stories
1. As a developer, I want to ingest local PDF, Markdown, and Text files so that I can query my own knowledge base.
2. As a developer, I want to use a local embedding model (`all-MiniLM-L6-v2`) so that ingestion and retrieval are fast and low-cost.
3. As a developer, I want to query the system via a React UI and see the response stream in real-time.
4. As a developer, I want to see a visual trace of every query in Arize Phoenix so that I can debug retrieval failures.
5. As a developer, I want the system to automatically calculate "Faithfulness" for every response so that I can detect hallucinations.
6. As a developer, I want to see "Answer Relevancy" scores in a dashboard so that I can evaluate how well the system addresses user intents.
7. As a developer, I want a pre-indexing script to batch process my existing documents into Qdrant.
8. As a developer, I want to run the entire stack (UI, API, DB, Telemetry) using Docker Compose for a consistent local environment.
9. As a developer, I want the system to use OpenRouter so that I can easily switch between different LLM providers (GPT-4o, Claude, etc.).
10. As a developer, I want the Qdrant database to persist on my host machine so that my indexed data survives container restarts.

## Implementation Decisions
- **Architecture**: Integrated Monolith (FastAPI) for backend logic, with Qdrant and Arize Phoenix as standalone containers.
- **Frontend**: React (Vite) with a streaming chat interface.
- **Backend**: FastAPI with asynchronous background tasks for Evals.
- **Service Layer**: Explicitly defined `retrieve()`, `rerank()`, and `generate()` functions to maintain control over the LangChain orchestration.
- **Vector DB**: Qdrant with host-mounted volume for persistence.
- **Embeddings**: Local `all-MiniLM-L6-v2` via Sentence-Transformers.
- **LLM**: OpenRouter (OpenAI SDK compatible).
- **Telemetry**: Arize Phoenix using OpenTelemetry/Phoenix SDKs for tracing.
- **Evals**: Ragas metrics computed using an OpenRouter model as a judge, with results exported to Arize Phoenix.

## Testing Decisions
- **Integration Tests**: Focus on the end-to-end RAG chain (Input -> Retrieval -> Output) to ensure structural correctness.
- **Logic Tests**: Unit tests for the `rerank()` and `context_construction` logic.
- **Mocking**: Mock OpenRouter API calls for CI/CD tests to prevent costs.
- **Verification**: Use "Golden Datasets" to verify that Ragas metrics correctly identify known bad/good responses.

## Out of Scope
- Multi-user authentication and role-based access control.
- Ingestion of large-scale document corpuses (>100 docs).
- Production-grade auto-scaling infrastructure.
- Complex human-in-the-loop labeling UI.

## Further Notes
The system is designed to be a "developer's workbench" for RAG experimentation. The focus is on the developer's ability to see and measure the system's internal state.
