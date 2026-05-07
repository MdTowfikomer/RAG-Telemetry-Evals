# PRD: Persistent RAG Workbench with Telemetry and Evals

## Problem Statement
Developing and debugging Retrieval-Augmented Generation (RAG) systems is often a "black box" process. Developers struggle to understand why an LLM provided a specific answer, whether the retrieval phase was successful, or if the generated response is faithful to the source material. 

Furthermore, existing tools often lack session persistence, making it difficult to audit past responses or track iterative improvements. Without granular tracing and directly exposed metrics, identifying bottlenecks and building user trust in the RAG chain is difficult.

## Solution
A modular RAG system that prioritizes observability, automated quality measurement, and session persistence. The system provides:
1.  **Visual Tracing**: Nested step-by-step visualization of the RAG pipeline (Query -> Retrieval -> Context Construction -> Generation) via Arize Phoenix.
2.  **Automated Evals**: Background computation of "RAGAS" metrics (Faithfulness, Answer Relevancy) for every query, now versioned and persisted.
3.  **Real-time Metric Surfacing**: Direct exposure of accuracy (Ragas) and performance (Latency) metrics in the UI, pushed via SSE as they are computed.
4.  **Persistent History**: A relational storage layer to save chat sessions, messages, and evaluation results, accessible via a history sidebar.
5.  **Modular Architecture**: Clean separation between retrieval, reranking, and generation logic (Ports & Adapters) to allow for easy component swapping.

## User Stories
1. As a developer, I want my chat sessions to be saved automatically, so that I can resume my work later or audit past responses.
2. As a user, I want to see a history of my past conversations in a sidebar, so that I can easily switch between different topics.
3. As a developer, I want to see accuracy scores (Faithfulness, Relevancy) for every assistant response directly in the UI, so that I can verify the RAG pipeline is working correctly.
4. As a user, I want to see performance metrics like latency, so that I can understand the responsiveness of the system.
5. As a developer, I want to see the reasoning behind a low evaluation score, so that I can identify if the issue lies in the retrieval or the generation phase.
6. As a developer, I want to manually trigger a re-evaluation of a specific response, so that I can test changes to my evaluation judge without re-running the whole query.
7. As a user, I want to see a 'pending' status while an evaluation is running, so that I know the system is still working on calculating the metrics.
8. As a user, I want to see high-level accuracy badges on a message card, so that I can get a quick sense of the response quality without opening the full metrics view.
9. As a developer, I want to see a detailed breakdown of all metrics in a dedicated sidebar tab, so that I have a focused space for deep analysis.
10. As a developer, I want to ingest local PDF, Markdown, and Text files so that I can query my own knowledge base.
11. As a developer, I want the chat stream to start immediately regardless of evaluation status, so that the user experience remains snappy and responsive.
12. As a system, I must store evaluations as versions, so that I can track how scores change if I re-evaluate with different models or parameters.

## Implementation Decisions
- **Architecture**: Integrated Monolith (FastAPI) for backend logic, with Qdrant and Arize Phoenix as standalone containers.
- **Persistence Layer**: Use SQLModel with SQLite for relational storage of Sessions, Messages, and Evaluations. 
- **Service Layer**: Ports & Adapters pattern with explicit interfaces for `Retriever`, `Reranker`, and `Generator`.
- **Frontend Orchestration**: Dedicated `useChat` hook and API adapter to isolate networking logic from UI components.
- **Evaluation Flow**: Ragas metrics computed asynchronously in background tasks. Backend uses Server-Sent Events (SSE) to notify the UI of score updates.
- **Status Tracking**: Explicit state transitions for evaluations (`pending` -> `completed` / `failed`) to handle async processing.
- **Vector DB**: Qdrant with host-mounted volume for persistence.
- **LLM**: OpenRouter (OpenAI SDK compatible) for both generation and evaluation judge.
- **Telemetry**: Arize Phoenix using OpenTelemetry for tracing.

## Testing Decisions
- **Relational Integrity**: Unit tests for SQLModel schemas to ensure cascades and relationships work as expected.
- **Pipeline Orchestration**: Unit tests for `RAGPipeline` using mock adapters to verify sequence and error handling.
- **State Machine**: Test evaluation status transitions using mock background tasks.
- **Frontend Logic**: Vitest tests for `useChat` hook and `api` adapter to verify state transitions and SSE handling.
- **Integration Tests**: Focus on the end-to-end RAG chain (Input -> Retrieval -> Output) and persistence.

## Out of Scope
- User authentication and multi-tenancy.
- Ingestion of large-scale document corpuses (>100 docs).
- Production-grade auto-scaling infrastructure.
- Complex human-in-the-loop labeling UI.

## Further Notes
The system follows a "developer's workbench" philosophy. Portability (SQLite), developer transparency (exposed metrics), and modularity are prioritized over high-scale multi-user features.
