# Persistent RAG Workbench

This project implements a modular Retrieval-Augmented Generation (RAG) system with a strong focus on observability, automated quality measurement, and session persistence. It addresses the "black box" problem often encountered in RAG development by providing granular tracing, automated evaluation metrics, and a persistent history of interactions.

## Key Features

- **Automated Evals**: Background computation of "RAGAS" metrics (Faithfulness, Answer Relevancy) for every query, with versioned and persisted results.
- **Real-time Metric Surfacing**: Direct exposure of accuracy (Ragas) and performance (Latency) metrics in the UI, pushed via Server-Sent Events (SSE) as they are computed.
- **Persistent History**: A relational storage layer to save chat sessions, messages, and evaluation results, accessible via a history sidebar.
- **Modular Architecture**: Clean separation between retrieval, reranking, and generation logic (Ports & Adapters) to allow for easy component swapping.
- **Local File Ingestion**: Ability to ingest local PDF, Markdown, and Text files to query against a custom knowledge base.
- **Responsive UI**: Chat stream starts immediately regardless of evaluation status for a snappy user experience.

## Tech Stack

- **Backend**: Python 3.12+, FastAPI
- **Frontend**: TypeScript, React, Vite
- **Vector Database**: Qdrant
- **RAG Framework**: LangChain
- **LLM Integration**: LangChain OpenAI, Cohere, OpenRouter (OpenAI SDK compatible)
- **RAG Evaluation**: Ragas
- **Persistence**: SQLModel (with SQLite / PostgreSQL)
- **Package Management**: uv

## Prerequisites

- **Python 3.12+** (via uv or virtual environment)
- **Node.js 18+** (for frontend development)
- **pnpm** (recommended for JavaScript package management) or npm/yarn
- **Docker** (for Qdrant)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Persistent-RAG-Workbench.git
cd Persistent-RAG-Workbench
```

### 2. Environment Setup

Create a `.env` file in the root directory and configure necessary environment variables.

```bash
cp .env.example .env
```

```ini
# .env example
OPENROUTER_API_KEY="sk-or-v1-..." # Required: Your OpenRouter API key
COHERE_API_KEY="your-cohere-key"  # Required: For generating dense embeddings
QDRANT_URL="http://localhost:6333" # URL to your Qdrant instance
```

### Required

| Variable | Description | Example |
| -------- | ----------- | ------- |
| `OPENROUTER_API_KEY` | API key for OpenRouter (must not be empty) | `sk-or-v1-...` |
| `COHERE_API_KEY` | API key for Cohere embeddings | `...` |

### Optional

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `QDRANT_URL` | Full URL to the Qdrant service | `http://localhost:6333` |
| `QDRANT_API_KEY` | API key for Qdrant (if using managed cloud) | `None` |
| `COLLECTION_NAME`| The Qdrant collection name | `rag_collection` |
| `EMBEDDING_MODEL`| Embedding model identifier | `embed-english-v3.0` |
| `OPENROUTER_MODEL`| Model to use for chat generation | `openrouter/free` |
| `RAGAS_EVAL_MODEL`| Model to use for evaluation via Ragas | `openrouter/free` |
| `DATABASE_URL` | SQLite / PG connection string | `sqlite:///sqlite.db` |
| `CORS_ORIGINS` | Comma-separated list of allowed origins | `["http://localhost:5173", ...]` |

### 3. Start Qdrant Services

Start Qdrant using Docker Compose:

```bash
docker compose up -d
```

### 4. Setup Backend

Install dependencies via `uv`:

```bash
uv pip install -e ".[dev]"
```

Run database migrations to initialize SQLite:

```bash
python backend/app.py migrate
```

### 5. Ingest Documents

Place your PDFs, Markdown, and Text files in the `data/` folder and run the ingestion script to chunk and embed them:

```bash
uv run backend/ingest.py
```

### 6. Start the Servers

**Terminal 1: FastAPI Backend**
```bash
uv run backend/app.py
```
*API available at `http://localhost:8000`*

**Terminal 2: React Frontend**
```bash
cd frontend
pnpm install
pnpm dev
```
*Frontend available at `http://localhost:5173`*

---

## Architecture & Code-Level Flow

### Directory Structure

```
├── .venv/                     # Python virtual environment
├── backend/                   # FastAPI application
│   ├── adapters/              # Plugins/implementations (QdrantRetriever, OpenRouterGenerator)
│   ├── api/                   # HTTP interface (routes, schemas, dependencies)
│   ├── core/                  # Domain logic (Models, RAGPipeline, Infrastructure, Interfaces)
│   ├── evaluation/            # Quality control (RagasEvaluator, MockEvaluator)
│   ├── services/              # Orchestrators (chat_service, evaluation_service)
│   ├── tests/                 # Backend tests
│   ├── app.py                 # FastAPI application entry point
│   └── ingest.py              # Script for document chunking & ingestion
├── data/                      # Local files for ingestion (PDFs, Markdowns, TXT)
├── frontend/                  # React/TypeScript application
│   ├── src/                   # Components, hooks (useChat), API
│   └── package.json           # Frontend dependencies
├── qdrant_storage/            # Persistent storage for Qdrant
├── docker-compose.yml         # Docker config for Qdrant
├── pyproject.toml             # Python project configuration
└── README.md                  # Project documentation
```

### Request Lifecycle (Chat Interaction)

1. **User Query**: User inputs a query in the React frontend.
2. **API Call**: Request hits the `POST /chat` endpoint in `backend/api/routes/chat.py`.
3. **Service Layer**: `ChatService.chat` is invoked, saving the user message to SQLite and triggering the pipeline.
4. **Pipeline Execution**: `RAGPipeline.execute()` runs:
   - **Retrieval**: `QdrantRetriever` takes the raw query and passes it to `QdrantVectorStore`. The query is embedded via Cohere (`embed_query`) and FastEmbed, then matched against Qdrant.
   - **Generation**: Relevant chunks are formatted into a context prompt and sent to OpenRouter via the `Generator` adapter.
5. **Response**: Backend streams the response back to the UI via Server-Sent Events (SSE).
6. **Async Evaluation**: A background task uses Ragas to compute metrics (Faithfulness, Relevancy) and saves the results to the database.

### Detailed Code Flow: The Query embedding path

The exact code path from an API request to vector embedding is tightly controlled:

1. **API receives query:** `request.query` arrives at `chat_endpoint`.
2. **Service calls Retriever:** `await self.retriever.retrieve(query, k)`
3. **Adapter calls VectorStore:** `self.vectorstore.similarity_search(query, k*3)`
4. **LangChain Embeds the Query:** Inside the Qdrant Langchain integration:
   ```python
   # Dense embedding
   query_vector = self.embedding.embed_query(query)
   # Sparse embedding (hybrid mode)
   sparse_vector = self.sparse_embedding.embed_query(query)
   # Issue the search to Qdrant, passing both vectors
   hits = self.client.search(query_vector=query_vector, sparse_vectors=...)
   ```

### Database Schema (SQLite/SQLModel)

```
sessions
├── id (UUID, PK)
├── title (string)
├── created_at (datetime)
└── updated_at (datetime)

messages
├── id (UUID, PK)
├── session_id (UUID, FK -> sessions)
├── role (string, "user" or "assistant")
├── content (text)
├── token_count (int)
└── latency_ms (int)

evaluations
├── id (UUID, PK)
├── message_id (UUID, FK -> messages)
├── faithfulness (float, optional)
├── answer_relevancy (float, optional)
├── context_precision (float, optional)
├── status (string, "pending", "completed", "failed")
└── version (int) # Allows tracking re-evaluations
```

---

## Available Scripts

| Command | Description |
| ------- | ----------- |
| `uv pip install -e ".[dev]"` | Install backend dependencies |
| `uv run backend/app.py` | Start the FastAPI server |
| `python backend/app.py migrate`| Run database migrations |
| `uv run backend/ingest.py` | Run document ingestion |
| `uv run pytest backend/tests/` | Run all backend tests |
| `pnpm dev` (in `/frontend`) | Start frontend server |

## Testing

**Backend Tests:**
Run the backend Minitest/Pytest suite:
```bash
uv run pytest backend/tests/
```
Tests cover SQLModel schemas, RAGPipeline (via mocks), and evaluation status transitions.

**Frontend Tests:**
```bash
cd frontend
pnpm test
```
Vitest covers the `useChat` hook, API adapters, and state transitions.

## Deployment

This project uses Docker Compose for local ease. For production deployment, you can containerize the backend and frontend separately.

### Docker deployment

Build and run the entire application using Docker:

```bash
# Build the backend image
docker build -t rag-backend -f backend/Dockerfile .

# Start Qdrant and the backend
docker compose up -d
```
