# Persistent RAG Workbench

This project implements a modular Retrieval-Augmented Generation (RAG) system with a strong focus on observability, automated quality measurement, and session persistence. It addresses the "black box" problem often encountered in RAG development by providing granular tracing, automated evaluation metrics, and a persistent history of interactions.

## Key Features

- **Visual Tracing**: Nested step-by-step visualization of the RAG pipeline (Query -> Retrieval -> Context Construction -> Generation) via Arize Phoenix.
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
- **Observability**: Arize Phoenix (via OpenTelemetry)
- **RAG Framework**: LangChain
- **LLM Integration**: LangChain OpenAI, LangChain HuggingFace, OpenRouter (OpenAI SDK compatible)
- **RAG Evaluation**: Ragas
- **Persistence**: SQLModel (with SQLite)
- **Package Management**: uv

## Prerequisites

- **Python 3.12+**
- **Node.js 18+** (for frontend development)
- **pnpm** (recommended for JavaScript package management) or npm/yarn
- **Docker** (for Qdrant and Arize Phoenix)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Persistent-RAG-Workbench.git
cd Persistent-RAG-Workbench
```

### 2. Environment Setup

Create a `.env` file in the root directory and configure necessary environment variables.
A `.env.example` file is typically provided for reference.

```ini
# .env example
OPENAI_API_KEY="sk-..." # Your OpenAI API key or OpenRouter key
OPENAI_BASE_URL="https://openrouter.ai/api/v1" # Example for OpenRouter
QDRANT_HOST="localhost"
QDRANT_PORT="6333"
PHOENIX_HOST="localhost"
PHOENIX_PORT="6006"
```

### 3. Install Backend Dependencies

```bash
uv pip install -e ".[dev]"
```

### 4. Start Docker Services

Start Qdrant and Arize Phoenix using Docker Compose:

```bash
docker compose up -d
```

### 5. Run Backend Migrations

Initialize the SQLite database for session persistence:

```bash
# This will create a local sqlite.db file
python backend/app.py migrate
```

### 6. Start the Backend Server

```bash
uv run backend/app.py
```

The backend API will be available at `http://localhost:8000`.

### 7. Install Frontend Dependencies

Navigate to the `frontend` directory and install dependencies:

```bash
cd frontend
pnpm install # or npm install or yarn install
```

### 8. Start the Frontend Development Server

```bash
pnpm dev # or npm run dev or yarn dev
```

Open your browser to `http://localhost:5173` (or the port indicated by Vite).

## Architecture

### Directory Structure

```
├── .venv/                     # Python virtual environment
├── backend/                   # FastAPI application
│   ├── adapters/              # Interfaces for RAG components (Retriever, Reranker, Generator)
│   ├── core/                  # Core RAG logic
│   ├── evaluation/            # Ragas evaluation logic
│   ├── services/              # Business logic services
│   ├── tests/                 # Backend tests
│   ├── app.py                 # Main FastAPI application entry point
│   └── ingest.py              # Script for document ingestion
├── data/                      # Local files for ingestion (PDFs, Markdowns, TXT)
├── frontend/                  # React/TypeScript application
│   ├── public/                # Static assets
│   ├── src/                   # Frontend source code (components, hooks, API)
│   ├── index.html             # Frontend entry point
│   ├── package.json           # Frontend dependencies and scripts
│   └── vite.config.ts         # Vite configuration
├── qdrant_storage/            # Persistent storage for Qdrant vector database
├── .gitignore
├── .python-version
├── docker-compose.yml         # Docker configuration for Qdrant and Phoenix
├── main.py                    # Main script (if any for overall orchestration)
├── PRD.md                     # Product Requirements Document
├── pyproject.toml             # Python project configuration and dependencies
├── README.md                  # This file
├── test_chat.py               # Integration tests for chat functionality
└── uv.lock                    # Dependency lock file for uv
```

### Request Lifecycle (Chat Interaction)

1.  **User Query**: User inputs a query in the React frontend.
2.  **Frontend API Call**: Frontend sends the query to the FastAPI backend.
3.  **Backend Processing**:
    *   `RAGPipeline` orchestrates retrieval, context construction, and generation.
    *   **Retrieval**: `Retriever` adapter fetches relevant documents from Qdrant.
    *   **Generation**: `Generator` adapter uses an LLM (e.g., OpenRouter) to generate a response.
4.  **Streaming Response**: Backend streams the LLM response back to the frontend using Server-Sent Events (SSE).
5.  **Asynchronous Evaluation**: In the background, `Ragas` metrics are computed, and results are pushed to the frontend via SSE.
6.  **Persistence**: Chat messages and evaluation results are saved to the SQLite database via SQLModel.

### Data Flow

```
User Action (Frontend)
    ↓
API Call (Frontend)
    ↓
FastAPI Backend (RAGPipeline: Retrieve → Generate)
    ↓
Qdrant (Vector DB for Retrieval)
    ↓
LLM (e.g., OpenRouter for Generation)
    ↓
Streaming Response (SSE to Frontend)
    ↓
Ragas Evaluation (Background task in Backend)
    ↓
Arize Phoenix (Telemetry Tracing)
    ↓
SQLite Database (Session History, Messages, Evaluations)
    ↓
UI Update (Frontend via SSE)
```

### Key Components

**Backend (FastAPI)**

-   **`app.py`**: Main FastAPI application. Defines API endpoints for chat, ingestion, and history.
-   **`adapters/`**: Contains interfaces and implementations for `Retriever`, `Reranker`, and `Generator` to ensure modularity.
-   **`core/`**: Houses the core `RAGPipeline` logic, orchestrating the steps of the RAG process.
-   **`evaluation/`**: Logic for computing Ragas metrics and managing evaluation states.
-   **`services/`**: Implements business logic, including interaction with the persistence layer.
-   **SQLModel**: Used for ORM with SQLite to manage chat sessions, messages, and evaluation results.

**Frontend (React/TypeScript)**

-   **`src/`**: Contains React components, hooks (e.g., `useChat` for state management), and API integration logic.
-   **`vite.config.ts`**: Frontend build configuration using Vite.

**External Services**

-   **Qdrant**: Vector database for efficient similarity search during retrieval. Persistent data stored in `./qdrant_storage`.
-   **Arize Phoenix**: Observability platform for visual tracing of the RAG pipeline. Accessible via `http://localhost:6006`.

### Database Schema (Conceptual)

```
sessions
├── id (UUID, PK)
├── created_at (datetime)
└── updated_at (datetime)

messages
├── id (UUID, PK)
├── session_id (UUID, FK -> sessions)
├── role (string, "user" or "assistant")
├── content (text)
├── created_at (datetime)
└── updated_at (datetime)

evaluations
├── id (UUID, PK)
├── message_id (UUID, FK -> messages)
├── faithfulness (float, optional)
├── answer_relevancy (float, optional)
├── latency (float, optional)
├── status (string, "pending", "completed", "failed")
├── version (int) # Allows tracking re-evaluations
├── created_at (datetime)
└── updated_at (datetime)
```

## Environment Variables

### Required

| Variable         | Description                                     | How to Get                                  |
| :--------------- | :---------------------------------------------- | :------------------------------------------ |
| `OPENAI_API_KEY` | API key for OpenAI or OpenAI-compatible service | From OpenAI, OpenRouter, or other providers |
| `OPENAI_BASE_URL` | Base URL for the OpenAI-compatible API (e.g., for OpenRouter) | Provided by your LLM API provider |

### Optional

| Variable        | Description                                     | Default      |
| :-------------- | :---------------------------------------------- | :----------- |
| `QDRANT_HOST`   | Hostname for the Qdrant service                 | `localhost`  |
| `QDRANT_PORT`   | Port for the Qdrant service                     | `6333`       |
| `PHOENIX_HOST`  | Hostname for the Arize Phoenix service          | `localhost`  |
| `PHOENIX_PORT`  | Port for the Arize Phoenix service              | `6006`       |
| `DATABASE_URL`  | Connection string for the SQLite database | `sqlite:///sqlite.db` |


## Available Scripts

### General

-   `uv pip install -e ".[dev]"`: Install backend dependencies.
-   `docker compose up -d`: Start Qdrant and Arize Phoenix services.
-   `docker compose down`: Stop Docker services.

### Backend

-   `uv run backend/app.py`: Start the FastAPI backend server.
-   `python backend/app.py migrate`: Run database migrations (for SQLModel/SQLite).
-   `uv run backend/ingest.py`: Run the document ingestion script.
-   `uv run pytest backend/tests/`: Run backend tests.

### Frontend

-   `cd frontend && pnpm install`: Install frontend dependencies.
-   `cd frontend && pnpm dev`: Start the frontend development server.
-   `cd frontend && pnpm build`: Build the frontend for production.
-   `cd frontend && pnpm test`: Run frontend tests.

## Testing

### Running Tests

To run all backend tests:

```bash
uv run pytest backend/tests/
```

To run all frontend tests:

```bash
cd frontend
pnpm test
```

### Test Structure

-   **Backend (`backend/tests/`)**:
    *   Unit tests for SQLModel schemas (relational integrity, cascades).
    *   Unit tests for `RAGPipeline` (using mock adapters for sequence, error handling).
    *   Tests for evaluation status transitions (mocking background tasks).
-   **Frontend (`frontend/src/tests/` or similar)**:
    *   Vitest tests for `useChat` hook and API adapter (state transitions, SSE handling).
-   **Integration Tests**:
    *   Focus on end-to-end RAG chain (Input -> Retrieval -> Output) and persistence.

## Deployment

This project is designed with Docker Compose for easy local deployment. For production, consider deploying the backend (FastAPI) and frontend separately, using a cloud provider of your choice.

### Docker

You can build and run the entire application using Docker:

```bash
# Build the backend image (if not using uv run directly)
docker build -t rag-backend -f backend/Dockerfile .

# Start services (including Qdrant, Phoenix, and potentially the backend)
docker compose up
```

Ensure your environment variables are correctly set for the Docker containers.

## Troubleshooting

### Docker Services Not Starting

**Error:** `port is already allocated` or similar network errors.

**Solution:**
Ensure no other services are running on ports `6333`, `6334`, `6006`, or `8000`. Stop any conflicting processes or change the port mappings in `docker-compose.yml`.

### Database Connection Issues

**Error:** `sqlite.db` not found or connection errors.

**Solution:**
Ensure you have run the migrations to create the SQLite database:
```bash
python backend/app.py migrate
```

### Frontend Build/Run Issues

**Error:** `vite` command not found or dependency issues.

**Solution:**
Navigate to the `frontend` directory and ensure all dependencies are installed and the correct package manager command is used:
```bash
cd frontend
pnpm install
pnpm dev
```

### RAG Pipeline Errors

**Error:** LLM errors, retrieval failures.

**Solution:**
1.  Check `OPENAI_API_KEY` and `OPENAI_BASE_URL` in your `.env` file.
2.  Verify Qdrant is running: `docker ps`.
3.  Check Arize Phoenix UI (`http://localhost:6006`) for detailed traces to pinpoint the exact failure point in the RAG chain.

## Further Notes

This system follows a "developer's workbench" philosophy, prioritizing portability (SQLite), developer transparency (exposed metrics), and modularity over high-scale multi-user features. It's an excellent foundation for iterating on RAG systems.
