# 📄 Resume & Production Roadmap: RAG Workbench

This document serves as a guide for expanding your **Persistent RAG Workbench** to make it resume-ready for AI/Software Engineering roles and preparing it for a secure public deployment.

---

## 🛠️ Section 1: Resume-Boosting Features & Enhancements

Recruiters look for optimization, system design, and evaluation-driven development rather than simple API wrappers. Implementing the following features will significantly elevate your technical portfolio.

### 1. Two-Stage Retrieval (Cross-Encoder Reranking)
* **Current Bottleneck:** The current [passthrough_reranker.py](file:///D:/Programming/major-projects/RAG/backend/adapters/rerankers/passthrough_reranker.py) returns chunks as-is. Standard semantic search retrieves documents based on vector similarity, which often misses the structural nuances of the user's question.
* **Implementation:**
  - Create a custom reranker using the lightweight CPU-based **FlashRank** library or integrate the **Cohere Rerank API**.
  - Update the `Reranker` interface to sort the retrieved chunks by relevance score and discard low-scoring contexts.
* **Resume Impact:** 
  > *"Implemented a two-stage retrieval pipeline using dense vector search and cross-encoder reranking, reducing noise in context construction and improving answer relevance by X%."*

### 2. Hybrid Search (Dense + Sparse Retrieval)
* **Current Bottleneck:** Dense vector search is excellent for general semantic understanding but struggles with specific keywords, product codes, or acronyms.
* **Implementation:**
  - Update [qdrant_retriever.py](file:///D:/Programming/major-projects/RAG/backend/adapters/retrievers/qdrant_retriever.py) to extract sparse document tokens (using BM25/Lexical indexing).
  - Query Qdrant for both dense embeddings and sparse tokens.
  - Merge the result rankings using **Reciprocal Rank Fusion (RRF)**.
* **Resume Impact:**
  > *"Engineered a hybrid search architecture fusing semantic vector embeddings with sparse lexical search (BM25) via Reciprocal Rank Fusion, improving exact-keyword search precision."*

### 3. Parent-Child Chunking
* **Current Bottleneck:** Large chunks contain too much filler text, while tiny chunks lack context.
* **Implementation:**
  - When ingesting, segment text into small "child" nodes (100 tokens) associated with larger parent chunks (500-1000 tokens).
  - Query the vector store against the child chunks, but feed the parent chunk context to the Generator.
* **Resume Impact:**
  > *"Optimized text chunking strategy by implementing parent-child document mappings, ensuring high-granularity retrieval while feeding comprehensive parent context to the LLM."*

---

## 🔒 Section 2: Pre-Deployment & Production Checklist

Before exposing your RAG workbench to the public internet, you must address several infrastructure and security vulnerabilities.

### 1. Rotate Exposed Credentials
* **Urgent Risk:** As identified in [ARCHITECTURAL_REVIEW.md](file:///D:/Programming/major-projects/RAG/ARCHITECTURAL_REVIEW.md#L79-L97), live API keys for OpenRouter and Hugging Face were previously committed to Git history.
* **Action Steps:**
  1. Revoke and delete the current keys in your API provider dashboards immediately.
  2. Generate new keys and store them only locally in an untracked `.env` file.
  3. Set up environment variables directly in your cloud hosting provider (Vercel, Render, Railway, etc.) without committing them.

### 2. Configure Production CORS Boundaries
* **Risk:** The wildcard setup `allow_origins=["*"]` allows any client origin to query your backend.
* **Action Steps:**
  - Update [backend/app.py](file:///D:/Programming/major-projects/RAG/backend/app.py#L42-L48) to pull authorized origins from your configuration settings:
    ```python
    CORS_ORIGINS = ["https://your-frontend-domain.vercel.app"]
    ```

### 3. Transition from SQLite to PostgreSQL
* **Risk:** SQLite databases are file-based and ephemeral. When hosting on serverless platforms (e.g., Render/Railway), the file system is reset on container updates, resulting in complete database history loss.
* **Action Steps:**
  - Use a serverless PostgreSQL database (e.g., **Neon** or **Supabase**).
  - Update your database connection string env variable. SQLModel will automatically support it with no changes to model definitions.

### 4. Deploy Qdrant to Qdrant Cloud
* **Action Steps:**
  - Instead of self-hosting Qdrant inside a Docker container in production, create a free-tier cluster (1GB) on **Qdrant Cloud**.
  - Configure the backend to point to the remote cluster URI and API Key.

---

## 📝 Section 3: Resume Bullet Points & Project Framing

Use these high-impact, action-oriented bullet points to detail your work on this project:

* **Modular System Architecture**
  > *"Architected a modular Retrieval-Augmented Generation (RAG) system utilizing a Clean Ports & Adapters design, decoupling retrieval, reranking, and generation engines for high testability and swappable backends."*

* **Decoupled Telemetry & Real-Time Performance**
  > *"Designed an asynchronous pipeline evaluator leveraging Ragas and Arize Phoenix/OpenTelemetry, deploying FastAPI background tasks to compute faithfulness and relevancy metrics without adding overhead to query generation latency."*

* **Reactive UX Orchestration**
  > *"Engineered a server-sent events (SSE) broadcasting system in FastAPI to stream LLM responses immediately, while asynchronously pushing downstream pipeline quality evaluation metrics to a React client."*

* **Relational Database Design**
  > *"Designed a persistent storage schema using SQLModel and SQLite/PostgreSQL to track chat session history, response latency metrics, token consumption, and version-controlled evaluation evaluations."*
