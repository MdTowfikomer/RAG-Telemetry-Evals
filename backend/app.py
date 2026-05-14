import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.dependencies import factory
from backend.api.routes import chat_router, evaluations_router, scores_router, sessions_router


def run_startup_migrations() -> None:
    engine = factory.get_engine()
    if engine.dialect.name != "sqlite":
        return

    with engine.connect() as connection:
        chatmessage_columns = {
            row[1] for row in connection.exec_driver_sql("PRAGMA table_info(chatmessage)")
        }
        if "latency_ms" not in chatmessage_columns:
            connection.exec_driver_sql("ALTER TABLE chatmessage ADD COLUMN latency_ms INTEGER")
        if "token_count" not in chatmessage_columns:
            connection.exec_driver_sql("ALTER TABLE chatmessage ADD COLUMN token_count INTEGER")

        evaluation_columns = {
            row[1] for row in connection.exec_driver_sql("PRAGMA table_info(evaluation)")
        }
        if "reasoning" not in evaluation_columns:
            connection.exec_driver_sql("ALTER TABLE evaluation ADD COLUMN reasoning VARCHAR")
        connection.commit()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    factory.init_db()
    run_startup_migrations()
    yield


def create_app() -> FastAPI:
    application = FastAPI(title="Modular RAG API", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(chat_router)
    application.include_router(sessions_router)
    application.include_router(evaluations_router)
    application.include_router(scores_router)
    return application


app = create_app()


if __name__ == "__main__":
    import sys

    import uvicorn

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(app, host="0.0.0.0", port=8000)
