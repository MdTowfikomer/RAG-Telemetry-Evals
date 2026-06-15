import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.dependencies import factory
from backend.api.routes import (
    chat_router,
    documents_router,
    evaluations_router,
    scores_router,
    sessions_router,
)


# what does lifespan do?
@asynccontextmanager
async def lifespan(_app: FastAPI):
    factory.init_db()
    yield


def create_app() -> FastAPI:
    application = FastAPI(title="Modular RAG API", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=factory.settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(chat_router)
    application.include_router(documents_router)
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
