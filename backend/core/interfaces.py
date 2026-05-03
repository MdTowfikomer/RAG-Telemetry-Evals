from abc import ABC, abstractmethod
from typing import AsyncGenerator

from .models import Document


class Retriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, k: int) -> list[Document]:
        raise NotImplementedError


class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        raise NotImplementedError


class Generator(ABC):
    @abstractmethod
    async def generate(self, query: str, docs: list[Document]) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream(self, query: str, docs: list[Document]) -> AsyncGenerator[str, None]:
        raise NotImplementedError
