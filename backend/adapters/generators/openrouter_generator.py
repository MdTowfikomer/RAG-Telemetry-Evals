import asyncio
from typing import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from backend.core import Document, Generator


class OpenRouterGenerator(Generator):
    def __init__(
        self,
        api_key_provider: Callable[[], SecretStr],
        default_model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        stream_delay_seconds: float = 0.01,
    ):
        self.api_key_provider = api_key_provider
        self.default_model = default_model
        self.base_url = base_url
        self.stream_delay_seconds = stream_delay_seconds

    def _build_chain(self, context_text: str):
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the following question based ONLY on the provided context.
            If the answer is not in the context, say that you don't know.

            Context:
            {context}

            Question: {question}
            """
        )

        llm = ChatOpenAI(
            api_key=self.api_key_provider(),
            base_url=self.base_url,
            model=self.default_model,
        )

        return (
            {"context": lambda _: context_text, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    async def generate(self, query: str, docs: list[Document]) -> str:
        context_text = "\n\n".join([doc.page_content for doc in docs])
        chain = self._build_chain(context_text)

        response = await asyncio.to_thread(chain.invoke, query)

        return response

    def stream(self, query: str, docs: list[Document]):
        context_text = "\n\n".join([doc.page_content for doc in docs])
        chain = self._build_chain(context_text)

        async def _iterate_tokens():
            async for chunk in chain.astream(query):
                yield chunk
                if self.stream_delay_seconds > 0:
                    await asyncio.sleep(self.stream_delay_seconds)

        return _iterate_tokens()
