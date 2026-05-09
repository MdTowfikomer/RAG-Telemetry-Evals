from datetime import UTC, datetime
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField
from sqlmodel import Relationship, SQLModel


class Document(BaseModel):
    page_content: str
    metadata: dict = Field(default_factory=dict)


class ChatSession(SQLModel, table=True):
    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    title: str
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))

    messages: List["ChatMessage"] = Relationship(back_populates="session")


class ChatMessage(SQLModel, table=True):
    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    session_id: UUID = SQLField(foreign_key="chatsession.id")
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))

    session: Optional[ChatSession] = Relationship(back_populates="messages")
    evaluations: List["Evaluation"] = Relationship(back_populates="message")


class Evaluation(SQLModel, table=True):
    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    message_id: UUID = SQLField(foreign_key="chatmessage.id")
    version: int
    status: str  # pending | completed | failed

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    error_message: str | None = None

    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))

    message: Optional[ChatMessage] = Relationship(back_populates="evaluations")
