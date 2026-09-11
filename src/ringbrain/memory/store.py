from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from ringbrain.db.models import Customer, MemoryEntry
from ringbrain.memory.embedder import Embedder


@dataclass
class RecalledMemory:
    content: str
    distance: float


class MemoryStore:
    """The 'self-learning brain': writes conversation summaries as embeddings,
    and recalls the most relevant past context for a new conversation so the
    agent never makes a customer repeat themselves.
    """

    def __init__(self, session: Session, embedder: Embedder):
        self.session = session
        self.embedder = embedder

    def get_or_create_customer(self, phone_number: str) -> Customer:
        customer = self.session.scalar(
            select(Customer).where(Customer.phone_number == phone_number)
        )
        if customer is None:
            customer = Customer(phone_number=phone_number)
            self.session.add(customer)
            self.session.flush()
        return customer

    def remember(self, customer_id: str, content: str, conversation_id: str | None = None) -> MemoryEntry:
        entry = MemoryEntry(
            customer_id=customer_id,
            conversation_id=conversation_id,
            content=content,
            embedding=self.embedder.embed(content),
        )
        self.session.add(entry)
        self.session.flush()
        return entry

    def recall(self, customer_id: str, query: str, top_k: int = 3) -> list[RecalledMemory]:
        query_vector = self.embedder.embed(query)
        distance = MemoryEntry.embedding.cosine_distance(query_vector)
        rows = self.session.execute(
            select(MemoryEntry.content, distance.label("distance"))
            .where(MemoryEntry.customer_id == customer_id)
            .order_by(distance)
            .limit(top_k)
        ).all()
        return [RecalledMemory(content=row.content, distance=row.distance) for row in rows]
