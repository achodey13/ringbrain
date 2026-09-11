import math
from dataclasses import dataclass

from ringbrain.memory.embedder import Embedder
from ringbrain.memory.store import RecalledMemory


@dataclass
class SimpleCustomer:
    id: str
    phone_number: str
    name: str | None = None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class _Entry:
    customer_id: str
    content: str
    embedding: list[float]


class InMemoryMemoryStore:
    """Drop-in replacement for MemoryStore that needs no Postgres — used by
    the CLI simulator and eval harness so RingBrain's agent logic is fully
    testable without any infrastructure.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._customers: dict[str, SimpleCustomer] = {}
        self._entries: list[_Entry] = []

    def get_or_create_customer(self, phone_number: str) -> SimpleCustomer:
        for customer in self._customers.values():
            if customer.phone_number == phone_number:
                return customer
        customer = SimpleCustomer(id=f"cust-{len(self._customers) + 1}", phone_number=phone_number)
        self._customers[customer.id] = customer
        return customer

    def remember(self, customer_id: str, content: str, conversation_id: str | None = None) -> None:
        self._entries.append(
            _Entry(customer_id=customer_id, content=content, embedding=self.embedder.embed(content))
        )

    def recall(self, customer_id: str, query: str, top_k: int = 3) -> list[RecalledMemory]:
        query_vector = self.embedder.embed(query)
        candidates = [e for e in self._entries if e.customer_id == customer_id]
        scored = sorted(
            candidates, key=lambda e: _cosine_similarity(e.embedding, query_vector), reverse=True
        )
        return [
            RecalledMemory(content=e.content, distance=1 - _cosine_similarity(e.embedding, query_vector))
            for e in scored[:top_k]
        ]
