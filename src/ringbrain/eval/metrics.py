import re
from dataclasses import dataclass, field

PRICE_PATTERN = re.compile(r"\$\s?\d+")

OUTCOME_MATCHES = {
    "book": {"book"},
    "escalate": {"escalate"},
    "info_only": {"info_only"},
    "any": {"book", "escalate", "info_only", "incomplete"},
}


@dataclass
class PersonaResult:
    persona_name: str
    expected_outcome: str
    final_outcome: str  # "book" | "escalate" | "info_only" | "incomplete"
    turns_used: int
    transcript: list[dict]
    hallucination_detected: bool = False
    judge_scores: dict | None = None

    @property
    def matches_expected(self) -> bool:
        return self.final_outcome in OUTCOME_MATCHES.get(self.expected_outcome, set())


@dataclass
class EvalReport:
    results: list[PersonaResult] = field(default_factory=list)

    @property
    def task_completion_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.matches_expected for r in self.results) / len(self.results)

    @property
    def hallucination_rate(self) -> float:
        if not self.results:
            return 0.0
        flagged = sum(r.hallucination_detected for r in self.results)
        return flagged / len(self.results)

    @property
    def average_turns(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.turns_used for r in self.results) / len(self.results)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "task_completion_rate": round(self.task_completion_rate, 3),
                "hallucination_rate": round(self.hallucination_rate, 3),
                "average_turns": round(self.average_turns, 2),
                "n_personas": len(self.results),
            },
            "results": [
                {
                    "persona_name": r.persona_name,
                    "expected_outcome": r.expected_outcome,
                    "final_outcome": r.final_outcome,
                    "matches_expected": r.matches_expected,
                    "turns_used": r.turns_used,
                    "hallucination_detected": r.hallucination_detected,
                    "judge_scores": r.judge_scores,
                    "transcript": r.transcript,
                }
                for r in self.results
            ],
        }


def detect_price_hallucination(agent_replies: list[str]) -> bool:
    """RingBrain is never given real pricing data in these eval scenarios, so
    any dollar figure in a reply means the model invented one.
    """
    return any(PRICE_PATTERN.search(reply) for reply in agent_replies)
