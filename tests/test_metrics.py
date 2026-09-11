from ringbrain.eval.metrics import EvalReport, PersonaResult, detect_price_hallucination


def test_detect_price_hallucination_true():
    assert detect_price_hallucination(["Sure, that's just $10 flat!"]) is True


def test_detect_price_hallucination_false():
    assert detect_price_hallucination(["I'm not able to quote exact pricing over the phone."]) is False


def test_eval_report_aggregates():
    report = EvalReport(
        results=[
            PersonaResult("a", "book", "book", 2, [], hallucination_detected=False),
            PersonaResult("b", "escalate", "escalate", 1, [], hallucination_detected=False),
            PersonaResult("c", "info_only", "book", 3, [], hallucination_detected=True),
        ]
    )
    assert report.task_completion_rate == 2 / 3
    assert report.hallucination_rate == 1 / 3
    assert report.average_turns == 2.0


def test_persona_result_matches_expected_any():
    result = PersonaResult("d", "any", "incomplete", 4, [])
    assert result.matches_expected is True
