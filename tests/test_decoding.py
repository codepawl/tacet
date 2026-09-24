import numpy as np
import pytest

from tacet.decoding import confidence_from_probabilities, decode_answers, usage_of
from tacet.packing import PackedRequest


def test_confidence_is_one_minus_normalized_entropy():
    assert confidence_from_probabilities(np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert confidence_from_probabilities(np.array([0.25, 0.25, 0.25, 0.25])) == pytest.approx(0.0)
    assert confidence_from_probabilities(np.array([1.0])) == 1.0


def packed(questions, markers):
    return PackedRequest(question_ids=[f"q{index}" for index in range(len(questions))], questions=questions,
                         token_ids=list(range(40)), segment_ids=[3] * 40, markers=markers)


def test_each_type_decodes_to_the_api_shape():
    questions = [
        {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B", "c": "C"}},
        {"type": "score", "instructions": "x", "criteria": ["low", {"what": "high"}]},
        {"type": "noul", "instructions": "x"},
    ]
    logits = np.full((3, 3), -1e4, dtype=np.float32)
    logits[0, :3] = [0.0, 2.0, 0.0]
    logits[1, :2] = [0.0, 0.0]
    logits[2, :2] = [0.0, np.log(3.0)]
    answers = decode_answers(logits, packed(questions, [[1, 2, 3], [5, 6], [8, 9]]))

    assert answers["q0"]["choice"] == "b"
    assert set(answers["q0"]["probabilities"]) == {"a", "b", "c"}
    assert sum(answers["q0"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert answers["q1"]["score"] == pytest.approx(0.5)
    assert answers["q1"]["probabilities"] == {"0": 0.5, "1": 0.5}
    assert answers["q1"]["legend"] == {"0": "low", "1": {"what": "high"}}
    assert answers["q2"] == {"type": "noul", "noul": 0.75, "confidence": 0.75}


def test_usage_reports_tokens_and_truncation():
    request = packed([{"type": "noul", "instructions": "x"}], [[1, 2]])
    assert usage_of(request) == {"input_tokens": 40, "output_tokens": 0}
    request.state_truncated = True
    assert usage_of(request)["state_truncated"] is True
