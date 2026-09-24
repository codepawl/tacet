import pytest

import tacet
from tacet import RequestError, choice, noul, score


def test_load_reads_the_name_and_defaults(tiny_model):
    assert tiny_model.name == "tacet-tiny"
    assert tiny_model.max_length == 1536
    assert tiny_model.device.type == "cpu"


def test_decide_returns_the_api_shape(tiny_model, good_request):
    response = tiny_model.decide(good_request["state"], good_request["questions"])
    assert response["model"] == "tacet-tiny"
    assert set(response) == {"model", "answers", "usage"}
    answers = response["answers"]
    assert answers["route"]["choice"] in {"billing", "tech"}
    assert sum(answers["route"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= answers["urgency"]["score"] <= 2.0
    assert answers["urgency"]["legend"] == {"0": "low", "1": "medium", "2": "high"}
    assert 0.0 <= answers["refund"]["noul"] <= 1.0
    assert response["usage"]["output_tokens"] == 0
    assert response["usage"]["input_tokens"] > 10


def test_question_helpers_build_the_wire_shape(tiny_model):
    questions = {"route": choice("which team?", {"billing": "payments", "tech": "bugs"}),
                 "urgency": score("how urgent?", ["low", "high"]),
                 "refund": noul("money back?", {"true": "they ask"})}
    assert questions["refund"] == {"type": "noul", "instructions": "money back?", "criteria": {"true": "they ask"}}
    assert set(tiny_model.decide("ticket", questions)["answers"]) == set(questions)


def test_batch_matches_one_at_a_time(tiny_model, good_request):
    requests = [good_request, {"state": "ticket " * 50, "questions": {"q": noul("refund?")}}]
    batched = tiny_model.decide_batch(requests, batch_size=2)
    for request, response in zip(requests, batched):
        alone = tiny_model.decide(request["state"], request["questions"])
        assert response["usage"] == alone["usage"]
        for question_id, answer in alone["answers"].items():
            for key, value in answer.items():
                if isinstance(value, float):
                    assert response["answers"][question_id][key] == pytest.approx(value, abs=2e-3)


def test_invalid_request_raises_request_error(tiny_model):
    with pytest.raises(RequestError) as raised:
        tiny_model.decide("s", {"q": {"type": "rank", "instructions": "x"}})
    assert raised.value.param == "questions.q.type"


def test_questions_that_do_not_fit_raise_with_a_code(tiny_model):
    levels = ["ticket " * 45] * 60
    with pytest.raises(RequestError) as raised:
        tiny_model.decide("s", {"q": score("how urgent?", levels)})
    assert raised.value.code == "questions_too_long"


def test_max_length_bounds(tiny_model_directory):
    with pytest.raises(ValueError):
        tacet.load(tiny_model_directory, device="cpu", max_length=8192)
    long_model = tacet.load(tiny_model_directory, device="cpu", max_length=4096)
    response = long_model.decide("ticket " * 5000, {"q": noul("refund?")})
    assert response["usage"]["input_tokens"] == 4096
    assert response["usage"]["state_truncated"] is True


def test_missing_folder_config_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError):
        tacet.load(str(tmp_path), device="cpu")
