"""Request validation, ported from the hosted API's tests (pricing and billing removed)."""

import json

import pytest

from tacet import validation
from tacet.validation import RequestError

SERVED = "tacet-sonata"


def test_valid_request_passes(good_request):
    state, questions = validation.validate_systemone(good_request, SERVED)
    assert state == good_request["state"]
    assert set(questions) == {"route", "urgency", "refund"}


@pytest.mark.parametrize("model", [None, SERVED, "codepawl/tacet-sonata", "tacet", "tacet-1", "tacet-latest",
                                   "codepawl/tacet-1"])
def test_served_name_and_hosted_aliases_are_accepted(good_request, model):
    if model is not None:
        good_request["model"] = model
    validation.validate_systemone(good_request, SERVED)


@pytest.mark.parametrize("mutate, status, param", [
    (lambda body: body.pop("state"), 400, "state"),
    (lambda body: body.update(state=5), 400, "state"),
    (lambda body: body.update(questions={}), 400, "questions"),
    (lambda body: body.update(model="gpt-4"), 404, "model"),
    (lambda body: body["questions"]["route"].update(type="rank"), 400, "questions.route.type"),
    (lambda body: body["questions"]["route"].update(criteria={"only": "one"}), 400, "questions.route.criteria"),
    (lambda body: body["questions"]["urgency"].update(criteria=["one"]), 400, "questions.urgency.criteria"),
    (lambda body: body["questions"]["refund"].update(criteria={"maybe": "x"}), 400, "questions.refund.criteria"),
    (lambda body: body["questions"]["refund"].update(instructions=""), 400, "questions.refund.instructions"),
    (lambda body: body.update(state="x" * (validation.MAX_STATE_CHARS + 1)), 413, "state"),
])
def test_invalid_requests_are_refused(good_request, mutate, status, param):
    mutate(good_request)
    with pytest.raises(RequestError) as raised:
        validation.validate_systemone(good_request, SERVED)
    assert raised.value.status == status
    assert raised.value.param == param


def test_too_many_questions():
    body = {"state": "s", "questions": {f"q{index}": {"type": "noul", "instructions": "ok?"}
                                        for index in range(validation.MAX_QUESTIONS + 1)}}
    with pytest.raises(RequestError):
        validation.validate_systemone(body, SERVED)


def test_every_error_has_a_code(good_request):
    body = json.loads(json.dumps(good_request))
    body["questions"]["route"]["type"] = "rank"
    with pytest.raises(RequestError) as raised:
        validation.validate_systemone(body, SERVED)
    assert raised.value.code == "invalid_request"
    with pytest.raises(RequestError) as raised:
        validation.validate_systemone({**good_request, "state": "x" * (validation.MAX_STATE_CHARS + 1)}, SERVED)
    assert raised.value.code == "state_too_large"
    with pytest.raises(RequestError) as raised:
        validation.validate_systemone({**good_request, "model": "gpt-4o"}, SERVED)
    assert raised.value.code == "model_not_found"
    assert raised.value.body()["error"]["type"] == "invalid_request_error"


def test_object_and_array_instructions_are_accepted_and_rendered_to_text():
    body = {"state": "x", "questions": {
        "severity": {"type": "score", "instructions": {"what": "How severe?", "examples": ["crash"]},
                     "criteria": ["low", "high"]},
        "refund": {"type": "noul", "instructions": ["Did they ask", "for money back?"]},
    }}
    _, questions = validation.validate_systemone(body, SERVED)
    assert questions["severity"]["instructions"] == '{"what": "How severe?", "examples": ["crash"]}'
    assert questions["refund"]["instructions"] == '["Did they ask", "for money back?"]'
    assert body["questions"]["severity"]["instructions"] == {"what": "How severe?", "examples": ["crash"]}


def test_empty_object_instructions_are_refused():
    body = {"state": "x", "questions": {"q": {"type": "noul", "instructions": {}}}}
    with pytest.raises(RequestError):
        validation.validate_systemone(body, SERVED)


def test_chat_adapter_reads_last_user_message(good_request):
    chat = {"model": "tacet-1", "messages": [
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": json.dumps(good_request)},
    ]}
    state, questions = validation.systemone_from_chat(chat, SERVED)
    assert state == good_request["state"]
    assert "route" in questions


def test_chat_adapter_accepts_content_parts_and_the_provider_prefix(good_request):
    chat = {"model": "codepawl/tacet-1", "messages": [
        {"role": "user", "content": [{"type": "text", "text": json.dumps(good_request)}]}]}
    _, questions = validation.systemone_from_chat(chat, SERVED)
    assert len(questions) == 3


def test_chat_adapter_refuses_unknown_models_like_systemone(good_request):
    chat = {"model": "gpt-4o", "messages": [{"role": "user", "content": json.dumps(good_request)}]}
    with pytest.raises(RequestError) as raised:
        validation.systemone_from_chat(chat, SERVED)
    assert raised.value.status == 404
    assert raised.value.code == "model_not_found"


@pytest.mark.parametrize("body", [
    {"messages": [{"role": "user", "content": "is this urgent?"}]},
    {"messages": [{"role": "user", "content": "[1, 2]"}]},
    {"messages": [{"role": "system", "content": "{}"}]},
    {"messages": []},
    [],
])
def test_chat_adapter_refuses_what_is_not_a_request(body):
    with pytest.raises(RequestError) as raised:
        validation.systemone_from_chat(body, SERVED)
    assert raised.value.status == 400


def decide_response(usage):
    return {"model": SERVED, "answers": {"refund": {"type": "noul", "noul": 0.9, "confidence": 0.9}},
            "usage": usage}


def test_chat_completion_shapes():
    response = decide_response({"input_tokens": 120, "output_tokens": 0})
    completion = validation.chat_completion(response)
    assert completion["model"] == SERVED
    assert completion["usage"] == {"prompt_tokens": 120, "completion_tokens": 0, "total_tokens": 120}
    assert json.loads(completion["choices"][0]["message"]["content"]) == {"answers": response["answers"]}
    stream = validation.chat_completion(response, stream=True)
    assert stream.endswith("data: [DONE]\n\n")
    assert stream.count("data: ") == 3


def test_chat_reply_carries_truncation_flags():
    usage = {"input_tokens": 120, "output_tokens": 0, "state_truncated": True, "options_truncated": True}
    response = decide_response(usage)
    content = json.loads(validation.chat_completion(response)["choices"][0]["message"]["content"])
    assert content == {"answers": response["answers"], "state_truncated": True, "options_truncated": True}


def test_models_listing_keeps_the_fields_old_clients_read():
    model = validation.models_listing(SERVED, 1536, 0)["data"][0]
    assert model["id"] == SERVED
    assert model["name"] == SERVED
    assert model["object"] == "model"
    assert model["context_length"] == 1536
    assert model["max_output_tokens"] == 0


def test_bearer_key():
    assert validation.bearer_key("Bearer abc") == "abc"
    assert validation.bearer_key("bearer abc") == "abc"
    assert validation.bearer_key("BEARER  abc ") == "abc"
    for header in (None, "", "Basic abc", "Bearer "):
        with pytest.raises(RequestError) as raised:
            validation.bearer_key(header)
        assert raised.value.status == 401
