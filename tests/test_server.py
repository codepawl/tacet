import json

import pytest
from fastapi.testclient import TestClient

from tacet.server import MAX_BODY_BYTES, create_app


@pytest.fixture(scope="module")
def client(tiny_model):
    with TestClient(create_app(tiny_model)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def keyed_client(tiny_model):
    with TestClient(create_app(tiny_model, api_key="local-test-key")) as test_client:
        yield test_client


def test_systemone_answers(client, good_request):
    response = client.post("/v1/systemone", json=good_request)
    assert response.status_code == 200
    assert response.headers["X-Request-Id"].startswith("req_")
    body = response.json()
    assert body["model"] == "tacet-tiny"
    assert body["answers"]["urgency"]["legend"] == {"0": "low", "1": "medium", "2": "high"}


def test_concurrent_requests_share_the_batcher(client, good_request):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(lambda _: client.post("/v1/systemone", json=good_request), range(12)))
    assert all(response.status_code == 200 for response in responses)


def test_validation_errors_use_the_envelope(client, good_request):
    good_request["model"] = "gpt-4o"
    response = client.post("/v1/systemone", json=good_request)
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "model_not_found"
    assert error["param"] == "model"


def test_questions_too_long_is_a_400(client):
    levels = ["ticket " * 45] * 60
    body = {"state": "s", "questions": {"q": {"type": "score", "instructions": "how?", "criteria": levels}}}
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "questions_too_long"


def test_bad_json_and_large_bodies_are_refused(client):
    response = client.post("/v1/systemone", content=b"{not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "The request body must be valid JSON."
    response = client.post("/v1/systemone", content=b"x" * (MAX_BODY_BYTES + 1))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_chat_completions(client, good_request):
    chat = {"model": "tacet", "messages": [{"role": "user", "content": json.dumps(good_request)}]}
    response = client.post("/v1/chat/completions", json=chat)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert set(json.loads(body["choices"][0]["message"]["content"])["answers"]) == {"route", "urgency", "refund"}
    assert body["usage"]["completion_tokens"] == 0


def test_chat_completions_stream(client, good_request):
    chat = {"stream": True, "messages": [{"role": "user", "content": json.dumps(good_request)}]}
    response = client.post("/v1/chat/completions", json=chat)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.endswith("data: [DONE]\n\n")


def test_models_and_health(client):
    listing = client.get("/v1/models").json()
    assert listing["object"] == "list"
    assert listing["data"][0]["id"] == "tacet-tiny"
    assert listing["data"][0]["context_length"] == 1536
    assert client.get("/v1/health").json() == {"ok": True, "model": "tacet-tiny"}
    assert client.get("/health").json()["ok"] is True


def test_unknown_routes_and_methods_use_the_envelope(client):
    response = client.get("/v2/nothing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    response = client.get("/v1/systemone")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_no_key_needed_by_default(client, good_request):
    response = client.post("/v1/systemone", json=good_request, headers={"Authorization": "Bearer anything"})
    assert response.status_code == 200


def test_api_key_is_enforced_when_set(keyed_client, good_request):
    missing = keyed_client.post("/v1/systemone", json=good_request)
    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "missing_api_key"
    wrong = keyed_client.post("/v1/systemone", json=good_request, headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    assert wrong.json()["error"]["code"] == "invalid_api_key"
    right = keyed_client.post("/v1/systemone", json=good_request,
                              headers={"Authorization": "Bearer local-test-key"})
    assert right.status_code == 200
    assert keyed_client.get("/v1/health").status_code == 200
