"""Checks on a real model folder. Run with TACET_TEST_CHECKPOINT=<folder> pytest -m slow."""

import os

import pytest

import tacet
from tacet import choice, noul, score

CHECKPOINT = os.environ.get("TACET_TEST_CHECKPOINT")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not CHECKPOINT, reason="set TACET_TEST_CHECKPOINT to a Tacet model folder"),
]


@pytest.fixture(scope="module")
def model():
    return tacet.load(CHECKPOINT, device="cpu")


def test_obvious_ticket_is_answered_sensibly(model):
    response = model.decide(
        state={"ticket": "I was charged twice for my March invoice and I want the second charge refunded."},
        questions={
            "route": choice("Which team should handle this ticket?",
                            {"billing": "payments, invoices and refunds", "tech": "bugs, crashes and outages"}),
            "urgency": score("How urgent is this ticket?", ["low", "medium", "high"]),
            "refund": noul("Does the customer ask for money back?"),
        },
    )
    answers = response["answers"]
    assert answers["route"]["choice"] == "billing"
    assert answers["route"]["probabilities"]["billing"] > 0.9
    assert 0.0 <= answers["refund"]["noul"] <= 1.0
    assert 0.0 <= answers["urgency"]["score"] <= 2.0


def test_batch_agrees_with_single_calls(model):
    requests = [
        {"state": "The app crashes every time I open settings.",
         "questions": {"route": choice("Which team?", {"billing": "payments", "tech": "bugs"})}},
        {"state": "Please refund my last payment, it was a mistake.",
         "questions": {"refund": noul("Does the customer ask for money back?")}},
    ]
    batched = model.decide_batch(requests)
    for request, response in zip(requests, batched):
        alone = model.decide(request["state"], request["questions"])
        assert response["answers"].keys() == alone["answers"].keys()
        for question_id, answer in alone["answers"].items():
            if answer["type"] == "noul":
                assert response["answers"][question_id]["noul"] == pytest.approx(answer["noul"], abs=1e-3)
            else:
                assert response["answers"][question_id]["choice"] == answer["choice"]
