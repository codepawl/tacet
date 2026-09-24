"""Turning the model's per-option logits into the answers the API returns.

`confidence_from_probabilities` is Laya's normalized-entropy confidence
(https://huggingface.co/convaiinnovations/laya, Apache 2.0, see NOTICE).
"""

import math

import numpy as np

DECIMALS = 4


def softmax(scores):
    exponentials = np.exp(scores - scores.max())
    return exponentials / exponentials.sum()


def confidence_from_probabilities(probabilities):
    """1 - H(p) / log(k): 1 when all mass is on one option, 0 when it is spread evenly."""
    option_count = len(probabilities)
    if option_count < 2:
        return 1.0
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -(probabilities * np.log(clipped)).sum()
    return float(np.clip(1.0 - entropy / math.log(option_count), 0.0, 1.0))


def _rounded(value):
    return round(float(value), DECIMALS)


def decode_choice(question, probabilities):
    names = list(question["criteria"].keys())
    return {
        "type": "choice",
        "choice": names[int(probabilities.argmax())],
        "probabilities": {name: _rounded(value) for name, value in zip(names, probabilities)},
        "confidence": _rounded(confidence_from_probabilities(probabilities)),
    }


def decode_score(question, probabilities):
    levels = np.arange(len(probabilities))
    return {
        "type": "score",
        "score": _rounded((levels * probabilities).sum()),
        "probabilities": {str(level): _rounded(value) for level, value in enumerate(probabilities)},
        "confidence": _rounded(confidence_from_probabilities(probabilities)),
        # Level number to the criterion as it was sent. Jev returns the same field.
        "legend": {str(level): criterion for level, criterion in enumerate(question["criteria"])},
    }


def decode_noul(probabilities):
    probability_true = float(probabilities[1])
    return {
        "type": "noul",
        "noul": _rounded(probability_true),
        "confidence": _rounded(max(probability_true, 1.0 - probability_true)),
    }


def decode_answers(logits, packed_request):
    """Answers for one request from its [questions, options] logits."""
    answers = {}
    for index, question_id in enumerate(packed_request.question_ids):
        question = packed_request.questions[index]
        option_count = len(packed_request.markers[index])
        probabilities = softmax(logits[index, :option_count])
        if question["type"] == "choice":
            answers[question_id] = decode_choice(question, probabilities)
        elif question["type"] == "score":
            answers[question_id] = decode_score(question, probabilities)
        else:
            answers[question_id] = decode_noul(probabilities)
    return answers


def usage_of(packed_request):
    """`usage` as the API reported it: the packed length, and flags when something was cut."""
    usage = {"input_tokens": len(packed_request.token_ids), "output_tokens": 0}
    if packed_request.state_truncated:
        usage["state_truncated"] = True
    if packed_request.options_truncated:
        usage["options_truncated"] = True
    return usage
