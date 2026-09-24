"""Turning a request into the one token sequence the model reads.

All questions go first and the state once after them:

  [CLS] <q1 type> question: instructions [MASK] opt [MASK] opt [SEP]
        <q2 type> question: instructions [MASK] opt ... [SEP]
        ...
        state [SEP]

Each question's options are read at its own [MASK] markers. Every token also gets a segment
id: the type of the question block it belongs to (choice, score, noul), or the state segment.

The option and state rendering (`QUESTION_TYPES`, `render_criterion`, `render_options`,
`serialize_state`) is the prompt format the model was trained on. It comes from Laya
(https://huggingface.co/convaiinnovations/laya, Apache 2.0, see NOTICE) and is copied here so
that the text the model reads can never change under it with another package's release.
"""

import json
from dataclasses import dataclass

import torch

QUESTION_TYPES = {"choice": 0, "score": 1, "noul": 2}
STATE_SEGMENT = 3
SEGMENT_COUNT = 4
MAX_OPTION_TOKENS = 48
# The questions go first, so they must leave the state at least this much room.
MIN_STATE_ROOM = 64


class QuestionsTooLong(ValueError):
    """The question blocks alone fill the sequence; the request cannot be answered."""


@dataclass
class PackedRequest:
    """One request, tokenized and ready to batch."""

    question_ids: list
    questions: list
    token_ids: list
    segment_ids: list
    markers: list
    state_truncated: bool = False
    options_truncated: bool = False


def serialize_state(state):
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value):
    """One criterion as text: strings pass through, anything structured becomes compact JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(question_type, criteria):
    """Option texts in label order. A noul question is always [false, true]."""
    if question_type == "choice":
        # Only None and "" mean "no description"; 0 and False are real criterion values.
        return [name if description is None or description == ""
                else "%s: %s" % (name, render_criterion(description))
                for name, description in criteria.items()]
    if question_type == "score":
        return ["level %d: %s" % (level, render_criterion(text)) for level, text in enumerate(criteria)]
    criteria = criteria or {}
    false_text = criteria.get("false")
    true_text = criteria.get("true")
    if false_text in (None, ""):
        false_text = "no, the statement does not hold"
    else:
        false_text = render_criterion(false_text)
    if true_text in (None, ""):
        true_text = "yes, the statement holds"
    else:
        true_text = render_criterion(true_text)
    return ["false: " + false_text, "true: " + true_text]


def instructions_text(instructions):
    if isinstance(instructions, str):
        return instructions
    return json.dumps(instructions)


def _without_mask(text, tokenizer):
    """A literal mask token in user text would add a marker the model reads as an option."""
    return text.replace(tokenizer.mask_token, " ")


def _token_ids(tokenizer, text):
    return tokenizer(_without_mask(text, tokenizer), add_special_tokens=False)["input_ids"]


def build_packed_sequence(tokenizer, state, questions, max_length):
    """Token ids, segment ids and per-question marker positions for one request.

    `questions` is a list of {"type", "instructions", "criteria"} in the order they are packed.
    """
    token_ids = [tokenizer.cls_token_id]
    segment_ids = [STATE_SEGMENT]
    markers = []

    for question in questions:
        options = render_options(question["type"], question.get("criteria"))
        heading = "%s question: %s" % (question["type"], instructions_text(question["instructions"]))
        block = _token_ids(tokenizer, heading)
        block_markers = []
        for option in options:
            block_markers.append(len(token_ids) + len(block))
            option_ids = _token_ids(tokenizer, " " + option)[:MAX_OPTION_TOKENS]
            block.append(tokenizer.mask_token_id)
            block.extend(option_ids)
        block.append(tokenizer.sep_token_id)
        token_ids.extend(block)
        segment_ids.extend([QUESTION_TYPES[question["type"]]] * len(block))
        markers.append(block_markers)

    if len(token_ids) > max_length - MIN_STATE_ROOM:
        raise QuestionsTooLong(
            f"the questions take {len(token_ids)} tokens, which leaves no room for the state "
            f"within {max_length}; send fewer or shorter questions")

    room = max(0, max_length - len(token_ids) - 1)
    state_ids = _token_ids(tokenizer, serialize_state(state))[:room]
    token_ids.extend(state_ids)
    token_ids.append(tokenizer.sep_token_id)
    segment_ids.extend([STATE_SEGMENT] * (len(state_ids) + 1))
    return token_ids, segment_ids, markers


def any_option_truncated(tokenizer, questions):
    """Whether any option is longer than the MAX_OPTION_TOKENS the model reads of it."""
    for question in questions:
        for option in render_options(question["type"], question.get("criteria")):
            if len(_token_ids(tokenizer, " " + option)) > MAX_OPTION_TOKENS:
                return True
    return False


def pack_request(tokenizer, state, questions, max_length):
    """A validated request as a PackedRequest. Raises QuestionsTooLong."""
    question_ids = list(questions.keys())
    ordered = [questions[question_id] for question_id in question_ids]
    token_ids, segment_ids, markers = build_packed_sequence(tokenizer, state, ordered, max_length)
    state_token_count = len(_token_ids(tokenizer, serialize_state(state)))
    # [CLS] and the closing [SEP] also carry the state segment.
    state_tokens_kept = segment_ids.count(STATE_SEGMENT) - 2
    return PackedRequest(
        question_ids=question_ids,
        questions=ordered,
        token_ids=token_ids,
        segment_ids=segment_ids,
        markers=markers,
        state_truncated=state_tokens_kept < state_token_count,
        options_truncated=any_option_truncated(tokenizer, ordered),
    )


def collate(packed_requests, pad_id):
    """Pad packed requests to one batch of tensors."""
    longest = max(len(request.token_ids) for request in packed_requests)
    most_questions = max(len(request.markers) for request in packed_requests)
    most_options = max(len(block) for request in packed_requests for block in request.markers)
    batch_size = len(packed_requests)

    input_ids = torch.full((batch_size, longest), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, longest), dtype=torch.long)
    segment_ids = torch.full((batch_size, longest), STATE_SEGMENT, dtype=torch.long)
    marker_positions = torch.zeros((batch_size, most_questions, most_options), dtype=torch.long)
    marker_mask = torch.zeros((batch_size, most_questions, most_options), dtype=torch.bool)

    for row, request in enumerate(packed_requests):
        length = len(request.token_ids)
        input_ids[row, :length] = torch.tensor(request.token_ids)
        attention_mask[row, :length] = 1
        segment_ids[row, :length] = torch.tensor(request.segment_ids)
        for question_index, block in enumerate(request.markers):
            marker_positions[row, question_index, :len(block)] = torch.tensor(block)
            marker_mask[row, question_index, :len(block)] = True
    return {"input_ids": input_ids, "attention_mask": attention_mask, "segment_ids": segment_ids,
            "marker_positions": marker_positions, "marker_mask": marker_mask}
