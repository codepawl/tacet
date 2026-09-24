import pytest

from tacet.packing import (MAX_OPTION_TOKENS, QUESTION_TYPES, STATE_SEGMENT, QuestionsTooLong,
                           build_packed_sequence, collate, pack_request, render_options)


def test_render_options_keeps_the_trained_prompt_format():
    assert render_options("choice", {"billing": "payments", "tech": None, "zero": 0}) == [
        "billing: payments", "tech", "zero: 0"]
    assert render_options("score", ["low", {"what": "high"}]) == ["level 0: low", 'level 1: {"what": "high"}']
    assert render_options("noul", None) == ["false: no, the statement does not hold",
                                            "true: yes, the statement holds"]
    assert render_options("noul", {"true": "they ask"}) == ["false: no, the statement does not hold",
                                                            "true: they ask"]


def test_questions_come_first_and_every_marker_is_a_mask(tokenizer, good_request):
    questions = list(good_request["questions"].values())
    token_ids, segment_ids, markers = build_packed_sequence(tokenizer, good_request["state"], questions, 512)

    assert token_ids[0] == tokenizer.cls_token_id
    assert token_ids[-1] == tokenizer.sep_token_id
    assert len(token_ids) == len(segment_ids)
    assert [len(block) for block in markers] == [2, 3, 2]
    for question, block in zip(questions, markers):
        for position in block:
            assert token_ids[position] == tokenizer.mask_token_id
            assert segment_ids[position] == QUESTION_TYPES[question["type"]]
    state_start = token_ids.index(tokenizer.sep_token_id, markers[-1][-1]) + 1
    assert segment_ids[state_start - 1] == QUESTION_TYPES["noul"]
    assert all(segment == STATE_SEGMENT for segment in segment_ids[state_start:])


def test_a_mask_token_in_user_text_does_not_become_a_marker(tokenizer):
    questions = [{"type": "noul", "instructions": "is <mask> this urgent?"}]
    token_ids, _, markers = build_packed_sequence(tokenizer, "the <mask> ticket", questions, 512)
    assert token_ids.count(tokenizer.mask_token_id) == len(markers[0]) == 2


def test_long_state_is_cut_to_fit_and_flagged(tokenizer, good_request):
    long_state = "ticket " * 2000
    packed = pack_request(tokenizer, long_state, good_request["questions"], 256)
    assert len(packed.token_ids) == 256
    assert packed.state_truncated
    short = pack_request(tokenizer, "ticket", good_request["questions"], 256)
    assert not short.state_truncated


def test_long_options_are_flagged(tokenizer):
    questions = {"q": {"type": "choice", "instructions": "which?",
                       "criteria": {"a": "ticket " * (MAX_OPTION_TOKENS + 5), "b": "no"}}}
    assert pack_request(tokenizer, "s", questions, 1024).options_truncated


def test_questions_that_leave_no_room_for_the_state_are_refused(tokenizer):
    levels = ["ticket " * 40] * 20
    questions = [{"type": "score", "instructions": "how urgent?", "criteria": levels}]
    with pytest.raises(QuestionsTooLong):
        build_packed_sequence(tokenizer, "s", questions, 256)


def test_collate_pads_and_masks(tokenizer, good_request):
    first = pack_request(tokenizer, "ticket", good_request["questions"], 512)
    second = pack_request(tokenizer, "ticket " * 30, {"q": {"type": "noul", "instructions": "yes?"}}, 512)
    batch = collate([first, second], tokenizer.pad_token_id)
    assert batch["input_ids"].shape == (2, max(len(first.token_ids), len(second.token_ids)))
    assert batch["marker_positions"].shape == (2, 3, 3)
    assert batch["marker_mask"][1].sum().item() == 2
    assert batch["attention_mask"][0].sum().item() == len(first.token_ids)
