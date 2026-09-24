"""Request validation and the wire format shared by `decide` and `tacet serve`.

The wire format is the one TypeSafe documents for Jev, which the hosted Tacet API also used:

    POST /v1/systemone
    {"state": <string | object | array>, "model": "tacet",
     "questions": {"<id>": {"type": "choice" | "score" | "noul",
                            "instructions": "...", "criteria": ...}}}

    -> {"model": "...", "answers": {"<id>": {...}},
        "usage": {"input_tokens": N, "output_tokens": 0}}
"""

import json
import time
import uuid

# Names every server answers to, besides the loaded model's own name. "tacet-1" and
# "tacet-latest" were the hosted API's names, so clients moving off it keep working.
MODEL_ALIASES = frozenset({"tacet", "tacet-1", "tacet-latest"})
# Routers such as OpenRouter send the model with the provider's prefix.
PROVIDER_PREFIX = "codepawl/"

# Every error carries a code a client can branch on; these are the defaults by status.
DEFAULT_ERROR_CODES = {400: "invalid_request", 401: "invalid_api_key", 404: "not_found",
                       405: "method_not_allowed", 413: "request_too_large", 500: "internal_error"}

MAX_QUESTIONS = 32
MAX_OPTIONS = 64
MAX_TEXT_CHARS = 4_000
MAX_STATE_CHARS = 200_000
QUESTION_TYPES = ("choice", "score", "noul")


class RequestError(ValueError):
    """A request the model cannot answer, with the HTTP status and code the server sends."""

    def __init__(self, status, message, error_type="invalid_request_error", code=None, param=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.error_type = error_type
        self.code = code or DEFAULT_ERROR_CODES.get(status, "error")
        self.param = param

    def body(self):
        return {"error": {"message": self.message, "type": self.error_type,
                          "code": self.code, "param": self.param}}


def render_text(value):
    """Text as sent, or anything else JSON can carry rendered as JSON text."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _check_text(value, where, limit=MAX_TEXT_CHARS):
    if not isinstance(value, str) or not value.strip():
        raise RequestError(400, f"{where} must be a non-empty string.", param=where)
    if len(value) > limit:
        raise RequestError(400, f"{where} is longer than {limit} characters.", param=where)


def _check_instructions(value, where):
    """Instructions are text, or an object or array that renders to text, as Jev accepts."""
    if isinstance(value, (dict, list)):
        if not value:
            raise RequestError(400, f"{where} must not be empty.", param=where)
        _check_text(render_text(value), where)
        return
    _check_text(value, where)


def _check_criterion(value, where):
    """A criterion is text, or anything JSON can carry that renders to text."""
    if len(render_text(value)) > MAX_TEXT_CHARS:
        raise RequestError(400, f"{where} is longer than {MAX_TEXT_CHARS} characters.", param=where)


def _check_choice_criteria(criteria, where):
    if not isinstance(criteria, dict) or len(criteria) < 2:
        raise RequestError(400, f"{where}.criteria must map at least two option names to descriptions.",
                           param=f"{where}.criteria")
    if len(criteria) > MAX_OPTIONS:
        raise RequestError(400, f"{where} has more than {MAX_OPTIONS} options.", param=f"{where}.criteria")
    for name, description in criteria.items():
        _check_text(name, f"{where}.criteria key", limit=200)
        if description is not None:
            _check_criterion(description, f"{where}.criteria.{name}")


def _check_score_criteria(criteria, where):
    if not isinstance(criteria, list) or len(criteria) < 2:
        raise RequestError(400, f"{where}.criteria must list at least two levels.", param=f"{where}.criteria")
    if len(criteria) > MAX_OPTIONS:
        raise RequestError(400, f"{where} has more than {MAX_OPTIONS} levels.", param=f"{where}.criteria")
    for index, level in enumerate(criteria):
        _check_criterion(level, f"{where}.criteria[{index}]")


def _check_noul_criteria(criteria, where):
    if criteria is None:
        return
    if not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}:
        raise RequestError(400, f"{where}.criteria for noul may only have 'true' and 'false'.",
                           param=f"{where}.criteria")
    for name, description in criteria.items():
        _check_criterion(description, f"{where}.criteria.{name}")


def validate_question(question_id, question):
    where = f"questions.{question_id}"
    if not isinstance(question, dict):
        raise RequestError(400, f"{where} must be an object.", param=where)
    question_type = question.get("type")
    if question_type not in QUESTION_TYPES:
        raise RequestError(400, f"{where}.type must be one of {', '.join(QUESTION_TYPES)}.",
                           param=f"{where}.type")
    _check_instructions(question.get("instructions"), f"{where}.instructions")
    criteria = question.get("criteria")
    if question_type == "choice":
        _check_choice_criteria(criteria, where)
    elif question_type == "score":
        _check_score_criteria(criteria, where)
    else:
        _check_noul_criteria(criteria, where)


def with_text_instructions(questions):
    """The questions with object or array instructions rendered to the text the model reads."""
    return {question_id: {**question, "instructions": render_text(question["instructions"])}
            for question_id, question in questions.items()}


def validate_request(state, questions):
    """Checks a state and its questions; returns the questions with text instructions."""
    if state is None:
        raise RequestError(400, "state is required.", param="state")
    if not isinstance(state, (str, dict, list)):
        raise RequestError(400, "state must be a string, an object or an array.", param="state")
    if len(render_text(state)) > MAX_STATE_CHARS:
        raise RequestError(413, f"state is longer than {MAX_STATE_CHARS} characters.", code="state_too_large",
                           param="state")
    if not isinstance(questions, dict) or not questions:
        raise RequestError(400, "questions must be an object with at least one question.", param="questions")
    if len(questions) > MAX_QUESTIONS:
        raise RequestError(400, f"At most {MAX_QUESTIONS} questions per request.", param="questions")
    for question_id, question in questions.items():
        _check_text(question_id, "question id", limit=200)
        validate_question(question_id, question)
    return with_text_instructions(questions)


def canonical_model(model):
    """The model name without a router's provider prefix."""
    if isinstance(model, str) and model.startswith(PROVIDER_PREFIX):
        return model[len(PROVIDER_PREFIX):]
    return model


def validate_systemone(body, served_model):
    """Checks a `/v1/systemone` body and returns (state, questions).

    A body without `model` is answered by the served model; any other name must be the served
    model's own name or one of MODEL_ALIASES.
    """
    if not isinstance(body, dict):
        raise RequestError(400, "The request body must be a JSON object.")
    model = canonical_model(body.get("model", served_model))
    if model != served_model and model not in MODEL_ALIASES:
        raise RequestError(404, f"Model '{model}' does not exist. Use '{served_model}'.", code="model_not_found",
                           param="model")
    state = body.get("state")
    questions = validate_request(state, body.get("questions"))
    return state, questions


def _last_user_text(messages):
    user_messages = [message for message in messages if isinstance(message, dict) and message.get("role") == "user"]
    if not user_messages:
        raise RequestError(400, "Put the request in a user message.", param="messages")
    content = user_messages[-1].get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content
                          if isinstance(part, dict) and part.get("type") == "text")
    if not isinstance(content, str):
        raise RequestError(400, "The last user message must be text.", param="messages")
    return content


def systemone_from_chat(body, served_model):
    """Reads a `/v1/chat/completions` body whose last user message is a systemone request.

    Tacet does not generate text, so the chat surface is an adapter: the caller puts
    `{"state": ..., "questions": ...}` as JSON in the last user message and gets the
    answers back as a JSON string in the assistant message.
    """
    if not isinstance(body, dict):
        raise RequestError(400, "The request body must be a JSON object.")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError(400, "messages must be a non-empty array.", param="messages")
    content = _last_user_text(messages)
    try:
        request = json.loads(content)
    except json.JSONDecodeError:
        raise RequestError(400, 'The last user message must be JSON: {"state": ..., "questions": {...}}.',
                           param="messages")
    if not isinstance(request, dict):
        raise RequestError(400, 'The last user message must be a JSON object with "state" and "questions".',
                           param="messages")
    request = dict(request)
    # The chat body's model wins: it is the one a router or SDK actually set.
    request["model"] = body.get("model", request.get("model", served_model))
    return validate_systemone(request, served_model)


def _completion_chunk(completion_id, created, model, delta, finish_reason, usage_block=None):
    chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    if usage_block is not None:
        chunk["usage"] = usage_block
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def chat_completion(response, stream=False):
    """An OpenAI-style chat completion carrying a `decide` response's answers as JSON text."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = response["model"]
    usage = response["usage"]
    reply = {"answers": response["answers"]}
    # Chat clients only see the message text, so what /v1/systemone reports in usage goes in there too.
    for flag in ("state_truncated", "options_truncated"):
        if usage.get(flag):
            reply[flag] = True
    content = json.dumps(reply, ensure_ascii=False, separators=(",", ":"))
    usage_block = {"prompt_tokens": usage["input_tokens"], "completion_tokens": 0,
                   "total_tokens": usage["input_tokens"]}
    if not stream:
        return {
            "id": completion_id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": usage_block,
        }
    first = _completion_chunk(completion_id, created, model, {"role": "assistant", "content": content}, None)
    last = _completion_chunk(completion_id, created, model, {}, "stop", usage_block)
    return first + last + "data: [DONE]\n\n"


def models_listing(served_model, context_length, created):
    """`GET /v1/models`: the hosted API's flat fields, without pricing or datacenter details."""
    return {
        "object": "list",
        "data": [{
            "id": served_model,
            "object": "model",
            "created": created,
            "owned_by": "codepawl",
            "name": served_model,
            "description": ("Typed decisions from one encoder pass. Send a state and typed questions (choice, "
                            "score, noul) as JSON; get a probability for every option back as JSON. Writes no "
                            "text, so completion tokens are always 0."),
            "context_length": context_length,
            "max_output_tokens": 0,
            "features": ["typed_decisions", "choice", "score", "noul", "calibrated_probabilities"],
        }],
    }


def bearer_key(authorization_header):
    """The key from an Authorization header; the scheme name is case-insensitive, as HTTP says."""
    scheme, _, key = (authorization_header or "").strip().partition(" ")
    if scheme.lower() != "bearer":
        raise RequestError(401, "Send your API key as 'Authorization: Bearer <key>'.", "authentication_error",
                           code="missing_api_key")
    key = key.strip()
    if not key:
        raise RequestError(401, "The API key is empty.", "authentication_error", code="missing_api_key")
    return key
