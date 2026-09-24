"""`tacet serve`: the hosted Tacet API's routes, answered by a local model.

Routes:
  POST /v1/systemone          the typed-decision call, Jev wire format
  POST /v1/chat/completions   OpenAI-compatible adapter (JSON request in the last user message)
  GET  /v1/models             model listing
  GET  /v1/health, /health

Requests that arrive within a few milliseconds of each other are answered in one forward
pass, with the model call in a worker thread so the event loop keeps accepting requests.
Request contents are never logged.
"""

import asyncio
import hmac
import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .validation import (RequestError, bearer_key, chat_completion, models_listing, systemone_from_chat,
                         validate_systemone)

MAX_BATCH = 8
BATCH_WAIT_SECONDS = 0.004
# A request body above this is refused before it is parsed. The largest valid state is
# 200,000 characters, and questions that fit the sequence are far smaller.
MAX_BODY_BYTES = 2_000_000


class Batcher:
    """Gathers requests for a few milliseconds and answers them in one forward pass."""

    def __init__(self, model, max_batch=MAX_BATCH, wait_seconds=BATCH_WAIT_SECONDS):
        self.model = model
        self.max_batch = max_batch
        self.wait_seconds = wait_seconds
        self.queue = None
        self.worker = None

    async def answer(self, packed_request):
        if self.worker is None:
            self.queue = asyncio.Queue()
            self.worker = asyncio.create_task(self.run_forever())
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((packed_request, future))
        return await future

    async def gather_batch(self):
        pending = [await self.queue.get()]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.wait_seconds
        while len(pending) < self.max_batch:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                pending.append(await asyncio.wait_for(self.queue.get(), remaining))
            except asyncio.TimeoutError:
                break
        return pending

    async def run_forever(self):
        while True:
            pending = await self.gather_batch()
            packed_requests = [packed_request for packed_request, _ in pending]
            try:
                responses = await asyncio.to_thread(self.model.run, packed_requests)
            except Exception as error:
                for _, future in pending:
                    if not future.done():
                        future.set_exception(error)
                continue
            for (_, future), response in zip(pending, responses):
                if not future.done():
                    future.set_result(response)


async def read_json_body(request):
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise RequestError(413, f"The request body is larger than {MAX_BODY_BYTES:,} bytes.",
                           code="request_too_large")
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise RequestError(413, f"The request body is larger than {MAX_BODY_BYTES:,} bytes.",
                           code="request_too_large")
    try:
        return json.loads(raw)
    except ValueError:
        raise RequestError(400, "The request body must be valid JSON.")


def check_api_key(request, api_key):
    """Nothing to check when the server runs without a key."""
    if api_key is None:
        return
    sent = bearer_key(request.headers.get("authorization"))
    if not hmac.compare_digest(sent.encode(), api_key.encode()):
        raise RequestError(401, "This API key is not valid.", "authentication_error", code="invalid_api_key")


def error_response(error, request_id):
    return JSONResponse(error.body(), status_code=error.status, headers={"X-Request-Id": request_id})


def create_app(model, api_key=None, max_batch=MAX_BATCH):
    """The HTTP app for a loaded TacetModel. With `api_key`, the /v1 POST routes need
    `Authorization: Bearer <api_key>`."""
    batcher = Batcher(model, max_batch)
    started_at = int(time.time())
    app = FastAPI(title="Tacet", docs_url=None, redoc_url=None, openapi_url=None)

    # Every response carries an X-Request-Id, and every error, including an unknown route or
    # a crash, comes back in the same envelope as the API's own errors.
    @app.middleware("http")
    async def request_id_and_errors(request, call_next):
        request_id = f"req_{uuid.uuid4().hex[:20]}"
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception as error:
            # Only the error type: its message may quote the request.
            print(f"{request_id} unhandled {type(error).__name__}")
            failure = RequestError(500, "Something went wrong in the server. See its log for the request id.",
                                   "api_error")
            response = JSONResponse(failure.body(), status_code=500)
        response.headers.setdefault("X-Request-Id", request_id)
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_errors(request, error):
        messages = {404: "No such route. The API lives under /v1/.", 405: "This route does not take that method."}
        message = messages.get(error.status_code, str(error.detail))
        return JSONResponse(RequestError(error.status_code, message).body(), status_code=error.status_code)

    async def answer(request, parse):
        """The parsed body and the model's response. Raises RequestError."""
        check_api_key(request, api_key)
        body = await read_json_body(request)
        state, questions = parse(body, model.name)
        packed_request = await asyncio.to_thread(model.prepare, state, questions)
        return body, await batcher.answer(packed_request)

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        request_id = request.state.request_id
        try:
            _, response = await answer(request, validate_systemone)
        except RequestError as error:
            return error_response(error, request_id)
        return JSONResponse(response, headers={"X-Request-Id": request_id})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        request_id = request.state.request_id
        try:
            body, response = await answer(request, systemone_from_chat)
        except RequestError as error:
            return error_response(error, request_id)
        headers = {"X-Request-Id": request_id}
        if body.get("stream"):
            return Response(chat_completion(response, stream=True), media_type="text/event-stream", headers=headers)
        return JSONResponse(chat_completion(response), headers=headers)

    @app.get("/v1/models")
    async def models():
        return models_listing(model.name, model.max_length, started_at)

    @app.get("/health")
    @app.get("/v1/health")
    async def health():
        return {"ok": True, "model": model.name}

    return app
