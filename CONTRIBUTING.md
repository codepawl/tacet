# Contributing

Issues and pull requests are welcome. For anything larger than a fix, open an issue first so we
can agree on the shape before you write it.

## Setup

```bash
uv sync --extra benchmark
uv run pytest
```

The fast tests build a tiny random model and need no download. The tests marked `slow` load a
real model folder; point `TACET_TEST_CHECKPOINT` at one to run them:

```bash
TACET_TEST_CHECKPOINT=path/to/tacet-sonata uv run pytest -m slow
```

## Code style

- Descriptive full names. No one letter or abbreviated names outside tiny loop indices.
- One statement per line and small functions.
- Comments only where the reason is not obvious from the code.
- Keep the wire format and error codes compatible with the hosted API unless a change is agreed
  in an issue: clients depend on them.
- The prompt rendering in `src/tacet/packing.py` is what the released weights were trained on.
  Changing it changes the model's answers.

## Pull requests

- Add or update a test for any change in behaviour.
- Run `uv run pytest` before you push.
- Never commit weights, `.env` files, keys or tokens.

By contributing you agree that your contribution is licensed under the Apache License 2.0.
