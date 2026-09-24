"""The `tacet` command line: `tacet serve` runs the HTTP API on a local model."""

import argparse
import os

from .model import DEFAULT_MAX_LENGTH, load

API_KEY_ENVIRONMENT_VARIABLE = "TACET_SERVE_API_KEY"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def build_parser():
    parser = argparse.ArgumentParser(prog="tacet")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="serve the Tacet HTTP API on a local model")
    serve.add_argument("--model", default="codepawl/tacet-small",
                       help="Hugging Face repo id or local folder (default: codepawl/tacet-small)")
    serve.add_argument("--revision", default=None, help="Hub revision (branch, tag or commit)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    serve.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                       help=f"packed sequence length in tokens, up to 4096 (default {DEFAULT_MAX_LENGTH})")
    serve.add_argument("--api-key", default=None,
                       help=f"require 'Authorization: Bearer <key>' (or set {API_KEY_ENVIRONMENT_VARIABLE})")
    return parser


def serve(arguments):
    import uvicorn

    from .server import create_app

    api_key = arguments.api_key or os.environ.get(API_KEY_ENVIRONMENT_VARIABLE) or None
    if api_key is None and arguments.host not in LOOPBACK_HOSTS:
        print(f"warning: serving on {arguments.host} without an API key; anyone who can reach it can use it")
    model = load(arguments.model, device=arguments.device, max_length=arguments.max_length,
                 revision=arguments.revision)
    print(f"serving {model.name} on {model.device}, max length {model.max_length}, "
          f"at http://{arguments.host}:{arguments.port}/v1")
    uvicorn.run(create_app(model, api_key=api_key), host=arguments.host, port=arguments.port)


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    if arguments.command == "serve":
        serve(arguments)


if __name__ == "__main__":
    main()
