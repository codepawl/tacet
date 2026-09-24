"""Loading a Tacet model and answering requests with it."""

import json
import os

import torch

from .decoding import decode_answers, usage_of
from .network import TacetNetwork
from .packing import QuestionsTooLong, collate, pack_request
from .validation import RequestError, validate_request

DEFAULT_MAX_LENGTH = 1536
# mmBERT reads up to 8192 positions, but Tacet was trained and measured up to 4096.
LONGEST_MAX_LENGTH = 4096
SHORTEST_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 8
# A released model folder has config.json; research checkpoints have rl_agent_config.json.
CONFIG_FILE_NAMES = ("config.json", "rl_agent_config.json")
MODEL_FILES = ["config.json", "model.safetensors", "encoder/*", "tokenizer/*"]


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def model_directory(name_or_path, revision=None):
    """A local folder as is, or a Hugging Face repo id downloaded to the local cache."""
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=name_or_path, revision=revision, allow_patterns=MODEL_FILES)


def read_config(directory):
    for file_name in CONFIG_FILE_NAMES:
        path = os.path.join(directory, file_name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
    raise FileNotFoundError(f"{directory} has no config.json; is it a Tacet model folder?")


def build_network(directory, config):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    encoder_config = AutoConfig.from_pretrained(os.path.join(directory, "encoder"))
    encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa")
    network = TacetNetwork(encoder, head_layers=config.get("head_layers", 2))
    network.load_state_dict(load_file(os.path.join(directory, "model.safetensors")), strict=True)
    return network


def check_max_length(max_length):
    if not SHORTEST_MAX_LENGTH <= max_length <= LONGEST_MAX_LENGTH:
        raise ValueError(f"max_length must be between {SHORTEST_MAX_LENGTH} and {LONGEST_MAX_LENGTH}, "
                         f"got {max_length}")


def load(name_or_path="codepawl/tacet-small", device="auto", max_length=DEFAULT_MAX_LENGTH, revision=None):
    """Loads a Tacet model from a Hugging Face repo id or a local folder.

    `device` is "auto" (CUDA when available), "cpu", "cuda" or "cuda:N". On a GPU the model
    runs in bfloat16 autocast. `max_length` caps the packed sequence in tokens; the state is
    cut to fit, and the answer's usage says so.
    """
    from transformers import AutoTokenizer

    check_max_length(max_length)
    directory = model_directory(name_or_path, revision)
    config = read_config(directory)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(directory, "tokenizer"))
    network = build_network(directory, config)
    name = config.get("name") or os.path.basename(os.path.normpath(name_or_path))
    return TacetModel(network, tokenizer, name, resolve_device(device), max_length)


class TacetModel:
    """A loaded model. `decide` answers one request, `decide_batch` many."""

    def __init__(self, network, tokenizer, name, device, max_length=DEFAULT_MAX_LENGTH):
        self.network = network.to(device).eval()
        self.tokenizer = tokenizer
        self.name = name
        self.device = device
        self.max_length = max_length

    def decide(self, state, questions):
        """Answers `questions` about `state` in one encoder pass.

        Returns {"model", "answers": {id: answer}, "usage"}, the shape the hosted API returned.
        Raises RequestError when the request is invalid or its questions do not fit.
        """
        return self.run([self.prepare(state, questions)])[0]

    def decide_batch(self, requests, batch_size=DEFAULT_BATCH_SIZE):
        """Answers a list of {"state", "questions"} requests, `batch_size` per forward pass."""
        packed_requests = [self.prepare(request["state"], request["questions"]) for request in requests]
        responses = []
        # ponytail: batches keep the given order; sorting by length would cut padding if throughput matters.
        for start in range(0, len(packed_requests), batch_size):
            responses.extend(self.run(packed_requests[start:start + batch_size]))
        return responses

    def prepare(self, state, questions):
        """Validates and tokenizes one request. Raises RequestError."""
        text_questions = validate_request(state, questions)
        try:
            return pack_request(self.tokenizer, state, text_questions, self.max_length)
        except QuestionsTooLong as error:
            raise RequestError(400, str(error), code="questions_too_long", param="questions") from error

    @torch.no_grad()
    def run(self, packed_requests):
        """One forward pass over prepared requests; one response per request, in order."""
        batch = collate(packed_requests, self.tokenizer.pad_token_id)
        inputs = {key: tensor.to(self.device) for key, tensor in batch.items()}
        on_gpu = self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=on_gpu):
            logits = self.network(**inputs)
        logits = logits.float().cpu().numpy()
        responses = []
        for row, packed_request in enumerate(packed_requests):
            responses.append({"model": self.name,
                              "answers": decode_answers(logits[row], packed_request),
                              "usage": usage_of(packed_request)})
        return responses
