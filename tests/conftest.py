"""A tiny random-weight Tacet model folder, in the released layout, for fast tests."""

import json
import os

import pytest
import torch

WORDS = ("the customer was charged twice refund billing tech bugs payments which team should handle this "
         "how urgent is low medium high does ask for money back question choice score noul level yes no "
         "statement holds not false true ticket").split()


def build_tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    special = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"]
    vocabulary = {token: index for index, token in enumerate(special + sorted(set(WORDS)))}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", cls_token="<bos>",
                                   sep_token="<eos>", mask_token="<mask>", unk_token="<unk>")


def build_tiny_encoder_config(vocabulary_size):
    from transformers import ModernBertConfig

    return ModernBertConfig(vocab_size=vocabulary_size, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=4096,
                            pad_token_id=0, eos_token_id=1, bos_token_id=2, cls_token_id=2, sep_token_id=1)


def write_tiny_model(directory):
    from safetensors.torch import save_file
    from transformers import AutoModel

    from tacet.network import TacetNetwork

    torch.manual_seed(0)
    tokenizer = build_tiny_tokenizer()
    encoder_config = build_tiny_encoder_config(len(tokenizer))
    network = TacetNetwork(AutoModel.from_config(encoder_config), head_layers=2)
    tokenizer.save_pretrained(os.path.join(directory, "tokenizer"))
    encoder_config.save_pretrained(os.path.join(directory, "encoder"))
    state_dict = {name: tensor.contiguous() for name, tensor in network.state_dict().items()}
    save_file(state_dict, os.path.join(directory, "model.safetensors"))
    with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as handle:
        json.dump({"name": "tacet-tiny", "head_layers": 2}, handle)


@pytest.fixture(scope="session")
def tiny_model_directory(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tacet-tiny")
    write_tiny_model(str(directory))
    return str(directory)


@pytest.fixture(scope="session")
def tiny_model(tiny_model_directory):
    import tacet

    return tacet.load(tiny_model_directory, device="cpu")


@pytest.fixture(scope="session")
def tokenizer(tiny_model):
    return tiny_model.tokenizer


GOOD_REQUEST = {
    "state": {"ticket": "The customer was charged twice."},
    "questions": {
        "route": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "payments", "tech": "bugs"}},
        "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]},
        "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
    },
}


@pytest.fixture
def good_request():
    return json.loads(json.dumps(GOOD_REQUEST))
