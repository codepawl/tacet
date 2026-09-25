import importlib.util
import json
import os
import sys

from conftest import write_tiny_model

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "prepare_hf_repo.py")


def load_script():
    specification = importlib.util.spec_from_file_location("prepare_hf_repo", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def write_research_checkpoint(directory):
    """The tiny model in the training layout: rl_agent_config.json with training details."""
    write_tiny_model(directory)
    os.remove(os.path.join(directory, "config.json"))
    training_config = {"encoder": "jhu-clsp/mmBERT-small", "head_layers": 2, "max_len": 4096,
                       "fine_tuned_from": "/work/checkpoints/somewhere", "training_history": [{"epoch": 0}]}
    with open(os.path.join(directory, "rl_agent_config.json"), "w", encoding="utf-8") as handle:
        json.dump(training_config, handle)


def test_assembles_a_loadable_folder_without_training_details(tmp_path, monkeypatch):
    checkpoint = str(tmp_path / "checkpoint")
    out = str(tmp_path / "release")
    os.makedirs(checkpoint)
    write_research_checkpoint(checkpoint)
    monkeypatch.setattr(sys, "argv", ["prepare_hf_repo.py", "--checkpoint", checkpoint, "--name", "tacet-sonata",
                                      "--out", out])
    load_script().main()

    assert sorted(os.listdir(out)) == ["LICENSE", "NOTICE", "README.md", "config.json", "encoder",
                                       "model.safetensors", "tokenizer"]
    with open(os.path.join(out, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    assert config == {"tacet_format": 1, "name": "tacet-sonata", "backbone": "jhu-clsp/mmBERT-small",
                      "head_layers": 2, "trained_max_length": 4096}
    with open(os.path.join(out, "README.md"), encoding="utf-8") as handle:
        card = handle.read()
    assert "{{" not in card
    assert "codepawl/tacet-sonata" in card
    assert "base_model: jhu-clsp/mmBERT-small" in card
