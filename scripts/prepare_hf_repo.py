"""Assembles a Hugging Face model folder from a Tacet checkpoint folder, ready to upload.

The output holds what `tacet.load` reads and what the licences require, nothing else:

  config.json          name, backbone, head layers and the length the model was trained at
  model.safetensors    the weights, copied unchanged
  encoder/config.json  the mmBERT encoder configuration
  tokenizer/           the tokenizer files
  README.md            the model card, filled from model_card/README.md
  LICENSE, NOTICE      Apache 2.0 and the attributions (mmBERT's MIT notice travels with the weights)

The checkpoint's own training config is not copied: it carries training history and local paths.
After assembling, the folder is loaded with `tacet.load` and answers one request, as a check.

Usage:
    python scripts/prepare_hf_repo.py --checkpoint path/to/tacet-v3-small --name tacet-sonata --out hf-release/tacet-sonata
"""

import argparse
import json
import os
import shutil

from safetensors import safe_open

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_CARD_TEMPLATE = os.path.join(REPOSITORY_ROOT, "model_card", "README.md")
HUB_ORGANIZATION = "codepawl"
CONFIG_FORMAT_VERSION = 1


def read_checkpoint_config(checkpoint):
    for file_name in ("config.json", "rl_agent_config.json"):
        path = os.path.join(checkpoint, file_name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
    raise SystemExit(f"{checkpoint} has no rl_agent_config.json or config.json")


def release_config(checkpoint_config, name):
    backbone = checkpoint_config.get("backbone") or checkpoint_config.get("built_from") or checkpoint_config["encoder"]
    return {
        "tacet_format": CONFIG_FORMAT_VERSION,
        "name": name,
        "backbone": backbone,
        "head_layers": checkpoint_config.get("head_layers", 2),
        "trained_max_length": checkpoint_config.get("trained_max_length") or checkpoint_config.get("max_len"),
    }


def count_parameters(weights_path):
    total = 0
    with safe_open(weights_path, "pt") as weights:
        for key in weights.keys():
            count = 1
            for dimension in weights.get_slice(key).get_shape():
                count *= dimension
            total += count
    return total


def readable_count(parameters):
    return f"{round(parameters / 1e6)}M"


def display_name(name):
    """tacet-sonata -> Tacet Sonata."""
    return " ".join(part.capitalize() for part in name.split("-"))


def fill_model_card(config, parameters):
    with open(MODEL_CARD_TEMPLATE, encoding="utf-8") as handle:
        card = handle.read()
    values = {"name": config["name"], "display_name": display_name(config["name"]),
              "repo_id": f"{HUB_ORGANIZATION}/{config['name']}",
              "backbone": config["backbone"], "parameters": readable_count(parameters)}
    for key, value in values.items():
        card = card.replace("{{" + key + "}}", value)
    return card


def copy_model_files(checkpoint, out):
    shutil.copy2(os.path.join(checkpoint, "model.safetensors"), os.path.join(out, "model.safetensors"))
    os.makedirs(os.path.join(out, "encoder"))
    shutil.copy2(os.path.join(checkpoint, "encoder", "config.json"), os.path.join(out, "encoder", "config.json"))
    shutil.copytree(os.path.join(checkpoint, "tokenizer"), os.path.join(out, "tokenizer"))
    for file_name in ("LICENSE", "NOTICE"):
        shutil.copy2(os.path.join(REPOSITORY_ROOT, file_name), os.path.join(out, file_name))


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def check_folder(out):
    import tacet

    model = tacet.load(out, device="cpu")
    response = model.decide("The customer asks for a refund of a double charge.",
                            {"refund": tacet.noul("Does the customer ask for money back?")})
    print(f"  check: loaded {model.name}, noul {response['answers']['refund']['noul']}")


def main():
    parser = argparse.ArgumentParser(description="Assemble a Hugging Face model folder from a Tacet checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="folder with model.safetensors, encoder/, tokenizer/")
    parser.add_argument("--name", required=True, help="model name, such as tacet-sonata or tacet-symphony")
    parser.add_argument("--out", required=True, help="output folder; must not exist yet")
    parser.add_argument("--skip-check", action="store_true", help="do not load the folder after assembling it")
    arguments = parser.parse_args()

    if os.path.exists(arguments.out):
        raise SystemExit(f"{arguments.out} already exists; pick a new folder")
    config = release_config(read_checkpoint_config(arguments.checkpoint), arguments.name)
    os.makedirs(arguments.out)
    copy_model_files(arguments.checkpoint, arguments.out)
    write_text(os.path.join(arguments.out, "config.json"), json.dumps(config, indent=2) + "\n")
    parameters = count_parameters(os.path.join(arguments.out, "model.safetensors"))
    write_text(os.path.join(arguments.out, "README.md"), fill_model_card(config, parameters))
    print(f"assembled {arguments.out}: {config['name']}, {readable_count(parameters)} parameters, "
          f"backbone {config['backbone']}")
    if not arguments.skip_check:
        check_folder(arguments.out)


if __name__ == "__main__":
    main()
