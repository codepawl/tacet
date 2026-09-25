"""Scores a Tacet model on the LocalLLaMA/typed-decisions test split.

400 cases, 2,000 decisions. Accuracy, soft accuracy, Brier and ECE use the definitions from
Laya's typed-decisions evaluation notebook (https://huggingface.co/convaiinnovations/laya,
Apache 2.0, see NOTICE), including the 1e-6 probability floor and the 15 bin ECE, so the
numbers are comparable with the ones Laya and the dataset's leaderboard publish.

Optionally, every choice question is asked again with its options listed in other random
orders (`--orderings N`); the flip rate says how often the answer depends on the listing order.

Usage, from a clone of the repository:
    uv sync --extra benchmark
    uv run python scripts/benchmark.py --model codepawl/tacet-sonata
    uv run python scripts/benchmark.py --model path/to/folder --device cuda --orderings 5 --out result.json
"""

import argparse
import collections
import json
import os
import random
import time

import numpy as np

import tacet

DATASET = "LocalLLaMA/typed-decisions"
# The dataset commit the published numbers were measured on.
DATASET_REVISION = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
PROBABILITY_FLOOR = 1e-6
ECE_BINS = 15


def load_cases(split, limit=None):
    import pyarrow.parquet
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, f"all/{split}-00000-of-00001.parquet", repo_type="dataset",
                           revision=DATASET_REVISION)
    cases = []
    for row in pyarrow.parquet.read_table(path).to_pylist():
        cases.append({
            "id": row["id"],
            "workflow": row["workflow"],
            "state": json.loads(row["state"]),
            "questions": json.loads(row["questions"]),
            "gold": json.loads(row["gold"]),
        })
    if limit:
        return cases[:limit]
    return cases


def normalised(values):
    array = np.asarray(values, dtype=np.float64)
    total = array.sum()
    if total > 0:
        return array / total
    return np.full(len(array), 1.0 / len(array))


def compare(predicted, target, correct):
    return {"correct": float(correct), "confidence": float(predicted.max()),
            "soft": float((predicted * target).sum()),
            "brier": float(((predicted - target) ** 2).sum()), "score_error": None}


def score_choice(question, answer, gold):
    names = list(question["criteria"].keys())
    predicted = normalised([answer["probabilities"].get(name, PROBABILITY_FLOOR) for name in names])
    target = normalised([gold["probabilities"].get(name, PROBABILITY_FLOOR) for name in names])
    return compare(predicted, target, answer["choice"] == str(gold["label"]))


def score_noul(answer, gold):
    probability_true = float(answer["noul"])
    gold_true = float(gold.get("noul", gold.get("probabilities", {}).get("true", 0.5)))
    predicted = np.array([1.0 - probability_true, probability_true])
    target = np.array([1.0 - gold_true, gold_true])
    predicted_label = "true" if probability_true >= 0.5 else "false"
    return compare(predicted, target, predicted_label == str(gold["label"]).lower())


def score_score(question, answer, gold):
    level_count = len(question.get("criteria", []))
    predicted = normalised([answer["probabilities"].get(str(level), 0.0) for level in range(level_count)])
    target = normalised([gold["probabilities"].get(str(level), 0.0) for level in range(level_count)])
    gold_level = int(gold.get("label", int(round(gold.get("score", 0.0)))))
    result = compare(predicted, target, int(np.argmax(predicted)) == gold_level)
    result["score_error"] = abs(float(answer["score"]) - float(gold.get("score", 0.0)))
    return result


def score_decision(question, answer, gold):
    """Correctness, the confidence used for ECE, soft accuracy (probability mass on the
    annotators' distribution), Brier against that distribution, and for score questions the
    absolute error of the expected score."""
    if question["type"] == "choice":
        return score_choice(question, answer, gold)
    if question["type"] == "noul":
        return score_noul(answer, gold)
    return score_score(question, answer, gold)


def expected_calibration_error(confidences, correct, bins=ECE_BINS):
    if len(confidences) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > low) & (confidences <= high)
        if in_bin.any():
            error += in_bin.mean() * abs(confidences[in_bin].mean() - correct[in_bin].mean())
    return float(error)


def answer_all(model, cases, questions_per_case, batch_size):
    requests = [{"state": case["state"], "questions": questions}
                for case, questions in zip(cases, questions_per_case)]
    return model.decide_batch(requests, batch_size=batch_size)


def reorder_choice_options(questions, generator):
    """The same questions with every choice question's options listed in a new order. noul has
    a fixed false/true layout and score levels are ordinal, so neither is reordered."""
    reordered = {}
    for question_id, question in questions.items():
        if question["type"] != "choice":
            reordered[question_id] = question
            continue
        names = list(question["criteria"].keys())
        generator.shuffle(names)
        reordered[question_id] = {**question, "criteria": {name: question["criteria"][name] for name in names}}
    return reordered


def order_robustness(model, cases, orderings, batch_size, first_responses):
    """How often a choice answer changes when only the option order changes."""
    choice_keys = [(case_index, question_id) for case_index, case in enumerate(cases)
                   for question_id, question in case["questions"].items() if question["type"] == "choice"]
    chosen_by_ordering = [[first_responses[case_index]["answers"][question_id]["choice"]
                           for case_index, question_id in choice_keys]]
    for ordering in range(1, orderings + 1):
        reordered = [reorder_choice_options(case["questions"], random.Random(ordering * 100_003 + case_index))
                     for case_index, case in enumerate(cases)]
        responses = answer_all(model, cases, reordered, batch_size)
        chosen_by_ordering.append([responses[case_index]["answers"][question_id]["choice"]
                                   for case_index, question_id in choice_keys])
    gold = [str(cases[case_index]["gold"][question_id]["label"]) for case_index, question_id in choice_keys]
    flipped = sum(len({chosen[index] for chosen in chosen_by_ordering}) > 1 for index in range(len(choice_keys)))
    accuracy_by_ordering = [float(np.mean([pick == label for pick, label in zip(chosen, gold)]))
                            for chosen in chosen_by_ordering]
    return {"choice_decisions": len(choice_keys), "orderings": orderings + 1,
            "flip_rate": flipped / max(len(choice_keys), 1),
            "accuracy_by_ordering": accuracy_by_ordering,
            "accuracy_spread": max(accuracy_by_ordering) - min(accuracy_by_ordering)}


def score_responses(cases, responses):
    decisions = []
    for case, response in zip(cases, responses):
        for question_id, question in case["questions"].items():
            scored = score_decision(question, response["answers"][question_id], case["gold"][question_id])
            scored.update({"case": case["id"], "question": question_id, "type": question["type"],
                           "workflow": case["workflow"]})
            decisions.append(scored)
    return decisions


def accuracy_by(decisions, field):
    groups = collections.defaultdict(list)
    for decision in decisions:
        groups[decision[field]].append(decision["correct"])
    return {name: {"accuracy": float(np.mean(values)), "n": len(values)} for name, values in sorted(groups.items())}


def summarize(decisions):
    correct = np.array([decision["correct"] for decision in decisions])
    confidences = np.array([decision["confidence"] for decision in decisions])
    score_errors = [decision["score_error"] for decision in decisions if decision["score_error"] is not None]
    return {
        "decisions": len(decisions),
        "accuracy": float(correct.mean()),
        "soft_accuracy": float(np.mean([decision["soft"] for decision in decisions])),
        "brier": float(np.mean([decision["brier"] for decision in decisions])),
        "ece": expected_calibration_error(confidences, correct),
        "score_mae": float(np.mean(score_errors)) if score_errors else None,
        "by_type": accuracy_by(decisions, "type"),
        "by_workflow": accuracy_by(decisions, "workflow"),
    }


def print_result(result):
    print(f"  accuracy        {result['accuracy']:.4f}   ({result['decisions']:,} decisions)")
    print(f"  soft accuracy   {result['soft_accuracy']:.4f}")
    print(f"  Brier           {result['brier']:.4f}")
    print(f"  ECE (15 bins)   {result['ece']:.4f}")
    if result["score_mae"] is not None:
        print(f"  score MAE       {result['score_mae']:.4f}")
    for name, entry in {**result["by_type"], **result["by_workflow"]}.items():
        print(f"    {name:<28} {entry['accuracy']:.4f}  (n={entry['n']})")
    if "order_robustness" in result:
        robustness = result["order_robustness"]
        print(f"  choice flip rate over {robustness['orderings']} orderings  {robustness['flip_rate']:.4f}"
              f"   accuracy spread {robustness['accuracy_spread']:.4f}")
    print(f"  mean input tokens per case  {result['mean_input_tokens_per_case']:.0f}")
    print(f"  seconds per case            {result['seconds_per_case']:.3f}")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="codepawl/tacet-sonata", help="Hub repo id or local folder")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N cases")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="requests per forward pass (1 matches how the published numbers were measured)")
    parser.add_argument("--orderings", type=int, default=0, help="extra random option orderings per choice question")
    parser.add_argument("--out", default=None, help="write the full result as JSON here")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    cases = load_cases(arguments.split, arguments.limit)
    model = tacet.load(arguments.model, device=arguments.device, max_length=arguments.max_length)
    parameters = sum(parameter.numel() for parameter in model.network.parameters())
    print(f"{model.name}  {parameters / 1e6:.1f}M parameters  on {model.device}  {len(cases)} cases\n")

    started = time.perf_counter()
    responses = answer_all(model, cases, [case["questions"] for case in cases], arguments.batch_size)
    elapsed = time.perf_counter() - started

    result = {"model": arguments.model, "name": model.name, "parameters": parameters, "device": str(model.device),
              "max_length": model.max_length, "split": arguments.split, "cases": len(cases),
              "dataset_revision": DATASET_REVISION}
    decisions = score_responses(cases, responses)
    result.update(summarize(decisions))
    if arguments.orderings:
        result["order_robustness"] = order_robustness(model, cases, arguments.orderings, arguments.batch_size,
                                                      responses)
    result["mean_input_tokens_per_case"] = float(np.mean([response["usage"]["input_tokens"]
                                                          for response in responses]))
    result["seconds_per_case"] = elapsed / len(cases)
    result["per_decision"] = [{"case": decision["case"], "question": decision["question"],
                               "correct": decision["correct"], "brier": decision["brier"]}
                              for decision in decisions]
    print_result(result)

    if arguments.out:
        out_path = os.path.abspath(arguments.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
        print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
