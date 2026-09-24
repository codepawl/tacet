---
license: apache-2.0
base_model: {{backbone}}
language:
- multilingual
pipeline_tag: text-classification
tags:
- tacet
- typed-decisions
- calibration
- mmbert
datasets:
- LocalLLaMA/typed-decisions
- AmazonScience/massive
- PolyAI/banking77
- bitext/Bitext-customer-support-llm-chatbot-training-dataset
- google/civil_comments
---

# {{name}}

{{name}} answers typed questions about a piece of state and gives a probability for every
option. It is an encoder, not a text generator: all questions and the state are packed into one
sequence and answered in one forward pass. {{parameters}} parameters, fine tuned from
[{{backbone}}](https://huggingface.co/{{backbone}}).

- **choice**: pick one of N named options, with a probability per option and a confidence.
- **score**: place the state on an ordered rubric, with a distribution over the levels and the expected score.
- **noul**: yes or no, with the probability of yes.

## Use

```bash
pip install tacet
```

```python
import tacet
from tacet import choice, score, noul

model = tacet.load("{{repo_id}}")
result = model.decide(
    state={"ticket": "I was charged twice for my March invoice. Please fix this today."},
    questions={
        "route": choice("Which team should handle this?",
                        {"billing": "payments, invoices, refunds", "tech": "bugs and outages"}),
        "urgency": score("How urgent is this?", ["low", "medium", "high"]),
        "refund": noul("Does the customer ask for money back?"),
    },
)
print(result["answers"]["route"]["probabilities"])
```

Or serve the Tacet HTTP API (`/v1/systemone`, `/v1/chat/completions`) on localhost:

```bash
tacet serve --model {{repo_id}}
```

The code, the wire format and the error codes are documented at
[github.com/codepawl/tacet](https://github.com/codepawl/tacet).
TODO(release): confirm the GitHub URL before upload.

The weights are not a `transformers` model on their own: the head and the packed input format
live in the `tacet` package.

## Evaluation

LocalLLaMA/typed-decisions test split, 400 cases, 2,000 decisions, scored with
`scripts/benchmark.py` in the tacet repository:

| accuracy | Brier | ECE (15 bins) |
|---|---|---|
| to be filled from the release report | | |

The train split of this benchmark is part of the training data, so these are in distribution
numbers for its four workflows.

## Limits

- Strong on the benchmark's four workflows (customer service, invoice processing, security
  incidents, agent trace observability) and on routing and triage questions like them.
- Weaker on long free text, on questions that need date arithmetic, and on counting. If a
  decision depends on a number, compute it in code and put it in the state.
- The probabilities are calibrated on the training distribution. Check them against your own
  labelled examples before you gate anything on a confidence threshold.
- Reads at most 1536 tokens by default and up to 4096 with `max_length`. A longer state is cut and
  the response says so (`usage.state_truncated`).
- Does not generate text or explain its answers.

## Training data

TODO(release): reconcile with the final dataset list in the phase 3 release report.

| data | licence |
|---|---|
| LocalLLaMA/typed-decisions (train split) | Apache 2.0 |
| AmazonScience MASSIVE | CC BY 4.0 |
| PolyAI banking77 | CC BY 4.0 |
| Bitext customer support | CDLA Sharing 1.0 (no obligations on Results, section 3.5) |
| google/civil_comments | CC0 1.0 |
| Synthetic cases and labels from Qwen3-30B-A3B-Instruct-2507 and Qwen3-30B-A3B-Thinking-2507 | Apache 2.0 |

## License

Apache 2.0. The backbone, mmBERT by JHU CLSP, is MIT licensed; its notice and the other
attributions are in the NOTICE file in this repository.

Contact: hello@codepawl.com
