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
model-index:
- name: {{name}}
  results:
  - task:
      type: text-classification
      name: Typed decisions
    dataset:
      name: LocalLLaMA/typed-decisions
      type: LocalLLaMA/typed-decisions
      split: test
    metrics:
    - type: accuracy
      value: 0.7625
      name: Accuracy
    - type: brier_score
      value: 0.0678
      name: Brier score
    - type: ece
      value: 0.095
      name: Expected calibration error
---

<p align="center">
  <img src="https://raw.githubusercontent.com/codepawl/tacet/main/assets/tacet-mark.png" alt="Tacet" width="88">
</p>

<h1 align="center">{{display_name}}</h1>

<p align="center">
  <a href="https://huggingface.co/codepawl/tacet-sonata"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-tacet--sonata-FFD21E?logo=huggingface&logoColor=000"></a>
  <a href="https://github.com/codepawl/tacet"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-codepawl%2Ftacet-181717?logo=github&logoColor=fff"></a>
  <a href="https://pypi.org/project/tacet/"><img alt="PyPI" src="https://img.shields.io/pypi/v/tacet?logo=pypi&logoColor=fff&label=PyPI&color=3775A9"></a>
  <a href="https://tacet.codepawl.com/docs"><img alt="Docs" src="https://img.shields.io/badge/Docs-tacet.codepawl.com-4F7FE0?logo=readthedocs&logoColor=fff"></a>
  <a href="https://codepawl.com/discord"><img alt="Discord" src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=fff"></a>
  <a href="https://github.com/codepawl/tacet/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-2F2F2F"></a>
</p>

{{display_name}} (`{{repo_id}}`) answers typed questions about a piece of state and gives a probability for every
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
    state={"ticket": "I was charged twice for my March invoice. Please refund the second charge."},
    questions={
        "route": choice("Which team should handle this?",
                        {"billing": "payments, invoices, refunds", "tech": "bugs and outages"}),
        "urgency": score("How urgent is this?", ["low", "medium", "high"]),
        "refund": noul("Is the customer asking for a refund?"),
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

The weights are not a `transformers` model on their own: the head and the packed input format
live in the `tacet` package.

## Evaluation

<img src="https://raw.githubusercontent.com/codepawl/tacet/main/assets/benchmark-light.png" alt="Tacet Sonata against Laya: 211 vs 14.9 requests per second, 32.5 vs 62.4 ms, 144M vs 421M parameters, accuracy 0.7625 vs 0.7675 (a tie), calibration error 0.095 vs 0.215">

LocalLLaMA/typed-decisions test split, 400 cases, 2,000 decisions, scored with
`scripts/benchmark.py` in the tacet repository:

| model | parameters | accuracy | Brier | ECE (15 bins) |
|---|---|---|---|---|
| {{display_name}} | {{parameters}} | 0.7625 | 0.0678 | 0.095 |
| Laya | 421M | 0.7675 | 0.0615 | 0.215 |

Both rows were scored with the same script on the same 2,000 decisions. The accuracy gap is not
significant (paired McNemar p = 0.63), so treat it as a tie. Tacet's probabilities are closer to how
often it is right (lower ECE), while Laya has the lower Brier score. The train split of this
benchmark is part of the training data for both models, so these are in distribution numbers for its
four workflows.

Other checks, on data outside that benchmark:

| evaluation | {{display_name}} | first release candidate (benchmark only) |
|---|---|---|
| our hand written free text suite, 386 cases in 8 languages, 1,192 questions (not public) | 0.539 | 0.379 |
| MASSIVE intent and domain, test split, 16 languages | 0.778 | 0.370 |

Chance on the free text suite is about 0.35. The suite's cases were written for evaluation only and
never trained on.

## Limits

- Strong on the benchmark's four workflows (customer service, invoice processing, security
  incidents, agent trace observability) and on routing and triage questions like them.
- Yes or no answers depend a lot on wording. They are more reliable when the question uses the
  same words as the text than when it paraphrases.
- Weaker on long free text, on questions that need date arithmetic, and on counting. If a
  decision depends on a number, compute it in code and put it in the state.
- The probabilities are calibrated on the training distribution. Check them against your own
  labelled examples before you gate anything on a confidence threshold.
- Reads at most 1536 tokens by default and up to 4096 with `max_length`. A longer state is cut and
  the response says so (`usage.state_truncated`).
- Does not generate text or explain its answers.

## Training data

| data | cases | licence |
|---|---|---|
| Synthetic cases in 16 languages, written and labelled by Qwen3-30B-A3B-Instruct-2507, disputed questions relabelled by Qwen3-30B-A3B-Thinking-2507 | 14,355 | Apache 2.0 |
| LocalLLaMA/typed-decisions (train split) | 1,200 | Apache 2.0 |
| AmazonScience MASSIVE | 601 | CC BY 4.0 |
| PolyAI banking77 | 137 | CC BY 4.0 |
| Bitext customer support | 141 | CDLA Sharing 1.0 (no obligations on Results, section 3.5) |
| google/civil_comments | 134 | CC0 1.0 |

The labelled datasets were turned into typed decision cases: their labels became the answers to
choice, score and yes or no questions. Nothing with a share alike licence or an unknown upstream
licence is in the mix. The weights are an average of three training runs with different seeds.

## License

Apache 2.0. The backbone, mmBERT by JHU CLSP, is MIT licensed; its notice and the other
attributions are in the NOTICE file in this repository.

Contact: hello@codepawl.com
