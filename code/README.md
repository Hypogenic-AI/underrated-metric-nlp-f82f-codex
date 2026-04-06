# Cloned Repositories

## Repo 1: lm-evaluation-harness
- URL: https://github.com/EleutherAI/lm-evaluation-harness
- Purpose: Standardized LLM evaluation framework (task orchestration, metrics, reproducible evaluation configs)
- Location: `code/lm-evaluation-harness/`
- Key files:
  - `code/lm-evaluation-harness/lm_eval/`
  - `code/lm-evaluation-harness/docs/interface.md`
  - `code/lm-evaluation-harness/examples/`
- Notes:
  - Install with backend extras, e.g. `pip install "lm_eval[hf]"`.
  - Useful as the core harness for running prompts/tasks and collecting model outputs for novelty analysis.

## Repo 2: mauve
- URL: https://github.com/krishnap25/mauve
- Purpose: Distributional text-generation metric (MAUVE) for comparing model output distributions against human text
- Location: `code/mauve/`
- Key files:
  - `code/mauve/src/mauve/`
  - `code/mauve/examples/`
  - `code/mauve/README.md`
- Notes:
  - Install package with `pip install mauve-text` (or editable install from source).
  - Supports direct text inputs, token inputs, or feature embeddings.
  - Strong candidate metric for this project’s novelty-quality tradeoff evaluation.

## Repo 3: CommonGen
- URL: https://github.com/INK-USC/CommonGen
- Purpose: CommonGen benchmark utilities (dataset/evaluation references and baseline methods)
- Location: `code/CommonGen/`
- Key files:
  - `code/CommonGen/evaluation/`
  - `code/CommonGen/methods/`
  - `code/CommonGen/README.md`
- Notes:
  - README points to maintained evaluation repository: https://github.com/allenai/CommonGen-Eval
  - Useful for constrained generation baselines and evaluation scripts around commonsense/novel composition.

## Quick Validation Performed
- Verified each repository cloned successfully.
- Read top-level README files to identify installation expectations and likely entry points.
- No full training/evaluation runs were executed in this phase (resource-gathering only).
