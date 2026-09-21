# Examples

[`eval.py`](eval.py) evaluates Jev and Sentence Transformers rerankers on
compatible [Nano-set benchmarks](https://huggingface.co/hakari-bench/datasets?search=nano).
NanoBEIR-en / NanoHotpotQA is the default; `--split` selects another benchmark,
and `--dataset` with `--revision` selects another compatible dataset repository.
Run it from the repository root so it uses this checkout.

For Jev, set `TYPESAFE_API_KEY` in the environment or `.env`:

```sh
uv run --locked --group examples --extra tokenizer python examples/eval.py \
  --task relevance --target en
```

This scores the full hybrid pool with the default relevance threshold of 0.2.
Use `--task rerank` for ordinary reranking with threshold 0.0.

To use Gemma token counting with explicit, smaller split budgets:

```sh
uv run --locked --group examples --extra tokenizer python examples/eval.py \
  --task relevance --tokenizer google/embeddinggemma-300m \
  --document-max-length 4000 \
  --split-state-budget 16000 --split-request-budget 30000
```

The budgets use Gemma token estimates and control request grouping. They do not
change the candidate count. The default budgets remain 26000 and 48000; local
token estimates may differ from the provider's counts.

For a smaller, controlled evaluation with ten candidates containing every positive:

```sh
uv run --locked --group examples --extra tokenizer python examples/eval.py \
  --task relevance --target en --top-k 10
```

The default candidate count is unlimited; `--top-k` here controls input candidates,
not the number of output results. Relevance defaults to threshold 0.2 and ordinary
reranking to 0.0. Both report nDCG@10 and filtering statistics.

See the [evaluation script guide](../docs/eval.md) for installation, Japanese data,
Sentence Transformers, thresholds, metrics, saved outputs, and offline tests.
