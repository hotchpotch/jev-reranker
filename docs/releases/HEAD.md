# HEAD

## Performance

- Reuse exact listwise group length estimates for budget checks and detailed
  request records. Pointwise and pairwise requests also reuse their preflight
  estimate when recording details.
- Cache exact tokenizer counts across calls on a reranker instance, with LRU
  eviction bounded to 2,048 strings and 4,000,000 retained Unicode code points.
  Oversized strings are counted without being retained. The cache stores text
  and counts in memory; it does not write them to disk. Keep a supplied tokenizer
  and its encoding configuration unchanged while using the instance.
- Reuse the initial document tokenization when truncating a newly counted
  document. Decoded text is still checked against the configured length limit.
- These changes preserve exact serialized-payload counting, document order,
  split budgets, and scoring prompts. They do not approximate request lengths
  by adding document lengths. Custom `length_fn` calls are not memoized.
