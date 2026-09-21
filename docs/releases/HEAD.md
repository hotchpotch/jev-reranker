# HEAD

- The default listwise relevance preset now stores its evaluation rubric once
  per request and uses shorter per-document references to reduce repeated prompt
  text. Its scoring rules combine evidence usefulness with retrieval relevance
  and explicit anchors for partial and supporting facts. Scores, request groups,
  and filtering decisions can change; the default threshold remains 0.2.
  Custom prompts, pointwise relevance, and the ordinary reranking preset retain
  their existing request formats.
- The evaluation script now uses all queries in the selected dataset split by
  default. Use `--query-limit N` to
  evaluate at most N queries in sorted query-ID order.
- The evaluation script accepts `--split-state-budget` and
  `--split-request-budget` to adjust Jev request grouping estimates without
  changing the candidate count or per-document length limit.
