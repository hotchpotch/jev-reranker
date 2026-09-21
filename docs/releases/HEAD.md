# HEAD

- The evaluation script now uses all queries in the selected dataset split by
  default. Use `--query-limit N` to
  evaluate at most N queries in sorted query-ID order.
- The evaluation script accepts `--split-state-budget` and
  `--split-request-budget` to adjust Jev request grouping estimates without
  changing the candidate count or per-document length limit.
