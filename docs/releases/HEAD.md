# HEAD

- Run listwise split groups concurrently within the instance-wide HTTP limit, including
  groups created after a provider context-limit rejection. Preserve sibling cancellation
  and original error types on failure.
  A single split listwise call can now have up to `max_concurrency` requests in flight
  (default 4 for listwise), instead of one, so rate-limit responses may be more likely.
  Review limits previously tuned for multiple concurrent calls. On failure, sibling
  requests already in flight are cancelled but may still be billed by the provider.
  Detail split ordering and request ID assignment are not deterministic for split
  listwise calls; final scores and stable tie ordering are unaffected.
