# HEAD

- Run listwise split groups concurrently within the instance-wide HTTP limit, including
  groups created after a provider context-limit rejection. Preserve sibling cancellation
  and original error types on failure.
