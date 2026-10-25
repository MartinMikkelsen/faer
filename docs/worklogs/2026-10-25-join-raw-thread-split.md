# Width split in `join_raw`

## Decisions

- **Split the width so the halves add up to it.** `utils::thread::join_raw` is documented as
  "splitting the amount of parallelism between the two", but it gave both halves
  `Par::Rayon(n - n/2)`. For an odd `n` the two halves overshoot, and the recursive splits used by the
  dense factorizations (23 call sites) reach the next power of two in leaves: a requested width of 3
  produces 4 single-thread leaves, 5 and 6 produce 8. The halves now get `n - n/2` and `n/2`, so a
  recursive split ends in exactly `n` leaves. `n >= 2` on that branch, so `n/2` is never zero.
- **Why it matters to tensor4all.** The CPU placement design gives each operation a thread budget and
  requires that what runs under it stays within it
  (`tensor4all-meta/docs/design/core-boundary-and-execution-ownership.md`, "Where placement is
  decided"). A faer call handed `Par::Rayon(budget)` must not create more concurrent leaves than the
  budget. Rounding the width down to a power of two on the caller's side was considered and rejected:
  it wastes threads (a budget of 6 would run on 4) to work around a defect that is a few lines here.
- **Origin.** The code is upstream faer's (author sarah-quiñones; last touched by the reformat
  `fb501bd`, the behaviour predates it) and is present at the fork's upstream base `0ecd846`. It is
  not a fork regression. Upstream feedback is deferred: nothing was reported upstream, and this
  record is the trace.

## Verification conclusions and constraints

- Focused test `utils::thread::tests::join_raw_leaves_do_not_exceed_requested_width`: in a 16-thread
  pool, so the pool never bounds the result, a recursive `join_raw` split for `n` in
  {2, 3, 5, 6, 7, 8} ends in exactly `n` leaves, and the peak number of concurrently running leaves
  is at most `n`. With the original split restored it fails at `n = 3` with 4 leaves.
- Whole library suite: 243 passed, 3 failed, 2 ignored. The three failures are the pre-existing
  `linalg::matmul::internal::tests::*strict_triangle*` cases already recorded in `PATCHSET.md`, in
  files this change does not touch; the one new pass is the focused test.
- Limits: no timing was taken and none is claimed. For an odd width the uneven split also changes
  how work is balanced between the halves (each half now receives the threads it is told about);
  whether that moves any factorization's time has not been measured.
