# The blocked partial-pivoting LU's trailing-update size floor

Ledger row: `Trailing-update size floor in the blocked partial-pivoting LU`.
Consumer: the campaign's `lu_factor` row for `512 x 512` `f64`, which reads 2.83 ms
at one thread and 1.53 ms at eight (1.85x) where MKL reaches 3.00 -> 0.61 (4.9x).

## Decisions

- **The level's two products are gated on the trailing update's size, nothing else
  changes.** `lu_in_place_recursion` splits its panel in half at every level and runs,
  per level, a panel solve (`solve_unit_lower_triangular_in_place`) and a trailing
  product (`A11 -= A10 A01`). Both were handed the caller's `par` unchanged at every
  depth, so the recursion opened a fork and a join around a trailing update that the
  next level halves again: at `512 x 512` that is roughly sixty regions, most of them
  a few thousand flops. They now take `Par::Seq` while `A11.nrows() * A11.ncols()` is
  not above `params.par_threshold`. The recursion still forwards the caller's `par` to
  its children, so each level decides for its own trailing update, and the panel
  subtree that a level already ran is untouched by that level's decision.
- **The floor is the existing `par_threshold`, not a new constant.** The measured
  crossover is `128 * 128`, which is `PartialPivLuParams::par_threshold`'s default, and
  the swap stage below already reads that field as the size at which parallelism stops
  paying. One knob, and its doc comment ("threshold at which size parallelism should be
  disabled") now describes both of its uses. A separate constant would need its own
  sweep and its own justification for a value the sweep picks anyway.
- **Measured by sweep, not by the cell that regressed.** A trailing-update floor was
  swept over 0, 1024, 2048, 4096, 8192, 16384, 24576, 32768 and 65536 at four threads,
  on `256 x 256` (where the region overhead is the largest share of the call),
  `512 x 512` and `1024 x 1024`: 16384 is the best of the nine on all three, 8192 and
  24576 are within 3%, and 65536 loses on all three. The sweep is what fixes the value;
  `256 x 256` is not the shape the change was chosen on.
- **Rejected: gating the swap stage's region too.** The swap stage is 5-10% of the call
  and it too measures worse under the caller's `par` than under `Par::Seq`: in both
  instrumented runs its bucket is larger at four threads than at one (0.1075 -> 0.1679
  and 0.1586 -> 0.2249 ms). It has its own predicate,
  `m * (ncols - n) > par_threshold`, which already refuses the top-level swap at
  `512 x 512`. Tightening it is a different predicate with its own crossover, and in the
  interleaved comparison below the same swap bucket moves to 0.1382 without a line of its
  code changing, so this record could not separate a swap gate from that drift. Left
  alone rather than moved on an unmeasured reading.
- **Not rejected: the recursion itself.** Nothing about the panel/blocking structure
  changed. The vendor gap at `512 x 512` is not closed by this: the remaining serial
  part is the leaf kernel (`lu_in_place_unblocked`, 12-19% of the call and sequential by
  construction) plus the level-0 matmul and solve, and the change removes only the
  region overhead below the floor.

## Verification conclusions and constraints

- Establishes: the `512 x 512` scaling gap is partly this library's own scheduling, and
  not only the vendor kernels. Instrumenting one `lu_in_place` (phase counters, that
  build only) splits the time at four threads into trailing product and panel solve
  (both bucketed by output size), the swap stage and the sequential leaf. The products
  with an output of at least 16384 elements scale, the 4096..16383 bucket and below do
  not: with the floor open vs forced at 16384, per-call mean over eight calls, the
  smallest such mean over ten interleaved repetitions, `512 x 512` `f64`, four threads
  (ms; each column is its own minimum, so the columns do not sum to the wall column) --
  - trailing product, >= 16384: 0.2336 -> 0.2276; 4096..16383: 0.4648 -> 0.3982;
    < 4096: 0.0480 -> 0.0121
  - panel solve, >= 16384: 0.2487 -> 0.2390; 4096..16383: 0.0872 -> 0.0745;
    < 4096: 0.0302 -> 0.0290
  - swap stage: 0.1601 -> 0.1382; sequential leaf: 0.2974 -> 0.2814; total:
    1.5960 -> 1.4134.
  The two buckets the change touches account for roughly 0.12 of the 0.18 ms; the rest is
  drift on buckets the change cannot reach, which is why the same comparison is also run
  on the shipped pair of builds below.
- Establishes, on the shipped code, two separately built binaries of the parent and the
  child interleaved: best of four calls, smallest over fourteen interleaved repetitions,
  `f64`, four threads (ms, parent -> child) -- `128 x 128` 0.0980 -> 0.0582 (1.68x),
  `192 x 192` 0.1597 -> 0.1120 (1.43x), `256 x 256` 0.3108 -> 0.2550 (1.22x),
  `384 x 384` 0.6046 -> 0.4806 (1.26x), `512 x 512` 1.2799 -> 1.1673 (1.10x),
  `1024 x 1024` 5.2400 -> 5.1188 (1.02x). At one thread the two builds are the same
  code path and measure the same (`512 x 512`: 1.8818 -> 1.8724); at two threads
  `512 x 512` is 1.4943 -> 1.4308. The win is entirely in the deep levels, so it is
  largest where the deep levels are the largest share of the call.
- Limits: **four physical cores**, CPUs 0-3 (pinned throughout; the campaign holds
  4-11 and their siblings), so the eight-physical-core row of the campaign's page is
  not reproduced here and the four-thread crossover is not the eight-thread one. The
  floor is a constant that a machine with more cores would want lower, and this
  measurement cannot say where.
- Limits: with an eight-thread rayon pool asked for eight workers on those same four
  cores, `512 x 512` `f64` takes 57 ms on the parent -- fifty times slower, not a
  marginal effect. It is reproducible and it is in the parent build, so it is not this
  change; it is recorded so that the 8T column above is not read as an eight-core
  result.
- The focused test is
  `linalg::lu::partial_pivoting::factor::tests::threaded_lu_matches_the_sequential_one`
  (96, 128, 192 below `128 * 128` at the top level, 256 and 512 above it, sequential and
  four-thread factors compared entry-wise, permutations and transposition counts
  compared exactly). The gate schedules the same arithmetic, so the test cannot fail on
  a wrong constant; it fixes that both sides of the floor agree.
- Gate: `cargo test -p t4a-faer --lib` on this revision passes, 205 passed, 0 failed,
  2 ignored. That includes
  `sparse::linalg::qr::tests::test_numeric_simplicial_qr_1_transpose`, which an earlier
  record blamed on the parent revision: it passes on this parent too, so the blame was
  misattributed, as `2026-10-14-bidiag-parallel-hint.md` had already found.
