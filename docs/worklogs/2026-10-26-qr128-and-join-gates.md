# QR 128 on a team, and the gates of the join regions

Continuation of [2026-10-26-remaining-gates.md](2026-10-26-remaining-gates.md). Design and constants: tensor4all/threadteam-rs
`docs/design/thresholds.md` (sections 1, 2.3, 3.3 to 3.5). Evidence and raw files: faer-benchmark
`data/probes/20261026-qr128-regions/` (README, `tables.md`, `regions.md`, `raw/`, `processes.log`). Code: `dd0bc19`,
`801d781`, `a9813bf`, `7b60550`. One Apple M5 Max, QoS `UserInitiated`, a shared laptop with the desktop session
running (48 of 221 logged processes are marked busy); everything below is rough and decides nothing.

## Task A: the QR 128 regression on a team (86 us to 102 us, serial 121 to 128 us)

Method: the `par-trace` feature records every gate decision with its dimensions, units, sections, estimate and
dispatch cost; the old decision is recomputed from the dimensions; every region that changed is timed alone,
serial against the team forced parallel.

* A 128 x 128 QR (block 8, sub block 4) asks 512 gate decisions. Only one gate differs between `9cd64f0` and
  `6788b20`: the column split of the block apply (`n_tasks`), 10 of 59 parallel under the old rule
  (`M*K*4M >= 589824`, `M` 56 to 128) and 5 under the new one (`M` 96 to 128). The `upgrade_householder_factor`
  reductions (83), the row split (37), the rank one path (never reached) and the triangular product and solve
  joins (170) are serial in both.
* **The estimate was wrong, not the section count.** `2MNK` multiply-adds at the GEMM rate was 0.58 to 0.77 of the
  measured serial time of an `N = 8` block (`M` 56 to 88). One `for_each_with_par`, so `sections = 1` is right.
  The apply also streams its `M x K` target; the fit `t = 2MNK * GEMM + MK * rank one rate` over 40 shapes (`M`
  24 to 512, `N` 2 to 32, f64 and c64) has a median error of 7 and 4 percent, so the gate uses it
  (`cost::block_apply_estimate`) with no new constant. The estimate is now 0.85 to 1.05 of the measured time from
  `M = 56`. A flat rate per multiply-add (0.054) fitted `N = 8` but slowed QR 256 by 8 to 12 percent (wide bases
  run at 0.036 to 0.043); the two term form replaced it.
* **The old parallel decision was not luck at `M` 56 to 72**, it was a lower floor. Forced parallel wins there
  by 1.1, 0.9 and 2.0 us per call; the estimates are 2.2, 2.9 and 3.7 us against `PAYS * 1.4 = 4.2 us`. The
  break-even is about 1.2 times the dispatch for a width 6 section, so the margin is the multiple 3. Rule kept.
  In one process: team 100.5 us, previous gate 96.8, no column split 131.6, serial 125.0.
* Result: QR 128 on a team 102.5 us before, 99 us now (0.82 to 0.80 of serial), 87 to 94 us with the gates of
  `9cd64f0`. The difference left is the three regions above (4 us) and 2 to 4 us that were not isolated (the
  gate calls cost 0.7 us per factorization serial; a sequential token now returns false before computing its
  estimate).
* Found on the way: a join (a split of the budget in two) is a section of width 2, not of the full degree. A 64 x 64
  triangular product is 7.0 us serial and 8.1 us on the team at an estimate of 4.8 us (72 x 72: 11.8 against
  16.7); it only wins from 80. `join_pays` decides every join at the width 2 dispatch cost (3.0 us team, 14 us
  outside the pool on the M5 table), including the existing triangular product and solve joins. This is not a
  measured cost of `Team::split`; the LU swap stage split costs about 13 us.

## Task B: the regions without a gate

| region | sections | estimate | constant |
| --- | --- | --- | --- |
| `bidiag_svd` recursion join | 1, width 2 | `2 (n/2)^2` | 53 ns per `m^2` (f64; real types only) |
| `bidiag_svd` merge products (`update_v`, `update_u`, both) | 1 each, width 2 | their multiply-adds | GEMM rate |
| `tridiag_evd` recursion join | 1, width 2 | `2 (n/2)^2` | 31 ns per `m^2` |
| `tridiag_evd` merge products | 1, width 2 | `(n1^2 + (n-n1)^2) n` multiply-adds | GEMM rate |
| `triangular_inverse` join | 1, width 2 | `2 (n/2)^3` | 0.0096 (f64), 0.0344 (c64) ns per `m^3` |
| sparse numeric product | 1, width = degree | the symbolic flop count | 1.5 (f64), 1.8 (c64) ns per multiply-add |
| norm reductions, `zip`, elementwise kernels | none | | no parallel region exists |

* Recurrence: the work of a join of size `n` is the two subproblems it runs, `2 W(n/2)`, `W(m)` the serial time of a
  whole solve at one thread. `W` is not cubic: the divide and conquer SVD is `45 m^2 + 0.023 m^3` ns (`m` 64 to 512,
  Gaussian bidiagonal data, nothing deflates), the EVD about `31 m^2`, so the unit is `m^2`; the triangular
  inverse is cubic. The merge after a join is not counted (it is not parallel through that join).
* The default recursion thresholds (128 and 64) are above the floors of the two divide and conquer joins on every
  token, so their effect is in the merge product gates; the join gates matter for smaller `recursion_threshold`.
* The sparse gate also covers the case where the rule fails: the numeric product then runs its column ranges as
  one range (`nthreads = 1` for the partition, the workspace keeps its columns).
* No `par_threshold` field was added: nothing was configurable in these regions.
* Norms and `zip`: no `Par` argument, no Rayon call, serial pairwise SIMD reductions (`linalg/reductions`,
  `linalg/zip.rs`). There is no region to convert; a parallel norm would be a new feature with an API question.

## Results (6 threads, ratio to one thread serial, f64; plain from outside the pool, team of 6)

| cell | plain before | plain after | team before | team after |
| --- | --- | --- | --- | --- |
| QR 128, 256, 512 | 1.02, 1.14, 0.68 | 1.02, 1.16, 0.71 | 0.82, 0.61, 0.41 | 0.80, 0.62, 0.47 (separate check 0.41 to 0.47) |
| SVD (U, V) 128, 256, 512 | 0.88, 0.88, 0.75 | 0.83, 0.87, 0.76 | 0.76, 0.61, 0.44 | 0.76, 0.63, 0.42 |
| EVD (eigenvectors) 128, 256, 512 | 0.98, 0.86, 0.67 | 0.99, 0.85, 0.66 | 0.97, 0.56, 0.39 | 0.98, 0.58, 0.39 |
| triangular inverse 128, 256, 512 | 6.62, 1.67, 0.96 | 1.00, 1.00, 0.90 | 0.81, 0.48, 0.29 | 1.04, 0.52, 0.31 |
| sparse product 500, 2000, 8000 | 15.0, 3.09, 0.83 | 1.02, 1.01, 0.84 | 0.44, 0.27, 0.25 | 0.47, 0.28, 0.25 |

The triangular inverse 128 on a team gives back what the ungated recursion won (0.81 to 1.04): the model puts the two
64 wide halves at 5.0 us (the rate at `m` 64 is 2.3 times the median one) where they take 11.6 us, and 9 us is the
floor of a width 2 section. A size dependent rate would need a second constant per family; not done.

## Tests

`cargo test --release --lib` with `--features threadteam` and with default features: everything passes except the
three known `strict_triangle` cases (281 and 267 passed). New: `utils::cost::tests` (`halves_join_gates_follow_the_rule`,
`join_gates_on_a_team_follow_the_width_two_cost`, `block_apply_split_starts_where_the_estimate_pays`, the `GATES`
table with the sparse rate), `utils::cost::join_results` (parallel equals serial for the triangular inverse, the
bidiagonal SVD, the tridiagonal EVD and the sparse product at sizes above the floors of `Par::rayon(4)`).
