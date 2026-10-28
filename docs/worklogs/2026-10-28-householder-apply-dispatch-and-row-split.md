# The block apply's regions and the row split

## Decisions

- **A region runs on the worker group the caller holds.** `utils::thread::par_for_each` dispatched to
  a plain Rayon `iter.for_each` for `Par::Rayon`. When a caller holds the worker group
  (`spindle::with_lock`, which is how a caller with a thread budget is meant to call faer and what
  the campaign harness does), every thread of the pool is inside that group, so a region dispatched
  that way runs on the calling thread alone. Three regions used the helper: the block Householder
  apply's row split, `upgrade_householder_factor`'s `V_bot^H V_bot` split, and
  `copy_across_layouts`'s column split. All three call `thread::for_each` (spindle-backed for
  `Rayon`, team-backed for `Team`, sequential for `Seq`) and the helper is deleted, so the route
  cannot come back by accident. The rule this restores is already written down in the fork's own
  notes on the `perf/svd-dqds` branch ("never use a raw rayon iterator in this crate"); the helper was
  the one place that broke it, and that file is not on `main` yet.
- **The block apply's row split goes with it.** It split `tmp = V^H C` over rows into one `N x K`
  partial per thread and, while taken, capped the column split at four tasks. It was adopted when the
  reduction measured 2.03 ms at eight threads for a `4096 x 16` basis against a `4096 x 240` target
  -- against a region that was itself starved by the defect above. Measured after the fix (best of
  9, group held, column-only / row split, ms):

  | basis / target | C 4T | C 8T | F 4T | F 8T |
  | --- | --- | --- | --- | --- |
  | 4096x16 / 240 | 0.677 / 0.745 | 0.357 / 0.508 | 0.638 / 0.644 | 0.383 / 0.549 |
  | 4096x32 / 224 | 1.052 / 1.105 | 0.603 / 0.684 | 1.131 / 1.192 | 0.740 / 0.901 |
  | 4096x64 / 192 | 1.767 / 1.842 | 0.966 / 1.154 | 2.113 / 2.108 | 1.603 / 1.393 |
  | 4096x96 / 160 | 2.171 / 2.316 | 1.371 / 1.458 | 2.865 / 2.768 | 2.399 / 2.005 |
  | 4096x256 / 256 | 9.511 / 9.879 | 5.305 / 7.284 | 10.990 / 11.125 | 8.522 / 7.906 |
  | 1024x64 / 64 | 0.153 / 0.182 | 0.083 / 0.161 | 0.195 / 0.195 | 0.122 / 0.162 |
  | 512x512 / 496 | 3.972 / 3.962 | 2.110 / 2.122 | 3.970 / 3.979 | 2.170 / 2.167 |

  The column split wins or ties on 11 of these 14 pairs; the row split wins three column-major cells
  at eight threads by 8-16%. It also costs a serial add of `threads` partials (0.96 ms of the 8.47 ms
  `4096x256` f64 8T call) and caps the column split at four tasks; uncapping that split while
  keeping the row split recovers part of it (7.28 -> 5.98 ms) and still loses to not having it.
  `upgrade_householder_factor`'s split stays: its output is `n x n` for a panel width `n`, so the
  reduction really is depth-bound there and the partials are tiny.
- **The issue's "since 0.25.0" is not a library regression.** `50c2b98` (0.24.4) and `0e15729`
  (0.25.0) measured by one harness are within ~1% of each other on the tall `householder_apply`
  cells, and both are 2-4x behind `faer-upstream` at 4T/8T. The 0.24.x page was recorded before the
  harness held the worker group, so it compared a protocol, not a revision.
- **Attribution by region timing, not by reading the diff.** The reduction region measured 12.1 ms of
  a 14.6 ms call at eight threads against 11.6 ms for the same reduction on one thread, which said
  "this region is not parallel" before any theory about which change introduced it.
- **The regression test watches dispatch, not a clock.** A `#[cfg(test)]` note inside the surviving
  region records whether a group worker (`spindle::current_thread_index().is_some()`) ran a chunk,
  and the test arms that note only for its own call so other tests' regions do not write to it. It
  fails on the previous dispatch with "no worker of the held group of 4 ran a region chunk".
- **Origin.** Both the helper and the row split are this fork's own (`fae766c` and `c061812`). Not
  upstream's; nothing to report under the deferred-feedback rule.

## Verification conclusions and constraints

- Focused test `linalg::householder::tests::upgrade_region_reaches_the_held_group` fails on the
  pre-change dispatch and passes after it. Whole library suite: 461 passed, 0 failed, 138 ignored.
- Campaign pair, boost off, `linalg-tall` + `linalg-mid`, three process sets, f64 and c64, 1/4/8T,
  baseline `2592bad` against the branch: the 32 tall `householder_apply` cells at 4T/8T go from
  2.0-4.1x behind `faer-upstream` to 0.65-1.14x, 31 of 32 inside 1.10. The one cell above the band
  (`f64` column-major `4096x256` block at 8T, 1.14) has an upstream row whose own block and sequence
  forms differ by 30% for identical work (5.07 vs 6.57 ms) while the fork's two agree (5.79 / 5.71
  ms), so that band crossing is not a fork mechanism. Guards: no cell of the 496 compared regressed
  past 1.10, and the tall `qr_factor`, `qr`, `svd_thin` and `svd_values` families are 13-34% faster
  than the same cells at the baseline.
- LAPACK reference: the OpenBLAS arm has no block-apply row, as expected: reference `DLARFB` is
  `W = V2^T C2` (DGEMM), `W = T^T W` (DTRMM), `C -= V W` (DGEMM), with its parallelism inside each
  call over the output and no row-split analogue. The declared `gemm` legs that mirror the apply
  (`16x4096x240` and `4096x16x240`) measure 0.55-0.96 of OpenBLAS at 1/4/8T, and the `qr_factor`
  composite 0.40-0.82, so on the calls LAPACK can express the fork is at or ahead of it.
- Limits. Turbo was off for every number above (`cpufreq_boost=0`); earlier attempts in the same
  session ran with boost on and are diagnostics only. `cpufreq_boost` is a host setting the account
  that measured cannot change without root. Three column-major cells at 8T lost 8-16% with the row
  split removed; they are recorded in the table above rather than tuned away with a threshold.
  `upgrade_factor` and `eig` are declared families the harness skips, so the
  `upgrade_householder_factor` route is covered through `qr_factor`/`qr` composites only, and the
  other tall losses on the same page (`qr`, `svd_thin`, `svd_values` versus OpenBLAS in column-major
  `c64`) were not attributed.
