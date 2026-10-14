# The bidiagonalisation's parallel hint is not dropped

PR: [tensor4all/faer#11](https://github.com/tensor4all/faer/pull/11). Ledger
row: `Bidiagonalisation parallel hint`.

## Decisions

- **No code change; the experiment is dropped.** The lead closed here was left as
  a hypothesis in `2026-10-14-compact-factor-region-floor.md`: that
  `bidiag_in_place_scratch` discards its hint (`_ = par; _ = params;`) and that
  `BidiagParams::par_threshold` is never read, so the bidiagonalisation is
  sequential however the caller calls it. The first half is true of the
  *scratch-size* function only, and correctly so — the workspace is one `n`-vector
  per side and does not depend on the thread count. The worker,
  `bidiag_in_place`, reads `params.par_threshold` on every column and forwards
  `par` into the trailing update (the fused op's column split and both trailing
  matmuls). There is no missing `par`. Measured `bidiag_in_place` at `Par::Seq`
  and `Par::rayon(8)`, f64 and c64: the two are identical wherever the gate is
  closed and differ only where it is open.
- **What actually makes `128 x 128` sequential.** The gate is
  `(m - k - 1) * (n - k - 1) < params.par_threshold` and it *latches*: once the
  trailing area drops below `192 * 256 = 49152`, `par` is `Par::Seq` for the rest
  of the factorisation. At `128 x 128` the first column already has
  `127 * 127 = 16129`, so the whole call is sequential by construction. That is
  the same shape-of-the-work decision #9 makes for the compact factor ("no region
  below the floor"), applied to the trailing update: below the floor the update
  would fork and join for work the same predicate calls too small to split.
- **Rejected, with the numbers that rejected it: remove the gate** (forward the
  caller's `par` unconditionally). It regresses every measured shape. Isolation
  driver on `bidiag_in_place`, CPUs 0-3, `RAYON_NUM_THREADS=8`, best of 7,
  interleaved A/B, min of 3 pairs, ms at `Par::rayon(8)`, gate → no gate:
  `128 x 128` f64 **0.319 → 0.797** (2.50x), c64 **0.617 → 1.093** (1.77x);
  `256 x 256` f64 2.328 → 3.345 (1.44x), c64 4.646 → 5.852 (1.26x);
  `512 x 512` f64 15.05 → 16.29 (1.08x), c64 22.38 → 25.14 (1.12x). End to end
  (`linalg::svd::svd` as tlinalg's provider calls it) at `128 x 128`: `svd_values`
  f64 0.625 → 1.134 (1.81x), `svd_thin` f64 0.932 → 1.421 (1.52x), c64 1.419 →
  1.879 (1.32x). The fork-and-join cost at these sizes is real; the gate is
  load-bearing.
- **Rejected: keep the caller's hint for the two trailing
  `upgrade_householder_factor` calls.** The latch reaches those calls too: for
  `4096 x 256` it drops to `Par::Seq` at `k >= 243`, so both compact-factor
  upgrades run sequential although #9's own `deep` floor would thread them.
  Preserving the caller's hint there changed nothing — no row moved more than 6% in a
  2-pair interleaved A/B, f64 and c64, from `128 x 128` to `4096 x 256` — so the
  starved upgrades are not a measurable loss on these cores.
- **Upstream.** `faer/src/linalg/svd/` is byte-identical to the upstream mirror
  (`git diff` against the GitHub mirror clone `sarah-ek/faer-rs` at `ddbd2ae` is
  empty), so the numbers above are upstream's behaviour for this file and not a
  fork divergence. The source of truth is Codeberg. No upstream binary was
  measured separately (**none measured**).

## Verification conclusions and constraints

- Establishes: the campaign's flat `svd_values` at 128 and 256 is not a dropped
  parallel hint. The bidiagonalisation is sequential because the work is below the
  floor the same code already uses, and threading it does not pay on the measured
  cores: at `128 x 128` `Par::Seq` and `Par::rayon(8)` are identical (0.319 / 0.319
  ms f64, 0.617 / 0.617 c64), at `256 x 256` the threaded call is 4.7% slower
  (2.223 → 2.328 ms f64), and only `512 x 512` pays (f64 16.06 → 15.05, c64 32.73
  → 22.38). End to end at the campaign's shapes, 1T / 8T, f64 / c64 (faer API;
  `svd_values` with the QR params the provider uses): `128 x 128` 0.640 / 0.625 and
  0.930 / 0.928, `256 x 256` 3.484 / 3.642 and 6.039 / 6.009. `svd_thin` scales
  where the vector applications do, not where the bidiagonalisation does:
  `128 x 128` 1.040 → 0.932 (f64) / 1.792 → 1.419 (c64), `256 x 256` 5.811 → 4.214
  / 11.58 → 7.549.
- Limits: **four cores.** Every number here is pinned to CPUs 0-3 (the campaign
  holds 4-11 and their SMT siblings), so `Par::rayon(8)` runs eight workers on
  four cores and the eight-physical-core crossover is not measured. A shape that
  loses here could be a small win on eight idle cores; the campaign page's own
  `svd_values` rows (1T vs 8T) say it is not for 128 and 256, but that is the
  provider path, not this driver.
- Limits: these are faer-API numbers, not the tlinalg page's rows. At `128 x 128`
  f64 my `svd_values` call is 0.64 ms while the page's `faer-1lane` row reads
  1.41 ms, so ~0.8 ms sits in the provider (per-item scratch reuse and the output
  push in `tlinalg/src/svd.rs`) rather than in `bidiag_in_place`. The page's
  `svd_thin` row also carries a reconstruction check and a possible QR retry that
  this driver does not, so this record does not attribute the page's gap to MKL
  to faer's SVD core.
- Gate: `cargo test -p t4a-faer --lib` on this revision passes — 203 passed,
  0 failed, 2 ignored. `sparse::linalg::qr::tests::
  test_numeric_simplicial_qr_1_transpose` passes as well (five reruns); the
  failure recorded in `2026-10-14-compact-factor-region-floor.md` did not
  reproduce on `9de95f7`.
- Remains unverified, and these are hypotheses only:
  - The eight-core crossover for the bidiag gate. The four-core A/B cannot see it.
  - A blocked bidiagonalisation (LAPACK's `dgebrd` shape: a panel plus a `GEMM`
    trailing update) is the candidate for the algorithm-level single-thread
    difference, not threading. This record measures no blocked variant, and at
    these shapes the faer-API call is already at or below the campaign page's MKL
    row, so a faer-versus-MKL gap at `128 x 128` is a provider-level question
    first.
