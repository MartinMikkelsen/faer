# The compact factor's parallel region floor

PR: [tensor4all/faer#9](https://github.com/tensor4all/faer/pull/9) (merged as
`04542fae`). Ledger row: `No parallel region below the compact factor's own floor`.
Consumer: [tensor4all/tlinalg-rs#56](https://github.com/tensor4all/tlinalg-rs/pull/56).

## Decisions

- **Ask the floor once, before opening anything.** `upgrade_householder_factor`
  threads its `V^H V` products, and its own `deep` predicate already states when that
  reduction has the shape to pay for it: deeper than `2 * n`, at least one row per
  thread, over the `8 * 1024` work floor. Below that floor the products still opened a
  rayon region each — and a blocked factorisation calls this function **once per
  block**, at a block size the recursion halves every level. The change is: below the
  floor the call takes `Par::Seq`, above it the caller's `par` is forwarded exactly as
  before. Thirteen lines, no new constant, nothing computed differently.
- **Why it was worth changing.** A threaded 128x128 QR measured **4.2-4.6x slower**
  than the sequential call on the same input — the fork-join cost of `n`-wide
  reductions paid about fifty times for work the same floor calls too small to split.
  Isolation: `qr_in_place` (the call tlinalg's `qr` family makes), f64/c64,
  CPUs 4-11, best of 7. `Par::rayon(8)` before → after, ms: 128x128 1.05 → **0.235**,
  256x256 2.46 → **0.917**, 512x512 4.59 → **2.85**, 1024x64 0.65 → **0.466**,
  4096x256 6.32 → 6.31 (unchanged, its reduction is above the floor).
  At tlinalg level the pair of campaign pages `bf88c0a6e568-mkl` → `7fdba8a53ba3-mkl`
  gives `faer-pool` medians of **0.576** (f64) / **0.742** (c64) for `qr`, with the
  sequential rows and the unchanged MKL binary at 1.00 as the floor; `qr` 128x128 b128
  c64 goes 121.77 → **40.79** ms against MKL's 39.29 (3.10x behind → **1.04x**).
- **Rejected: gate the per-block apply on the trailing area**
  (`(m - row) * (n - col) < par_threshold`, the rule the column-pivoted sibling uses
  for itself). It fixes the small and medium shapes and **regresses the tall ones**:
  4096x256 f64 6.36 → 6.82 ms, c64 16.53 → 18.23. A tall block update is long in depth
  and small in area, so an area threshold sends real wins to `Par::Seq` — the same
  reason the fork's own block-apply gate is a depth-against-width inequality and not
  an area.
- **Rejected: force the `n x n x n` head product of the `prev_block_size < 8` branch to
  `Par::Seq`.** 512x512 f64 2.49 → 3.43 ms (38% worse), neutral elsewhere.
- **Interacts with** the fork's rows `Row-split reduction in the compact-WY factor's
  deep half` and `Row-split reduction in the block Householder apply`: those splits are
  *inside* this function and stay exactly as measured; this change is the gate in front
  of them, so a shape that used to open one region now opens none, and a shape above the
  floor behaves as before.
- **Upstream.** Upstream's `upgrade_householder_factor` forwards `par` to the same
  products and the same `join_raw` recursion with no floor at all, so the pattern is
  upstream's, not the fork's; upstream's `qr/no_pivoting/factor.rs` likewise declares a
  `par_threshold` that nothing reads. The fork cannot port the fix back verbatim, because
  the floor it reuses is itself a fork addition from the row-split work.
  *What was measured* is the published upstream release, `faer-0.24.4` from crates.io
  (linked by path from the registry source), with the same driver, host and pins as the
  fork's numbers on this page. *What was read* is the GitHub mirror clone
  `sarah-ek/faer-rs` at `ddbd2ae`; the source of truth is Codeberg, so treat the line
  numbers there as the mirror's, not upstream's. Upstream feedback is deferred for now
  (see `AGENTS.md`).

## Verification conclusions and constraints

- Establishes: a threaded no-pivot QR is at worst at parity with the sequential call and
  wins from `n = 224` upward (0.88x at 224, 0.37x at 768), and the tall shapes the
  ledger's other rows serve are unchanged within 1.2%.
- Limits: the tlinalg-level pair is a **scan**, not a paired comparison (`--aa 1`), so no
  pooled cell is claimed; one pooled cell moved the wrong way and is not attributable
  (`householder_apply` 32x32 b128 f64 reads 0.246 → 0.472 ms, a shape that fans the batch
  out over lanes with a sequential inner call and therefore does not contain this change
  at all). One pre-existing failure in the fork is unrelated:
  `sparse::linalg::qr::tests::test_numeric_simplicial_qr_1_transpose` panics in
  `triangular.rs:990` on the parent commit too.
- Remains unverified, and these are hypotheses only — each was found by reading the
  source while measuring this change, none has an isolated measurement yet:
  - `svd/bidiag.rs`: `bidiag_in_place_scratch` discards its hint (`_ = par; _ =
    params;`) and `BidiagParams::par_threshold` is never read, so the bidiagonalisation
    is sequential however the caller calls. Consistent with `svd_values` being flat from
    1T to 8T (128x128 b1 f64 1.412 → 1.433 ms against MKL's 0.704 → 0.678). Measured
    and closed: `bidiag_in_place` (the worker) does read the threshold, and its floor
    is load-bearing — see `2026-10-14-bidiag-parallel-hint.md`.
  - `evd/tridiag.rs`: **corrected on re-reading, and the correction is the same lesson as the
    bidiagonalisation lead above.** The scratch function ignoring its hint is the workspace sizing
    it correctly (shape-sized, not thread-sized), and `tridiag_in_place` *does* read
    `params.par_threshold` — `rem * rem / 2 < par_threshold` latches to `Par::Seq` below the
    floor, which at 128x128 (`127² / 2 = 8064 < 49152`) closes it for the whole factorisation.
    The flat `eigh` reading (1.26 → 1.24 ms against MKL's 1.08 → 0.586) is therefore *this
    library's own floor doing its job*, not a dropped hint: MKL threads that size and faer
    declines to. Whether the floor sits where it should was a measured question about the
    crossover — the `M ~ 2K` episode in the ledger is the warning about moving such a constant
    from the wrong cell — and it **has since been measured**: forcing the gate open costs
    3.1x at 128x128 and 1.5x at 256x256, and no lower floor buys anything, so the floor is where
    it belongs. See `2026-10-14-evd-svd-floor-sweep.md`, which sweeps the bidiagonalisation's
    floor the same way; both floors end up DROPPED, i.e. kept as they are.
  - `lu/partial_pivoting/factor.rs` is *not* missing its hint (panel solve, trailing
    matmul and recursion all take `par`, and its `par_threshold` is read), but
    `lu_factor` 512x512 b1 f64 scales only 1.85x from 1T to 8T where MKL reaches 4.9x.
    This one needs a phase profile before any threshold is touched; the `M ~ 2K` episode
    in this ledger is the lesson about choosing a constant from the wrong cell.
