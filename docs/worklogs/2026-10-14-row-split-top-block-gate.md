# The block Householder row split's top block

PR: [tensor4all/faer#13](https://github.com/tensor4all/faer/pull/13). Ledger row:
`Row-split gate for the block Householder apply's top block`.
Consumer: [tensor4all/faer#12](https://github.com/tensor4all/faer/issues/12), the
campaign blocker the `householder_apply` page could not be recorded past.

## Decisions

- **Take the row split only while one chunk swallows the top `n` rows.** The split sums
  `tmp = V^H C` from `n_chunks` partials over the target's rows. The first `n` rows of `V`
  are its unit-lower triangular head, and their contribution is applied through that head's
  own `n x n` factor (`essentials_top.transpose()`, `UnitTriangularUpper`); the rows below
  go through the rectangular `essentials_bot`, which slices correctly. A chunk whose rows are
  a strict *subrange* of `[0, n)` would need the head factor's columns `lo..top_hi` -- a
  `top_hi x (top_hi - lo)` unit-upper trapezoid, a shape `triangular::matmul` cannot state
  because its precondition wants a square triangular operand. The predicate asked only for
  depth (`M >= 2K`), a thread floor and a work floor, so any shape with
  `ceil(M / threads) < n` handed the full `n x n` factor to a shorter row chunk and tripped
  `lhs_ncols == rhs_nrows` inside `qr_in_place`. One more conjunct,
  `M.div_ceil(threads) >= n`, is the whole change; no product is computed differently.
- **The owner is the library, not the harness.** The abort is reachable with no harness at
  all: `qr_in_place` on a 2048x2048 `f64` at 8 threads with its *own*
  `recommended_block_size` (64) aborts, and so does 4096x4096 at 4 and 8 threads, before any
  `householder_apply` call runs. The harness's own fixture is legal: it passes
  `block_size = min(m, n)` to `qr_in_place_scratch`/`qr_in_place`, whose only contract on that
  argument is `> 0` and `Q_coeff.ncols() == min(A.nrows(), A.ncols())`. The campaign's
  256x256 case is exactly this shape, and the assertion fires *inside* `qr_in_place`, not in
  the apply that follows it. The row split is fork-only (`fa22a69`, `c061812`, `c8fea58`);
  upstream's generic apply has only the column split, whose operands cannot mismatch. So this
  is a fork defect and no upstream revision was measured for it; there is nothing to report
  upstream.
- **Rejected: slice the head factor's columns instead of gating.** The correct partial is
  `(V_top^H)[:, lo..top_hi] * C[lo..top_hi, :]`; expressing it needs the `lo x (top_hi - lo)`
  strict-upper block and the `(top_hi - lo) x (top_hi - lo)` unit upper block as two products
  into disjoint row ranges of the partial, plus an explicit zero fill of the rows the head
  leaves empty. That is 15-20 lines and a new invariant ("the tail of every partial stays
  zero") to buy parallel work on the top `n` rows -- a range the ledger's own numbers do not
  show the split paying for. Rejected for now; the gate is one predicate.
- **Not rejected: the row split itself.** Every shape the split was measured on has
  `chunk >= n` and is untouched by the gate: the tall class is a 4096-row target with a 16-wide
  basis (`chunk = 512`, `K = 240`) and the recorded tests use 2048x4 and 4096x16, so none of
  the split's recorded wins is removed. That is a shape argument, not a re-measurement: no
  timing was taken for this change.

## Verification conclusions and constraints

- Establishes: a threaded no-pivot QR is correct again at every block size, and the failing
  shape class computes the same map as the sequential call.
- The focused test is
  `linalg::householder::tests::threaded_panel_narrower_than_the_row_chunk_matches_the_sequential_one`
  (a 256x128 basis with a 64-column target at 4 threads, `chunk = 64 < n = 128`). On the
  parent it fails with the same assertion as the issue (`lhs_ncols = 128`, `rhs_nrows = 64`);
  with the gate it passes. An isolated `qr_in_place` sweep confirms the broader fix:
  256x256, 2048x2048 and 4096x4096 at 1/2/4/8 threads and block sizes 64/128/256 no longer
  abort, where 2048x2048 at 4 and 8 threads aborted on the parent with the *recommended* block
  size.
- Reduces the campaign blocker: the harness's `verify` over the whole declared `householder`
  shape grid (4, 8, 16, 32, 64, 256, 1024x64, 4096x256; `f64`/`c64`; both orders; 1, 4 and 8
  threads; fork and upstream rows) reports "all comparisons within tolerance" and exits 0.
  That is the check `record_run.py` makes before publishing, so a page for a population
  containing `householder_apply` is recordable again.
- Limits: the gate is a *restriction*, not a better top block. A deep reduction with a chunk
  narrower than its panel now runs its top rows on the column path; that shape class has not
  been measured, and no win is claimed or lost for it. The whole-fork gate was
  `cargo test -p t4a-faer --lib` (204 passed, 0 failed, 2 ignored); it carries no pre-existing
  failure. In particular `sparse::linalg::qr::tests::test_numeric_simplicial_qr_1_transpose`,
  which the issue names as failing on 22f23cf, passes here on both 22f23cf and 9de95f7 -- its
  calls are all `Par::Seq`, so it cannot reach the split at all. The second reproducer in the
  issue was therefore misattributed; the isolated `qr_in_place` sweep above is what
  establishes the library ownership.
