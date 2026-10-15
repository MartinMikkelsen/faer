# Row-major adaptation for the column-pivoted QR, both LU flavours, and `to_owned`

Ledger rows: `Row-major col-pivoting QR`, `Row-major LU`, `Row-major to_owned`.
Companion to the `qr_in_place` record (`2026-10-15-qr-row-major-layout.md`); the
copy itself, its tiling, its thread split and its guard for the no-pivoting QR are
documented there.

## Decisions

- **The same copy, at the same layer, for three more entry points.**
  `qr::col_pivoting::factor::qr_in_place` and both `lu_in_place`s now copy a
  row-major matrix into a column-major buffer, factor that, and copy the factor
  back, so the arithmetic is the column-major path's and so are the factor, the
  `tau` matrix, the permutations and the transposition count. The tile and the
  column split are one helper (`linalg::copy_across_layouts`), which the
  no-pivoting factorisation uses too. What each caller guarantees exactly is the
  *adapted* shape's: a tall or square LU, and any column-pivoted shape with more
  than one row and column, is the column-major factorisation bit for bit; a shape
  a guard leaves on the direct path keeps its own kernels and is only equal up to
  rounding.
- **Each caller states its own guard, because the guard is a cost model.**
  `qr_in_place` adapts a tall or square matrix only (the panel is `size`
  reflectors over `m` rows against an `m * n` copy). Partial-pivoting LU uses the
  same one: its unblocked core walks each pivot column twice (the search, then the
  scaling) and its swap stage swaps *inside* a column, so it too is `size * m`
  against `m * n`. The column-pivoted factorisation does not: it computes the norm
  of every one of the `n` columns before it starts
  (`for j in 0..n { A.col(j).norm_l2() }`) and re-reduces them per step when the
  incremental-norm path is closed, so its row-major path touches `m * n` cache
  lines on *every* shape; its guard is `nrows() > 1 && ncols() > 1`, wide
  included. Its gain is not only locality: a strided column norm is a scalar
  `hypot` walk where a contiguous one is a SIMD accumulation.
- **`MatRef::to_owned` is the same defect one level up, and it is fixed for a
  canonical source.** Every composite factorisation begins by copying its argument
  into an owned column-major `Mat`; `Mat::from_fn` fills that column by column, so
  for a row-major source every read steps by the source's row stride. It was
  1.01 ms against 0.057 for `512x512` f64, 6.36 against 0.36 at `1024x1024`, 5.08
  against 0.37 at `4096x256`, and 4.30 against 0.36 at `256x4096` -- about 17.7x,
  17.7x, 14.1x and 11.9x. It now stages a tile when the source's columns are not
  contiguous. The destination is `Mat::zeros`, not an uninitialised allocation:
  `Conjugate::Canonical: ComplexField` already makes that available.
- **Only an already-canonical source takes the tile.** `MatRef::canonical()` is a
  *reinterpretation* of the source's memory, documented as "an unconjugated view",
  while `to_owned` has to apply `Conj::apply`, which is not a reinterpretation for
  a conjugated view. The first attempt gated on strides alone and broke the five
  `sparse::linalg::cholesky` tests, which call `A.adjoint().to_owned()`; the gate is
  `T::IS_CANONICAL`, and a conjugated source keeps the elementwise path. `cloned`
  is left alone for a different reason -- `T: Clone` rules out `Mat::zeros` -- so a
  row-major `cloned` still pays the stride.
- **Dropped, measured, and not carried: the full-pivoting LU copy.**
  `lu::full_pivoting::factor::lu_in_place` already selects the contiguous
  orientation itself (`if mat.row_stride().abs() < mat.col_stride().abs()`), so a
  row-major input is factored through `mat.transpose_mut()`; its pivot search scans
  the whole remaining matrix rather than one column, which is a different cost
  model from the one the other three callers use. Measured with and without the
  hunk at `512x512` f64 one thread: 13.758 against 13.712 C order, 13.449 against
  13.415 F order, and at eight threads 9.419 against 9.395 -- a no-op, because the
  copy and the transposed view cost about the same. Two copies and an allocation
  for nothing is not worth carrying, so the hunk is out and the wide guard is
  stated for partial pivoting only.
- **The buffers come from the heap, not from `stack`,** for the reason the QR
  record gives: the `_scratch` functions are handed the shape and not the strides.

## Verification conclusions and constraints

- Establishes: the row-major penalty in these entry points is the same
  cache-line-per-element panel walk and it is gone. Paired, best of 21, `f64`,
  CPUs `4` / `4-7` / `4-11`, before this change against after it:

  | entry point | shape | t | before | after | F order (after) |
  |---|---|---|---|---|---|
  | `col_pivoting::qr_in_place` | 256x256 | 1 | 58.96 | 1.52 | 1.45 |
  | `col_pivoting::qr_in_place` | 256x256 | 8 | 59.61 | 1.19 | 1.54 |
  | `col_pivoting::qr_in_place` | 1024x64 | 1 | 22.23 | 0.726 | 0.472 |
  | `col_pivoting::qr_in_place` | 1024x64 | 8 | 22.45 | 0.509 | 0.536 |
  | `partial_pivoting::lu_in_place` | 512x512 | 1 | 12.14 | 3.37 | 3.04 |
  | `partial_pivoting::lu_in_place` | 512x512 | 4 | 7.02 | 2.01 | 1.88 |
  | `partial_pivoting::lu_in_place` | 512x512 | 8 | 6.34 | 1.89 | 1.81 |
  | `partial_pivoting::lu_in_place` | 1024x1024 | 1 | 59.80 | 22.91 | 19.89 |
  | `partial_pivoting::lu_in_place` | 1024x1024 | 8 | 22.93 | 8.21 | 6.65 |
  | `partial_pivoting::lu_in_place` | 256x256 | 1 | 1.530 | 0.486 | 0.412 |
  | `MatRef::to_owned` | 512x512 | 1 | 1.006 | 0.275 | 0.057 |
  | `MatRef::to_owned` | 1024x1024 | 1 | 6.357 | 2.007 | 0.399 |
  | `MatRef::to_owned` | 4096x256 | 1 | 5.078 | 2.045 | 0.387 |
  | `MatRef::to_owned` | 256x4096 | 1 | 4.302 | 1.774 | 0.389 |

  `c64` moves with it (`col_pivoting` 256x256 one thread 64.40 -> 3.29, 1024x64
  24.69 -> 1.64; partial-pivoting LU 512x512 one thread 18.08 -> 9.50; `to_owned`
  512x512 one thread 0.878 -> 0.625, 1024x1024 6.234 -> 4.491). The before column is
  the same run's own baseline and the F order is unchanged within its noise, so the
  C/F ratio is 38.8x -> 1.05 at `256x256` one thread for the column-pivoted
  factorisation and 3.60x -> 1.11 for LV, and the shapes the guards leave alone are
  unmoved.
- What the campaign sees of this: its `lu` family and its `qr` family call the
  *composite* (`MatRef::partial_piv_lu()`, `MatRef::qr()`), which reach these
  kernels through `to_owned`, so `to_owned` is what moves them and the four
  `*_in_place` adaptations are not reached at all. A probe at one thread, `f64`,
  `n = 512`, C order, against the recorded page (`bfe4041`, MKL 2026.1):
  `lu` 6.1417 -> 4.156 (eight threads 4.5329 -> 3.033), `qr` 12.5143 -> 11.185,
  `full_piv_lu` 12.95 -> 12.63 (its two copies are the no-op above), `cholesky`
  1.5173 -> 1.517, `eigh` 29.424 -> 29.618, `svd_values` 39.870 -> 39.766. The
  three unchanged families do not use `to_owned`: `Cholesky::new` allocates zeroes
  and copies the requested triangle, and the self-adjoint and SVD entry points hand
  the borrowed input to a routine that copies internally, transposing the view
  first when it is wide. So "every composite pays `to_owned`" is not true, and
  fixing `to_owned` does not close their input-copy penalty.
- Limits: the campaign records no low-level row-major `lu_in_place` and no
  column-pivoted QR at all, so the before/after tables above are this record's own
  driver and not a published page; the campaign's own `fbench verify` does not see
  those two either. It does check the family it declares: `qr_factor`'s check
  re-factors the *original* fixture through the composite rather than inspecting
  what the timed low-level call left, so the compact call's own output is covered
  by the unit test below and not by the campaign. The page's `qr`, `lu` and
  `full_piv_lu` rows are cross-run comparisons against the probe, not paired ones.
- Limits: the same host-variance caveat as the QR record -- absolute times drift
  between invocations, so the C/F ratio and the within-run before/after are the
  readable statements.
- Limits: the `to_owned` tile adds a `Mat::zeros` initialization pass (every entry
  is overwritten before it is read), and it is serial: `to_owned` has no `par`
  argument and is given `Par::Seq`, so at eight threads the copy is a larger share
  of a composite call than it would be if it could use the caller's budget.
- Revision measured: the parent of the commit carrying this change against it;
  upstream `faer-v0.24.4` was not measured.
- Test: `linalg::tests::test_row_major_adaptation_matches_the_column_major_one`
  covers the three new entry points -- the factor at every `(i, j)`, the `tau`
  matrix, every permutation and its inverse, and the transposition count, all bit
  for bit, `f64` and `c64`, sequential and threaded, with a plain row-major view
  and with one whose rows are walked backwards, over `1x1, 2x2, 16x16, 17x15,
  64x64, 300x17, 64x256, 300x8` for the column-pivoted factorisation and `1x1, 2x2,
  16x16, 17x15, 64x64, 128x128, 300x17` for both LU flavours;
  perturbing the copy so that it drops one column of each tile makes it fail. The
  no-pivoting factorisation has its own test beside its `qr_in_place`, and it is
  the one that carries the shapes its guard leaves alone (wide, one row, one
  column, empty), compared up to rounding rather than bit for bit. The fork's whole
  `--lib` suite (207 tests) passes, including the five
  `sparse::linalg::cholesky` tests that the first `to_owned` attempt broke.

## Upstream

All of this is upstream `faer-v0.24.4`'s code, and the fork's rules defer upstream
feedback: nothing was filed or written to an upstream tracker, and the decision is
the maintainer's. What the measurements above establish about upstream, in the three
places it is worth stating:

- **The column-pivoted factorisation is the one upstream neither adapts nor
  warns about.** `faer/src/linalg/qr/col_pivoting/factor.rs` at the parent revision
  `bfe4041` is byte-identical to the pinned upstream checkout, and that file has no
  `perf-warn` at all -- while `qr/no_pivoting/factor.rs`, `lu/{partial,full}_pivoting/factor.rs`
  and `cholesky/bunch_kaufman/factor.rs` all have one. So the 38x row-major penalty
  this record measures is upstream's, and upstream has no message telling a caller to
  transpose first. That is the closest thing here to an upstream oversight rather than
  a stance.
- **Partial-pivoting LU's warning claims something its own code does not deliver.**
  The message reads "LU with partial pivoting prefers column-major or row-major
  matrix. Found matrix with generic strides." The full-pivoting flavour earns the
  "or row-major" half -- it factors a row-major input through `mat.transpose_mut()`,
  which is why the copy measured a no-op there. The partial-pivoting flavour does not:
  it pays 3.6x at `512x512` f64 one thread, all of it inherited, since the fork's only
  change to that file at `bfe4041` is the `par_here` floor inside the recursion and a
  test.
- **The swap stage has a dead branch that looks like the lost half of exactly this
  fix.** `let swap = |mat| { ... if col.row_stride() == 1 { <loop> } else { <loop> } }`:
  both arms are the same loop, in upstream as well as here, so the row-major arm that
  the condition was written for never landed and a row-major matrix swaps elements
  *inside* a column -- one cache line per element, which is part of the 3.6x. Naming it
  is all this record does; the fork's adaptation makes it harmless rather than fixing
  it.
- Two smaller upstream properties, neither a bug: `MatRef::to_owned` fills through
  `Mat::from_fn`'s column-first walk, and `cloned`'s `T: Clone` bound is what rules out
  `Mat::zeros` there; and `MatRef::canonical()` is documented as "an unconjugated view"
  but is a reinterpretation of the source's memory, so it is only usable where the
  conjugation is irrelevant -- upstream's own use is a norm. The first `to_owned`
  attempt in this record used it for values and produced the wrong ones.
