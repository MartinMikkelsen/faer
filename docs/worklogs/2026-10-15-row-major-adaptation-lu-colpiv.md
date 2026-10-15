# Row-major adaptation for the column-pivoted QR and both LU flavours

Ledger rows: `Row-major col-pivoting QR`, `Row-major LU`. Companion to the
`qr_in_place` record (`2026-10-15-qr-row-major-layout.md`); the copy itself, its
tiling and its thread split are documented there.

## Decisions

- **The same copy, at the same layer, for three more entry points.** A row-major
  matrix is copied into a column-major buffer, factored, and copied back, so the
  arithmetic is the column-major path's and the factor, the `tau` matrix and the
  permutations are the column-major ones. The tile and the column split are one
  helper now (`linalg::copy_across_layouts`), which the no-pivoting factorisation
  uses too.
- **Each caller states its own guard, because the guard is a cost model.**
  `qr_in_place` adapts a tall or square matrix only (the panel is `size`
  reflectors over `m` rows against a `m * n` copy). `lu_in_place`, partial and
  full, uses the same one: its unblocked core walks each pivot column twice (the
  search, then the scaling) and its swap stage swaps *inside* a column, so it is
  also `size * m` against `m * n`. The column-pivoted factorisation does not:
  it computes the norm of every one of the `n` columns before it starts
  (`for j in 0..n { A.col(j).norm_l2() }`), and its incremental-norm fast path is
  gated on `A.row_stride() == 1` and unwraps a column-major view, so its row-major
  path touches `m * n` cache lines on *every* shape. Its guard is therefore
  `nrows() > 1 && ncols() > 1`, wide included.
- **The buffers come from the heap, not from `stack`,** for the reason the QR
  record gives: the `_scratch` functions are handed the shape and not the strides.
- **Not taken: fixing `MatRef::to_owned`/`cloned` instead of, or as well as,
  these.** Every composite entry point (`MatRef::lu`, `.cholesky`, `.qr`, `.eigh`,
  `.svd_*`) already starts by copying its argument into an owned column-major
  `Mat`, so it never reaches a row-major `*_in_place` at all -- and that copy is
  itself 17x slower for a row-major source than a column-major one (512x512 f64:
  1.01 ms against 0.057; 1024x1024: 6.36 against 0.36; 4096x256: 5.08 against
  0.36). It is the same defect and it would move the campaign's composite cells,
  but the destination is an owned `Mat`, `Mat::from_fn` fills it column by column
  so the source read is the strided one, `Mat::zeros` is unavailable for the
  `T: Clone` bound that `cloned` carries, and an uninitialised owned `Mat` with a
  panic-safety guard is a change to a core allocator. It is its own row; the
  measurement above is the whole of what this record establishes about it.
- **Not taken: gating the LU adaptation's copy on `nrows() > ncols()`.** The
  recursion supports `m < n` (it solves the trailing columns afterwards), so the
  guard is written as the QR one rather than as an equality; nothing in the
  campaign measures a wide LU, and the harness refuses one for both providers.

## Verification conclusions and constraints

- Establishes: the row-major penalty in these three entry points is the same
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

  `c64` moves with it (`col_pivoting` 256x256 one thread 64.40 -> 3.29, 1024x64
  24.69 -> 1.64; partial-pivoting LU 512x512 one thread 18.08 -> 9.50). The
  before column is the same run's own baseline and the F order is unchanged
  within its noise, so the C/F ratio is 38.8x -> 1.05 at `256x256` one thread for
  the column-pivoted factorisation and 3.60x -> 1.11 for LU, and the wide shapes
  the guards leave alone are unmoved.
- Limits: the campaign's `lu`, `full_piv_lu`, `cholesky`, `solve`, `eigh` and
  `svd_*` cells call the *composite* (`MatRef::lu()` and friends), which copies
  its argument into an owned column-major `Mat` first and so never reaches these
  entry points. Nothing in the campaign records a low-level row-major `lu_in_place`
  or a column-pivoted QR at all, so the numbers above are this record's own driver
  and not a published page; the campaign's own `fbench verify` does cover both
  through the composite and passes (see below).
- Limits: the same host-variance caveat as the QR record -- absolute times drift
  between invocations, so the C/F ratio and the within-run before/after are the
  readable statements.
- Revision measured: the parent of the commit carrying this change against it;
  upstream `faer-v0.24.4` was not measured.
- Test: `linalg::tests::test_row_major_adaptation_matches_the_column_major_one`
  covers all three entry points -- factor, `tau` and every permutation, bit for
  bit, `f64` and `c64`, sequential and threaded, over `1x1, 2x2, 16x16, 17x15,
  64x64, 300x17, 64x256` for the column-pivoted factorisation and `1, 2, 16, 17,
  64, 128` for both LU flavours; perturbing the copy so that it drops one column
  of each tile makes it fail. The fork's `linalg::` suite (152 tests) passes, and
  `fbench verify` reports `all comparisons within tolerance` for `linalg-small`,
  `linalg-medium`, `linalg-large`, `linalg-tall` and `linalg-wide` at one thread
  in both dtypes and both orders.
