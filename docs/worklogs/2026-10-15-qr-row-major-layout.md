# QR on a row-major matrix: the panel is not layout-aware

Ledger row: `Row-major QR layout adaptation`. Driver: the `faer-kernels`
campaign's C-order cell, where `qr_factor` at `4096x256` was 1.7x the vendor at
one thread and 7.1x at eight. Consumer: `tlinalg-rs#46`.

## Decisions

- **Copy, then factor column-major, then copy back.** `qr_in_place` now adapts a
  matrix whose columns are not the contiguous direction. Nothing in the panel is
  layout-aware: `make_householder_in_place` reads the pivot column
  (`tail.norm_l2()`) and then scales it, and the sequential rank-one update
  beside it walked the block with `z!`, which is one element per cache line when
  the matrix is row-major. LAPACK's row-major interface does exactly this copy;
  faer's `perf-warn` used to tell the caller to do it instead, and that warning is
  gone because it no longer describes anything.
- **The arithmetic is unchanged, so the result is not a new factorization.** The
  copy lands the same matrix values in a column-major buffer and the factor is
  copied back, so for an *adapted* shape the factorization — the reflectors,
  `tau`, `R`, the rank decision — is the column-major one, bit for bit. A shape a
  guard leaves on the direct path is not adapted and keeps its own kernels, so its
  two orders agree only up to rounding. The focused test asserts the first group
  bit for bit and the second up to rounding, across `0x0, 3x0, 0x3, 1x7, 7x1, 2x3, 3x2, 4x4, 17x15,
  16x16, 64x64, 300x17, 17x300, 256x64, 64x256` at `f64` and `c64`, sequential and
  threaded, and again with a view whose rows are walked backwards; a variant with
  an exactly zero leading column covers the one place the unblocked core stores a
  reflector's essential part in a column other than the one it eliminates.
  Perturbing the copy so that it drops the last column of each tile makes it fail.
- **The buffer is heap, not `stack`.** `qr_in_place_scratch` is handed the shape
  and not the strides, so it cannot know whether the caller will pass a row-major
  matrix; reserving `m * n` unconditionally would tax every column-major call.
  The fork's two row splits already allocate their partials inside a public call
  and the campaign's own blind-spot list records that boundary as it is, so this
  is the same trade and not a new one.
- **Only a tall or square matrix is adapted, and the guard is about the panel's
  own work.** The panel touches about `size * (m - size / 2)` elements one cache
  line at a time -- one pivot column per reflector -- against `m * n` elements
  the copy walks contiguous twice. For a tall or square matrix those are within a
  factor of two of each other and the stride is worth far more than that. A wide
  one is `size == m` over `n` columns, so `256 x 4096` would copy eight megabytes
  to save a couple of per cent of the panel: measured at one thread it is 16.9 ms
  before the adaptation and 20.7 with it, worse by 22%, and with the guard it is
  back at 17.2. A single row and a single column are the limit cases of the same
  thing (`qr_in_place` eliminates at most one column of a one-row matrix), and a
  `m x 1` column vector has a unit row stride and never reaches the guard at all.
  The wide matrix's own C-order penalty is *not* this one and is not closed here.
- **The copy is tiled, and each side is read and written along its own contiguous
  direction.** Reading column-wise out of a row-major buffer is the same stride
  problem in miniature, so the copy stages `16 x 16` tiles: it reads the tile
  along whichever side is row-major and writes it along whichever is column-major,
  and each tile is one L1 transposition. Getting that round the wrong way for the
  copy *back* is worth 3-11% (`512x512` f64 one thread: 5.89 -> 5.24 ms). When the
  caller asked for threads the copy splits the columns over disjoint
  column ranges. Rejected: `zip!(dst, src)`, one line, measured at `4096x256` f64
  one thread at 23.1 ms against 20.4 for the tiled copy on a paired run; rejected:
  a `32 x 32` tile, worse on the large shapes than `16 x 16` (21.9 against 20.4).
- **Dropped, measured, and not carried: routing the unblocked core's single
  reflectors through the shared block-apply.**
  `apply_block_householder_on_the_left_in_place_with_conj` picks its kernel from
  the layout — the single-reflector SIMD path for a column-major matrix, and the
  generic path's two packing products for a row-major one — so it is a real fix
  for the same defect at the kernel level. It is worth 6.8x on an *unblocked*
  `4096x256` row-major QR (320.2 -> 46.9 ms) and about 2% on the column-major
  blocked path it would also touch (7.22 -> 7.05 ms at eight threads), but the
  adaptation above means the unblocked core never sees a row-major matrix through
  `qr_in_place` at all, so the 2% is the whole of what is left and a second change
  in one experiment is not worth it. It is a candidate for its own row.
- **Not taken: making the column-major copy a view.** A row-major `m x n` buffer
  *is* a column-major `n x m` buffer, so `A.transpose()` is free and contiguous —
  but that view is `A^T`, whose factorization is `A`'s LQ-style factor (`R^T` in
  the lower trapezoid, reflectors stored in rows), not `A`'s QR. The values are the
  same set, the storage convention is not, and the caller's contract is faer's QR
  convention.

## Verification conclusions and constraints

- Establishes: the row-major `qr_factor` no longer has a layout-specific defect.
  Paired, same process, best of 21, second of two invocations, CPUs `4` / `4-7` /
  `4-11`, `qr_in_place` through faer's own `recommended_block_size` at `bfe4041`
  (before) and at the commit carrying this change (after), ms in C order; the F
  order column is the same run's always-column-major path.

  | shape | dtype | t | before | after | F order (after) |
  |---|---|---|---|---|---|
  | `4096x256` | f64 | 1 | 47.83 | 19.44 | 15.77 |
  | `4096x256` | f64 | 4 | 42.85 | 10.03 | 8.35 |
  | `4096x256` | f64 | 8 | 42.96 | 8.39 | 7.23 |
  | `4096x256` | c64 | 1 | 106.87 | 64.34 | 55.78 |
  | `4096x256` | c64 | 8 | 90.47 | 21.00 | 18.13 |
  | `1024x64` | f64 | 1 | 1.090 | 0.587 | 0.462 |
  | `1024x64` | f64 | 8 | 2.919 | 0.678 | 0.628 |
  | `512x512` | f64 | 1 | 9.44 | 5.24 | 4.68 |
  | `512x512` | f64 | 8 | 12.73 | 3.23 | 3.03 |
  | `256x256` | f64 | 1 | 1.455 | 0.925 | 0.807 |
  | `256x256` | f64 | 8 | 2.315 | 0.905 | 0.860 |

  and, for the wide class the guard deliberately leaves alone, the same run
  before -> after (`m` under `n`, so no adaptation either way):

  | shape | dtype | t | before | after | F order (after) |
  |---|---|---|---|---|---|
  | `256x4096` | f64 | 1 | 16.90 | 17.20 | 16.90 |
  | `256x4096` | f64 | 8 | 5.21 | 5.06 | 3.22 |
  | `256x4096` | c64 | 1 | 60.23 | 60.54 | 64.04 |
  | `64x1024` | f64 | 1 | 0.587 | 0.591 | 0.334 |

  The row-major path also scales now, which it did not before: `4096x256` f64 is
  19.4 / 10.0 / 8.4 ms at one, four and eight threads against 47.8 / 42.9 / 43.0
  before, so the deficit was not the copy and not the threads but one serial
  panel. The C/F ratio — the statement that survives the host's variance — is
  3.03 -> 1.23 at `4096x256` f64 one thread and 5.94 -> 1.16 at eight, and the
  wide class, which it does not touch, stays where it was (1.02 and 1.57 at
  `256x4096` f64, one and eight threads) and stays within 5% of the vendor.
- Limits: absolute times on this host move by up to 40% between invocations for
  an *unchanged* path (`512x512` f64 F order four threads: 4.05 and 2.87 ms in the
  two paired runs), so what the before/after column claims is a large, consistent
  effect and not a decimal; the within-run C/F ratio is the stable number. The
  vendor comparison the campaign page carries (`bfe4041`, MKL 2026.1,
  `faer-kernels`: `27.73 / 8.66 / 8.20` ms for the same C-order cell) is a
  cross-run comparison: it moves `4096x256` f64 from 1.72x / 4.95x / 5.24x behind
  to 0.70x / 1.18x / 1.02x on paper, and the paired ratio above is the evidence
  for the direction and the size of that move, not for its third digit.
- Limits: only the no-pivoting factorization is adapted, only for a tall or
  square matrix, and only when the caller passes a row-major matrix to
  `qr_in_place` itself. `MatRef::qr()`'s own `A.to_owned()` already hands the
  factorization a column-major copy, so the high-level path is unaffected.
  `qr::col_pivoting` and both LU flavours were, in a companion row; the shapes
  each guard leaves alone are covered by the records beside this one.
- Limits, recorded and not taken: the wide class keeps a C-order penalty at four
  and eight threads that this is not (`256x4096` f64: 6.28 against 4.51 at four,
  5.06 against 3.22 at eight; one thread is parity). It is not a column walk the
  copy would remove — the panel there is `m` reflectors over a `256`-element
  column — and it is not a vendor lag: the campaign's MKL C-order column for that
  cell is 12.10 / 4.82, so the fork is at 0.53 / 1.05 of it. Not measured
  further.
- Limits: the residual overhead on the C-order cells is the copy and nothing
  isolates it. At eight threads it is the whole end-to-end difference between the
  two orders, 1.16 ms at `4096x256` f64 and 0.05 ms at `1024x64`; at one thread it
  is 3.67 and 0.13 ms. A copy that avoided the per-call allocation (the buffer is
  one `m * n` `Vec` per call, freed at the end, so a small repeated row-major QR
  pays a malloc and, above the allocator's mmap threshold, fresh-page faults) or
  that used non-temporal stores is not measured here.
- Revision measured: fork `bfe4041` (parent) against the commit carrying this
  change; upstream `faer-v0.24.4` is not implicated and no upstream revision was
  measured for this record.
- Test: `qr::no_pivoting::factor::tests::test_row_major_factor_is_the_column_major_one`,
  and the fork's `linalg::` suite (151 tests) passes at the revision above. The
  campaign's own gate agrees: `fbench verify` at rev `e3d388b` reports
  `all comparisons within tolerance` for `qr_factor` and `qr` over every regime
  that carries them -- `linalg-small` (4..64) at 8T, `linalg-medium` (256) at 1T
  and 8T, `linalg-large` (512) at 4T, `linalg-tall` (`1024x64`, `4096x256`) at 1T
  and 8T, `linalg-wide` (`64x1024`, `256x4096`) at 1T -- in both dtypes, both
  orders and all three engine rows. That is a check of each provider's
  reconstruction residual and compact-`Q` orthonormality, run against the
  harness rather than the test above.
