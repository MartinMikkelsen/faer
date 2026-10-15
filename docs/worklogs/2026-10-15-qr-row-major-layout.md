# QR on a row-major matrix: the panel is not layout-aware

Ledger row: `Row-major QR layout adaptation`. Driver: the
`faer-kernels` campaign's C-order cell, where `qr_factor` at `4096x256` was 1.7x
the vendor at one thread and 7.1x at eight. Consumer: `tlinalg-rs#46`.

## Decisions

- **Copy, then factor column-major, then copy back.** `qr_in_place` now adapts a
  matrix whose columns are not the contiguous direction. Nothing in the panel is
  layout-aware: `make_householder_in_place` reads the pivot column
  (`tail.norm_l2()`) and then scales it, and the sequential rank-one update
  beside it walked the block with `z!`, which is one element per cache line when
  the matrix is row-major. LAPACK's row-major interface does exactly this copy;
  faer's `perf-warn` used to tell the caller to do it instead, and that warning is
  now gone because it no longer describes anything.
- **The arithmetic is unchanged, so the result is not a new factorization.** The
  copy lands the same matrix values in a column-major buffer and the factor is
  copied back, so the factorization — reflectors, `tau`, `R`, the rank decision —
  is the column-major one entry for entry. The focused test asserts equality
  across `1x1, 2x3, 3x2, 4x4, 17x15, 16x16, 64x64, 300x17, 17x300, 256x64,
  64x256` at `f64` and `c64`, plus the threaded copy at two budgets; perturbing
  the copy so that it drops one column of each tile makes it fail at `16x16`.
- **The buffer is heap, not `stack`.** `qr_in_place_scratch` is handed the shape
  and not the strides, so it cannot know whether the caller will pass a row-major
  matrix; reserving `m * n` unconditionally would tax every column-major call.
  The fork's two row splits already allocate their partials inside a public call
  and the campaign's own blind-spot list records that boundary as it is, so this
  is the same trade and not a new one.
- **The copy is tiled and split over columns.** Reading column-wise out of a
  row-major buffer is the same stride problem in miniature, so the copy stages
  `16 x 16` tiles and walks each side in its own contiguous direction, and it
  hands `parallelism_degree(par)` disjoint column ranges to rayon when the caller
  asked for threads. Rejected: `zip!(dst, src)`, which is one line and was
  measured at `4096x256` f64 one thread (23.1 ms against 20.4 for the tiled copy,
  on the same paired run); rejected: a `32 x 32` tile, which was worse on the
  large shapes than `16 x 16` (21.9 against 20.4).
- **Dropped, measured, and not carried: routing the unblocked core's single
  reflectors through the shared block-apply.** `apply_block_householder_on_the_left_in_place_with_conj`
  picks its kernel from the layout — the single-reflector SIMD path for a
  column-major matrix, and the generic path's two packing products for a
  row-major one — so it is a real fix for the same defect at the kernel level. It
  is worth 6.8x on an *unblocked* `4096x256` row-major QR (320.2 -> 46.9 ms) and
  about 2% on the column-major blocked path it would also touch (7.22 -> 7.05 ms
  at eight threads), but the adaptation above means the unblocked core never sees
  a row-major matrix through `qr_in_place` at all, so the 2% is the whole of what
  is left and a second change in one experiment is not worth it. It is a
  candidate for its own row.
- **Not taken: making the column-major copy a view.** A row-major `m x n` buffer
  *is* a column-major `n x m` buffer, so `A.transpose()` is free and contiguous —
  but that view is `A^T`, whose factorization is `A`'s LQ-style factor
  (`R^T` in the lower trapezoid, reflectors stored in rows), not `A`'s QR. The
  values are the same set, the storage convention is not, and the caller's
   contract is `faer`'s QR convention.

## Verification conclusions and constraints

- Establishes: the row-major `qr_factor` no longer has a layout-specific defect.
  Paired, same process, best of 21, second of two invocations, CPUs `4` / `4-7` /
  `4-11`, `bfe4041` against `044f947`, `qr_in_place` through faer's own
  `recommended_block_size` (ms, C order; the F-order column is the same run's
  unchanged path):

  | shape | dtype | t | before | after | F order (after) |
  |---|---|---|---|---|---|
  | `4096x256` | f64 | 1 | 47.83 | 20.44 | 15.85 |
  | `4096x256` | f64 | 4 | 42.85 | 10.43 | 8.33 |
  | `4096x256` | f64 | 8 | 42.96 | 8.45 | 7.26 |
  | `4096x256` | c64 | 1 | 106.87 | 65.92 | 55.82 |
  | `4096x256` | c64 | 8 | 90.47 | 21.35 | 18.11 |
  | `1024x64` | f64 | 1 | 1.090 | 0.628 | 0.463 |
  | `1024x64` | f64 | 8 | 2.919 | 0.683 | 0.622 |
  | `512x512` | f64 | 1 | 9.44 | 5.89 | 4.71 |
  | `512x512` | f64 | 8 | 12.73 | 3.36 | 3.09 |
  | `256x256` | f64 | 1 | 1.455 | 0.950 | 0.814 |
  | `256x256` | f64 | 8 | 2.315 | 0.917 | 0.849 |

  The row-major path also scales now, which it did not before: `4096x256` f64 is
  20.4 / 10.4 / 8.5 ms at one, four and eight threads against 47.8 / 42.9 / 43.0
  before, so the deficit was not the copy and not the threads but one serial
  panel. Against the campaign's vendor column for the same C-order cell, MKL is
  27.73 / 8.66 / 8.20 ms (`bfe4041`, MKL 2026.1, `faer-kernels` page), so this
  moves `4096x256` f64 from 1.72x / 4.95x / 5.24x behind to 0.74x / 1.20x / 1.03x.
- Limits: absolute times on this host move by up to 40% between invocations for
  an *unchanged* path (`512x512` f64 F order four threads: 4.05 and 2.91 ms in
  the two paired runs), so the before/after column and the within-run C/F ratio
  are the numbers to read, not a cross-run difference. The C/F ratio is the
  stable statement and it is 3.00 -> 1.29 at `4096x256` f64 one thread and
  5.57 -> 1.16 at eight.
- Limits: only the no-pivoting factorization is adapted.
  `qr::col_pivoting::factor::qr_in_place` still walks a row-major matrix with a
  column stride, and its `perf-warn` is still in place; nothing in the campaign
  measures it. Only `qr_in_place` itself is adapted, so the residual overhead on
  the C-order cells is the copy: 1.1-1.4x over the F-order path at the shapes
  above, which is 4.6 ms of the `4096x256` f64 eight-thread cell.
- Limits: the copy is one `m * n` allocation per call, freed at the end, so a
  repeated small row-major QR pays a malloc and a fresh-page fault each time; at
  `m * n` above the allocator's mmap threshold those pages are also faulted in on
  first touch. Nothing here measures an allocation-free variant.
- Revision measured: fork `bfe4041` (parent) against `044f947`; upstream
  `faer-v0.24.4` is not implicated (`qr_in_place` on a row-major matrix is this
  fork's cell and upstream's panel behaves the same, but no upstream revision was
  measured for this record).
- Test: `qr::no_pivoting::factor::tests::test_row_major_factor_is_the_column_major_one`,
  and the fork's `linalg::` suite (151 tests) passes at the revision above.
