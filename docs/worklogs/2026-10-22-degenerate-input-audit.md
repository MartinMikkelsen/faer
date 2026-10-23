# Degenerate inputs in the dense factorizations (#33-#38)

Six reported defects where a finite but degenerate input (an exactly zero pivot
column, a zero matrix, an exactly zero scale, a subnormal scale, a reused output
buffer) produced NaN factors, a wrong factor, or a silently wrong spectrum. Every
fix follows what the reference implementation does at the same point rather than
inventing a policy; the reference for each is named in the code comment.

## Decisions

- **LU partial pivoting (#38)**: a zero pivot column is a singular factor, not a
  division. OpenBLAS's own `dgetf2_k.c` skips the column scaling when the pivot is
  exactly zero (`if (temp1 != ZERO) ... else info = j+1`), and reference `dgetf2`
  divides elementwise below `SFMIN` so that a subnormal pivot's reciprocal does
  not overflow. Both branches are taken: the multipliers below a zero pivot stay
  zero, `U` keeps the singularity, and the factors still reconstruct the input.
  The subnormal branch rescales both operands by `max_positive` (an exact power
  of two) before dividing, because `num-complex`'s `Div` -- unlike the Fortran
  complex division `dgetf2` relies on -- forms the *squared* denominator and
  underflows it for a subnormal pivot. The blocked path needed nothing: its panel
  solve is a *unit* triangular solve, which already ignores the stored diagonal.
- **Column-pivoted QR (#35)**: LAPACK does not scale the matrix at all
  (`dgeqp3` relies on `dnrm2`/`dlarfg` being scale-safe), so this scaling is
  faer's own; a zero matrix simply has no scale, and `dlarfg` treats an all-zero
  column as `H = I` (`XNORM == 0 -> TAU = 0`). The factor now stays unscaled
  instead of multiplying every entry by `+inf`.
- **`norm_l2` strided fallback (#36)**: the fallback accumulated the norm with
  `hypot` (the same scale/`ssq` recurrence reference `dnrm2` uses) and then
  squared that accumulator *unscaled* before forming the three range sums, which
  overflowed or underflowed a finite norm. Squaring after scaling each of the
  three sums restores the contract the SIMD path already implements, so the
  strided and contiguous paths agree again.
- **Fixing #36 exposes two reciprocal overflows that the bug had been hiding.**
  Both are `dlarfg`'s situation -- a normaliser whose reciprocal overflows -- but
  the fix is an adaptation, not `dlarfg`'s arithmetic: `dlarfg` divides by
  `SFMIN = safmin/eps`, rescales by its reciprocal and *recomputes* the norm,
  while these two sites rescale the operand by the exact power of two `2^52`
  (`2^-1074 * 2^52 = 2^-1022` is still normal, so the scaled reciprocal is
  finite) and keep their existing single-pass normalisation. The row
  normalisation in `bidiag_in_place` (`norm.recip()`) and the complex phase
  normalisation in `compute_bidiag_cplx_svd` (`norm1.recip()`) overflowed for a
  subnormal scale and produced `0 * inf` NaNs downstream; the complex one is
  also the reason the complex path cannot simply divide elementwise, since its
  quotient squares the denominator. Reachability is not new: a column-major
  (contiguous) complex input reached the same NaN through the SIMD path, which
  never had the #36 defect. An SVD whose *whole* scale is subnormal is still
  unsupported: the bidiagonal iteration's own `max.recip()` overflows there, and
  that is left as it was, outside these issues.
- **LBLT reconstruction (#34)**: reference `dsytrf` keeps `D` inside the packed
  factor; this fork splits it out (`L` has a unit diagonal, `B_diag`/`B_subdiag`
  hold the block diagonal) for the high-level wrapper, so `reconstruct` has to
  read its `diagonal` argument for both block sizes, not `L`'s diagonal. The
  existing low-level test passed `L.diagonal()` as that argument and so could
  not see the mismatch.
- **GEVD vector outputs (#37)**: LAPACK's `dgges` initialises `VSL`/`VSR` with
  `dlaset('Full', n, n, ZERO, ONE, ...)` before `dhgeqz` accumulates into them.
  This wrote only the diagonal, so a reused buffer's off-diagonal entries
  survived into the eigenvectors.
- **Divide-and-conquer SVD zero block (#33)**: the two-row storage keeps rows
  `0` and `n` of a subproblem's `U`, and reference `dlasd1` documents that block
  as the subproblem's own singular vectors. For a zero block that is the
  identity, so the bottom row's boundary entry is `1`; writing `0` dropped the
  nonzero singular value of the parent's merge and left only the values-only and
  U-only callers broken (a `V` request takes the full-storage path, which was
  already right).

## Verification conclusions and constraints

- Each defect has one focused regression test -- eight in all, at four dtypes where
  the guard is dtype-dependent (two dtypes where the fixture is real-only) -- and each fails on the unfixed code and passes
  after it (checked by reverting each fix individually): `linalg::norm_l2::tests::
  test_norm_l2_strided_range`, `linalg::lu::partial_pivoting::factor::tests::
  zero_pivot_column_reconstructs` (zero *and* subnormal pivot),
  `linalg::qr::col_pivoting::factor::tests::zero_matrix_reconstructs`,
  `linalg::cholesky::lblt::reconstruct::tests::
  test_reconstruct_separate_diagonal`, `linalg::gevd::t::reused_eigenvector_buffers`,
  `linalg::svd::tests::test_zero_block_at_split`,
  `linalg::svd::tests::test_extreme_scale` for the scale-aware consequence of
  #36 (`NoConvergence` at `1e160`, a one-third relative reconstruction error at
  `1e-300`, both before the change), and
  `linalg::svd::tests::test_rank_one_subnormal_cascade`, which is the only
  fixture that drives the two subnormal normalisations (a rank-one matrix
  cascades into subnormal bidiagonal entries; without the guards it ends in
  `NoConvergence`).
- Dtype coverage per fixture, because most guards compare against a dtype-sized
  constant (`min_positive`, `max_positive`, `eps`) or rescale by an
  exponent-dependent power of two: `f32`/`c32`/`f64`/`c64` for `norm_l2`, the LU
  zero and subnormal pivot, the col-pivoted zero matrix and the SVD
  zero-block/scale/cascade fixtures; `f32`/`f64` for the LBLT reconstruction and
  the GEVD reused buffer (both take a real scalar type: `reconstruct` is generic
  but the fixture is not instantiated for complex, and `gevd_real` is the real
  entry point, so the complex `gevd_cplx` path is covered only by the existing
  suite). The non-column-major layouts the issues also report were not re-run
  under test.
- The six reported reproductions also pass through the public wrappers
  (`singular_values`, `lblt().reconstruct()`, `col_piv_qr()`, `row.norm_l2()`,
  `gevd_real`, `partial_piv_lu()`) at `f64` (and `c64` where the issue reports it).
- Workspace gates: 227 library tests pass (2 existing ignored) and 37 doctests
  pass. `cargo clippy --all-targets` reports nothing in `faer/src`.
- Correctness only: no timing was recorded and none is claimed. Every change adds
  at most one comparison per pivot column / per diagonal entry, or one fill per
  requested vector output; no measured path changes its algorithm or its
  parallel schedule. The existing `test_special`, `test_real`, `test_cplx`,
  `test_tiny_diagonal_at_split` and LU/QR suites are unchanged and pass, which is
  the guard against the added branches moving anything else.
- Known limitation, unchanged by this work: `norm_l2`'s contract is three *squared*
  range sums, so a norm above `sqrt(max_positive)` (~9e307 for `f64`) still has no
  representable middle sum and comes back as `inf` — in the strided fallback and in
  the SIMD path alike, since both form the same sums. Extending it past that point
  means returning a scaled pair instead of squared sums, which is a contract change
  and not part of #36's report (its cases are `1e160` and `1e-300`, both inside the
  range that is now correct).
- Known limitation, diagnosed but deliberately not changed: a matrix whose *whole*
  scale is subnormal loses its entries before the SVD proper, in the shared
  Householder kernel -- `householder::make_householder_imp` treats
  `|head| < min_positive` as zero (`*head = zero()`), where reference `dlarfg` only
  zeroes the *tail* when `XNORM == 0` and rescales a small `BETA` instead of dropping
  `ALPHA`. Measured on the branch: `f64` `2^-1030 * [[1, 0.5], [0, 1]]` collapses to a
  zero spectrum, and returns `1.113e-310 / 6.786e-311` (the correct `1.2808 * s` and
  `0.7808 * s`, exact reconstruction) only when that zeroing is disabled locally. It is
  not touched here because it is a denormal policy decision in a kernel shared by
  every QR, LU and bidiagonalisation caller (denormals also cost 10-100x on x86), not a
  defect of these six reports. The two `max.recip()` normalisations inside the
  bidiagonal SVD that the same investigation exposed were left as they were for the
  same reason: with the head zeroing in place no test reaches a subnormal `max`, and
  rescaling there alone turns `NoConvergence` into a *wrong* finite spectrum rather
  than a right one.
- What remains unverified: the LAPACK/OpenBLAS comparisons are read from their
  sources (reference `dgetf2`, `dgeqp3`, `dlarfg`, `dsytrf`, `dgesvd`/`dlasd1`,
  `dgges`, and OpenBLAS's `getf2_k.c` and `nrm2` kernels) as the stated intent of
  each fix, not as a numerical cross-check against a linked OpenBLAS build.
