# Reported defects in the dense factorizations and solvers (#40-#45)

Six reported defects where a finite, non-degenerate input produced a NaN, an
overflowed value, a non-orthogonal factor or a wrong reconstruction. Each fix is
local to the site the report names; none adds an algorithm, a parameter or an
API change.

## Decisions

- **SVD least-squares solve (#40)**: `solve_lstsq` reciprocated every singular
  value, so an exactly zero singular value turned its (zero) projected
  right-hand side into NaN. An exactly zero direction now contributes zero. The
  alternative -- the rank tolerance `pseudoinverse` next to it uses
  (`eps * max(m, n) * smax`), i.e. the truncated solve `dgelsd` performs -- was
  rejected: it would silently drop a representable direction such as
  `diag(1, 2^-54)`, whose exact finite solution `[0, 2^54]` this solver returns
  today, and that is a contract change no report asks for. The pseudoinverse
  keeps its own tolerance; `solve_lstsq` only stops dividing by zero.
- **Full SVD factors (#41)**: the tall (and, after transposition, wide) QR path
  fills the bottom rows of `U` with zeros and the trailing block with an
  identity *before* applying the QR Householder sequence, because the extra
  columns of a full `U` are `Q * [0; I]`. The top-right block (rows `0..n`,
  cols `n..m`) was left at whatever the caller's buffer held, so `Q` acted on
  stale entries. It is zeroed with the rest of the pre-image now; reference
  `dorgqr` starts from the identity for the same reason.
- **2x2 self-adjoint eigenvalues (#42)**: the shortcut formed the trace as
  `(a + d) * 0.5`, which overflows for a finite spectrum near the
  floating-point limit even though the average and both eigenvalues are
  representable. The trace is now formed as `a * 0.5 + d * 0.5` *only when
  `a + d` is not finite*; everywhere else the direct form stays, because
  halving is only exact down to the normal range -- `q = f64::from_bits(1)` is
  `q/2 + q/2 = 0`, while `(q + q) * 0.5 = q`. The sum of two halves cannot
  overflow when the average is representable. This is not a reference-behaviour
  claim: LAPACK's `dlaev2` forms `A + D` as well. The largest eigenvalue
  `t1 + t0` can still overflow, which is the legitimate case of a spectrum that
  has no finite representation.
- **Scalar triangular solve (#43)**: the 1x1 base case multiplied by the
  reciprocal of the pivot, and `1 / d` overflows for a subnormal `d` even when
  `d * x = b` has the exactly representable solution `d / d = 1`. A *real*
  pivot now divides elementwise (`x / x` is exactly one for every finite nonzero
  `x`) when the reciprocal is not finite, and keeps the existing multiply
  otherwise. Both restrictions are deliberate rather than cautious:
  - The finiteness condition keeps every previously working case bit-identical.
    faer's `recip` maps a non-finite value to zero, and the block Householder
    apply relies on that: its 1x1 solve of the compact `T` factor is reached
    with a non-finite entry in the existing complex
    `hessenberg`/`tridiag`/`bidiag` tests, where an unconditional division
    turned the previously finite result into NaN and failed eight complex tests.
  - The real-only restriction is `num_complex`'s division, which squares the
    denominator components: for `c64` `(1e-310, 0)` those squares underflow to
    zero and the same division gives `0/0 = NaN`, i.e. *worse* than the
    reciprocal path's `inf`. Robust complex division needs `zladiv`-style
    component scaling, which is a separate change (see below).
  The upper-triangular entry point reaches the same base case through its
  reversed view.
- **Empty self-adjoint EVD (#44)**: the scratch query asked for the Householder
  sequence's workspace with `n - 1`, which underflows for `n == 0` and panics
  before a `Result` exists. It uses `saturating_sub(1)`; with the scratch fixed,
  the 0x0 decomposition returns `Ok`, an empty spectrum and a 0x0 eigenvector
  matrix.
- **Pivoted LLT early rank (#45)**: the rank exit happens before the failing
  column is factorized, so the trailing columns of `a` still held the input
  matrix. `reconstruct` reads the whole lower triangle as `L L^H` and has no
  rank argument, so those stale entries were multiplied in. The trailing
  columns are zeroed at the exit, which makes the stored factor the rank-`rank`
  one whose `L L^H` is the truncated matrix. The retained columns keep their
  multipliers below the diagonal -- `reconstruct` then forms
  `L(:, 0..r) L(:, 0..r)^H`, not a leading block's `L_r L_r^H`, and that is what
  makes the truncation come out at the right scale (a diagonal entry near `1`,
  not near `3` for the reported `ones + eps I`). The failing pivot's diagonal
  entry, which the unfixed code recorded at `a[(rank, rank)]`, is now zero as
  part of the cleared columns; nothing in the crate reads it (`solve` and
  `inverse` ignore the rank, and the CG preconditioner reads the leading
  `rank x rank` block).

## Verification conclusions and constraints

- One focused regression test per report, each failing on the unfixed code and
  passing after it (checked by reverting the six production hunks together):
  `linalg::solvers::tests::svd_lstsq_rank_deficient` (compares against
  `pseudoinverse() * b` *and* the exact `[2, 0]`),
  `linalg::svd::tests::full_factors_ignore_old_output_entries` (tall `4x2` and
  wide `2x4`, both requested factors prefilled with ones; checks `U^T U = I`,
  `V^T V = I` and reconstruction for each),
  `linalg::evd::scale_and_empty_tests::two_by_two_finite_spectrum_at_the_range_limit`
  (`f64` `1e308` and `f32` `1e38`) plus
  `linalg::evd::scale_and_empty_tests::two_by_two_subnormal_diagonal` (`f64`
  `from_bits(1)`, which the halve-first form rounds to zero),
  `linalg::evd::scale_and_empty_tests::empty_matrix`,
  `linalg::triangular_solve::tests::tiny_pivot_scalar_solve` (`f64` `1e-310` and
  `f32` `1e-44`, each with and without the implicit conjugation) and
  `linalg::cholesky::llt_pivoting::reconstruct::tests::test_early_rank_reconstruct`
  (the reported `ones + eps I`, rank 1, every stored lower-triangular entry of
  the reconstruction within `1e-12` of `1`).
- Fixtures are at `f64` except the two whose guard is a dtype-sized quantity
  (`f32`/`f64` for the 2x2 range limit and the scalar pivot). The other four
  fixes introduce no dtype-sized constant: #41 and #45 are pure storage
  clearing, #44 an unsigned subtraction, and #40 scales its tolerance by
  `eps::<T::Real>()` by construction.
- The six reported reproductions pass through the public entry points
  (`svd().solve_lstsq`, low-level `svd` with full factors, `self_adjoint_eigen`,
  `solve_lower_triangular_in_place`, and `llt_pivoting` factor + `reconstruct`).
- Workspace gates: 232 library tests pass with the two existing ignored, and 37
  doctests; `cargo clippy --all-targets` reports nothing in `faer/src`. The two
  suite tests that spend many minutes under the suite's parallel load
  (`linalg::evd::schur::complex_schur::tests::test_n`,
  `matrix_free::eigen::tests::test_arnoldi_cplx`) were filtered out of the batch
  run and re-run on their own, where they pass in about two seconds; neither
  touches a changed path.
- Correctness only: no timing was recorded and none is claimed. Every fix either
  replaces one arithmetic operation with an equally cheap one or adds one fill
  of a buffer the caller already supplied; no measured path changes its
  algorithm, blocking or parallel schedule.

## What remains unverified

- The reference comparison is the *intent* read from `dgelsd`, `dorgqr` and
  `dtrsv`-style elementwise division, not a numerical cross-check against a
  linked LAPACK build; #42 is explicitly not claimed as reference behaviour.
- `reconstruct` still has no rank argument, so a caller that wants the
  truncation error rather than the truncated matrix must inspect
  `PivLltInfo::rank`. This was not added: it is an API change, and the factor
  now carries the rank-`rank` matrix.
- The full-factor fix covers the QR path the tall/wide ratio selects; the
  square-ish path (`compute_squareish_svd`) already produced full factors from
  the decomposition itself and was not changed.
- A complex 1x1 triangular solve with a subnormal pivot still returns a
  non-finite value: `num_complex`'s quotient forms the squared denominator, so
  it underflows for the same inputs the real branch now handles. Fixing it means
  scaling both operands LAPACK-`zladiv`-style or using `div_real` against the
  pivot's magnitude, which is a separate change from the reported (real) case;
  the complex path keeps its previous result rather than gaining a NaN.
