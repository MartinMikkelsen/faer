# Schur eigenvector extraction at the range limit

`tensor4all/faer#77`. `eigenvectors_from_schur_real` / `_cplx` (and the dense
`evd_real` / `evd_cplx` drivers through `evd_from_real_schur_imp` /
`evd_from_cplx_schur_imp`) reported success with a wrong direction for a finite
Schur form near the top of the range.

## Decisions

- The shift clamp of the shifted back substitution is `eps * sum |a_ij|`, and
  the plain sum overflows to `inf` for a finite matrix at the range limit
  (`2.5 * 2^1023` for the reported 2x2). The clamp is now computed once as
  `eps_norm` and recomputed term by term — `eps * |re| + eps * |im|` per entry —
  **only** when the plain `eps * sum` is not finite.
  Scaling every term unconditionally was rejected: it changes the subnormal
  floor. For two `2^-1023` entries the plain form gives `eps * 2^-1022 =
  2^-1074` (the smallest subnormal, and the only thing that keeps the single
  eigenvector of a repeated subnormal eigenvalue), while the term-by-term form
  rounds both products to zero. `subnormal_shift_clamp_is_unchanged` fails with
  `NoConvergence` under the unconditional variant.
- The solvers now take `eps_norm` instead of `norm`, so the finite path performs
  the same multiplication once per column instead of once per clamp test; the
  clamp sites are otherwise unchanged and every finite input keeps its previous
  arithmetic and its previous result.
- The 2x2 block solves of both shifted solvers form `a^2`, `b * c` and `a * r`,
  products of two entries that overflow for a block above `sqrt(MAX)` even when
  the solution is representable. A first version rescaled such a block by a
  power of two in a loop; the post-implementation review showed it was wrong
  three ways: an overflowed shifted diagonal `A[i, i] - p = inf` never leaves
  the loop (a hang on finite input), uniform shrinking flushes the small entry
  of a wide-range block that base solves exactly
  (`[[0, -2^1023], [2^-1022, 0]]`, determinant 2.25), and the complex solver
  compared the scaled determinant with the unscaled clamp. The block arithmetic
  is therefore left as it was. At the top of the range it fails with a typed
  error (`NoConvergence`), not a wrong `Ok`, which is what the issue asks for;
  a range-safe 2x2 solve is left as a separate item.
- `q = sqrt(|A[k, k-1]|) * sqrt(|A[k-1, k]|)` in `evd_from_real_schur_imp` needs
  no change: it is the geometric mean of two finite entries and cannot exceed
  either, so it is finite whenever the Schur form is.

## Verification conclusions and constraints

- Focused tests `linalg::evd::range_limit_vector_tests::*` at `f32`, `f64`,
  `c32` and `c64`, at scale 1 and at the reported power of two: the reported
  2x2, the complex triangular analogue, the 3x3 conjugate-pair block (at the
  range limit it must fail or return eigenvectors, never a wrong `Ok`; the
  same for a pair whose shifted diagonal overflows), and the wide-range pair
  above, which must keep its exact solution.
  Residuals are measured on the dimensionless `t / scale` with a small relative
  1-norm residual and unit 2-norm asserted. At base `two_by_two_at_the_range_limit`
  fails with relative residual 2/3 (the number in the report).
- The ignored LAPACK validation (`--include-ignored`, the nonsymmetric and Schur
  groups) produces byte-identical `LAPACK_CASES` summaries and 1286 identical
  diagnostic lines before and after, and an identical failing set: the
  pre-existing GET22 residual diagnostics and the pre-existing self-adjoint
  "known spectrum" diagnostics, neither of which this change touches.
- Limits left unchanged. (a) A complex shift near the top of the range still
  breaks the complex reciprocal in `faer_traits::math_utils::recip_impl`: its
  `(re * small)^2 + (im * small)^2` overflows once a part is around `2^1023`
  (`f64`) / `2^127` (`f32`), the reciprocal becomes zero and the eigenvector
  component is dropped. Measured on `c64`: `recip` of both `(2^1023, 0)` and
  `(2^1023, 2^1023)` returns `0`, where the values are `2^-1023` and
  `2^-1024 (1, -1)`. The shifts the regression tests exercise are `2^1022`
  (`f64`) and `2^126` (`f32`), one halving below that, and they pass; the
  reported inputs do not reach it either. This is upstream `faer-traits` code,
  unchanged and not measured against upstream `main`. (b) The 2x2 block solve
  above.
