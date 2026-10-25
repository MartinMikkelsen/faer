# Tridiagonal EVD at the range limits

Issue: [tensor4all/faer#57](https://github.com/tensor4all/faer/issues/57).

## Decisions

- **Rescale in the tridiagonal backend, in both paths, through one helper.** `normalize_range` divides
  `(diag, offdiag)` by their largest entry when that entry leaves
  `[sqrt(min_positive)/eps, sqrt(max_positive)]` — the window in which an entry squared stays in range
  (`(eps * anorm)^2` does not underflow, `anorm^2` does not overflow) and a reciprocal stays finite —
  and returns the factor that takes the computed eigenvalues back. The callers apply it before they
  touch the input, so the input arrays and the eigenvalues the caller receives keep the caller's scale.
  Applying it in the backend rather than at one entry point covers both reported surfaces
  (`tridiagonal_self_adjoint_evd` and the tridiagonal step of `self_adjoint_evd`), which duplicate the
  QR / divide-and-conquer dispatch.
- **Divide instead of multiplying by the reciprocal.** The QR fallback normalized by `anorm.recip()`
  unconditionally, so a subnormal norm (no finite reciprocal) turned every entry into `inf`/`NaN` and
  the iteration reported `NoConvergence`; `x / anorm` is exact where the reciprocal does not exist.
  In-window spectra now skip the pass altogether instead of round-tripping through `* 1/anorm * anorm`.
- **Fix the 2x2 shortcut by scaling, not by rewriting its formula.** `hypot` overflows above about
  `9e307` (`(x * sqrt(min_positive))^2` leaves the range), and `a - d` overflows for the reported
  `[-1e308, 1e308]`, so the closed form cannot be made safe with algebra alone at that scale. On the
  normalized matrix it is the unchanged upstream code, and the two eigenvalues are multiplied back.
- **Rejected: rescaling in `divide_and_conquer` only.** That is where the silent wrong answer appears
  for `n >= 128` (the fork's report covers `n = 3`, but at `n = 129` and `n = 300` the reported input
  family returned `inf` and `NaN`), but the subnormal `NoConvergence` is in the QR fallback, so a
  single-path fix would leave one of the two reported matrices failing.
- **Rejected: LAPACK's scale-to-`ssfmax` window (`sqrt(safmin)/eps^2`).** Scaling a spectrum of order one
  down to `3e-123` needs `dlascl`'s staged multiplication to scale it back, because `anorm / ssfmax`
  overflows; normalizing to unit size reaches the same protection with one division and one
  multiplication.

## Verification conclusions and constraints

- Focused tests `linalg::evd::tridiagonal_range_tests`: the reported diagonal `[-1e308, 1e308]` returns
  exactly `[-1e308, 1e308]` and its coupled 2x2 sibling `±sqrt(2) * 1e308`; the reported subnormal
  `([2s, 2s, 2s], [-s, -s])`, `s = 1e-310`, returns `s * (2 - sqrt 2, 2, 2 + sqrt 2)` to `1e-13`
  relative; and a constant tridiagonal of `n = 5` at `1e308` and at `1e-310` matches
  `a + 2b cos(k pi / (n + 1))` to `1e-13 * anorm` through both the QR fallback (`usize::MAX` threshold)
  and divide-and-conquer (`threshold = 4`). On the parent the first two fail (`inf` and
  `NoConvergence`) and the divide-and-conquer cells return `inf`/`NaN` at the top of the range.
- Whole library suite: 246 passed, 3 failed, 2 ignored. The three failures are the pre-existing
  `linalg::matmul::internal::tests::*strict_triangle*` cases, which also fail on the parent commit on
  this machine; no new failure and no new clippy warning.
- Limits: no timing was taken and none is claimed; the added work is one pass of comparisons over
  `diag`/`offdiag` plus a division that the QR fallback already performed as a multiplication. The
  scaling window is a property of this algorithm's own squares and reciprocals, not a substitute for
  a wider audit: `self_adjoint_evd` still tridiagonalizes the input at its original scale, so an
  extreme-magnitude dense matrix can overflow there before the tridiagonal backend is reached, and
  `math_utils::hypot` itself still returns `inf` above about `9e307`. Both are outside this report and
  are not fixed here.
