# Tridiagonal QR spectrum accuracy

Issue: [tensor4all/faer#76](https://github.com/tensor4all/faer/issues/76).

## Decisions

- **The QR chase, not the reduction, carries the error.** On the failing fixture
  (`s[i] = 1 - 2 i / (n - 1)` behind adjacent 0.6/0.8 rotations, `n = 127`, kind 2) the
  reduced tridiagonal is a fixed input for both tridiagonal solvers. Feeding that identical
  `(diag, offdiag)` to `qr_algorithm` and to `divide_and_conquer` (threshold 8) gave, against
  the intended spectrum, 168 f32 epsilons via the QR path and 3.5 via divide-and-conquer
  (f64: 65.5 and 3.5). Promoting the f32 reduction to f64 put the reduced tridiagonal's own
  spectrum within 3 epsilons of the intended one, so the Householder reduction is not the
  source and was left alone.
- **Replace the diagonal update, not the shift, the deflation test or the direction.** The
  inner step used to rebuild each diagonal from the explicit 2x2 similarity,
  `c (c d - s e) - s (c e - s d')`, which re-rounds an entry of size `|d|` on every rotation;
  the rotation count of a full solve is `O(n^2)`, so each diagonal is touched `O(n)` times and
  the rounding walks. Adopted LAPACK `dsteqr`'s QR inner loop, whose `p`/`g` recurrence
  updates a diagonal as `g + p` — the correction the previous rotation applied — and defers
  the trailing corner to `d[end] -= p`. Measured on the same fixed tridiagonal (ratio to the
  intended spectrum, f32/f64): kind 2 `n = 127` 168/65.5 → 11.0/14.5, kind 2 `n = 129`
  128.5/26.5 → 21.0/11.0, kind 4 `n = 127` 27.0/11.0 → 9.0/8.0, kind 4 `n = 257` 77.0 → 11.0.
- **Take `dsteqr`'s shift expression too.** `g = d[start] - d[end] + e / (g0 + sign(g0) hypot(g0, 1))`
  replaces `d[start] - mu` with `mu` computed through `e^2 / (delta + h)`. They agree in exact
  arithmetic, but the first keeps the low bits of the shift when a block's two end diagonals are
  close. It matters for the degenerate kind 5 fixture (±1 with multiplicity 64 and 65, `n = 129`):
  the first form lands at 49 f64 epsilons, the second at 60, and the reduced tridiagonal's own
  spectrum is already 49 epsilons from the intended one — so the first form reaches the floor
  the reduction sets and the second does not.
- **Rejected: LAPACK's QL/QR direction choice.** Measured on a reversed-array QL emulation it
  changed nothing for kinds 2, 5 and 7 and was much worse for the clustered kind 4 (1006 vs 27
  f32 epsilons), and the direction is chosen once per block here rather than per sweep as in
  `dsteqr`. Not adopted.
- **Rejected: LAPACK's `e^2 <= eps^2 |d_m d_{m+1}| + safmin` deflation test.** Worse on the
  fixtures that matter (kind 2 `n = 127` 11.0 → 15.0 f32) and unusable without a scaled
  comparison in the subnormal fixtures that the range tests cover; `hypot` stays.
- **Rejected: a closed-form 2x2 block solve (`dlae2`).** No change at all on these fixtures
  (the ratio and the rotation count were identical wherever it applied).
- **Rejected: rewriting the shift to LAPACK's sign convention everywhere.** Keeping the
  existing `make_givens` rotation for the eigenvector columns and flipping only the sine used by
  the scalar recurrence leaves the whole `u` path untouched, and the existing `U S U^T`
  reconstruction tests confirm the two stay consistent.

## Verification conclusions and constraints

- `dense_self_adjoint_qr_spectrum_accuracy` (`linalg::evd::tests`) runs the kind 2 `n = 127`
  fixture through `self_adjoint_evd` at thresholds 8 and 128 with and without eigenvectors for
  f32, f64, c32 and c64, and requires the CHKST bound of 60 epsilons on both the spectrum and the
  residual. It fails on the parent commit for f32 (1.68e2), c32 (1.5e2) and f64 (6.55e1) and
  passes here for all four types.
- The opt-in LAPACK groups `lapack_self_adjoint_evd_*` (16 tests) and
  `lapack_tridiagonal_evd_*` (12 tests) pass for all four scalar types; on the parent commit the
  `self_adjoint_evd` case and parallel groups failed with spectrum and mode-agreement ratios
  around 1.5e2 (f32/c32) and 65-66 (f64). The unrelated `lapack_nonsymmetric_evd` groups fail
  identically before and after (same diagnostics counts).
- The whole library suite passes: 465 passed, 0 failed, 138 ignored (the opt-in LAPACK groups
  among the ignored).
- Cost (operation counts only; no timing was taken). The sweeps and rotations are the same within
  a few percent on the fixed fixtures (identical for f64 kind 2 `n = 127`/`n = 129`; `-5%` for f32
  kind 4 `n = 257`; `+8%` for the degenerate f64 kind 5), and the per-rotation work drops from about
  25 to about 15 multiply/adds. No square root or division is added per rotation. The shift costs
  one `hypot` and two divisions per sweep (`dsteqr`'s form), against one `hypot` and one division on
  the previous ordinary path, so a sweep has one more division. The early exit on a zero bulge
  element is dropped (`dsteqr` has none); on the measured fixtures it never fired, since the
  rotation counts are unchanged, but that is an observation on these fixtures, not a proof that it
  never would.
- Limits: the QR fallback still computes with only one shift per sweep
  and no QL direction, so a fixture whose spectrum needs many sweeps from the bottom would still
  be worse than a LAPACK-quality solver, and the `n >= 128` vector path is unaffected because it
  uses divide-and-conquer. The `hypot`-based deflation test is deliberately kept, and the
  degenerate-spectrum accuracy is bounded by the reduction's own error, not by the solver.
