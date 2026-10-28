# Divide-and-conquer SVD merge boundary rotation (#75)

## Decisions

- Fix the merge's own boundary rotation. Not the secular root finder, not the
  deflation tolerance, not a reorthogonalisation pass. The measurement that
  decided it: instrumenting the recursive merge of the issue's
  127x127 f32 logarithmic bidiagonal reports the first non-orthogonal merged
  block at a subproblem of size 15, while the children's factors (`u_lhs1`,
  `u_lhs2`, `vm`), the secular roots, the repaired `zhat` and the secular
  residual of every root at that merge agree with its sibling merges. So the
  error enters between `compute_svd_of_m` and the merge's output update, in the
  two-entry rotation that `divide_and_conquer` builds from `alpha * lambda` and
  `beta * phi` and applies to the first and last column of the merged left
  factor.
- Why that expression fails. `alpha` and `beta` are entries of a subproblem
  that `divide_and_conquer` has already rescaled to max 1, and the fixture
  spans `eps^-2 .. eps^2`, so at a deep merge `alpha * lambda` and
  `beta * phi` are genuinely tiny. Measured at the first failing merge:
  `al = 1.02407e-24`, `bp = -4.03613e-22`, so `al^2` flushes to zero and
  `bp^2` is subnormal (`1.6255e-43`, 116 subnormal steps, seven significant
  bits); `(al^2 + bp^2).sqrt()` gives `4.0317568e-22`, off by `1.1e-3`, instead of
  `4.03615e-22`, and `s0 = bp / r0 = -1.0010858`. `c0^2 + s0^2 = 1.00218`, so
  the "rotation" scales the two columns instead of rotating them: the merged
  block's 1-norm Gram error is `2.0e-3`, i.e. `1.14e3` of the ORT01 tolerance
  at n=15 and `1.4e2` at n=127, the absolute error staying put while the
  denominator `n` grows up the recursion.
- Why the reported factor is V and not U. For a lower-bidiagonal input
  `svd_imp` hands the solver the transposed bidiagonal (`diag`/`subdiag` are
  copied from `bid[(i, i)]`/`bid[(i, i + 1)]` with the conjugate) and passes
  the caller's `v` as the solver's left factor, so the broken left factor of
  the merge surfaces as the public `V`. The solver's own right factor, and
  therefore the public `U`, never carries it.
- `perturb_col0` is not the cause. It forms the same repaired `zhat` as
  LAPACK's update (`zhat_i^2 = |d_i^2 - sigma_i^2|`) and its secular residual
  at the failing merge matches its siblings; a common scale error in `zhat`
  would have shown up as a residual, and it does not.
- Keep the historical arithmetic where it is exact. The scaled hypot and the
  ratio-normalised Givens pair are taken only when `al^2 + bp^2` is subnormal
  or non-finite, the same guard shape this file already uses for `max.recip()`
  pre-scaling: inside the normal range the squares are exact to rounding, so
  every merge keeps its previous bits. An unguarded version that always used
  the scaled forms fixed the same cases but changed the last bits of every
  merge; it was dropped for the guarded one.
- Reference reading, not an import. `DLASD2` forms the same pair as
  `Z(1) = DLAPY2(Z1, Z(M))`, `C = Z1/Z(1)`, `S = Z(M)/Z(1)`, so LAPACK avoids
  the squaring underflow but keeps the same subnormal products and the same
  non-renormalised `C^2 + S^2`; the ratio-normalised pair is this fork's
  choice. No OpenBLAS or LAPACK source is copied.
- Rejected: raising the deflation tolerance or the `eps_coarse` floor (the
  failing merges' `alpha`/`beta` are legitimately tiny, not noise); a
  reorthogonalisation or per-merge renormalisation of the merged columns (an
  extra O(n^2) pass per merge); promoting f32 merges to f64 (allocation and a
  different kernel); a post-hoc column scaling fix-up (it would leave the
  rotation applied to the columns non-orthogonal).

## Verification conclusions and constraints

- Baseline fork `d6cb4ee` (main after #73/#80). The new ordinary regression
  `linalg::svd::tests::lapack_svd_f32::lapack::bidiagonal_merge_boundary_rotation_is_orthogonal`
  (also instantiated for f64/c32/c64) fails on the baseline with the issue's
  own numbers -- `seed=0x4c415041434b, n=127, threshold=8: V orthogonality
  ratio=1.4395331e2` and `seed=0x4c4150414351, n=126, threshold=8: 2.6362531e4`
  -- and passes after the fix. The third case, `seed=0x4c415041434d, n=128` at
  the default threshold 128, only fails on f32.
- Bounded neighbour scan (17 seeds from the baseline seed, n=7/8/9/126/127/128/129,
  thresholds 8/128, full/full public `svd`, ORT01-style ratios with the
  denominator `n * eps`), V orthogonality, before -> after: f32 44 violations,
  max ratio 3.28e4 -> 0 violations, max 0.27; c32 48, 3.03e4 -> 0, max 0.29;
  f64 4, 2.73e10 -> 0, max 0.53; c64 0 -> 0. U orthogonality and
  reconstruction stay inside their bounds in every cell (max 1.21 and 1.18 of
  the tolerance over the four types), so the fix is not trading one factor for
  another. The issue's three named cases go from 1.44e2 / 3.28e4 / 1.20e4 to
  0.0157 / 0.0179 / 0.0078.
- The whole SVD test selection, including the ignored LAPACK groups, passes for
  all four types (82 tests; the earlier `rank_one_at_recursion_splits` failures
  belong to #73 and are fixed at the baseline). The default library suite is
  468 passed, 0 failed, 138 ignored.
- Cost: the two replaced expressions are O(1) and evaluated once per merge, so
  the merge's operation count is unchanged -- the D&C merge is dominated by its
  `(k+1)x(k+1)` and `(rem+1)x(rem+1)` products against `n+1` columns, and no
  pass over a matrix is added, removed or reordered. Only the underflowing
  merges take the scaled branch, which costs one divide and one square root.
  No timing was measured; no performance claim is made.
- What remains unverified: the rotation's *angle* is still formed from products
  that can be subnormal, so it can carry up to a few parts in a thousand of
  relative error when both products are subnormal. The pair is now orthogonal,
  so that no longer costs orthogonality, and the scan shows no visible effect
  on the factors or the reconstruction (the three named cases land at
  0.0078-0.0179 against a tolerance of 50); removing it would need the products
  to be formed with a power-of-two split, which no failing case required. Only
  f32/c32 and one f64 seed family exercise the scaled branch locally (x86_64
  Linux, Rust 1.99.1); other platforms and fixture families were not scanned. Stable rustfmt reports the
  same pre-existing differences on baseline and candidate (355, none in the
  changed blocks).
- No upstream checkout was measured and no upstream feedback is part of this
  change. No performance experiment is involved; the ledger entry is a
  correctness `KEEP`.
