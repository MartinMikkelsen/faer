# Singular-value order after a subnormal deflation in the divide-and-conquer SVD

Issue: [tensor4all/faer#82](https://github.com/tensor4all/faer/issues/82). Regression of
`t4a-faer` 0.24.9 (`2c5d45a`), which the reporter narrowed to restoring the 0.24.8
`reductions/norm_l2.rs`; the mechanism was unknown.

## Decisions

- **The defect is in `deflate`, not in the norm that exposed it.** The `norm_l2` change is
  correct and stays: for the reported 128x128 `c64` matrix (three nonzero entries) it makes
  bidiagonalisation's row norm finite, and the bidiagonal then reaches the divide-and-conquer
  with a *normal* cascade of diagonal entries down to `3.1e-294`, plus one subnormal coupling
  `subdiag[67] = 1.9e-310`. That entry is the only thing the old code mishandled.
- **The owning defect is a classification, not a sort.** `deflate` groups the entries of the
  merged `diag` (`D` of the secular equation) with `|diag[i]| < min_positive` and treats that
  group as "already deflated": the group's members are placed by index, the final reversal puts
  the group at the end of `D`, and `compute_svd_of_m` expects exactly that (the trailing zeros
  it counts, and the ascending nonzero prefix it reverses into the output). Index order inside
  the group is only interchangeable while every member is zero. A nonzero subnormal value is
  *not* zero: `compute_svd_of_m` and `compute_singular_values` classify with exact-zero tests
  (`diag[k] == zero()`, `col0[k] == zero()`), so the same value was a deflated zero for the
  ordering permutation and an ordinary singular value for everything downstream. The reported
  output is the result: `[0.5968, 1.9e-310, 0, ..., 0, 6.3e-4, ...]` -- the subnormal landed in
  the middle of the zero tail, and the single adjacent-swap pass in `compute_svd_of_m` (which
  relies on `s` being ascending except at one place) only fixed the one inversion it saw.
- **Chosen fix: group by exact zero in the two ordering tests of `deflate`** (the `perm`
  construction and the `total_deflation` shift). A nonzero subnormal value now keeps its place
  in the magnitude-ordered two-run merge, and the value stays a value: for a genuinely subnormal
  singular value (a diagonal matrix with an entry below `min_positive`), dropping it with the
  zeros would be a correctness loss, not just a reordering.
- **Rejected: sorting `s` fully in `compute_svd_of_m`.** It would repair the order of this input
  and of any other order the permutation produces, but it leaves `D` non-monotone for
  `compute_singular_values`, whose bracketing (`left = diag[k]`, `right = diag[k']` at the next
  nonzero `col0`) then solves the secular equation over an interval that does not contain the
  wanted root. That path only stayed quiet here because the merge deflated to a single nonzero
  `col0` entry (`actual_n == 1`), so the bracketing was never used; a case with several
  surviving couplings would produce wrong values rather than wrong order.
- **The `sml` uses that are tolerances stay.** `eps_strict`'s floor, the `col0` zeroing, the
  small-`diag` deflation step, and the deflation sweep's range all decide *what to deflate*
  (LAPACK's role for `TOL = 8 eps max(...)`, floored at `SML`), not what order to emit. They
  behave identically after the change.
- **The 0.24.9 work is left alone.** Reverting `norm_l2` was never a candidate: #36's overflow and
  underflow were real, and the SVD must be correct for a bidiagonal with subnormal entries, which
  is what the new tests state directly.

## Verification conclusions and constraints

- Reproducer: the issue's 128x128 `c64` matrix, before the change
  `[0.5968, 1.9e-310, 0, ..., 6.3e-4 at index 64, ...]`; after, `[0.5968, 6.3152e-4, 1.06e-15,
  1.24e-32, ..., 3.13e-294, 1.90e-310, 0, ...]` -- nonnegative, nonincreasing, reconstruction and
  unitarity within `1e-14`, and the two nonzero values matching `zgesdd`.
- Focused tests, each failing before and passing after: `svd::tests::test_sorted_with_subnormal_at_split`
  (the reported matrix through the public `Mat::svd` contract) and
  `svd::bidiag_svd::tests::subnormal_diag_at_split` (a direct divide-and-conquer call at `n = 128`
  with a diagonal subnormal in the right half, `min_positive / 2` per type, at three split
  positions, at `recursion_threshold` 128 and 4, `f64` and `f32`; order, every value within the
  merge's normwise floor `8 eps ||B||` of the exact spectrum, and reconstruction). The subnormal
  value itself is only determined to that floor: at threshold 4 it comes back as another
  subnormal, and a zero subproblem can come back as `8 eps`, the deflation floor LAPACK `DLASD2`
  uses too. The fix is about order, not about relative accuracy of tiny singular values.
- Normal-range inputs cannot change: `|x| < min_positive` and `x == 0` differ only for subnormal
  `x`, so the classification -- and therefore every result -- is bit-identical otherwise. The
  LAPACK validation suites (`--include-ignored`, all four types) agree: the only failure is the
  pre-existing f32 `bidiagonal_family` V-orthogonality diagnostic (#75), with the same
  `ratio=1.4395331e2` as at the base commit, and #73's `rank_one_at_recursion_splits` already
  passes at this base (PR #80). Library suite: 466 pass, 0 fail.
- Limits: no timing was taken and none is claimed. The change swaps two comparisons for two
  comparisons and adds no pass, allocation or branch on the ordinary path. The defect is inherited
  from upstream faer's divide-and-conquer port (identical at the base `0ecd846`); nothing was
  reported upstream (deferred-feedback rule).
- Overlap with #75: both read the same merge, but the lines differ (#75 is the merge's boundary
  rotation); the two changes were combined and pass the LAPACK SVD suites together.
- The tridiagonal-EVD divide-and-conquer was checked for the same pattern: it does not group
  entries by a `min_positive` test.
