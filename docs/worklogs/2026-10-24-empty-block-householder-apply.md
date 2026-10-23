# Empty block Householder apply

Issue: [tensor4all/faer#51](https://github.com/tensor4all/faer/issues/51).

## Decisions

- **Return immediately from the generic block apply when the basis has no columns.** An empty block
  Householder transformation is the identity, so the call is a no-op for every caller. The blocked QR
  keeps only the nonzero columns of each block, so a block that is exactly zero reaches the apply
  with `N == 0`. The fix goes in the owning primitive rather than its QR call site, because the
  public apply's assertions admit an empty basis and any later caller would inherit the same abort.
- **The panic is fork-only, and the fix keeps upstream's result.** The row split added in `c061812`
  computes `partials.par_chunks_mut(n * k)`, where `n` is the basis' column count; at `N = 0` that is
  a zero chunk size, which rayon rejects. Upstream faer 0.24.4 (the fork's base `0ecd846`) reaches
  the same empty apply -- an instrumented build of the issue's reproducer prints six `N == 0` calls
  (`K = 2, 8, 2, 4, 2, 64`) -- but it only splits columns, where the zero-width products are a no-op,
  so it does not panic. The `A1.ncols() > 0` call site and the zero-column skip are shared with
  upstream; the fork's row split is what turned the empty call into an abort.
- **Rejected: adding `N > 0` to the row-split predicate alone.** That is the issue's proposed fix. It
  stops the panic but still sizes an `N x K` temporary and runs the zero-width products, and every
  future split would need the same conjunct. The early return is the same result with no work.
- **Rejected: gating only the QR call site on `local > 0`.** Correct for this report, but it leaves
  the public apply aborting on an input its own assertions accept.

## Verification conclusions and constraints

- Focused test
  `linalg::qr::no_pivoting::factor::tests::zero_column_block_does_not_panic_with_threads`: a
  256x96 `f64` matrix whose first 32 columns are zero, factored with `block_size = 32` in a
  2-thread pool. On the parent it panics at `factor.rs:242` with `chunk_size must not be zero`
  before any comparison; with the early return it passes, the rank is 64, and the threaded factor
  and rank equal the sequential ones.
- The issue's own 1024x96 / 32-zero-column reproducer at `Par::rayon(2)` completes after the fix.
- Whole library suite: 242 passed, 3 failed, 2 ignored. The three failures are the pre-existing
  `linalg::matmul::internal::tests::*strict_triangle*` cases in files this change does not touch
  (already recorded in `PATCHSET.md`); no new failure, and no new warning.
- Limits: no timing was taken and none is claimed. The early return is one comparison on a path that
  performs a block apply, and it removes the zero-width products rather than adding work. This is a
  fork regression, so nothing was reported upstream.
