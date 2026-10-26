# Focused f64 LAPACK SVD pilot (#31)

## Decisions

- Adapt selected LAPACK DCHKBD/DDRVBD matrix families and numerical properties
  in a private child of faer's existing SVD tests. Preserve the pinned
  OpenBLAS distribution revision, original LAPACK provenance, and BSD notices.
  A Rust port avoids requiring a Fortran compiler or external BLAS to run it.
- Use fixed orthogonal rotations and known diagonal spectra before undertaking
  the much larger DLATMS generator port. Keep residual products and reductions
  independent of faer's norms, normalize before multiplying extreme scales,
  and retain the source input file's threshold of 50.
- Exercise five output modes and both reduced/default recursion boundaries.
  Keep the validation groups explicitly selectable as required by #31; their
  checker self-tests run ordinarily. Known failures remain failures in the
  explicit suite, with no tolerance relaxation or expected-failure assertion.
- Test the bidiagonal adapter directly with DCHKBD checks 15–19, including
  the source's safe-minimum spectrum-comparison denominator without a dimension
  factor. Adapt faer's extra zero row explicitly while preserving the source's
  denominator n. Direct tests cover all four U/V-presence combinations.
- Put rank-one entries at the first, middle, and last diagonal positions,
  including the documented size-8 / threshold-4 case. The initial first-entry
  case did not expose the zero-child merge from #33; a passing first-entry
  case is not evidence that the split-position defect is fixed.
- Fill U, V and S with NaN before every call except in the reused-output
  group, so an unwritten output entry fails a check rather than passing as a
  zero.
- Separate the test port from production fixes. The split-position defect and
  the full-factor reused-buffer defect are tracked by #33 and #41; their cases
  stay as regression coverage.
- Reuse the numerical checks in a sibling test module for layout and lifecycle
  coverage. Extend known spectra with repeated and tightly clustered values,
  verify the fixtures against literals, and exercise extreme scales through
  padded/reversed inputs and both output orientations. Border guards and
  input snapshots distinguish wrong numerics from unintended writes.
- Reuse output and scratch buffers across A and -A/2 in the same strided
  layout, rather than treating initial prefill alone as proof of buffer reuse.
  Use dedicated 2-/4-thread pools and default algorithm thresholds on sizes
  that reach the parallel branches, and require each parallel spectrum to
  agree with the sequential one. Restrict those larger cases to thin/omitted
  factors so independent scalar checks stay inexpensive; do not compare
  non-unique vector entries.
- Keep LAPACK copyright notices in `lapack.rs`, which holds the adapted
  material; `svd_views.rs` holds faer's own cases and points to it.

- Run all SVD tests, including the ignored validation groups, in a separate
  weekly Ubuntu workflow with a manual trigger. Local timing supports a
  two-minute execution limit; reserve 30 minutes for the job because clean
  hosted setup and dependency compilation are unmeasured. Keep the push/PR gate
  unchanged. Save dependencies even on failure, retain logs for 30 days, and
  use a separate two-minute job that opens one labeled issue on the first
  failure or timeout of a default-branch run and comments on it for each later
  one.

## Verification conclusions and constraints

- On `main` at `5a27bcece2f9f0d1c1b94c9b42dd7bfc4a94dc39`, all 4,599
  configured calls pass. The largest of the 12,625 check ratios is 2.30
  against the threshold of 50, so the checkers' own rounding is not close to
  deciding any result; parallel and sequential spectra differ by at most 0.16
  on the same scale. The same source against fork revision
  `d3215290618aa49a1e246c7c2d8989915688c594`, before the #33 fix (#39) and the
  #41 fix (#47), fails four groups: the split-position rank-one cases through
  both the driver and the direct solver, the reused full factors, and 480
  full-factor orthogonality checks in the layout group on the QR side of the
  aspect-ratio boundary. No guard write, input change, convergence error or
  panic occurs at either revision. Checker and fixture self-tests establish
  zero/extreme-scale handling, the source's test-19 normalization, the guard
  checker, the literal new spectra, and rejection of corrupted factors,
  nonfinite values, and invalid spectra.
- The `linalg::svd` tests and the new self-tests pass. The ordinary workspace
  run has three triangular-matmul failures, the same three that `PATCHSET.md`
  records as failing on `main`; their names are listed with the detailed
  results. All 55 workspace doctests pass; formatting and whitespace were
  checked.
- At 512×128 and 128×512 the bidiagonal reduction of the 128×128 R factor is
  below its parallel-work floor, so those shapes run parallel code only outside
  the reduction; only 257×256 reaches the parallel reduction.
- The crate's `license` metadata is changed to `MIT AND BSD-3-Clause` for the
  adapted LAPACK test file. Existing LAPACK ports in production code kept the
  metadata unchanged, so this needs a maintainer decision.
- The workflow's build and test commands, run locally on the same revision
  and profile settings, built the test binary in 41.11 seconds with
  dependencies cached and passed all 26 selected tests in 7.29–7.34 seconds of
  test execution over three runs. These are local observations, not hosted or
  cold-build timings or a performance comparison. A failing exit status
  survives the workflow's `tee` under its `-eo pipefail` shell. The failure
  reporting script was run against a mocked GitHub API: first failure, comment
  on an existing issue, exclusion of pull requests and unrelated issues, the
  cancelled-result note, and an API error that fails the job.
- The workflow has not yet run on GitHub. Its first hosted run, after merge,
  sets the measured basis for the budget and cadence.
- [Coverage, adaptations, reproduction instructions, and measured results](../lapack-svd-pilot.md)
  are the detailed reference. Local runtime observations do not
  establish comparative performance. No upstream faer revision was tested.
- The checks are normwise; they do not establish relative accuracy of tiny
  singular values. The direct group's LAPACK families other than 16 have a
  zero off-diagonal and do not make the solver iterate. Full LAPACK
  fixtures/generators, other precisions, and arbitrary stride/thread/shape
  combinations remain outside this pilot.
