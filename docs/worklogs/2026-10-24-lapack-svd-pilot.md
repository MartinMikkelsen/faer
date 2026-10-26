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
- Separate the test port from production fixes. The full-factor reused-buffer
  defect is already tracked by #41 and is reproduced here.

## Verification conclusions and constraints

- At fork revision `d3215290618aa49a1e246c7c2d8989915688c594`, four of the
  five validation groups passed and the reused-output group reproduced #41 in
  tall U and wide V orthogonality. Rebased on `main` at
  `5a27bcece2f9f0d1c1b94c9b42dd7bfc4a94dc39`, which includes the #41 fix from
  #47, all 1,150 configured SVD calls pass. The largest of the 3,785 check
  ratios is 1.77 against the threshold of 50, so the checkers' own rounding is
  not close to deciding any result. Checker self-tests establish
  zero/extreme-scale handling and rejection of corrupted factors, nonfinite
  values, and invalid spectra.
- The `linalg::svd` tests and the new checker tests pass. The ordinary
  workspace run has three triangular-matmul failures, the same three that
  `PATCHSET.md` records as failing on `main`; their names are listed with the
  detailed results. All 55 workspace doctests pass; formatting, whitespace, and
  inclusion of the test source and LAPACK license in the package were checked.
- The crate's `license` metadata is changed to `MIT AND BSD-3-Clause` for the
  adapted LAPACK test file. Existing LAPACK ports in production code kept the
  metadata unchanged, so this needs a maintainer decision.
- [Coverage, adaptations, reproduction instructions, and measured results](../lapack-svd-pilot.md)
  are the detailed reference. The single local runtime observation does not
  establish comparative performance. No upstream faer revision was tested.
- The checks are normwise; they do not establish relative accuracy of tiny
  singular values. Full LAPACK fixtures/generators, other precisions, broader
  layouts/threading, direct bidiagonal entrypoints, and scheduled CI remain
  outside this pilot.
