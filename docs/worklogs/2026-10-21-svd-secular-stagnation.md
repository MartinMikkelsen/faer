# SVD secular-equation stagnation (#28)

## Decisions

- Fix the shared secular root finder, not the divide-and-conquer split, deflation
  thresholds, or the public SVD algorithm choice. The secant loop previously
  stopped when two function values differed by no more than machine epsilon,
  but returned its estimate without requesting bisection. A flat function value
  away from the root is not convergence.
- Keep the existing exact-root and relative-step convergence checks. When the
  function-value difference prevents secant interpolation, use the bracketed
  bisection that already handles unsuccessful interpolation. No new solver,
  output-residual scan, retry, workspace, dependency, or API is introduced.
- The incorrect smallest root also perturbs the merge's reconstructed update
  column (`perturb_col0`), explaining why orthonormal factors can reconstruct
  the wrong matrix. Checking orthogonality alone is insufficient.

## OpenBLAS/LAPACK comparison

Compared against OpenBLAS **v0.3.26**, tag commit
`6c77e5e314474773a7749357b153caba4ec3817d`, using its bundled
[`lapack-netlib/SRC`](https://github.com/OpenMathLib/OpenBLAS/tree/v0.3.26/lapack-netlib/SRC).
For explicit singular vectors the corresponding path is `DBDSDC` -> `DLASD0`
-> `DLASD1` -> `DLASD2` (deflation) / `DLASD3` -> `DLASD4` (roots).
`DLASD3` reconstructs the update column from the computed roots, corresponding
in purpose to faer's `perturb_col0`.

`DLASD4` accepts a root when `abs(W) <= EPS * ERRETM`, where `ERRETM` estimates
secular-function evaluation error. It safeguards interpolation within a bracket,
uses Newton/bracket steps when necessary, and returns `INFO=1` on failure to
converge within 400 iterations. It does not equate indistinguishable function
values with convergence. Faer retains its own algorithm and bisection; this is
a behavior comparison, not a LAPACK translation or import.
`DBDSDC` without vectors uses `DLASDQ`, so the divide-and-conquer oracle below
requests explicit vectors rather than measuring only singular values.

Direct calls to the system OpenBLAS **0.3.26+ds-1ubuntu0.1**, with its effective
thread count set and verified as **1**, give the following for the issue's
128x128 matrix (`d[64]=1e-13`). These are numerical residuals, not timings:

| implementation | relative Frobenius reconstruction error | s[63] | s[64] |
| --- | ---: | ---: | ---: |
| faer before | 1.9322022904322647e-3 | 4.8333386379073480e-2 | 5.8993518642901059e-6 |
| faer after | 2.7664330953404153e-15 | 4.8327490472264548e-2 | 1.2403473458920837e-14 |
| OpenBLAS DBDSDC / DGESDD | 1.4686644327950810e-15 | 4.8327490472264596e-2 | 1.2403473458920847e-14 |
| OpenBLAS DGESVD (QR) | 2.5101921527362456e-15 | 4.8327490472264582e-2 | 1.2403473458920842e-14 |

The system LAPACK calls also reconstruct the zero-padded 256x256 case, and
128x128 cases with `d[64]=0` or `1e-6`, within 2.7e-15. No changes to OpenBLAS
or LAPACK were made.

## Verification conclusions and constraints

- The baseline is fork `50c2b989484f063f4e1b132adc333715391d8b73`.
  Its bidiagonal-SVD file's last change is upstream
  `645768f3f2b62f3c822ff8151aaa8900b07ff197`. The original upstream checkout
  was not rebuilt separately; the defect was reproduced in this fork.
- The scalar regression reproduces the same spurious root with
  `f(x)=1-(tiny/x)^2`, `tiny=1e-13/sqrt(65)`: both secant samples round to 1,
  while their arguments are approximately 5.90e-6 and 2.95e-6. The regression
  and the public-SVD regression fail before the production change and pass
  afterwards.
- Root tests cover both ends of a bracket, the largest-root path, and tiny
  magnitudes down to 1e-30. SVD tests cover 128/256 square matrices, real and
  purely imaginary c64 inputs, `d[64]` from zero through 1e-6, and both/one/no
  vector outputs, comparing spectra with forced QR and checking available
  factors for orthogonality and reconstruction.
- The default workspace suite passes: 216 unit tests and 37 doctests; two
  pre-existing ignored tests remain ignored. The affected SVD and EVD suites
  pass, as do default and std-only builds and the rustdoc build.
- Changed Rust files pass nightly rustfmt and whitespace checks. Whole-workspace
  formatting has pre-existing differences outside these files. The unmodified
  `faer-traits/src/lib.rs:1271` NaN predicate (`value != value`) fails the
  current clippy `eq_op` lint; unrelated lint/format fixes are not included.
- This is a correctness-only change. No performance acceptance or timing claim
  is made. The additional bisection work is required only where secant
  interpolation previously returned an unconverged root. macOS/aarch64 and
  the ignored large-fixture tests were not run locally. Publication,
  downstream lockfiles, and removing downstream residual retries are separate
  work; no upstream tracker was contacted.
