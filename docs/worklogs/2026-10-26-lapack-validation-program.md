# LAPACK numerical validation program (#31)

## Decisions

- Carry the bounded EIG/LIN candidates through all four native scalar types,
  including existing SVD, without implementing missing production interfaces.
  Independent scalar products and literal spectra/solutions calibrate residual,
  conjugation, permutation and separate-D storage checks. Provenance is pinned
  to LAPACK 3.12.0 in OpenBLAS 791b02a1a00c996b66a894f6df93421c53b62d80.
- Keep heavy groups opt-in and small calibrations ordinary. Numerical findings
  stay failed assertions; their fixes are separate work. A failing principal
  case does not prevent later cases/families; malformed factors do not feed
  dependent solves. Principal and dependent counts are reported separately.
- A library error or rank on a degenerate input is a finding unless the LAPACK
  routine being adapted reports the same outcome; the harness does not excuse a
  typed error where LAPACK returns a result. Production calls whose results feed
  checks, including Q formation and eigenvector back-transformation, run inside
  the panic-isolating wrappers as dependent calls.
- Reordering block identity uses the full Schur norm, since GET36 does not
  promise relative accuracy for tiny individual blocks. Successful moves check
  the exact block-adjusted destination. General EVD spectra are initialized
  before every mode and compared on normal fixtures; nonnormal forward-error
  claims require condition estimates that this port intentionally omits.
- Replace the SVD-only weekly workflow with one sequential all-family job,
  diagnostic artifacts and one labeled failure report. Preserve SVD's original
  cases/tolerances; add count instrumentation for the combined execution audit.

## Verification conclusions and constraints

- Development measurements: 59a62189fdc7fad26dafe1a89ed84501d6118af7, which is
  `main` 5a27bce plus one test in the `solvers.rs` test module (same library
  code). The final combined run repeats on this change's own commit, based on
  5a27bce, with identical per-case records. The pre-existing SVD work is
  included. All 148 initial source/lock identities are checked: six
  append-only test hooks are the only edits to existing numerical files.
  No kernel, public API, dependency, upstream measurement or performance
  improvement is part of this change.
- Every inventory family is implemented within the documented existing-API
  scope. The [catalog](../lapack-validation.md) records bounded manifests,
  adaptations, omissions, minimal findings and the original twelve reports.
  [Per-type results](../lapack-validation-results.json) separate port completion
  from numerical success. All 113 inspected source blob identities verify.
- All 325 selected tests execute: 297 pass and 28 fail. All 71,856 principal
  calls are attempted, plus 50,952 dependent calls. Hessenberg, Schur, reordering,
  balance, LU, QR and least squares pass. SVD, general/self-adjoint EVD,
  empty-complex tridiagonal EVD, pivoted LLT/LDL and complex tiny-pivot TRS
  findings remain opt-in failures. An 80-digit reference confirms the f64
  clustered eigenvector residual observation independently of native summation.
- Ordinary workspace results are 423 pass / three pre-existing strict-triangle
  matmul failures / 138 ignored. Other workspace libraries and all 55 doctests
  pass. Native oracle calibrations, eight offline runner/reporter checks,
  changed-test formatting, packaging and workflow syntax pass.
- Local build/execution is 58.85s + 207.13s with cached third-party dependencies.
  Family limits sum to 20 minutes; build limit 20 minutes, combined step 44,
  validation job 50 and reporting job two. Cold hosted builds, Linux execution
  and actual failure-report writes remain unmeasured. Weekly/manual is the
  initial cadence; the scheduled job will remain red while findings persist.
- Pivoted LLT does not compare the first pivot with its rank tolerance: the
  check in `llt_pivoting/factor.rs` is guarded by `if j > 0`. A zero matrix of
  order n≥2 returns NonPositivePivot where DPSTRF returns rank 0, and the 1×1
  zero matrix returns rank one. The guard comes from upstream faer commit
  `64280357` (sarah-quiñones); it was measured only on this fork at `59a6218`,
  not on an upstream checkout. It is recorded, not fixed or reported upstream.
- Unverified: the GET34 rejected-swap check (T and Q bitwise unchanged) is in
  place, but no current fixture produces a rejection. `schur_swap_checked`
  applies its 2×2 standardization regardless of the returned status and can
  reject the second of two split 1×1 swaps after the first succeeded; whether
  either path changes T on a rejection is untested.
- [Complete evidence](evidence/2026-10-26-lapack-validation.tar.gz) preserves
  final logs, exact selections, environment/build timing, source proof,
  script checks and the high-precision reference data/calculation.
