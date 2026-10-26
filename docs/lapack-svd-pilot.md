# Focused f32/f64/c32/c64 LAPACK SVD validation

This is the first bounded Rust adaptation for [#31](https://github.com/tensor4all/faer/issues/31).
It exercises faer's low-level `svd` interface with selected LAPACK matrix
families and correctness checks. It requires neither OpenBLAS nor a Fortran
compiler. It is not the complete LAPACK SVD suite.

The remaining #31 families build on this pilot; they are catalogued in
[the combined validation catalog](lapack-validation.md).

## Running it

From the repository root, run all thirty-two validation groups explicitly:

```sh
CARGO_PROFILE_DEV_DEBUG=0 cargo test -p t4a-faer --lib lapack_svd -- --ignored --nocapture --test-threads=1
```

The `lapack_svd_f32`, `lapack_svd_f64`, `lapack_svd_c32`, and
`lapack_svd_c64` modules instantiate
the same eight groups. Use a module name as the filter to select one scalar
type. The groups are ignored in ordinary test execution by design.
They are not expected-failure tests: every violated
check fails the explicit run, including known defects. The parallel group
requires the `rayon` feature (enabled by default). Six small checker/fixture
self-tests run per scalar type, plus three complex calibrations per complex
precision and native-f32 scale calibrations for f32/c32. To run these thirty-two
pilot checker/fixture tests with the existing SVD tests:

```sh
CARGO_PROFILE_DEV_DEBUG=0 cargo test -p t4a-faer --lib linalg::svd::tests -- --test-threads=1
```

## Scheduled validation and budget

The SVD-only job is now replaced by
[`lapack-validation.yml`](../.github/workflows/lapack-validation.yml).
Current all-family coverage, runtime, budget and failure reporting are recorded
in [the combined validation catalog](lapack-validation.md). The following
subsection preserves the original pilot measurements and configuration.

### Historical SVD-only configuration

The original `svd-validation.yml` ran the entire
SVD test module with `--include-ignored`: the thirty-two pilot groups plus 45
ordinary SVD/checker tests. New tests under that module are selected without
editing the workflow. It runs weekly on Mondays at 03:23 UTC (12:23 JST), with
a manual `workflow_dispatch` trigger for changes that need an earlier check.
Weekly is the initial cadence while clean hosted compilation is unmeasured
and known cases still fail.
The regular push/PR build gate is unchanged; there is one Ubuntu runner and no
OS, precision, or feature matrix in this pilot workflow.

The selected limits are **two minutes for test execution**, **30 minutes for
the validation job**, and **two minutes for a separate failure-reporting job**.
The configured job limits sum to 32 minutes when both jobs run. The larger
validation limit reserves room for toolchain setup, dependency compilation,
cache handling, and log upload; it is not an expected runtime.
The expanded four-type run took 22.58 seconds of test execution locally, so these
limits are retained; its results and source revision appear below.

The workflow build/test commands were first measured with the f64-only pilot
locally on 2026-10-26,
on macOS arm64 with Rust 1.98.1 and the repository's optimized development
profile (`opt-level = 3`). Both debug information and incremental compilation
were disabled, as in the workflow. Production code remained at
`d3215290618aa49a1e246c7c2d8989915688c594`.

| Phase | Wall seconds, including Cargo | Test execution seconds | Result |
| --- | ---: | ---: | --- |
| Build test binary, dependencies already cached | 44.78 | — | Passed |
| Combined SVD run 1 | 7.803 | 6.61 | 16 passed, 4 failed, 0 ignored; exit 101 |
| Combined SVD run 2 | 7.357 | 6.63 | 16 passed, 4 failed, 0 ignored; exit 101 |
| Combined SVD run 3 | 6.857 | 6.41 | 16 passed, 4 failed, 0 ignored; exit 101 |

Each initial combined run includes all 4,543 f64 pilot calls and the ordinary SVD/checker
tests. The same four pilot groups reproduce #33/#41 on this checkout. No
failure is allowed to pass CI: Bash pipefail preserves Cargo's exit status
while retaining logs. The cache is saved on failure so these known defects do
not force repeated dependency builds. Environment, build timing, and numerical
logs are uploaded with 14-day retention when the upload step can run.

Local execution of the f64-only workflow's Bash blocks confirmed a successful
build (42.17 seconds, dependencies cached) followed by exit 101 from the test
pipeline (6.78 seconds of test execution, 7.80 including Cargo). All 20 tests
ran, with the same 16 passes and four failures. YAML parsing, Bash syntax,
failure-status preservation through `tee`, and five mocked issue-reporting
scenarios pass, including first creation, updating an existing issue, excluding
pull requests/unrelated issues, and propagating API errors. No GitHub API
mutation was performed during these checks.

The shared CI policy requires scheduled failures to reach maintainers. A
separate reporting job creates or updates one open issue titled
`Scheduled SVD validation is failing`, under the label
`scheduled-svd-validation`, in the repository running the workflow. It records
the failed run, commit, and job result, including build failures, cancellations,
or validation-job timeouts.
It does not suppress the failed job, create a new issue on every failure, or
write to the upstream tracker. Maintainers can close the issue after a green
run; the workflow does not close issues automatically.

This is an initial budget based on local observations, not a hosted timing
measurement or a performance comparison. A clean hosted dependency build and
the first Ubuntu execution remain unmeasured. Inspect their build/test logs
before adding a matrix or increasing frequency; investigate overruns rather
than silently raising the limits. No workflow has been pushed or dispatched
from this session. GitHub's [scheduled-workflow rules](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)
require this file on the default branch before the weekly schedule can run.

## Source and adaptation

Original project: LAPACK 3.12.0, distributed in OpenMathLib/OpenBLAS revision
[`791b02a1a00c996b66a894f6df93421c53b62d80`](https://github.com/OpenMathLib/OpenBLAS/tree/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib).
The sources used are:

- [`TESTING/EIG/dchkbd.f`](https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib/TESTING/EIG/dchkbd.f):
  matrix families 1–12 and 16; reconstruction, orthogonality, spectrum ordering,
  and values/vector-mode agreement, including the properties in tests 15–19.
- [`TESTING/EIG/ddrvbd.f`](https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib/TESTING/EIG/ddrvbd.f):
  corresponding checks at the complete SVD interface, with full, thin, and
  omitted factors.
- [`TESTING/EIG/dbdt03.f`](https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib/TESTING/EIG/dbdt03.f):
  the bidiagonal reconstruction ratio used by DCHKBD test 15.
- [`TESTING/EIG/schkbd.f`](https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib/TESTING/EIG/schkbd.f),
  [`sdrvbd.f`](https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib/TESTING/EIG/sdrvbd.f), and
  [`sbdt03.f`](https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib/TESTING/EIG/sbdt03.f):
  corresponding single-precision checks; SCHKBD test 19 has the same
  safe-minimum denominator and no dimension factor.
- [`TESTING/EIG/zchkbd.f`](https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib/TESTING/EIG/zchkbd.f),
  [`zdrvbd.f`](https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib/TESTING/EIG/zdrvbd.f),
  [`zbdt03.f`](https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib/TESTING/EIG/zbdt03.f), and
  [`zunt01.f`](https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib/TESTING/EIG/zunt01.f):
  complex reconstruction, real ordered spectra, and unitary-factor checks.
  ZCHKBD has fourteen tests; this port does not invent complex tests 15–19.
  The shared direct group extends the D/SCHKBD 15–19 properties to faer's
  complex phase-removal adapter, rather than translating nonexistent ZCHKBD
  divide-and-conquer checks.
- [`TESTING/EIG/cchkbd.f`](https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib/TESTING/EIG/cchkbd.f),
  [`cdrvbd.f`](https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib/TESTING/EIG/cdrvbd.f),
  [`cbdt03.f`](https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib/TESTING/EIG/cbdt03.f), and
  [`cunt01.f`](https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib/TESTING/EIG/cunt01.f):
  corresponding single-precision complex reconstruction, ordered real spectra,
  and unitary-factor checks. CCHKBD likewise has only fourteen tests; c32's
  direct phase-adapter cases extend D/SCHKBD 15–19 properties.
- [`TESTING/svd.in`](https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib/TESTING/svd.in):
  the fixed failure threshold of **50** and selected small dimensions.

The LAPACK adaptations live in `faer/src/linalg/svd/tests/lapack.rs`, and the
faer-specific view/lifecycle cases in its sibling `svd_views.rs`. Both files
are included under concrete `lapack_svd_f32`, `lapack_svd_f64`,
`lapack_svd_c32`, and `lapack_svd_c64` child modules of the existing SVD tests, with private
`Scalar` storage and `Real` precision aliases. This shares every
case, generator, and checker without a generic numeric-trait layer or a second
copy of the case loops. Both files retain source provenance and
copyright notices. The license terms are in `faer/COPYING.LAPACK.BSD`. The
package's SPDX metadata includes BSD-3-Clause for this adaptation. The original
complex calibration fixtures in `tests/complex.rs` carry the fork's MIT notice.

Rust matrix construction replaces the Fortran generators. Diagonal signs/phases and
orthogonal/unitary rotations are deterministic; the rotated families use fixed 3/5,
4/5 rotations rather than DLATMS or faer's QR. Family 16 uses faer's existing
rand dependency with seed `0x4C415041434B`, reset for each dimension, and entries
log-uniform on `[EPSILON², EPSILON⁻²]`. This does not reproduce LAPACK's random
sequence; f32 and f64 draws need not be identical. The pilot uses scales 1,
`sqrt(Real::MAX)`, and `sqrt(Real::MIN_POSITIVE)`, following the family descriptions rather than the
drivers' complete machine-parameter scaling machinery.

Fixtures, fixed rotations, SVD calls, and scalar reference products all use
native `f32`, `f64`, `c32`, or `c64` arithmetic. EPSILON, safe minimum, overflow limit, geometric
spacing, and cluster gaps come from the corresponding real precision. In f32/c32 the extreme
fixture scales are approximately `1.845e19` and `1.084e-19`, and EPSILON is
`2^-23`; f64 and c64 use `2^-52`. The ratio threshold stays **50** for
all four scalar types. F32/c32 matrices are generated with these limits directly;
they are not casts of f64/c64 extreme-scale fixtures. Calibration tests check their
large/small scales, geometric endpoint, and cluster spacing independently.

C32/c64 diagonal phases cycle through `i`, `0.6 + 0.8i`, `-0.8 + 0.6i`, and `-1`.
The fixed rotations use imaginary off-diagonal coefficients: each two-row
rotation is `[[0.6, 0.8i], [0.8i, 0.6]]`, with the corresponding two-column
rotation using 0.8 and 0.6. Conjugate coefficients make these rotations unitary.
This produces nonuniform complex phases and dense entries with both real and
imaginary components, while preserving independently known real spectra.
Rank-one and logarithmic-bidiagonal complex fixtures also carry unit phases.
The f32/f64 signs and positive logarithmic draws remain unchanged.

An independent calibration at each complex precision checks a 2×2 phased/rotated fixture against
four hand-derived complex entries. Additional checks require conjugation on
`A = [-2i], U = [1], S = [2], V = [i]`, verify `[i]^H [i] = [1]`, reject a
wrong V phase, and reject NaNs confined to imaginary components. Faer's
complex singular-value buffers must have exactly zero imaginary parts;
nonzero, infinite, and NaN imaginary outputs are rejected before using their
real parts for spectrum and reconstruction checks.
Complex parent guards have a nonzero imaginary component, and input snapshots
compare the bits of both components.

## Coverage

The base driver groups use sequential `f32`, `f64`, `c32`, and `c64` and five `(U, V)` output modes: `(Full, Full)`,
`(Thin, Thin)`, `(Thin, No)`, `(No, Thin)`, and `(No, No)`. Scratch is obtained
from the matching `svd_scratch` query. Zero-filled output buffers are used
except in the explicit reused-output group.

| Group | Cases and dimensions | SVD calls per scalar type |
| --- | --- | ---: |
| Diagonal families 1–7 | Zero, identity, linear/geometric/clustered spectra, and large/small linear spectra; shapes `(0,0)`, `(0,3)`, `(3,0)`, `(1,1)`, `(3,3)`, `(7,7)`, `(8,8)`, `(9,9)`, `(10,16)`, `(16,10)`; recursion thresholds 8 and 128 | 700 |
| Rotated families 8–12 | Known linear/geometric/clustered spectra and large/small linear spectra; shapes `(3,3)`, `(7,7)`, `(8,8)`, `(9,9)`, `(10,16)`, `(16,10)`; thresholds 8 and 128 | 300 |
| Bidiagonal family 16 | Lower bidiagonal inputs and their transposes; square sizes 7, 8, 9, 127, 128, 129; thresholds 8 and 128 | 120 |
| Direct DCHKBD 15–19 properties | Diagonal families 1, 2, 4–7 and logarithmic lower bidiagonal family 16; sizes 1, 7, 8, 9, 127, 128, 129; thresholds 8 and 128; plus the size-8 split-position rank-one case at threshold 4; all four U/V presence combinations. C32/c64 test the complex phase-removal adapter | 396 |
| Rank-one extension | A single nonzero entry at the first, middle, or last diagonal position; size 8 / threshold 4, plus sizes 127, 128, 129 / threshold 128, motivated by #33 | 60 |
| Reused-output extension | Identity inputs of shapes `(4,2)`, `(2,4)`, `(4,4)`; U/V buffers filled with ones, motivated by #41 | 15 |
| Layouts and actual buffer reuse | Six shapes × six spectrum families × four input views × two output orientations × five output modes × two successive calls, threshold 8 | 2,880 |
| Default parallel paths | Shapes `(257,256)`, `(512,128)`, `(128,512)`; sequential and dedicated 2-/4-thread pools; four thin/omitted-factor modes; two successive calls | 72 |

Total with default features: **18,172 SVD calls**, comprising **4,543 per
scalar type** (4,147 driver and 396 direct adapter calls each). The combined
total is 16,588 driver and 1,584 direct adapter calls.
Threshold 8 brings divide-and-conquer into small
cases; threshold 128 covers the shipped default and its 127/128/129 boundary.
The general bidiagonal-family group includes the driver's reduction stage.
The direct group calls `compute_bidiag_real_svd` with its matching
`bidiag_real_svd_scratch` query and bypasses that stage. It requests both
factors, U only, V only, and values only, with vector buffers prefilled with
ones. Faer's private solver stores an n-by-n lower bidiagonal as an
(n+1)-by-n matrix with a zero last row and an (n+1)-column U. The reconstruction
and U-orthogonality checks include that padding and retain DCHKBD's denominator
n, rather than weakening it to n+1.
C32/c64 instead call `compute_bidiag_cplx_svd` with `bidiag_cplx_svd_scratch`,
exercising phase removal/restoration around the f32/f64 real bidiagonal core.
The complex-phased bidiagonal inputs are a faer-specific adapter extension;
LAPACK's C/ZCHKBD themselves use real bidiagonal coefficients.

The rank-one extension includes the documented 8-by-8 matrix with the nonzero
entry at `(4,4)` and threshold 4. Merely placing the entry at `(0,0)` does not
exercise #33's zero-child merge at the divide-and-conquer split.

### Faer-specific views and lifecycle

The layout group uses base shapes `(4,2)`, `(17,7)`, `(17,8)`, `(17,9)`,
`(29,16)`, `(30,16)` and also tests their transposes. The four input layouts
are padded column-major, padded row-major, and both with rows and columns
reversed (negative strides). Output U/V are inset views of guarded parent
matrices, in either column-major or row-major orientation. The singular-value
output is guarded too and has a negative stride for row-major outputs.
These use safe view constructors; no new unsafe code is needed.

The six spectrum families are geometric, dominant-value clustered, large and
small linear spectra from DCHKBD, plus two faer extensions: repeated levels
`[4, ..., 4, 1, ..., 1]` and tightly clustered values
`s[i] = 1 - 8 * EPSILON * i`. The latter are generator kinds 8 and 9 in Rust,
not DCHKBD matrix-type numbers. Fixed rotations preserve the known spectra.
Literal fixture tests check both new families rather than trusting a generator
to define its own expected values unchecked.

For every mode and output orientation, the same U, V, singular-value storage,
and scratch buffer serve two successive calls: the strided input A and then
an owned input `-A/2`, whose known singular values are halved. U/V start with
ones; singular-value slots start with NaNs to detect incomplete overwrite.
After each call, the mathematical checks run, visible parent borders must
remain unchanged, and the current input must be bitwise unchanged. A small
self-test proves the guard checker rejects writes at each parent edge.
Driver, direct-bidiagonal, and view calls record panics and convergence errors
as failures while remaining cases
continue; they are never converted into successful results.

The reduced recursion threshold is 8, with minimum dimensions 7/8/9 on both
sides of that boundary. Aspect ratios 29/16 and 30/16 straddle the shipped QR
ratio threshold 11/6. The larger default-threshold recursion boundary 127/128/129
remains covered by the base groups.

The parallel group verifies the same known rotated linear spectra with
`Par::Seq`, then `Par::rayon(2)` and `Par::rayon(4)` inside dedicated pools of
the matching size, without changing global parallelism. It retains default
algorithm parameters, including recursion threshold 128. At 257×256 the
bidiagonal reduction starts above its `192 * 256` parallel-work floor; the
512×128 and 128×512 cases exercise the tall/wide blocked QR route above its
floor. Thin/thin, U-only, V-only, and values-only modes avoid expensive full
512-column reference Gram matrices. All use guarded buffers and actual reuse,
as above; this is selected coverage, not a full layout/thread Cartesian sweep.

## Numerical checks

- Reconstruction: `||A - U S Vᴴ||₁ / (||A||₁ max(m,n) EPSILON)` (Vᵀ for real data).
  A and S are divided by the largest absolute input entry before products are
  formed. A zero input requires an exactly zero reconstructed matrix.
- Orthogonality/unitarity: `||I - QᴴQ||₁ / (rows(Q) EPSILON)`, including all columns of
  full factors. Reconstruction alone cannot check unused full-factor columns.
- Spectrum: all singular values must be finite, nonnegative, and decreasing.
  Known spectra and agreement across output modes use a maximum absolute
  difference normalized by the largest reference singular value, dimension,
  and EPSILON. A zero reference spectrum requires exactly zero values.

Complex entry magnitudes use native standard-library `hypot(re, im)`, so the
matrix 1-norm sums complex moduli. Some LAPACK helpers use BLAS's
`abs(re) + abs(im)` convention; these Rust ratios are bounded mathematical
adaptations rather than bit-for-bit copies of those residuals. Reconstruction
and unitary Gram products remain scalar and independent of faer's matrix
products and norms. Transposed row-major input views remain ordinary transpose
views; the reconstruction oracle always conjugates V regardless of layout.

The direct bidiagonal checks map explicitly to DCHKBD:

| Test | Property |
| --- | --- |
| 15 | Bidiagonal reconstruction |
| 16 | Orthogonality of the left factor, including faer's extra padded column |
| 17 | Orthogonality of the right factor |
| 18 | Finite, nonnegative singular values in non-increasing order |
| 19 | Spectrum agreement with and without vectors |

Test 19 uses the source's denominator
`max(sqrt(MIN_POSITIVE) * max(S1[0], 1), EPSILON * max(S1[0], S2[0]))`.
It has **no dimension factor**, unlike the driver's spectrum comparison.
The same check additionally covers one-sided vector requests. A checker
self-test verifies that a difference of 64 EPSILON units scores 64, not 0.5
at dimension 128, and checks the safe-minimum floor and nonfinite rejection.

The checks use native-precision scalar reference products and reductions, independent of faer's
norm and matrix-product implementations. Full-factor driver cases are at most
129×129 in the base groups and 30×16 (or its transpose) in the layout group.
Larger parallel cases use only thin/omitted factors, at most 512×128 or
257×256; the padded direct-solver matrices are at most 130×129.
Nonfinite intermediates fail explicitly. Ratios at or above 50 fail. Failures
include family/seed, dimensions, recursion threshold, output mode, buffer fill,
check name, and ratio; numerical failures are collected across the group.

These are normwise accuracy checks, not a guarantee of high relative accuracy
for tiny singular values. Family 16 has no independently supplied exact
spectrum. Singular-vector entries are not compared because signs and bases
within repeated singular subspaces are not unique.

## Initial result (2026-10-24)

On macOS arm64, Rust 1.98.1, the repository's optimized development profile
with debug info disabled, and fork production code at
`d3215290618aa49a1e246c7c2d8989915688c594` plus the test-only additions:

| Group | Result |
| --- | --- |
| Diagonal families | Passed |
| Rotated families | Passed |
| Bidiagonal family | Passed |
| Rank one at default boundary (first-entry only) | Passed; this did not cover the split-position reproduction of #33 |
| Reused output buffers | Failed: full U for 4×2 and full V for 2×4 |

Both failures have orthogonality ratio `6.755399441055744e15` against threshold
50. Reconstruction did not fail. They reproduce existing
[#41](https://github.com/tensor4all/faer/issues/41); the full-factor QR path
leaves prior entries in its top-right output block. The port preserves the
failing checks and does not include a production fix.

The explicit five-group run took 0.68 seconds of test execution (1.16 seconds
including Cargo invocation). This is one local runtime observation, not a
paired performance comparison or a CI budget guarantee. No upstream faer
revision was run.

The ordinary workspace test run reached 219 passing library tests, 3 failures,
and 7 ignored tests (the five new opt-in groups plus two existing tests). All
three failures were reproduced separately on an untouched archive of starting
revision `d3215290618aa49a1e246c7c2d8989915688c594`, on the same host and profile:

- `linalg::matmul::internal::tests::empty_strict_triangle_preserves_parent_guards`
- `linalg::matmul::internal::tests::strict_triangle_preserves_nonempty_parent_guards`
- `linalg::matmul::internal::tests::strict_triangle_maps_preserve_stored_entries_and_parent_guards`

The first two fail on zero-length scratch alignment; the third fails its
indexed strict-triangle product check. They are pre-existing and outside this
SVD test port. The focused existing SVD tests and the two new checker tests
pass (9 tests total). All 37 workspace doctests pass. The new Rust test file
passes the available formatter's check, the diff has no whitespace errors,
and Cargo's package listing includes the new test module and LAPACK license.

Not covered yet: other scalar types, a general layout/thread sweep, all
LAPACK generators and fixtures, the independent DSVDCH spectral check,
selected-value algorithms, invalid-argument/workspace sweeps, or scheduled CI.

## Direct-solver and split-position result (2026-10-26)

The expanded suite still tests fork production code at
`d3215290618aa49a1e246c7c2d8989915688c594`; no numerical implementation was
changed. All 1,591 configured calls ran. The diagonal, rotated, and
driver-bidiagonal groups pass. All three checker self-tests pass.

Three groups fail on that checkout:

- `lapack_svd_f64_dchkbd_15_to_19`: the added size-8 split-position rank-one
  input fails when the direct solver omits U, whether V is requested or not.
  DCHKBD test 19 reports `4.503599627370496e15`; the independent known-spectrum
  check also fails. The 392 calls from the LAPACK input families pass. This
  isolates #33 without the general driver's reduction stage.
- `lapack_svd_f64_rank_one_at_recursion_splits`: the 8-by-8 input with pivot
  `(4,4)` at threshold 4 fails known-spectrum and output-mode agreement for
  U-only and values-only calls, reproducing #33. Each ratio is
  `5.62949953421312e14` against threshold 50.
- `lapack_svd_f64_prefilled_outputs`: the two full-factor orthogonality
  failures from #41 remain, with ratios `6.755399441055744e15`.

The issues are closed on GitHub; these failures describe the pinned local
checkout, not a claim about the latest remote code. No cases were removed or
changed to expected failures, and the failure threshold remains 50.

The combined run of six validation groups and three checker self-tests took
2.10 seconds of test execution on the same host/profile as the initial run.
This is a single local runtime observation, not a performance comparison.
The ordinary workspace run has 220 passing library tests, the same three
pre-existing matmul failures listed above, and eight ignored tests (six new
validation groups plus two pre-existing tests).
The focused ordinary SVD run passes all ten existing/checker tests and leaves
the six opt-in validation groups ignored as intended. All 37 workspace
doctests pass on the expanded source.

## View, reuse, and parallel result (2026-10-26)

The eight default-feature validation groups ran all **4,543 configured SVD
calls**, with the same pinned production code and unchanged threshold 50.
Four groups pass and four fail:

- The new parallel group passes all 72 calls, including sequential controls,
  both thread pools, and both calls on reused buffers.
- The new layout/reuse group runs all 2,880 calls. Its only numerical failures
  are 480 full-factor orthogonality checks: 240 for U in tall inputs and 240
  for V in wide inputs, evenly split between the first and reused calls.
  They occur on the QR side of the aspect-ratio boundary and match #41.
  The 29×16 cases and all thin/omitted-factor modes pass. No input mutation,
  parent-guard overwrite, convergence error, or panic was observed.
- The three existing failing groups still reproduce #33/#41, as detailed
  above. The shared driver-check refactor did not change their outcomes.

The full opt-in run took 4.67 seconds of test execution on the same host/profile
as the earlier runs; this is a single local observation, not a performance
comparison. The ordinary workspace run has 222 passing library tests, the
same three pre-existing matmul failures, and ten ignored tests (eight validation
groups plus two pre-existing tests).
All 12 ordinary SVD/checker tests and all 37 workspace doctests pass. Both Rust
test modules pass the available formatter's check; the diff has no whitespace
errors, and Cargo's offline package listing includes both modules and the
LAPACK license.

That result measured the f64-only pilot. Arbitrary stride patterns, all thread counts,
all shape/scale/layout combinations, full LAPACK generators, high relative
accuracy of tiny singular values, and production fixes remain outside the
numerical pilot. The scheduled workflow and its initial budget are
described above; hosted execution remains unverified. No current remote
revision or upstream faer was tested.

## F32 extension result (2026-10-26)

Both precisions use the same eight groups: **9,086 configured calls were
attempted**, 4,543 per precision. One f32 call panicked; the shared call checker
recorded it and continued, so no remaining cases were skipped. All other calls
returned successfully from the SVD interface, though some mathematical checks
failed. The combined ordinary/opt-in run selected all 36 tests, with **27
passing and nine failing**, none ignored. Seven validation groups pass and
nine fail; all twenty ordinary SVD/checker tests pass.

- F64 retains the same four failing groups and the same #33/#41 diagnostics.
- F32 diagonal, rotated, and parallel groups pass. Both precisions' parallel
  groups pass all 144 configured calls in total. The guarded view/reuse group
  has the same 480 full-factor orthogonality failures per precision consistent
  with #41, without input or guard corruption.
- F32 direct-bidiagonal and rank-one groups reproduce the split-position
  spectrum failures from #33. Additionally, the size-8 rank-one input with
  pivot `(0,0)`, threshold 4, and `(U,V) = (No,Thin)` panics at
  `bidiag_svd.rs`'s `usize` scratch allocation: the matching `svd_scratch`
  buffer has zero bytes remaining when two `usize` values (16 bytes on this
  host) are requested. The harness preserves this diagnostic and uses no
  extra workspace padding.
- The f32 logarithmic-bidiagonal family also fails V orthogonality at size
  127, threshold 8, lower orientation, seed `0x4C415041434B`, for full/full,
  thin/thin, and V-only modes. The native-precision ratio is `143.95331`
  against 50. An independent f64 Gram calculation on the returned f32 V
  gives `||I - V^T V||_1 = 2.1793563577362455e-3`, or ratio
  `143.95091478233962` using f32 epsilon. This confirms the failure is not
  native-f32 reference accumulation error. F32 random draws differ from the
  f64 fixture; this is not a same-matrix cross-precision comparison.

These additional findings describe the local checkout; no current remote or
upstream revision was tested and no production numerical fix was made. SVD
production sources remain at `d3215290618aa49a1e246c7c2d8989915688c594`. A concurrent
uncommitted bounds-helper extraction in `faer/src/mat/matref.rs` was present
during the final run (observed file blob
`2c35e54825b100b44ca1cc6bbe27651bd5a18318`); it was left untouched. The source
state is therefore not an untouched archive of that commit.

With the workflow settings, the build step took 41.449 seconds and the test
invocation took 57.393 seconds, including a recompilation after concurrent
source edits; actual test execution was **14.26 seconds**. These are local
observations with cached dependencies, not clean hosted measurements or a
performance comparison. Both the two-minute test limit and 30-minute job
limit remain unchanged. The existing workflow selects both precisions without
an added matrix or workflow edit.

The f32 fixture-calibration test failed against the original f64-only
generators and passes with native f32 fixtures. All thirteen checker/fixture
self-tests pass, including both panic/error-continuation tests. Both included
test files and the changed SVD module pass formatting; Cargo's offline package
listing includes both shared sources and the LAPACK license.
The final workspace library run has 230 passes, the same three pre-existing
matmul failures listed above, and eighteen ignored tests (sixteen pilot groups
plus two pre-existing tests). Doctests were not repeated for this test-only
extension.

## C64 extension result (2026-10-26)

All three scalar types use the same eight groups: **13,629 configured calls
were attempted**, 4,543 per type. The combined scheduled command selects all
53 tests, with **40 passing and thirteen failing**, none ignored. All 29
ordinary SVD/checker tests pass, including the three complex calibrations.
The f32/f64 failure diagnostics are identical to the preceding two-type run.
The one f32 scratch panic is still recorded; all other configured calls return
successfully from SVD, with numerical failures retained separately.

C64 diagonal, rotated, logarithmic-bidiagonal, and parallel groups pass.
Its four failing groups reproduce the #33/#41 failure patterns:

- The direct adapter loses the split-position rank-one spectrum at size 8,
  threshold 4, when internal U is omitted (V-only and values-only). Known-spectrum
  and mode-agreement ratios fail. The driver likewise fails its U-only and
  values-only modes on that split-position fixture.
- Prefilled full outputs fail U unitarity for the 4×2 fixture and V unitarity
  for its 2×4 counterpart. Reconstruction does not detect the corrupted unused
  columns, so the full-factor Gram check is essential.
- The guarded layout/reuse group reports 480 full-factor unitarity failures:
  240 for tall U and 240 for wide V, split evenly between the initial and reused
  calls. There are no input modifications, overwritten parent guards,
  reconstruction failures, or spectrum failures in this group. The 29×16
  cases pass; the 30×16 cases fail, as in the real pilots.

All **216 parallel-group calls** pass across the three types, including c64's
72 calls with sequential and dedicated 2-/4-thread execution. No tolerance,
scratch padding, expected-failure assertion, or production numerical fix was
introduced. Complex fixture/checker calibrations failed before their
implementations and pass with the shared conjugating checks and phased inputs.

The build step took 39.583 seconds with cached dependencies. The subsequent
test invocation took 42.960 seconds including a Cargo build-directory lock
wait; actual test execution was **16.41 seconds**. This is a local budget
observation, not a performance comparison or a clean hosted timing. The existing
two-minute execution and 30-minute validation-job limits remain unchanged.
The workflow selects c64 automatically without a workflow edit.

SVD production sources remain at `d3215290618aa49a1e246c7c2d8989915688c594`.
Concurrent uncommitted bounds-helper work in `faer/src/mat/matref.rs` was present
(observed file blob `fffd23b01f9fe04c6ebf61cda8b529c1252f416a`) and was left
untouched. No current remote or upstream faer revision was tested. The pinned
complex LAPACK references retain the existing BSD notice and package metadata;
the independent complex calibration file is MIT-licensed.

The workspace library run has **239 passing tests, three pre-existing matmul
failures, and 26 ignored tests** (24 pilot groups plus two pre-existing tests).
The failure names and prior starting-revision reproductions are recorded above.
Formatting, whitespace checks, and the offline Cargo package listing pass;
all three test sources and the LAPACK license are included. Doctests were not
repeated for this test-only extension.

C32, full LAPACK generators, arbitrary layout/thread combinations, high
relative accuracy of tiny singular values, and clean hosted execution remain
outside this validation. No workflow was pushed or dispatched.

## C32 extension and current-checkout result (2026-10-26)

The pilot now covers **f32/f64/c32/c64** with the same eight opt-in groups per
type. C32 uses native f32 fixtures, epsilon, safe minimum, scalar products,
complex phases, conjugate reconstruction, and unitary-factor checks. Its scale
calibration rejects the double-precision fixture implementation, and all 32
ordinary pilot checker/fixture tests pass at their native precisions.

Production sources in this run are the current checkout,
`59a62189fdc7fad26dafe1a89ed84501d6118af7`, whose ledger records upstream base
Codeberg `0ecd846`. This checkout already contains the fixes for #33 and #41;
the earlier results above describe an older source state. This test extension
changes no production routine, dependency, API, scratch padding, or threshold.
No current remote or separate upstream checkout was tested.

All **18,172 configured calls were attempted**, 4,543 per scalar type. Two
single-precision calls panic; their contexts are recorded and later calls
continue. The combined scheduled command selects 77 tests: **74 pass and three
fail**, none ignored. All 45 ordinary SVD/checker tests pass. Of the 32 opt-in
groups, 29 pass:

| Type | Groups passing / failing | Remaining failures |
| --- | --- | --- |
| f64 | 8 / 0 | None in this bounded suite |
| c64 | 8 / 0 | None in this bounded suite |
| f32 | 6 / 2 | Rank-one scratch panic; logarithmic-bidiagonal V orthogonality |
| c32 | 7 / 1 | Rank-one scratch panic |

Both single-precision scratch panics occur on an 8×8 rank-one driver input with
pivot `(0,0)`, recursion threshold 4, sequential execution, and `(U,V) =
`(No,Thin)`. The c32 input has `A[0,0] = i`; the real fixture has `A[0,0] = 1`.
The exact matching `svd_scratch::<Scalar>` query supplies the buffer. Inside
`bidiag_svd.rs`'s divide-and-conquer merge, the first `col_perm` allocation
requests two `usize` entries (16 bytes on this host) with zero bytes remaining.
A focused c32 rerun reproduces the panic through the complex phase adapter
into the f32 core. No workaround was added to the harness.

The f32 logarithmic lower-bidiagonal fixture retains the preceding result:
size 127, threshold 8, seed `0x4C415041434B`, V orthogonality ratio `143.95331`
against 50 in full/full, thin/thin, and V-only modes. The independently checked
f64 Gram result is recorded in the f32 section. C32's phased fixture passes;
these different inputs do not establish a cross-type accuracy comparison.

All **288 parallel-group calls pass** across the four types. All full-factor
prefill and guarded layout/reuse groups pass on this checkout, with no input
modifications or overwritten parent guards. The split-position spectrum cases
also pass, consistent with the already-present #33/#41 fixes.

The native c32 test-binary compilation took 45.60 seconds with dependencies
cached. The subsequent workflow build step reused the binary (0.717 seconds),
and the combined test invocation took 23.029 seconds, including **22.58 seconds
of test execution**. These local observations retain the two-minute execution
and 30-minute validation-job limits; they are not cold hosted timings or
performance comparisons. The workflow selects all four types without an edit.

The workspace test invocation reaches **279 passing library tests, three
previously recorded matmul failures, and 34 ignored tests** (32 pilot groups
plus two existing tests). The failures remain:

- `linalg::matmul::internal::tests::empty_strict_triangle_preserves_parent_guards`
- `linalg::matmul::internal::tests::strict_triangle_preserves_nonempty_parent_guards`
- `linalg::matmul::internal::tests::strict_triangle_maps_preserve_stored_entries_and_parent_guards`

That library failure stops Cargo before doctests, which were not separately
repeated. The changed Rust sources and inserted module block pass formatting;
the complete parent module has one pre-existing formatting difference in
`full_factors_ignore_old_output_entries`, left untouched. The diff has no
whitespace errors, and Cargo's offline package listing includes all three test
sources and their license texts.

Full LAPACK generators, high relative accuracy of tiny singular values,
arbitrary layout/thread combinations, other LAPACK families, production fixes,
and clean hosted execution remain outside this pilot. No commits or remote
mutations were made.
