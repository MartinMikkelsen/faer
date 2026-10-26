# Focused f64 LAPACK SVD validation

This is an opt-in f64 validation suite for faer's SVD. It adapts selected
LAPACK test families and checks from DCHKBD and DDRVBD to Rust, and adds
faer-specific cases. It tests the `svd` driver and, directly, the private real
bidiagonal solver. The faer-specific cases cover strided views, reused buffers,
recursion and QR-path boundaries, and parallel execution. It needs neither
OpenBLAS nor a Fortran compiler, and a weekly workflow runs it. It covers part
of [#31](https://github.com/tensor4all/faer/issues/31) and is not the complete
LAPACK SVD suite; the gaps are listed under "Not covered yet".

## Running it

From the repository root, run all eight validation groups explicitly:

```sh
CARGO_PROFILE_DEV_DEBUG=0 cargo test -p t4a-faer --lib lapack_svd_f64 -- --ignored --nocapture --test-threads=1
```

The groups share the `lapack_svd_f64` name prefix and are ignored in ordinary
test execution by design. They are not expected-failure tests: every violated
check fails the explicit run, including known defects. The parallel group
requires the `rayon` feature (enabled by default). Five small checker/fixture
self-tests run in ordinary tests; to run them with the existing SVD tests:

```sh
CARGO_PROFILE_DEV_DEBUG=0 cargo test -p t4a-faer --lib linalg::svd::tests -- --test-threads=1
```

## Scheduled validation and budget

[`svd-validation.yml`](../.github/workflows/svd-validation.yml) runs the entire
SVD test module with `--include-ignored`: the eight validation groups plus the
ordinary SVD and self-tests, 26 tests in all. New tests under that module are
selected without editing the workflow. It runs weekly on Mondays at 03:23 UTC
(12:23 JST), with a manual `workflow_dispatch` trigger. Weekly is the initial
cadence until a hosted run has measured the clean dependency build; the test
execution itself is a few seconds. The regular push/PR build gate is
unchanged; there is one Ubuntu runner and no OS, precision, or feature matrix.

The limits are **two minutes for test execution**, **30 minutes for the
validation job**, and **two minutes for a separate failure-reporting job**.
The larger validation limit reserves room for toolchain setup, dependency
compilation, cache handling, and log upload; it is not an expected runtime.

The workflow's build and test commands were measured locally on 2026-10-26 on
macOS arm64 with Rust 1.98.1 and the repository's optimized development profile
(`opt-level = 3`), with debug information and incremental compilation disabled
as in the workflow, on `main` at `5a27bcece2f9f0d1c1b94c9b42dd7bfc4a94dc39`
plus the test-only additions:

| Phase | Wall seconds, including Cargo | Test execution seconds | Result |
| --- | ---: | ---: | --- |
| Build test binary, dependencies already cached | 41.11 | — | Passed |
| SVD test run 1 | 8.65 | 7.34 | 26 passed, 0 failed; exit 0 |
| SVD test run 2 | 7.70 | 7.30 | 26 passed, 0 failed; exit 0 |
| SVD test run 3 | 7.69 | 7.29 | 26 passed, 0 failed; exit 0 |

The hosted runner is x86_64, so it also takes faer's x86 SIMD kernels rather
than the arm64 kernels measured here. An explicit `shell: bash` runs each step
with `-eo pipefail`, so a failing `cargo test` fails its step even though the
output is piped through `tee`. The cache is saved on failure so a failing run
does not force repeated dependency builds. Environment, build timing, and test
logs are uploaded with 30-day retention.

Scheduled failures are reported to maintainers through one open issue titled
`Scheduled SVD validation is failing`, under the label
`scheduled-svd-validation`, in the repository running the workflow. A separate
job creates it on the first failure and adds a comment with the run, event,
ref, commit, and job result on each later one. A job that exceeds its time
limit ends as `cancelled` and is reported too; so is a manual cancel, which the
comment marks as such. Manual runs on branches other than the default branch
are not reported. The job does not suppress the failed run, create a new issue
on every failure, write to the upstream tracker, or close the issue; maintainers
close it after a green run.

This is an initial budget from local observations, not a hosted timing
measurement or a performance comparison. The workflow has not yet run on
GitHub. Both the schedule and the manual trigger need the file on the default
branch ([scheduled-workflow rules](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)),
so the first hosted run happens after merge; dispatching it once manually then
measures the clean build and the x86_64 test time. Inspect those logs before
adding a matrix or changing the cadence, and investigate overruns rather than
raising the limits.

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
- [`TESTING/svd.in`](https://github.com/OpenMathLib/OpenBLAS/blob/791b02a1a00c996b66a894f6df93421c53b62d80/lapack-netlib/TESTING/svd.in):
  the fixed failure threshold of **50** and selected small dimensions.

The LAPACK adaptations live in `faer/src/linalg/svd/tests/lapack.rs`, and the
faer-specific view/lifecycle cases in its sibling `svd_views.rs`. Both are
children of the existing SVD test module. `lapack.rs` retains the source
provenance and copyright notices; `svd_views.rs` contains faer's own cases and
imports its generators and checks from `lapack.rs`. The license terms are in
`faer/COPYING.LAPACK.BSD`. The package's SPDX metadata includes BSD-3-Clause
for this adaptation.

Rust matrix construction replaces the Fortran generators. Diagonal signs and
orthogonal rotations are deterministic; the rotated families use fixed 3/5,
4/5 rotations rather than DLATMS or faer's QR. Family 16 uses faer's existing
rand dependency with seed `0x4C415041434B`, reset for each dimension, and entries
log-uniform on `[EPSILON², EPSILON⁻²]`. This does not reproduce LAPACK's random
sequence. The pilot uses scales 1, `sqrt(f64::MAX)`, and
`sqrt(f64::MIN_POSITIVE)`, following the family descriptions rather than the
drivers' complete machine-parameter scaling machinery.

## Coverage

The base driver groups use sequential `f64` and five `(U, V)` output modes: `(Full, Full)`,
`(Thin, Thin)`, `(Thin, No)`, `(No, Thin)`, and `(No, No)`. Scratch is obtained
from the matching `svd_scratch` query. U, V and S are filled with NaN before
each call, so an entry the routine leaves unwritten fails a check; the
reused-output group fills U and V with ones instead.

| Group | Cases and dimensions | SVD calls |
| --- | --- | ---: |
| Diagonal families 1–7 | Zero, identity, linear/geometric/clustered spectra, and large/small linear spectra; shapes `(0,0)`, `(0,3)`, `(3,0)`, `(1,1)`, `(3,3)`, `(7,7)`, `(8,8)`, `(9,9)`, `(10,16)`, `(16,10)`; recursion thresholds 8 and 128 | 700 |
| Rotated families 8–12 | Known linear/geometric/clustered spectra and large/small linear spectra; shapes `(3,3)`, `(7,7)`, `(8,8)`, `(9,9)`, `(10,16)`, `(16,10)`; thresholds 8 and 128 | 300 |
| Bidiagonal family 16 | Lower bidiagonal inputs and their transposes; square sizes 7, 8, 9, 127, 128, 129; thresholds 8 and 128 | 120 |
| Direct DCHKBD 15–19 | Diagonal families 1–7 and logarithmic lower bidiagonal family 16; sizes 1, 7, 8, 9, 127, 128, 129; thresholds 8 and 128; plus the size-8 split-position rank-one case at threshold 4; all four U/V presence combinations | 452 |
| Rank-one extension | A single nonzero entry at the first, middle, or last diagonal position; size 8 / threshold 4, plus sizes 127, 128, 129 / threshold 128, motivated by #33 | 60 |
| Reused-output extension | Identity inputs of shapes `(4,2)`, `(2,4)`, `(4,4)`; U/V buffers filled with ones, motivated by #41 | 15 |
| Layouts and actual buffer reuse | Six shapes × six spectrum families × four input views × two output orientations × five output modes × two successive calls in the same layout, threshold 8 | 2,880 |
| Default parallel paths | Shapes `(257,256)`, `(512,128)`, `(128,512)`; sequential and dedicated 2-/4-thread pools; four thin/omitted-factor modes; two successive calls; parallel spectra compared with the sequential run | 72 |

Total with default features: **4,599 SVD calls**: 4,147 driver calls and
452 direct bidiagonal calls.
Threshold 8 brings divide-and-conquer into small
cases; threshold 128 covers the shipped default and its 127/128/129 boundary.
The general bidiagonal-family group includes the driver's reduction stage.
The direct group calls `compute_bidiag_real_svd` with its matching
`bidiag_real_svd_scratch` query and bypasses that stage. It requests both
factors, U only, V only, and values only, with vector buffers filled with
NaN. Faer's private solver stores an n-by-n lower bidiagonal as an
(n+1)-by-n matrix with a zero last row and an (n+1)-column U. The reconstruction
and U-orthogonality checks include that padding and retain DCHKBD's denominator
n, rather than weakening it to n+1.

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
`-A/2` in the same strided layout, whose known singular values are halved.
U, V and the singular values start as NaN to detect incomplete overwrite.
After each call, the mathematical checks run, visible parent borders must
remain unchanged, and the current input must be bitwise unchanged. A small
self-test proves the guard checker rejects writes at each parent edge.
Panics and convergence errors are recorded as failures while remaining cases
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
floor; their 128×128 R factor is below the reduction's floor
(127² < 192 · 256), so in those shapes the bidiagonal reduction runs
sequentially. Each parallel call's singular values must also agree with the
sequential call on the same input and mode, normalized as the output-mode
agreement check is. Thin/thin, U-only, V-only, and values-only modes avoid expensive full
512-column reference Gram matrices. All use guarded buffers and actual reuse,
as above; this is selected coverage, not a full layout/thread Cartesian sweep.

## Numerical checks

- Reconstruction: `||A - U S Vᵀ||₁ / (||A||₁ max(m,n) EPSILON)`.
  A and S are divided by the largest absolute input entry before products are
  formed. A zero input requires an exactly zero reconstructed matrix.
- Orthogonality: `||I - QᵀQ||₁ / (rows(Q) EPSILON)`, including all columns of
  full factors. Reconstruction alone cannot check unused full-factor columns.
- Spectrum: all singular values must be finite, nonnegative, and decreasing.
  Known spectra and agreement across output modes use a maximum absolute
  difference normalized by the largest reference singular value, dimension,
  and EPSILON. A zero reference spectrum requires exactly zero values.

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

The checks use scalar reference products and reductions, independent of faer's
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

## Result (2026-10-26)

On macOS arm64, Rust 1.98.1, the repository's optimized development profile
with debug info disabled, and `main` at
`5a27bcece2f9f0d1c1b94c9b42dd7bfc4a94dc39` plus the test-only additions, all
eight groups pass:

| Group | Result |
| --- | --- |
| Direct DCHKBD 15–19 | Passed |
| Diagonal families | Passed |
| Rotated families | Passed |
| Bidiagonal family | Passed |
| Rank one at recursion splits | Passed |
| Reused output buffers | Passed |
| Layouts and actual buffer reuse | Passed |
| Default parallel paths | Passed |

The split-position rank-one cases are regression coverage for
[#33](https://github.com/tensor4all/faer/issues/33), fixed by #39, and the
reused-output and layout groups for
[#41](https://github.com/tensor4all/faer/issues/41), fixed by #47. The same
test source run against fork revision
`d3215290618aa49a1e246c7c2d8989915688c594`, which predates both fixes, fails
four groups and passes the other four, including the parallel group:

- `lapack_svd_f64_dchkbd_15_to_19`: the 8-by-8 case with the nonzero entry at
  `(4,4)` and threshold 4 fails when the direct solver omits U, with or without
  V: known-spectrum ratio `5.62949953421312e14` and DCHKBD test 19 ratio
  `4.503599627370496e15`. This isolates #33 without the driver's reduction.
- `lapack_svd_f64_rank_one_at_recursion_splits`: the same input through `svd`
  fails known-spectrum and output-mode agreement for U-only and values-only
  calls, each with ratio `5.62949953421312e14`.
- `lapack_svd_f64_prefilled_outputs`: full U for 4×2 and full V for 2×4 fail
  orthogonality with ratio `6.755399441055744e15`.
- `lapack_svd_f64_layouts_and_reused_buffers`: 480 full-factor orthogonality
  checks fail, 240 for U in tall inputs and 240 for V in wide inputs, on the
  QR side of the aspect-ratio boundary. The 29×16 cases and all thin/omitted
  factor modes pass. No parent-guard overwrite, input change, convergence
  error or panic occurs.

On `main`, the largest ratio across all 12,625 checks is 2.30 (U
orthogonality, kind 6, 7×17 padded row-major input, threshold 8), against the
threshold of 50. The largest parallel-versus-sequential difference is 0.16
(512×128, `Par::rayon(2)`, thin factors): the parallel spectra are not
bitwise identical to the sequential ones. Test 19 and the driver's
output-mode agreement are exactly 0 in every case. The ratios were collected
by temporarily logging every check; that logging is not part of the source.

The explicit run of the eight groups and five self-tests took 7.37 seconds of
test execution. This is one local runtime observation, not a paired
performance comparison or a CI budget guarantee. No upstream faer revision was
run.

The ordinary workspace test run reached 251 passing library tests, 3 failures,
and 10 ignored tests (the eight new opt-in groups plus two existing tests). The
three failures are the strict-triangle tests that `PATCHSET.md` already records
as failing on `main`, in files this port does not touch:

- `linalg::matmul::internal::tests::empty_strict_triangle_preserves_parent_guards`
- `linalg::matmul::internal::tests::strict_triangle_preserves_nonempty_parent_guards`
- `linalg::matmul::internal::tests::strict_triangle_maps_preserve_stored_entries_and_parent_guards`

The `linalg::svd` library tests pass (27 passed; 9 ignored: the eight new groups
and the existing `test_josef`). All 55 workspace doctests pass. Both test files
pass `rustfmt --check`, and the diff has no whitespace errors.

Not covered yet: other scalar types, all LAPACK generators and fixtures
(including the random dense families 13–15), the independent DSVDCH spectral
check, selected-value algorithms, or invalid-argument/workspace sweeps.
Within the f64 cases, 4 of the 9 U/V output-mode combinations
are not exercised; the layout and parallel groups are selected cases, not a
full shape/stride/thread sweep, and the parallel group requests no full
factors. The direct group tests lower bidiagonals only, through the private
solver. Its diagonal families have a zero off-diagonal, which the solver
resolves without iterating, so only family 16 and the split-position case
exercise its iteration; DCHKBD's bidiagonals from reduced families 8–15 are
not reproduced.
