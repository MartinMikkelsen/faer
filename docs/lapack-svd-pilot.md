# Focused f64 LAPACK SVD validation

This is the first bounded Rust adaptation for [#31](https://github.com/tensor4all/faer/issues/31).
It exercises faer's low-level `svd` interface with selected LAPACK matrix
families and correctness checks. It requires neither OpenBLAS nor a Fortran
compiler. It is not the complete LAPACK SVD suite.

## Running it

From the repository root, run all six validation groups explicitly:

```sh
CARGO_PROFILE_DEV_DEBUG=0 cargo test -p t4a-faer --lib lapack_svd_f64 -- --ignored --nocapture --test-threads=1
```

The groups share the `lapack_svd_f64` name prefix and are ignored in ordinary
test execution by design. They are not expected-failure tests: every violated
check fails the explicit run, including known defects. Three small checker
self-tests run in ordinary tests:

```sh
CARGO_PROFILE_DEV_DEBUG=0 cargo test -p t4a-faer --lib linalg::svd::tests::lapack::checkers
```

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

The implementation lives in `faer/src/linalg/svd/tests/lapack.rs`, a child of
the existing SVD test module. The original copyright notices are retained
there; the license terms are in `faer/COPYING.LAPACK.BSD`. The package's SPDX
metadata includes BSD-3-Clause for this adaptation.

Rust matrix construction replaces the Fortran generators. Diagonal signs and
orthogonal rotations are deterministic; the rotated families use fixed 3/5,
4/5 rotations rather than DLATMS or faer's QR. Family 16 uses faer's existing
rand dependency with seed `0x4C415041434B`, reset for each dimension, and entries
log-uniform on `[EPSILON², EPSILON⁻²]`. This does not reproduce LAPACK's random
sequence. The pilot uses scales 1, `sqrt(f64::MAX)`, and
`sqrt(f64::MIN_POSITIVE)`, following the family descriptions rather than the
drivers' complete machine-parameter scaling machinery.

## Coverage

Every driver case uses sequential `f64` and five `(U, V)` output modes: `(Full, Full)`,
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

Total: **1,647 SVD calls**: 1,195 driver calls and 452 direct bidiagonal calls.
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
norm and matrix-product implementations, on driver matrices no larger than
129×129 and padded direct-solver matrices no larger than 130×129.
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
six groups pass:

| Group | Result |
| --- | --- |
| Direct DCHKBD 15–19 | Passed |
| Diagonal families | Passed |
| Rotated families | Passed |
| Bidiagonal family | Passed |
| Rank one at recursion splits | Passed |
| Reused output buffers | Passed |

The split-position rank-one cases are regression coverage for
[#33](https://github.com/tensor4all/faer/issues/33), fixed by #39, and the
reused-output group for [#41](https://github.com/tensor4all/faer/issues/41),
fixed by #47. The same test source run against fork revision
`d3215290618aa49a1e246c7c2d8989915688c594`, which predates both fixes, fails
three groups and passes the other three:

- `lapack_svd_f64_dchkbd_15_to_19`: the 8-by-8 case with the nonzero entry at
  `(4,4)` and threshold 4 fails when the direct solver omits U, with or without
  V: known-spectrum ratio `5.62949953421312e14` and DCHKBD test 19 ratio
  `4.503599627370496e15`. This isolates #33 without the driver's reduction.
- `lapack_svd_f64_rank_one_at_recursion_splits`: the same input through `svd`
  fails known-spectrum and output-mode agreement for U-only and values-only
  calls, each with ratio `5.62949953421312e14`.
- `lapack_svd_f64_prefilled_outputs`: full U for 4×2 and full V for 2×4 fail
  orthogonality with ratio `6.755399441055744e15`.

The NaN-filled outputs cause no other failure at either revision.

On `main`, the largest ratio across all 5,237 checks is 1.77 (V orthogonality,
rotated family 10, 8×8, recursion threshold 128), against the threshold of 50.
In the direct group the largest is 1.0 (test 15, family 16, n=8, threshold 8).
Test 19 and the driver's output-mode agreement are exactly 0 in every case. The
ratios were collected by temporarily logging every check; that logging is not
part of the source.

The explicit run of the six groups and three checker self-tests took 2.06
seconds of test execution. This is one local runtime observation, not a paired
performance comparison or a CI budget guarantee. No upstream faer revision was
run.

The ordinary workspace test run reached 249 passing library tests, 3 failures,
and 8 ignored tests (the six new opt-in groups plus two existing tests). The
three failures are the strict-triangle tests that `PATCHSET.md` already records
as failing on `main`, in files this port does not touch:

- `linalg::matmul::internal::tests::empty_strict_triangle_preserves_parent_guards`
- `linalg::matmul::internal::tests::strict_triangle_preserves_nonempty_parent_guards`
- `linalg::matmul::internal::tests::strict_triangle_maps_preserve_stored_entries_and_parent_guards`

The `linalg::svd` library tests pass (25 passed; 7 ignored: the six new groups
and the existing `test_josef`). All 55 workspace doctests pass. The test file
passes `rustfmt --check`, and the diff has no whitespace errors.

Not covered yet: other scalar types, a general layout/thread sweep, all
LAPACK generators and fixtures (including the random dense families 13–15),
the independent DSVDCH spectral check, selected-value algorithms,
invalid-argument/workspace sweeps, or scheduled CI. Within the f64 cases, only
the reused-output inputs reach the QR pre-reduction path (aspect ratio above
`qr_ratio_threshold = 11/6`), and 4 of the 9 U/V output-mode combinations are
not exercised. The direct group tests lower bidiagonals only, through the
private solver. Its diagonal families have a zero off-diagonal, which the
solver resolves without iterating, so only family 16 and the split-position
case exercise its iteration; DCHKBD's bidiagonals from reduced families 8–15
are not reproduced.
