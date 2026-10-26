# Focused f64 LAPACK SVD validation

This is the first bounded Rust adaptation for [#31](https://github.com/tensor4all/faer/issues/31).
It exercises faer's low-level `svd` interface with selected LAPACK matrix
families and correctness checks. It requires neither OpenBLAS nor a Fortran
compiler. It is not the complete LAPACK SVD suite.

## Running it

From the repository root, run all five validation groups explicitly:

```sh
CARGO_PROFILE_DEV_DEBUG=0 cargo test -p t4a-faer --lib lapack_svd_f64 -- --ignored --nocapture --test-threads=1
```

The groups share the `lapack_svd_f64` name prefix and are ignored in ordinary
test execution by design. They are not expected-failure tests: every violated
check fails the explicit run, including known defects. Two small checker
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

Every case uses sequential `f64` and five `(U, V)` output modes: `(Full, Full)`,
`(Thin, Thin)`, `(Thin, No)`, `(No, Thin)`, and `(No, No)`. Scratch is obtained
from the matching `svd_scratch` query. Zero-filled output buffers are used
except in the explicit reused-output group.

| Group | Cases and dimensions | SVD calls |
| --- | --- | ---: |
| Diagonal families 1–7 | Zero, identity, linear/geometric/clustered spectra, and large/small linear spectra; shapes `(0,0)`, `(0,3)`, `(3,0)`, `(1,1)`, `(3,3)`, `(7,7)`, `(8,8)`, `(9,9)`, `(10,16)`, `(16,10)`; recursion thresholds 8 and 128 | 700 |
| Rotated families 8–12 | Known linear/geometric/clustered spectra and large/small linear spectra; shapes `(3,3)`, `(7,7)`, `(8,8)`, `(9,9)`, `(10,16)`, `(16,10)`; thresholds 8 and 128 | 300 |
| Bidiagonal family 16 | Lower bidiagonal inputs and their transposes; square sizes 7, 8, 9, 127, 128, 129; thresholds 8 and 128 | 120 |
| Rank-one extension | A single nonzero diagonal entry; square sizes 127, 128, 129 at the default threshold 128, motivated by #33 | 15 |
| Reused-output extension | Identity inputs of shapes `(4,2)`, `(2,4)`, `(4,4)`; U/V buffers filled with ones, motivated by #41 | 15 |

Total: **1,150 SVD calls**. Threshold 8 brings divide-and-conquer into small
cases; threshold 128 covers the shipped default and its 127/128/129 boundary.
The bidiagonal inputs go through the general SVD interface, including its
reduction stage; this pilot does not separately invoke the private bidiagonal
solver.

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

The checks use scalar reference products and reductions, independent of faer's
norm and matrix-product implementations, on matrices no larger than 129×129.
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
five groups pass:

| Group | Result |
| --- | --- |
| Diagonal families | Passed |
| Rotated families | Passed |
| Bidiagonal family | Passed |
| Rank one at default boundary | Passed |
| Reused output buffers | Passed |

The reused-output group is regression coverage for
[#41](https://github.com/tensor4all/faer/issues/41), fixed by #47. At fork
revision `d3215290618aa49a1e246c7c2d8989915688c594`, before that fix, it failed
full U for 4×2 and full V for 2×4 with orthogonality ratio
`6.755399441055744e15`; the other four groups passed there.

The largest ratio across all 3,785 checks is 1.77 (V orthogonality, rotated
family 10, 8×8, recursion threshold 128), against the threshold of 50. The
ratios were collected by temporarily logging every check; that logging is not
part of the source.

The explicit five-group run took 0.75 seconds of test execution. This is one
local runtime observation, not a paired performance comparison or a CI budget
guarantee. No upstream faer revision was run.

The ordinary workspace test run reached 248 passing library tests, 3 failures,
and 7 ignored tests (the five new opt-in groups plus two existing tests). The
three failures are the strict-triangle tests that `PATCHSET.md` already records
as failing on `main`, in files this port does not touch:

- `linalg::matmul::internal::tests::empty_strict_triangle_preserves_parent_guards`
- `linalg::matmul::internal::tests::strict_triangle_preserves_nonempty_parent_guards`
- `linalg::matmul::internal::tests::strict_triangle_maps_preserve_stored_entries_and_parent_guards`

The `linalg::svd` library tests pass (24 passed; 6 ignored: the five new groups
and the existing `test_josef`). All 55 workspace doctests pass. The new Rust
test file passes `rustfmt --check`, the diff has no whitespace errors, and
Cargo's package listing includes the new test module and LAPACK license.

Not covered yet: other scalar types, a general layout/thread sweep, all
LAPACK generators and fixtures (including the random dense families 13–15),
the independent DSVDCH spectral check, selected-value algorithms,
invalid-argument/workspace sweeps, or scheduled CI. Within the f64 cases, only
the reused-output inputs reach the QR pre-reduction path (aspect ratio above
`qr_ratio_threshold = 11/6`), outputs are zero-filled outside the reused-output
group, and 4 of the 9 U/V output-mode combinations are not exercised.
