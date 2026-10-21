# Public Schur, bidiagonal SVD, and balancing API

## Decisions

- Implements tensor4all/faer#26 as a correctness/API experiment on base
  `50c2b989484f063f4e1b132adc333715391d8b73`. No performance claim; no upstream
  revision was measured, and no upstream feedback is prepared or sent.
- Public wrappers live in the existing `linalg::evd` and `linalg::svd` modules.
  The doc-hidden Schur module's visibility and existing numerical kernels are
  unchanged. Real and complex Schur entry points follow the existing
  `_real`/`_cplx` naming. Scratch queries use the existing `StackReq` convention.
- Hessenberg wrappers always request Schur form, update an optional accumulator,
  expose the kernel's eigenvalue order (positive imaginary part first for real
  pairs), and clear the kernel's below-band workspace before returning a full
  Schur matrix. Empty/singleton inputs bypass the QR kernel's active-range
  arithmetic. Nonfinite input and nonconvergence return `EvdError::NoConvergence`.
- As explicitly authorized by the maintainer, the real swap dispatcher is
  factored into `schur_swap_checked`. It retains both recursive split calls and
  block standardization, and returns the first nonzero split status or the
  per-shape swap status. The old `schur_swap` ignores that status and still
  returns zero. Its existing internal callers therefore keep identical handling
  of rejected swaps. In particular the aggressive-deflation path that reads
  `ierr` still never sees a rejection: this is an observation for maintainers,
  not something this experiment fixes.
- Public real reordering uses the checked dispatcher and reports the moving
  block's current first row on rejection. It leaves the last successful Schur
  state, including the existing block-standardization step. Public callers
  always identify whole standardized blocks; a pair's second row is mapped to
  its first row, and destination positions follow `xTREXC` size adjustments.
  Complex adjacent swaps unconditionally return zero, so their wrapper has no
  rejection error. Fixed-size swap storage requires no dynamic scratch.
- Right-eigenvector wrappers zero the conversion workspace, call the existing
  back substitution, optionally multiply by Schur vectors, and reuse the EVD
  boundary's unit-norm normalization and finite-output check. Real pair columns
  represent the positive-imaginary eigenvector as `re + i im`.
- Bidiagonal SVD uses the existing complex phase reduction and real
  divide-and-conquer/QR solver with a trailing zero subdiagonal and padded left
  factor. An upper input is handled as the adjoint of a lower bidiagonal,
  exchanging the resulting factors. The public output is `V`, not `Vᴴ`.
  Caller-owned offdiagonal slices are unchanged; diagonal slices become real,
  nonnegative, descending singular values.
- Balancing is an original implementation of Parlett–Reinsch permutation and
  radix-two scaling with the LAPACK 3.5+ 2-norm/5% convergence criterion
  (including the diagonal in the norm) and safe-range guards (Parlett–Reinsch, Numerische Mathematik 13, 293–304, 1969;
  James–Langou–Lowery, arXiv:1401.5766). The diagonal-inclusive norm rule was
  cross-checked against the locally cached LAPACK reference. No LAPACK code
  is copied or adapted.
  Metadata explicitly records the final permutation and diagonal factors:
  `B[i,j] = A[perm[i],perm[j]] * scale[j] / scale[i]`. The returned active range
  is zero-based and half-open, possibly empty. Right eigenvectors transform by
  `P D`; the API documents the inverse similarity for matrix functions.
- Scaling by powers of two is reversible without rounding for normal values;
  underflow can prevent bit-for-bit recovery for arbitrary floating-point
  inputs. The documentation states this limit instead of claiming unconditional
  reversibility. Extreme finite input is protected by scaling guards.

## Verification conclusions and constraints

- Five focused integration suites exercise f32/f64/c32/c64 at sizes
  1, 2, 3, 8, 33, and 100, using reconstruction/residuals, orthogonality,
  ordering and Schur-form invariants. Coverage includes random and structured
  inputs, real pairs crossing singleton blocks in both directions, both SVD
  orientations, zero/repeated singular values, exact balancing similarity and
  recovery on normal values, isolation permutations, right-eigenvector
  back-transformation, nonfinite errors, and exact-size tridiagonal EVD scratch
  for both values-only and vector-producing paths (including divide-and-conquer).
- A finite ill-conditioned two-pair example reaches the real swap rejection.
  It verifies unchanged factors on an initial rejection and preservation of
  an earlier successful swap when a later swap rejects.
- QR's iteration-limit failure is not forced: its limit is internal, and no
  finite nonconvergent example was established. Nonfinite-input error paths
  are covered. The iteration status is forwarded unchanged.
- Ordinary Cargo gates (rerun with network, stable 1.99.1, macOS arm64): CI
  builds (default; `--no-default-features --features std`) with no warning
  beyond `main`'s; the five focused test targets (14 tests), `linalg::evd`
  (27) and `linalg::svd` (13) library tests and the `linalg::evd` doctests
  (16) pass in release. `RUSTDOCFLAGS=-Dwarnings cargo doc` stops on the
  empty-code-block error that `main` already has; no new item warns.
- Formatting uses the CI's nightly formatter, restricted to changed ranges in
  existing files. The workspace-wide check also reports pre-existing formatting
  differences in untouched files; those are intentionally not reformatted.
