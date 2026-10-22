# Public Schur, bidiagonal SVD, and balancing API

## Decisions

- Implements tensor4all/faer#26 as a correctness/API experiment on base
  `50c2b989484f063f4e1b132adc333715391d8b73`. No performance claim; no upstream
  revision was measured, and no upstream feedback is prepared or sent.
- Public wrappers live in the existing `linalg::evd` and `linalg::svd` modules.
  The doc-hidden Schur module's visibility and existing numerical kernels are
  unchanged. Real and complex Schur entry points follow the existing
  `_real`/`_cplx` naming. Scratch queries use the existing `StackReq` convention.
  A review asked for `_in_place`/`_into` names under the shared output-update
  naming rule; the maintainer withdrew it because `docs/review-rules.md` makes
  faer's conventions take precedence, so the faer-style names are kept and the
  rustdoc states which arguments are overwritten and that `x` is not read.
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
  rejection error. `reorder_schur_scratch` sizes the copy that lets a real
  swap be undone (see the review fixes below); the complex path uses none.
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

## Review fixes (PR #27; rebased on `064016e`)

### Real Schur reordering

- The review showed `Ok` results with a broken `q` (`||q t q^T - t|| / ||t||
  = 1`). faer's 2x1 and 1x2 kernels build their reflectors directly from the
  window, apply them without a stability test and always return 0. The public
  path now uses ports of LAPACK `dlaexc` for these cases (`real_schur.rs`:
  `lasy2_12_21` for `dlasy2`, `larfg3` using faer's Householder generator
  with LAPACK's `tau`, `larfx3_*` for `dlarfx`, `laexc_12`/`laexc_21`). Each
  swap is computed provisionally on a copy of the `3 x 3` window and rejected
  when the provisional entries that must vanish, or the moved 1x1 block's
  diagonal value, are off by more than `max(10 eps |d|_max, safmin / eps)`;
  this is `dlaexc`'s test, plus a rejection of NaN. The 1x1 (rotation) and
  2x2 (Sylvester, same test) kernels already had `dlaexc`'s shape and are
  reused. A flag on the dispatcher keeps faer's historical 2x1/1x2 kernels for
  the internal `schur_swap` callers (QR and its deflation).
- An earlier attempt used a custom acceptance criterion (similarity residual
  plus orthogonality of a trial on the window). It was dropped: it rejected
  the review example that LAPACK accepts, it is not LAPACK's test, and it
  could not see updates outside the window.
- Overflow outside the window: with a column near `1e308` beside the window,
  `dlarfx`'s `v^T c` can overflow before cancellation even though the
  reflected column is representable (Accelerate `dtrexc` returns it finite).
  `larfx3_vec` keeps `dlarfx`'s arithmetic and, only for a vector whose
  direct update is not finite, recomputes it scaled by the exact power of two
  `1/8`; since `|v_i| <= 1` and `tau <= 2`, every intermediate then stays
  below `max_positive`, so the result overflows only if the reflected vector
  itself is not representable.
- Contract guarantee: before each adjacent swap the public wrapper copies the
  rows and columns of `t` and the columns of `q` that the swap updates (into
  scratch now declared by `reorder_schur_scratch::<T>(dim)`, `O(n)` per
  swap). If the swap is rejected or leaves a non-finite entry there, the copy
  is restored and `SchurReorderError` is returned, so `t`/`q` are exactly as
  after the last successful swap; success implies finite output.
- Standardization: `lahqr_schur22` (faer's `dlanv2`) computes
  `sqrt(c^2 + z^2)` unscaled where `dlanv2` calls `dlapy2`; overflow made
  `cs = sn = 0`. A first fallback to faer's generic `hypot` still overflowed
  (`hypot(1e308, 1e297) = inf`, the generic magnitude squares scaled
  components). The fallback is now a port of `dlapy2` (`lapy2`), taken only
  when the unscaled value is not finite or below `sqrt(safmin / eps)`, so
  other inputs keep the historical arithmetic.
- Cross-check against macOS Accelerate `dtrexc`/`strexc` (harness outside the
  repository). The review examples agree: both review 2x1 cases, the
  mirrored 1x2 case, both outside-window columns and the standardization case
  are accepted with finite output. The existing two-pair case is rejected by
  both. A scan of 200,000 3x3/4x4 cases has diagonal and coupling entries with
  magnitudes `10^U(-300, 300)` or `10^U(-308, 308)`. One third of the cases
  have off-window entries up to `max_positive` and couplings of `1e300` to
  `1e308`. Results:
  - faer reports success in 199,898 cases, none corrupt (corrupt means
    non-finite, or residual or orthogonality error above `1e-12`).
  - faer never rejects a case where LAPACK returns a good result.
  - Where faer rejects (102 cases), LAPACK either rejects too (27) or
    reports success with corrupt output (75).
  - LAPACK reports success with corrupt output in 843 cases overall.
  - In 3 cases faer succeeds (residual below `6e-16`) where LAPACK rejects. In
    each, the window holds entries of about `1e308`, so LAPACK's provisional
    `dlarfx` on the window overflows; ours takes the rescaled path.

### Bidiagonal SVD

- The shared complex phase normalization (`norm1.recip()`), complex
  magnitude (`abs`, whose scaled squares flush below `min_positive` and
  overflow above about `max_positive / 2`), and QR and divide-and-conquer
  pre-scaling (`max.recip()`) failed at the ends of the range. The
  maintainer's eight cases (`[1e-40f32]`, `[1e-310f64]`,
  `[c32(1e-40, 1e-40)]`, `[c64(1e-310, 1e-310)]`, upper and lower) returned
  `Ok` with `inf`/NaN; `c64(1e-320, 1e-320)` gave a zero singular value and
  `c64(MAX/2, MAX/2)` `NoConvergence`. Each helper now rescales by the exact
  power of two `1 / min_positive` (or divides by `max`) only outside
  `[min_positive, 1 / min_positive]` or when the reciprocal is not finite and
  normal. A 480-case output hash (complex and real bidiagonal SVD, QR and
  divide-and-conquer, and `Mat::svd`) is identical to the pre-fix revision.
  The public entry also returns `SvdError::NoConvergence` instead of
  non-finite output.
- Observation, not changed here: `Mat::svd` returns `NoConvergence` for a
  matrix mixing `1e200` real parts with O(1) imaginary parts, before and
  after this change (general bidiagonalization path).

### Provenance

- The `dlaexc`, `dlasy2`, `dlarfx` (order 3) and `dlapy2` ports and the
  `dlanv2`-style fallback are translations of Reference-LAPACK v3.12.1
  (commit `6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca`; these routines are
  unchanged on `master` at the time of writing). `real_schur.rs` records the
  routines, URL and material changes at the port site. Licensing follows the
  repository's existing convention for ported code: the routines are added
  to the "ported algorithms" list of `faer/COPYING.LAPACK.BSD`, which carries
  the LAPACK copyright notice and modified-BSD terms and ships with the crate,
  and `NOTICE` lists Schur reordering on its LAPACK line. The port-site
  comment points there instead of repeating the notice, as the existing
  LAPACK ports do. `Cargo.toml` keeps `license = "MIT"`: as `NOTICE` states,
  that metadata "describes the crate as a whole; it does not relicense those
  files", whose terms are in the COPYING files.

### Tests

- Public contracts (`faer/tests/`): `public_schur_reorder`
  (`reviewer_2x1_and_mirrored_1x2_swaps_are_accepted_and_stable`,
  `swap_keeps_large_outside_window_column_finite`,
  `overflowing_swap_is_rejected_unchanged`,
  `standardization_of_extreme_pair_stays_orthogonal`,
  `ordinary_2x1_and_1x2_swaps`, f32 and f64) and `public_bidiagonal_svd`
  (`subnormal_one_by_one_in_every_scalar_type`, `reviewer_magnitude_boundaries`,
  `reviewer_subnormal_one_by_one`, `tiny_and_large_magnitudes`).
- Private helpers, in child test modules of their owners:
  `evd::schur::real_schur::tests::lapack_ports` (`lapy2`, `lahqr_schur22`
  fallback, `lasy2_12_21`, `larfg3`/`larfx3_vec`, `laexc_thresh`,
  `laexc_12`/`laexc_21` acceptance and NaN rejection; f32 and f64),
  `svd::tests::cplx_phase_reduction_outside_normal_range` (f32/f64/c32/c64)
  and `svd::bidiag_svd::tests::prescale_outside_normal_range` (QR and
  divide-and-conquer, f32/f64). Each was checked to fail with its fix
  disabled.
- `eigenvectors_from_schur_*` never reads `x`; a test pre-fills it with NaN.

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
- Ordinary Cargo gates (stable 1.99.1, macOS arm64, after the review fixes):
  CI builds (default; `--no-default-features --features std`) with no warning
  beyond `main`'s; the five focused test targets (24 tests), `linalg::evd`
  (34) and `linalg::svd` (18 discovered, one of them, `test_josef`, ignored
  upstream) library tests and the `linalg::evd` (16) and
  `linalg::svd` (2) doctests pass in release. Clippy stops on deny-level lints
  in untouched files that `main` already has. `RUSTDOCFLAGS=-Dwarnings cargo doc` stops on the
  empty-code-block error that `main` already has; no new item warns.
- Formatting uses the CI's nightly formatter, restricted to changed ranges in
  existing files. The workspace-wide check also reports pre-existing formatting
  differences in untouched files; those are intentionally not reformatted.
