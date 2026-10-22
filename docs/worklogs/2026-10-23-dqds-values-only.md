# dqds for values-only singular values

## Decisions

**What is missing.** `svd_values` loses to both vendors at every square size and the source says why:
LAPACK's `dgesdd(jobz='N')` never applies a rotation. It reaches `DBDSDC(compq='N')` ->
`DLASDQ(NCVT=0,NRU=0,NCC=0)` -> `DBDSQR`, which takes the `ROTATE` false branch and calls `DLASQ1`, the
Fernando-Parlett **dqds** iteration: O(n^2), no rotations, no vectors, high relative accuracy. faer has
no such path - below `recursion_threshold = 128` it runs a rotation-based implicit QR and above it a
divide and conquer that still builds merge intermediates for a values-only call. That explains the
whole measured pattern: 64 at 1.75x, 512 still at 1.22-1.39x, `svd_thin` *winning* at 512 (faer's
vector machinery is competitive, its values machinery is not), and the tall/wide shapes winning
because the bidiagonalisation dominates there.

**The change.** A new `faer/src/linalg/svd/dqds.rs`, `pub(crate)`, computing the singular values of a
real bidiagonal from `(d_i, e_i)` in place:

- Take `|d_i|` at entry: flipping the sign of a diagonal entry is an orthogonal transform, so the
  singular values are unchanged and the iteration only sees non-negative diagonals.
- Work on `q_i = e_i^2` and apply the shifted **dqds** transform, which is the differential form of
  the implicit QL iteration on `BB^T`:

  ```
  d~_1     = d_1 - tau
  for i in 1..n-1:
      q~_i     = q_i * d_{i+1} / d~_i
      d~_{i+1} = d_{i+1} + q~_i - q_i - tau
  shift += tau
  ```

  with the accumulated `shift` added back to a value when it deflates, so the converged `d_i + shift`
  are the true `sigma_i^2`.
- Shift: the smaller eigenvalue of the trailing 2x2 of `BB^T` (diagonal `d_{n-1}^2 + e_{n-2}^2`,
  `d_n^2 + e_{n-1}^2`, off-diagonal `d_{n-1} e_{n-1}`), computed from the trace and determinant rather
  than from the quadratic's roots, clamped to `[0, smallest entry of the trailing block]`. No
  Jiang-Yamamoto refinement: the simple shift is what the crossover in the literature needs and a
  smarter one is a later experiment, not this one.
- Split at `q_i == 0` (the problem separates), deflate when `q_i <= eps^2 * d_i * d_{i+1}`, and stop a
  block at `n * 200` sweeps with `SvdError::NoConvergence` rather than looping forever.
- A block that contains a zero diagonal entry, or a non-finite input, is handed to the existing
  rotation-based `qr_algorithm` instead: the dqds iteration divides by `d~_i`, and a zero diagonal is
  the one case where the cheap kernel needs care the benchmark does not exercise. Rare, documented,
  and it keeps the new code on the path it was written for.
- The caller keeps its sort: the result is unordered, which is what the existing path also returns
  before `svd()` sorts.

**Wiring.** `compute_bidiag_real_svd`'s values-only arm calls `dqds` in place of `qr_algorithm`, at
every size, and the ported `else if u.is_none() && v.is_none()` arm from the predecessor branch goes
away with it: with dqds there is nothing to gain from routing values to QR above 128 either, and that
branch's own measurement (1.20x at 128, slower at 512/8T) becomes moot.

**Rejected.**

- Copying or translating `dlasq1`/`dlasq2`/`dlasq3`: this fork's rule is own code, and the algorithm is
  published (Fernando & Parlett, *Accurate singular values and differential qd algorithms*, Numer.
  Math. 67 (1994); Parlett & Marques, *An implementation of the dqds algorithm (positive case)*, LAPACK
  Working Note) - the recurrence above is from those, not from the Fortran.
- Unshifted qd: converges linearly in the ratio of neighbouring squared singular values, so a clustered
  spectrum costs more sweeps than the rotations it saves. The shift is the point of dqds.
- A dqds that also produces vectors: faer's `svd_thin` is already faster than the vendors at 512, so
  there is nothing to win there and the vector bookkeeping is what makes LAPACK's own `svd_thin` slow.

## Verification conclusions and constraints

- The recurrence is unproven until it is measured against a reference: the first test compares
  `dqds` against `thin_svd` of the same bidiagonal for n = 2..40 over random, clustered, degenerate and
  badly scaled spectra, at 1e-10 relative for `f64` and 1e-3 for `f32`. A wrong sign or a misplaced
  term in the transform fails that immediately, which is the reason the design states the recurrence
  before writing it.
- `svd()`'s own contract (sorted, non-negative, `sqrt`-ed) stays the caller's; the new code returns
  squared values in an arbitrary order.
- Acceptance: the campaign's optimization loop first (64 target, 32 guard: `svd_values` at 64 must
  approach the vendors' 0.275 ms from 0.52), then the blast-radius arm for adoption (`svd_values` and
  `svd_thin` in every declared shape and both dtypes beside `eigh`, `qr_factor` and `gemm` as
  controls).
- Not claimed: that dqds beats divide and conquer at every size, or anything about dtypes other than
  the four the harness can now measure.

## First measurement, and what it leaves

Implemented and wired (values-only, non-rank-deficient -> `dqds`, everything else unchanged), tests
green: `dqds` against `thin_svd` for n = 2..40 over well-separated, negative-diagonal, all-ones and
badly-scaled spectra at 1e-10, plus a diagonal and a 1e150 scaling check; the lib is 217 passed.

`probes/svd-phases`, f64, best of 7, 1T on CPU 4 / 8T on 4-11, `svd_values` ms:

| n | baseline `50c2b98` | QR-at-all-sizes (predecessor) | **dqds** |
| --- | --- | --- | --- |
| 32 | 0.1423 | 0.1427 | 0.1354 |
| 64 | 0.5217 | 0.5251 | **0.4711** (-10%) |
| 96 | 1.1879 | 1.1857 | 1.0739 (-10%) |
| 128 | 2.6281 | 2.1958 | 1.9771 (-25%) |
| 256 | 12.2824 | 11.0868 | 10.3197 (-16%) |

So the right algorithm is in place and it is 10-25% faster, but it does **not** close the 64 gap
(0.47 against OpenBLAS's 0.275). The remainder after the bidiagonalisation is still 0.304 ms, and at
about 0.8 us per 64x64 sweep that is roughly **375 sweeps** - six per deflation. LAPACK's dqds
converges in a couple of sweeps per deflation because `dlasq4` refines the shift and `dlasq3` deflates
on a tolerance relative to the current shift, while this kernel uses the trailing 2x2 shift alone and
the strict `b_i <= eps^2 sqrt(a_i a_{i+1})` test.

**Next lever, in order:** the deflation tolerance (aggressive, relative to the shift, as `dlasq3`
does) and then the shift refinement (`dlasq4`). That is where the remaining 0.47 -> 0.275 sits, and it
is a tuning exercise on an implementation that now exists and is verified, not a new algorithm.

## The tuning attempt, and why it is not in the tree

The two levers the measurement pointed at were tried and are **not shipped**, because they trade the
accuracy the kernel was verified at for speed:

- **Deflation relative to the accumulated shift** (`TOL2 = (100 eps)^2`, both of `dlasq3`'s conditions)
  instead of the strict `b_i <= eps^2 sqrt(a_i a_{i+1})`.
- **Late-failure shift correction** (`tau = (tau + dmin)(1 - 2 eps)` when the minimum factor went
  negative and the one before it did not), a Gershgorin clamp on the candidate shift, and accepting a
  negative *final* factor by its magnitude, which is what lets a badly scaled block converge at all.

With all of them the badly scaled case (`d = [1e-6, O(1), O(1)]`) converges instead of falling back, but
the smallest singular value comes out at ~1.3e-6 relative error against `thin_svd` where the shipped
kernel is at 1e-10, i.e. the deflation is deflating a coupling that still carried six digits. That is
the classic trade-off and it needs the analytic two-eigenvalue deflation (`dlasq3`'s 2x2/3x3 solves)
rather than a threshold, plus a high-precision reference to check against - neither of which this
session had left.

So the tree keeps the strict test: `801c197`, tests green, 10-25% faster, and the remaining
0.47 -> 0.275 at n=64 stays open with the diagnosis and the two failed candidates recorded here.

A second attempt changed only the tolerance (keeping the strict structure) and the rule for a negative
last factor: it still does not converge on the mixed-scale block, which says the missing piece is
`dlasq3`'s **analytic two-eigenvalue deflation** (the trailing 2x2 solved directly, two values released
at once), not a looser threshold. That is the next unit of work; the tree keeps `801c197` until it
exists, verified at 1e-10.
