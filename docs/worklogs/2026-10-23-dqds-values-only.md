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

## Campaign loop measurement (64 target, 32 guard)

Paired in one stack (`--regimes linalg-small --families svd_values,svd_thin,qr_factor,eigh --shapes
32,64 --dtypes f64,c64,f32,c32 --threads 4,8 --reps 5 --aa 3`, MKL arm, three engines). Baseline
`20261022T223956Z-dqds-loop-baseline` at the pin, candidate `20261022T230352Z-dqds-loop-candidate` at
`118e427`; the comparator reports identical measured paths and the `faer-upstream` control reproduces
(median 1.0026 over 128 cells). Below 1 is faster:

| `svd_values` | f64 | c64 | f32 | c32 |
| --- | --- | --- | --- | --- |
| 64 (target), C order | 0.930 | 0.910 | **0.827** | **0.866** |
| 64, F order | 0.890 | 0.949 | **0.819** | 0.870 |
| 32 (guard), C order | 0.926 | 0.933 | 0.872 | 0.908 |
| 32, F order | 0.949 | 0.963 | 0.877 | 0.908 |

Both sizes improve in every dtype, so the guard is not paying for the target: the kernel is simply
cheaper at 32 as well. 32 cells faster and 3 slower beyond 5%: `qr_factor` f64 F 4T 1.13 (0.084 ->
0.102 ms), `qr_factor` c32 C 4T 1.06 and f32 F 4T 1.05. `qr_factor` does not reach this code, so those
are the binary-layout effect this campaign has measured before (a pair of one source moves a cell by
25% at that size), but they are **unresolved** and the blast-radius arm has to settle them before
adoption.

A third attempt combined the shift-relative tolerance with the **analytic two-value deflation** (the
trailing 2x2 solved directly and both values released) and without the Gershgorin clamp. It fails to
converge on `n = 4` with a negative diagonal: an unclamped shift makes an intermediate factor go
non-positive, the retry cascade cannot recover inside its budget, and the sweep gives up.

The three attempts together locate the missing piece precisely: it is **not** the deflation test and
**not** the two-value release, it is the **shift**. LAPACK's `dlasq4` produces a shift that is
admissible by construction - it starts from the previous sweep's smallest factor (`TAU = -DMIN`) and
refines it inside the trailing block - so `dlasq5`'s factors stay positive and `dlasq3` never has to
retry. A trailing-2x2 shift clamped to keep it small enough to stay admissible loses the speed a
shift is for; the clamp and the shift have to be designed together.

That is the next unit of work, and it is bigger than the deflation: a shift rule that is admissible
by construction, with the deflation tolerance switched back to shift-relative and the two-value
release on top. Until then the tree keeps `801c197`, verified at 1e-10.

## Blast-radius arm

`linalg-small,linalg-mid,linalg-tall`, families `svd_values,svd_thin,qr_factor,eigh`, four dtypes,
1T/4T/8T, MKL arm, paired in one stack: baseline `20261023T012045Z-dqds-blast-baseline` at the pin,
candidate `20261023T000823Z-dqds-blast` at `7452d24`; identical measured paths, `faer-upstream`
control 1.0016 over 720 cells.

`svd_values`, median of the 24 cells per shape (below 1 is faster):

| 32 | 64 | 96 | 128 | 256 | 512 | 1024x64 | 4096x256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.925 | **0.886** | 0.882 | 0.895 | 0.958 | 1.010 | 0.984 | 1.010 |

So the gain is real from 32 to 256 (peak 0.75 single cells at 128) and gone at 512 and above, where the
bidiagonalisation dominates and dqds and divide-and-conquer cost the same. 111 cells faster, 40 slower
beyond 5%: **36 are `qr_factor`** (1.09-1.24, mostly `f32`/`c32` at 4T/8T) and 4 are `svd_values`.

`qr_factor` does not call this code - `compute_bidiag_real_svd` is reached only by the two SVD entry
points - so those 36 are the binary-layout effect this campaign has measured before, but 36 cells is
above the 22-of-780 rate a same-source pair showed, and the protocol does not accept a guard cell on an
explanation. **Adoption is blocked on that**, and the next measurement is the one that settles it: the
same arm with the dqds call compiled but switched off (as the column-slab work did with its split), so
the layout moves and the algorithm does not.

## Status

Verified at 1e-10, in tree, faster from 32 to 256, neutral above, and **not adopted**: the `qr_factor`
guard needs its control, and the 512-and-up range needs the shift work above before dqds can claim it.

## The `qr_factor` guard, settled by the split-off control

The control compiled the kernel and never called it (`const DQDS_ENABLED: bool = false` in front of
the values-only branch), so the binary's layout moved with the change and the algorithm did not:
`20261023T043848Z-dqds-off-control`, same stack, `qr_factor,svd_values` at 64/96/128, four dtypes,
4T/8T. Three ratios: on/pin, off/pin, on/off.

- `svd_values` 128x128: on/pin 0.88-0.93, **off/pin 0.99-1.01**, on/off 0.87-0.92. The gain belongs to
  the call, and the control reproduces the pin exactly.
- `qr_factor` 128x128: on/pin 0.93-1.24, off/pin 0.87-1.06, **on/off 1.01-1.20**. `qr_factor` never
  reaches this code, and a one-bit change that only alters what the optimiser eliminates moves those
  cells by the same amount the change appeared to. So the 36 guard cells are the binary-layout effect,
  not a cost of this kernel.

The campaign's decision rule agrees: the unchanged `faer-upstream` row reproduced the baseline in every
pair (1.0016 and 1.0026), and a per-cell rule is exactly what the toolkit's own `control_check`
documentation warns against.

**Status: EXPERIMENT, not adopted.** The predeclared primary target was `svd_values/64` approaching the
vendors' 0.275 ms and it lands at 0.47 (0.886 against the pin), so the target is not met and the record
does not call it met. What the arm establishes is: verified to 1e-10, 0.88-0.96 from 32 to 256, neutral
at 512 and above, no attributable guard regression. The remaining work is the shift rule (`dlasq4`),
which is also what 512 and above need.

## Why 512 and above did not move, and the gate that fixes it

The probe answers it with the two stages separated, pin against this branch (f64, `svd_values`):

| n, threads | bidiagonalisation | values stage (pin) | values stage (dqds) | total (pin -> dqds) |
| --- | --- | --- | --- | --- |
| 256, 1T | 5.92 | 6.34 | 4.43 | 12.26 -> 10.31 |
| 512, 1T | 42.89 | 22.35 | 15.79 | 65.24 -> 58.75 |
| 512, 8T | 34.50 | 13.71 | 16.63 | 48.22 -> 51.14 |

Two causes, both measured:

1. **dqds is serial; the divide-and-conquer it replaced threads its merge.** At one thread dqds is
   29% cheaper on the values stage at 512; at eight threads the same stage is 13.71 ms threaded
   against 16.63 serial, so the exchange becomes a small loss.
2. **Dilution.** At 512 the bidiagonalisation is 66% of the call at 1T and 72% at 8T, so even the
   29% cut moves the total only 10%.

The campaign's per-shape median over four dtypes, two orders and three budgets hid both: 512 read
0.931 at 1T and 1.033/1.035 at 4T/8T, median 1.010.

**The route is therefore by (budget, size)**: dqds when the budget is one, or below `n = 256`;
divide and conquer otherwise. Measured in the same arm (`20261023T065035Z-dqds-gated`, 192 cells,
`svd_values` only):

| | 32 | 64 | 96 | 128 | 256 | 512 | 1024x64 | 4096x256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1T | 0.909 | 0.879 | 0.877 | **0.789** | 0.856 | 0.922 | 0.975 | 0.978 |
| 4T | 0.910 | 0.879 | 0.877 | 0.911 | 1.004 | 0.999 | 0.977 | 0.999 |
| 8T | 0.911 | 0.873 | 0.877 | 0.899 | 0.999 | 1.000 | 0.986 | 0.998 |

110 cells faster, **0 slower beyond 5%**. The remaining headroom at 512 is no longer in the values
stage: it is the bidiagonalisation, which is 66-72% of that call.
