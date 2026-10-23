# dqds: the shift, the release test, and the copies (`c88f886`)

## Decisions

**Why.** `docs/worklogs/2026-10-23-dqds-values-only.md` left the kernel 2.2-2.4x slower than the
vendor's values-only stage at every size and named the shift as the missing piece, on the evidence of
three tuning attempts that each changed several things at once. This unit started where the handoff
said to: counting. Temporary counters on the bidiagonal of a random dense matrix (the campaign's
shape), accepted and rejected sweeps, zero-shift sweeps and releases:

| n | accepted / value | rejected / value | zero-shift | entry updates |
| --- | --- | --- | --- | --- |
| 64 | 13.0 | 9.2 | 118 | 52.7K |
| 128 | 12.4 | 8.7 | 234 | 193K |
| 512 | 5.7 | 3.8 | 426 | 1357K |

Three causes, each visible in the counts:

1. **The shift always overshot.** The trailing 2x2's smaller eigenvalue is an *upper* bound on the
   block's smallest eigenvalue (interlacing), so a sweep at that shift fails whenever the eigenvector is
   not yet concentrated at the bottom, i.e. almost always; the loop then halved it twice and fell to
   zero, four wasted sweeps per failure (the 426 zero-shift sweeps at n = 512).
2. **Every attempt copied the block twice** (save before, restore after a rejection).
3. **The release test was in the wrong units.** `b_i <= eps^2 sqrt(a_i a_{i+1})` compares the coupling
   with the *shifted* factors, which go to zero as the accumulated shift approaches the eigenvalue, so
   a bottom entry that had converged in the eigenvalue's own units kept sweeping. The eigenvalue is
   `sigma + a`, and that is what the coupling has to be negligible against.

**The change, all in `faer/src/linalg/svd/dqds.rs`:**

- **Ping-pong buffers.** A sweep writes into the second buffer; acceptance swaps the roles, rejection
  costs nothing. One copy per block at the end if the result landed in the second buffer.
- **Release relative to the eigenvalue.** `b_{n-2} <= (100 eps)^2 (sigma + a_{n-1})` or
  `<= (100 eps)^2 a_{n-2}` releases the bottom value; when the coupling *above* the trailing 2x2 is
  negligible on the same test, the 2x2 is solved in closed form and both values are released; a block
  of one or two is closed form. `(100 eps)^2` in squared units is `100 eps` relative on the singular
  value, the usual negligibility of a QR-type iteration.
- **The shift.** When the smallest current factor is at the bottom, the trailing 2x2's eigenvalue
  `lam2` backed off by twice its second-order coupling to the entry above, weighted by the 2x2
  eigenvector's component on that side (`2 (a_{n-3} b_{n-3}) v_1^2 / gap`), floored at half the
  smallest factor; when the smallest factor is interior, half of it, no more than `lam2`. The first
  sweep of a block is unshifted, as the information the shift needs comes from a sweep.
- **Rejection.** A failure at the last factor alone says the bottom eigenvalue is at about
  `tau + delta` (`delta` the failed last factor, negative), so the retry is `(tau + delta)(1 - 4 eps)`;
  an interior failure cuts the shift by four; the third retry is zero, and a zero shift that fails is
  non-convergence as before.

Measured on the same bidiagonals, the shift's variants (temporary knobs, removed):

| back-off | accepted / value (512) | rejected / value | entry updates |
| --- | --- | --- | --- |
| none (`tau = lam2`) | 3.1 | 2.9 | 840K |
| coupling / gap, unweighted | 4.2 | 0.06 | 609K |
| 1x the weighted term | 3.1 | 2.9 | 840K |
| **2x the weighted term** | **3.5** | **0.23** | **517K** |
| 1.5x, 3x, 4x | 3.4-3.5 | 0.2-0.3 | 519-524K |
| the shift floor (0, 0.5, 0.75 of `dmin`) | no effect | | |

So the back-off's size is what trades rejections for sweeps, and twice the second-order term is where
the two are both low. At n = 64 the same choice gives 3.7 accepted and 0.3 rejected per value, from
13.0 and 9.2.

**Rejected.**

- Keeping the strict release test and only fixing the shift: the counts say the release test is a third
  of the sweeps on its own (the converged bottom keeps sweeping), and the test's units are a bug, not a
  tolerance.
- Flipping the block (reversing the array when the top converges faster) and the vendor-style
  nine-case shift table: not measured; the counts above are within 1.4x of the sweep count a
  nine-case shift reaches in the literature, and the remaining time is the latency-bound recurrence
  (`delta -> ahat -> a / ahat -> delta`, one division on the critical path per entry), not the sweep
  count.

## Verification conclusions and constraints

- Accuracy did not move: the reference test (n = 2..40, four spectra including a 1e-6-graded one,
  against `thin_svd` at 1e-10) passes unchanged, and a new per-value test on the bidiagonals of random
  dense matrices up to n = 200 holds a relative 1e-12 against `thin_svd` (measured 1e-14 at 512).
- The accuracy loss the earlier attempt recorded (1.3e-6 at the small value of the graded case) came
  with accepting a negative last factor by its magnitude, which places the eigenvalue at `tau + |delta|`
  instead of `tau - |delta|`. This unit never accepts a non-positive factor.
- lib: 218 passed / 2 ignored.

## Route

With the sweep count down, the values stage at 512 is 6.9 ms serial against 13.3 ms for the threaded
divide-and-conquer it was gated behind above n = 256 (1.6 against 4.4 at 256, 25 against 43 at 1024,
probe, 8T), so values-only takes dqds at every size and budget (`7eebfd4`), one rule again.

## Campaign measurement

Both arms are `svd_values,svd_thin` over `linalg-small,linalg-mid,linalg-tall`, four dtypes, 1T/4T/8T,
MKL arm, three engines, paired against the pin baseline `20261023T012045Z-dqds-blast-baseline`
(`50c2b98`); `scripts/candidate_compare.py --cells svd_values,svd_thin`, below 1 is faster.

`20261023T100330Z-blocked-dqds`, at `7eebfd4` (the blocked bidiagonalisation, this unit and the
route): `faer-upstream` control median 1.0022 over 384 cells, 0 outside their own range; **208 cells
faster, 6 slower beyond 5%**. `svd_values`, median ratio over dtypes and orders:

| | 32 | 64 | 96 | 128 | 256 | 512 | 1024x64 | 4096x256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1T | 0.622 | **0.599** | 0.622 | 0.577 | 0.656 | 0.850 | 0.889 | 0.938 |
| 4T | 0.619 | 0.599 | 0.620 | 0.664 | 0.709 | 0.616 | 0.896 | 0.938 |
| 8T | 0.621 | 0.599 | 0.621 | 0.662 | 0.710 | **0.568** | 0.902 | 0.927 |

f64 F absolute: 64 0.537 -> 0.296 ms, 128 2.62 -> 1.30, 512 65.4 -> 43.3 (1T) and 48.5 -> 25.8 (8T).

**The six flagged cells** are all `svd_thin`, complex, 1T, at 256 and 512 (1.05-1.11). `svd_thin`
computes vectors and never reaches this kernel or the route, and the blocked-bidiag run measured the
same cells with the same bidiagonalisation code at 0.977-0.980; the candidate's absolute times agree
between the two runs within 1% (c64 512: 205.7 and 207.3 ms). So the flag is the run-to-run drift of
the normalised ratio on that pair, recorded here and not resolved by this unit.

**Against the vendor in the same run** (`faer-fork` / `lapack-mkl`, F order, below 1 is faer faster):

| `svd_values` | 32 | 64 | 96 | 128 | 256 | 512 | 1024x64 | 4096x256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 1T | 1.37 | 1.16 | 1.11 | 1.13 | 1.02 | **0.90** | **0.94** | **0.85** |
| f64 8T | 1.37 | 1.14 | 1.12 | 1.16 | 1.18 | 1.15 | 1.86 | 1.31 |
| c64 1T | **0.98** | **0.83** | **0.73** | **0.77** | **0.68** | **0.59** | **0.74** | **0.73** |
| c64 8T | **0.98** | **0.82** | **0.72** | 1.10 | 1.33 | **0.91** | 1.00 | 1.33 |

The predeclared target (`svd_values/64` approaching 0.275 ms) lands at **0.296 ms against the vendor's
0.254**: the values stage is at parity (0.126 vs ~0.13 ms, probe), the remaining 0.04 ms is the
unblocked bidiagonalisation at 64 (0.17 ms), which the floor scan says is not helped by blocking.
Real 512 and the tall shapes are faster than the vendor at one thread; complex is faster at one thread
everywhere and at eight threads for 512. What is left above one thread is the panel's sequential part
(13.6 of 19.5 ms at 512) and the tall path's threaded overhead (1024x64 at 8T, 1.86), both outside
this unit.

## Follow-up

The numbers above were taken before the leading-dimension padding and the caller-held worker group
(`docs/worklogs/2026-10-23-blocked-bidiag.md`, its last three sections). The final campaign run
(`20261023T125347Z-blocked-dqds-caller-lock-2`) has `svd_values` at 0.60 of the pin at 64 and 0.61 at
512, 8T; the values stage itself is unchanged since `c88f886`.
