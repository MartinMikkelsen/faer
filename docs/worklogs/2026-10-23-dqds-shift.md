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

## Campaign measurement

(filled in below.)
