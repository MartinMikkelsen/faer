# Blocked bidiagonalisation (dlabrd)

## Decisions

**Why.** `bidiag_in_place` factorises one Householder pair at a time and applies it to the trailing
`(m-k) x (n-k)` block with two **level-2** products (`y2 = A21^H A22`, `z2 = A22 A12^H`, both
matvec-shaped). Measured on this host, that costs 42.9 ms at n=512 one thread and 34.5 ms at eight, and
the eight-thread figure is only 1.24x better than one thread. The two halves of a blocked version were
measured separately (`probes/rankk`, `probes/bidiag-panel`):

| | 1T | 8T |
| --- | --- | --- |
| trailing updates, rank-32 GEMM, `n/ib` of them | 8.8 ms | 1.4 ms |
| panel generation, `m/ib` unblocked panels of width 32 | 3.5 ms | 3.4 ms |
| today's whole bidiagonalisation | 42.9 ms | 34.5 ms |

Those two numbers are **kernel diagnostics, not a bound on this algorithm**: the first measures one
GEMM where the panel needs two products and two correction recurrences, the second measures a narrow
matrix and therefore omits the full-width right-reflector work. A blocked vendor `?gebrd` on the same
host and size costs 39.8 ms at one thread and 15.6 ms at eight, which is the honest reference point:
**the panel's own level-2 work is comparable to its level-3 update**, so the expected gain is at eight
threads (where the update is the part that threads) and may be small at one. The first arm after this
lands decides whether it stays.

**The change, in `faer/src/linalg/svd/bidiag.rs` only.** The panel is not "factor a slice and apply two
block helpers": a bidiagonalisation's right reflectors span the whole remaining row and its left ones the
whole remaining column, so the trailing block is updated by **correction matrices**, exactly as LAPACK's
`dlabrd` + `dgebrd` do it. Per panel of width `ib`:

1. **Panel recurrence, deferred update.** For `i` in the panel, first bring `A(i:m, i)` and `A(i, i+1:n)`
   up to date against the *already accumulated* correction columns with the two short products
   (`A(i:m,i) -= A(i:m,1:i-1) Y(i,1:i-1)^T`, `-= X(i:m,1:i-1) A(1:i-1,i)^T`, and the row analogue), then
   generate the column reflector and the row reflector, and build the two correction columns
   `Y(i+1:n, i)` and `X(i+1:m, i)` from the same short products, scaled by **`tl.recip()` and
   `tr.recip()`** - faer stores `H = I - v v^H / tau` (`householder.rs`) and the current loop already
   multiplies by reciprocals (`bidiag.rs:106,160,189,206`), while `Hl`/`Hr` keep `tl`/`tr` themselves.
2. **Trailing update, once per panel, level-3.** `A22 -= V Y^T + X U^T`, where `V = A(i+ib:m, i:i+ib)`
   and `U = A(i:i+ib, i+ib:n)` are the reflector tails already stored in `A`, through two
   `linalg::matmul::matmul` calls with the caller's `par`.

That is `dgebrd`'s own two-`dgemm` update (`A := A - V Y^T - X U^T`, LAPACK `dgebrd.f`), and the panel
part is `dlabrd`'s twelve short products per step. Nothing new is needed from `householder`: the
reflector generators stay as they are, and the two level-3 products are plain `matmul`.

**Views and conjugation, stated because the review asked.** The right reflectors are stored in `A`'s
rows, so the row-basis is passed as a **transposed** view with `Conj::Yes` where an adjoint is needed -
`bidiag.rs` already does exactly this for its final output-factor construction
(`V.transpose(), Conj::Yes`), and the left transpose applications likewise need `Conj::Yes` to obtain an
adjoint. The design does not invent a convention; it reuses the two the file already uses.

**What stays exactly as it is.** The reflector storage layout in `A` (unit lower triangle and unit
superdiagonal, with `Hl`/`Hr` holding the scalars), the meaning of `bidiag_in_place`'s arguments and
its output, and `par_threshold`'s role, with one correction the review demanded: the predicate is evaluated on the
**full remaining dimensions**, and a panel-local decision must not latch `Par::Seq` onto the trailing
update - the update keeps the caller's `par` for as long as the full remaining area clears the floor.

**The loop predicate, stated exactly** because the stopping rule decides both the size floor and the
tests: block while the *remaining* size `min(m - p, n - p) > 128` (strict), then run the existing per-`k`
loop on the tail from that point; any input with `min(m, n) <= 128` never enters the blocked loop at all.
The floor is decided **in this file and mirrored in the scratch query**, so the small sizes are not
merely arithmetically unchanged but also allocate what they allocate today. The guard is that
`svd_values` at 64 and 128 does not move.

**Entering the tail.** The panel's two products have already updated the trailing block, so the unblocked
tail must start with **fresh local indexing and no pending `y`/`z` pair** - the existing loop's
`k > 0` branch consumes the previous step's pending pair (`bidiag.rs:87-103,111-116`), which would apply
a second deferred update to a block that is already current. After that the tail keeps the existing
per-step arithmetic unchanged, and the output factors are upgraded once at the end.

**Scratch.** `bidiag_in_place_scratch` gains the two correction matrices, `X` (`m x ib`) and `Y`
(`n x ib`), which live through the panel and its update. No `any_of` sharing between them is available, and the two
trailing products both accumulate directly into `A22`, so they need no result buffer. The scratch query
grows and the existing production queries propagate the larger requirement (`svd/mod.rs`'s
`svd_imp_scratch` and the tall path's reduction) - no caller change is implied. The two `matmul` calls allocate nothing; do not claim this
plan removes the helpers' own allocations, because it does not use them for the update.

**Rejected.**

- Threading the existing level-2 updates harder. Measured: eight threads buy 1.24x, because the update
  is a matvec over a 512x512 block and memory-bound. More parallelism on the same traffic cannot fix it.
- A public block-size knob in `BidiagParams`. Nothing outside this file needs to choose it, and the two
  probe tables say the choice is flat between 16 and 64 on the update side and monotone on the panel
  side, so it is one internal constant plus one measurement.
- Building a second compact-WY subsystem for the right-side reflectors. The revised update does not use
  those helpers at all - `A22 -= V Y^T + X U^T` is two plain `matmul` calls - so there is no convention
  to match and no subsystem to build.
- A separate small-`K` GEMM workstream. The rank-32 update already runs at 31 GFLOP/s one thread and 186
  at eight (`probes/rankk`), so the kernel is not the limit here; that concern only exists if this
  change lands and the update share turns out larger than measured.

## Verification conclusions and constraints

- **Correctness first, and the rounding changes.** A blocked factorisation is not bit-identical to the
  unblocked one, so the lib suite (`cargo test -p t4a-faer --lib`, currently 217 passed / 2 ignored)
  plus `fbench verify` over the affected populations must be green before any timing.
- **A focused test that crosses panels.** The existing bidiag tests use sizes 4 and 8 and the SVD tests
  reach 150 at `Par::Seq`, so with a floor of 128 and a width of 32 none of them exercises two
  consecutive blocked panels plus the unblocked tail. Add one reconstruction test (real and complex)
  at a size the loop predicate above actually blocks - `min(m, n)` at 300 and a rectangular twin, since
  200 is only one panel - using the queried scratch, and for the threaded case **set
  `par_threshold = 0`**: with the default `192 * 256`, a `199^2` trailing area stays under the floor and
  the threaded call would test sequential work.
- **Predeclared measurement, with the gate where the expectation is.** The expectation is an eight-thread
  scaling gain, so the **primary gate is `svd_values` at 512, 8T: under 20 ms (from 49.1)**, and the
  one-thread figure is a **guard, not a target**: it must stay within the recorded spread rather than
  reach a number. Case list unchanged: `svd_values` and `svd_thin` over `linalg-small/mid/tall` plus the
  `1024x64` and `4096x256` shapes, four dtypes, 1T/4T/8T, MKL arm, paired in one stack against the pin;
  `svd_values` at 64 and 128 must not move; `eigh`, `qr_factor` and `gemm` are the collateral controls;
  the unchanged `faer-upstream` row is the machine check.
- **The block width and the size floor are measured, not chosen.** Both come from the first arm: if
  `ib = 32` and a floor of 128 leave the 64/128 guards unmoved and 512 improved, that is what the
  constant becomes; if the panel work turns out to dominate (the vendor's own blocked `?gebrd` suggests
  it can), the record says so and the change is dropped rather than tuned.
- **Not claimed:** that this closes the 512 gap to the vendor. Two thirds of that gap is this stage's
  *scaling*, and the remaining third is the values kernel (`docs/worklogs/2026-10-23-dqds-values-only.md`).

## Implementation and what the breakdown said (`555d84f`)

Implemented as designed, plus two things a per-step breakdown (temporary timers in the panel,
`probes/svd-phases`, n = 512, diagnostic) found that the design did not predict:

| stage of one step | 1T, as designed | 8T, as designed | 8T, after the two fixes |
| --- | --- | --- | --- |
| `A^H u` product (threaded) | 10.3 | 6.5 | 2.7 |
| `A r^H` product (threaded) | 10.6 | 7.5 | 3.7 |
| row update + right reflector (sequential) | 10.8 | 11.1 | 6.2 |
| short recurrence products (sequential) | 6.6 | 6.5 | 5.1 |
| trailing update, per panel | 5.4 | 1.4 | 1.2 |
| unblocked tail (128) | 1.4 | 1.4 | 1.4 |
| **whole bidiagonalisation** | **45.8** | **35.2** | **21.6** (ib = 32); **19.5** (ib = 16) |

1. **The row is strided.** A row of `A` has a stride of `m` elements (4 KiB at n = 512), and
   `matmul`'s vector paths require a contiguous destination or vector, so the row update fell to the
   general kernel and the right reflector's norm ran as a strided reduction: 15 ms at any budget. Each
   row is now gathered once into a contiguous buffer, updated, reflected and used as the vector of the
   `X` products there, and scattered back once (10.8 -> 6.2 ms).
2. **The worker group was not held.** `faer` never calls `spindle::with_lock`, and without it
   `spindle::for_each` is a plain rayon fork/join per region: workers are woken for each of the two
   products per step and get an arbitrary chunk each time. The blocked phase now runs inside
   `with_lock(nthreads)`: the workers spin between regions and thread `k` keeps column range `k` of the
   trailing block across them, so the block stays in that core's L2. The two products went from
   10.3/10.6 ms to 2.7/3.7 ms at eight threads. This is a threading-policy point for the maintainer:
   it is the substrate's own API for exactly this pattern, and it is the first use of it in the fork.

The block width and the floor were then scanned with both fixes in (min of 3, ms):

| ib, floor | 256 1T | 512 1T | 1024 1T | 256 8T | 512 8T | 1024 8T |
| --- | --- | --- | --- | --- | --- | --- |
| unblocked (today) | 5.9 | 42.9 | 315 | 6.7 | 34.5 | - |
| 32, 128 | 5.9 | 39.5 | 291 | 6.1 | 21.6 | 90 |
| 16, 128 | **5.4** | **37.2** | **282** | **5.5** | **19.5** | **83** |
| 8, 128 | 5.2 | 35.8 | 283 | 5.6 | 19.9 | 86 |
| 48, 128 | 6.3 | 42.1 | 308 | 6.6 | 23.9 | 103 |
| 16, 96 / 16, 64 | 5.3 | 37.0 | 283 | 5.5 | 19.2 | 83 |

8-16 is flat, 32 and 48 lose on the recurrence side, and the floor does not matter between 64 and
128; the constants are `BLOCK_SIZE = 16`, `BLOCK_FLOOR = 128` (the floor the guards were declared
on). The 1T figure moved after all (42.9 -> 37.2), because the strided row cost was also in the
unblocked loop's budget and the gather pays for itself.

What remains sequential at eight threads is 13.6 of 19.5 ms: the short recurrence products (five
to eight vector products of `n x i` per step, bandwidth-bound in L2), the row gather and scatter, and
the tail. The vendor's blocked `?gebrd` is at 15.6 ms on the same host and size.

## Campaign measurement

Both arms are `svd_values,svd_thin` over `linalg-small,linalg-mid,linalg-tall`, four dtypes, 1T/4T/8T,
MKL arm, three engines, paired against the pin baseline `20261023T012045Z-dqds-blast-baseline`
(`50c2b98`); `scripts/candidate_compare.py --cells svd_values,svd_thin`, below 1 is faster.

**This change alone** (`20261023T091426Z-blocked-bidiag`, at `555d84f`): `faer-upstream` control median
1.0006 over 384 cells, 0 outside their own range; **172 cells faster, 0 slower beyond 5%**.
`svd_values`, median ratio over dtypes and orders:

| | 32 | 64 | 96 | 128 | 256 | 512 | 1024x64 | 4096x256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1T | 0.912 | 0.882 | 0.877 | 0.788 | 0.806 | 0.829 | 0.980 | 0.970 |
| 4T | 0.914 | 0.879 | 0.876 | 0.909 | 0.932 | 0.737 | 0.989 | 0.979 |
| 8T | 0.915 | 0.879 | 0.877 | 0.898 | 0.929 | **0.688** | 0.996 | 0.975 |

32 to 96 are the gated dqds run's numbers unchanged (the floor keeps them out), 128 and 256 move at 1T
only through the row gather in the unblocked loop's neighbourhood of the floor, and 512 is the blocked
phase: f64 F 48.5 -> 32.6 ms at 8T, 65.4 -> 52.8 at 1T. `svd_thin` moves the same way at 512
(0.954 / 0.799 / 0.745 at 1T / 4T / 8T) and is flat elsewhere.

**The predeclared gate** was `svd_values` at 512, 8T, under 20 ms. It is not met by this change alone
(32.6 ms): the bidiagonalisation went to 19.1 ms and the values stage behind it was the threaded
divide-and-conquer at 13.3 ms. With the dqds unit after it (`docs/worklogs/2026-10-23-dqds-shift.md`)
the same cell is 25.8 ms, against the vendor's 22.3 on the same host. The 1T guard held: 512 improved
and 64/128 did not move at this change (the gated run's 0.879/0.789).

**The floor was re-scanned after the dqds unit** at 32, 64 and 128 on sizes 32 to 256 (probe, 1T and
8T): a floor of 64 is flat and a floor of 32 makes 64 slower (0.17 -> 0.19 ms at 1T, 0.22 at 8T), so
128 stays. The 64 cell's remaining distance to the vendor (0.296 vs 0.254 ms) is the unblocked loop's
0.17 ms bidiagonalisation, not this unit.

## The 8T gap, located (`5b2b5f9`)

With the dqds unit in, `svd_values` 512 at eight threads was 25.8 ms against the vendor's 22.3 while the
values stages were at parity (6.6 vs 6.9), so the whole gap was this stage's 19.5 against the vendor's
15.6. Three hypotheses, each with a probe (`probes/gemv-threads`, diagnostic):

| hypothesis | probe | result |
| --- | --- | --- |
| the short recurrence products pay `matmul` call overhead | one 512x16 vector product, `Par::Seq` | 1.2-1.3 us at 50-57 GB/s: bandwidth-bound in L2, nothing to remove |
| the regions pay synchronisation | one 8-job region, held group vs plain | 1.75 us vs 4.0 us; 768 regions are 1.3 ms |
| the power-of-two leading dimension aliases L1 sets | bidiag at n = 500, 512, 520; 512 with lda 520 and 528 | **500: 14.6, 512: 18.6, 520: 15.8; 512 at lda 520: 15.3 ms (8T)**; 1T 36.1 -> 33.5; 256: 5.3 -> 4.8 |

A column stride that is a multiple of 1 KiB maps a row's entries onto a few L1 sets (one set at 4 KiB,
n = 512 `f64`), and this stage gathers and scatters a row at every step; every campaign shape is a power
of two. The SVD driver now allocates its work copy with 64 more bytes per column whenever the stride
would be a multiple of 1 KiB and works on the top `m` rows (`work_rows` in `svd/mod.rs`); nothing else
changes. Probe, caller holding the worker group: `svd_values` 512 8T 25.6 -> 22.1 ms (vendor 22.3), 1T
43.3 -> 40.0 (vendor 48.0); `svd_thin` 512 8T 43.0 -> 37.5 (vendor 47.6).

## Who holds the worker group (`01e741c`, `1850d9e`)

`555d84f` held the spindle group inside `bidiag_in_place`. The maintainer's call: the lock is the
caller's, as upstream's own benchmark does (it wraps everything in `with_lock` and the library never
takes it), and as R-performance-11/14 place parallelism policy and session fixed costs with the caller's
context. So the library takes no lock; `faer::spindle` is re-exported and the crate docs (*parallelism*),
`Par::Rayon` and the README say to hold it. Nesting is allowed by spindle (a lock inside a lock borrows
the outer group's workers) and already happens on every threaded product (the native gemm kernel takes
one); the accidental double lock measured the same 22.2 ms as a single one. Without the lock this stage
is 25.5 ms at 512, 8T instead of 19.2, so a caller that does not hold it keeps about two thirds of the
gain.

The campaign follows: `fbench` holds the group around a faer row's timed loop when the budget is more than
one thread (`faer-benchmark` `a605778`), both faer rows alike, and the pin baseline was re-recorded under
it (`20261023T112343Z-caller-lock-baseline`). The first candidate under it (`20261023T120538Z`,
`01e741c`) found the fork's own problem: the block-apply split adopted on 2026-10-18 used raw rayon
`par_chunks_mut` iterators, which find the pool's threads occupied by the held workers and run starved
(`svd_values` 4096x256 8T at 1.83 of the vendor, from 1.31 without the lock). `1850d9e` routes them
through `spindle::for_each` like every other region.

## Final campaign measurement (`20261023T125347Z-blocked-dqds-caller-lock-2`, at `1850d9e`)

Against `20261023T112343Z-caller-lock-baseline` (the pin `50c2b98`, same harness): `faer-upstream`
control median 1.0027 over 384 cells, 0 outside their own range; **266 cells faster, 0 slower beyond 5%**.
`svd_values`, median ratio over dtypes and orders:

| | 32 | 64 | 96 | 128 | 256 | 512 | 1024x64 | 4096x256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1T | 0.627 | **0.604** | 0.629 | 0.568 | 0.599 | 0.674 | 0.895 | 0.918 |
| 4T | 0.627 | 0.604 | 0.628 | 0.666 | 0.672 | 0.621 | 0.728 | 0.628 |
| 8T | 0.627 | 0.605 | 0.630 | 0.667 | 0.673 | **0.611** | 0.698 | **0.570** |

`svd_thin`: 0.91 / 0.82 / 0.79 at 512 (1T / 4T / 8T), 0.58 and 0.51 at 4096x256 (4T, 8T), flat at 32-96.

Against the vendor in the same run (`faer-fork` / `lapack-mkl`, F order, below 1 is faer faster):

| `svd_values` | 32 | 64 | 96 | 128 | 256 | 512 | 1024x64 | 4096x256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 1T | 1.39 | 1.17 | 1.12 | 1.07 | **0.94** | **0.85** | **0.95** | **0.83** |
| f64 4T | 1.38 | 1.16 | 1.11 | 1.09 | 1.03 | **0.91** | 1.06 | **0.93** |
| f64 8T | 1.39 | 1.15 | 1.12 | 1.10 | 1.06 | **0.97** | 1.24 | 1.11 |
| c64 1T | **0.99** | **0.84** | **0.76** | **0.75** | **0.68** | **0.60** | **0.74** | **0.76** |
| c64 8T | **0.99** | **0.84** | **0.75** | 1.05 | 1.23 | **0.80** | **0.76** | 1.25 |

Real 512 is faster than the vendor at every budget and 4096x256 at one and four threads; complex is
faster at one thread everywhere and at eight threads for 512 and both tall shapes. What is left: the
`svd_values/64` target at 0.298 against 0.255 ms (the unblocked loop's 0.17 ms bidiagonalisation; blocking
does not help it), 32-128 by the same unblocked loop, and 8T at 256 and the tall shapes (1.06-1.25),
where the sequential part of the panel and the tall path's own regions set the time.
