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
