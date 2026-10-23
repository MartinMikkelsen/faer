# Performance tips for this fork

Read this before writing or reviewing performance-sensitive code here (R-performance-51). Each entry
is a lesson that cost a measurement to learn; the worklog it cites has the numbers and the host.
`docs/worklogs/` holds the history of each change, `PATCHSET.md` what the fork carries today; this
file holds what carries over to the next change.

## Measure the breakdown before designing

Two design reviews of the blocked bidiagonalisation did not predict either of the two effects that
decided it (strided rows, worker wake-ups). One per-stage timing pass did
(`docs/worklogs/2026-10-23-blocked-bidiag.md`). Put temporary timers around the stages of the loop
you are changing, at 1T and 8T, before choosing where to work; remove them before committing.

## Rows of a column-major matrix are strided operands

A row of `A` has stride `m`. `matmul`'s vector paths fall to the general kernel when the destination
or the vector is strided, and a reflector's norm becomes a strided reduction. Gather the row into a
contiguous buffer once, update it, reflect it and use it as the products' vector there, scatter once
(`bidiag_blocked_panel`'s `w`). The row-side steps of the panel went from 15 ms to 6 ms at n = 512.

## A leading dimension that is a multiple of 1 KiB aliases L1 sets

The L1 here is 48 KiB, 12-way, 64 sets: addresses 4 KiB apart share a set, and a 512-row `f64`
copy has exactly that column stride, so every entry of a row lands in one set and twelve of them fit.
The bidiagonalisation touches a row per step and ran a fifth slower at 512 than at 500 or 520. The
SVD driver pads its work copy by 64 bytes per column when the stride would be a multiple of 1 KiB
(`svd/mod.rs` `work_rows`): 18.6 -> 15.3 ms at eight threads. Benchmark shapes are powers of two,
so this is where it shows; a vendor's `?gebrd` at 512 matches our 500.

## The worker group is the caller's

`spindle::for_each` outside `spindle::with_lock` is a plain rayon fork/join: workers woken per
region, partition chosen afresh. Inside a held group they spin between regions and worker `k`
keeps part `k`, so a core's L2 keeps its slice of the trailing block across regions. An algorithm
of many short regions (two vector products per step) is a third slower without it (25.5 vs 19.2 ms,
512, 8T). The library does **not** take the lock: the owner of parallelism policy does, once per
session or region (`faer::spindle::with_lock`, re-exported; crate docs, *parallelism*). The
campaign harness holds it around a faer row's timed loop for the same reason.

Corollary: never use a raw rayon iterator (`par_chunks_mut`, `par_iter`, `rayon::join`) in this
crate. Under a held group the pool's threads are occupied and the raw iterator runs starved
(block-apply split, 4096x256 at 8T: 28.9 ms raw, 17.5 through `spindle::for_each`).

## A region costs about 2 us held, 4 us plain, and a threaded gemv of 512 is 6 us

Measured with `probes/gemv-threads` (faer-benchmark, diagnostic). Below `256 * 256` elements the
matvec is sequential by its own threshold. A step that issues two regions pays about 4 us of
synchronisation; 384 steps are 1.5 ms. Design the loop so that the regions are the big products,
not the short ones.

## Short products are bandwidth-bound in L2, not call-bound

A 512x16 vector product costs 1.2 us at 50 GB/s with `Par::Seq`. Fusing eight of them per step buys
little; halving the panel width halves them (8-16 flat, 32 slower).

## dqds: the sweep is latency-bound, so only the sweep count matters

One division sits on the dependency chain per entry (`delta -> ahat -> a / ahat -> delta`), about
10 ns at this clock, the same for everyone. Speed comes from fewer sweeps: a shift that is rarely
rejected (the trailing 2x2's eigenvalue backed off by twice its eigenvector-weighted second-order
coupling), a release test in the eigenvalue's units (`(100 eps)^2 (sigma + a)`, not the shifted
factors, which go to zero as the shift converges), ping-pong buffers so a rejection costs no copy
(`docs/worklogs/2026-10-23-dqds-shift.md`: 13 accepted and 9 rejected sweeps per value at 64 became
3.7 and 0.3). Count sweeps before touching the shift.

## Thresholds are measured, with the table beside the constant

Block width and floor (`bidiag.rs`), the dqds back-off factor, the padding threshold: each has its
scan in the worklog and a one-line summary next to the constant. A threshold without its table is a
guess.
