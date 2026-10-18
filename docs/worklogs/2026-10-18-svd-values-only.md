# SVD values-only: where the time is, and why the obvious shortcut lost

The issue asks what a values-only SVD pays that a thin SVD does not, and whether
faer's values path differs from Reference LAPACK's. This record answers both with
a phase split at the pinned revision, then reports the one candidate it produced
and the measurement that rejected it.

## What LAPACK does, and what faer did

Reference LAPACK's `DBDSDC` (https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dbdsdc.f)
takes two branches on the bidiagonal problem: with `COMPQ = 'N'` (values only) it calls
`DLASDQ` with `NCVT = NRU = NCC = 0`, and `DLASDQ` then calls `DLASQ1` — the `dqds`
iteration, which accumulates no vectors. With singular vectors it calls `DLASD0`, the
divide-and-conquer merge. faer's `compute_bidiag_real_svd` had one branch on size only:
below `recursion_threshold` (128) it used the implicit QR iteration, at or above it
`divide_and_conquer`, whatever the caller asked for. The values-only caller
(`MatRef::singular_values`) passes `u = None`, `v = None`, so `divide_and_conquer` runs
with `MatU::TwoRowsStorage` — a two-row dumping ground that the merge never reads back —
and still zeroes an `(n+1)^2` scratch matrix and builds every singular vector of every
merged subproblem into it with the closed-form formula before applying the Jacobi
rotations to those rows.

## The phase split (probe, fork `5578ccc`, f64, column-major fixture)

`bidiag_in_place` is public, so the probe times it on the same fixture and the two
composites beside it; the bidiagonal-SVD remainder is `svd_values - bidiag`. Both
sides include the same `m x n` copy, so the subtraction cancels it. Best of 7, CPUs
`4` / `4-7` / `4-11` for 1T / 4T / 8T, the observer off that L3 domain.

| n | T | bidiag | `svd_values` | `svd_thin` | remainder |
| --- | --- | --- | --- | --- | --- |
| 64 | 8 | 0.17 | 0.53 | 0.98 | 0.35 |
| 128 | 8 | 0.85 | 2.20 | 3.14 | 1.35 |
| 256 | 8 | 6.29 | 11.50 | 13.72 | 5.20 |
| 512 | 8 | 34.34 | 54.20 | 59.15 | 19.86 |
| 512 | 1 | 42.97 | 63.10 | 104.63 | 20.13 |

At 512 and eight threads the bidiagonalization is 71% of the values-only call and the
bidiagonal SVD the remaining 29%. The bidiagonalization gains only 1.26x from one to
eight threads; that is the parallel-threshold question
`2026-10-14-bidiag-parallel-hint.md` already answered (the shipped floor is
load-bearing, lowering it loses 1.8-2.5x), so it was not re-swept here.

## The candidate: route values-only to the implicit QR iteration at every size

One condition change in `compute_bidiag_real_svd` (`|| (u.is_none() && v.is_none())`)
and one in `bidiag_real_svd_scratch` (do not reserve the divide-and-conquer scratch),
plus `test_values_only_matches_thin_on_hard_inputs`. Correctness was clean: the
focused test passed, and `cargo test -p t4a-faer --lib` reported 210 passed / 2
ignored on the branch. The whole lib suite exercises values-only against the thin
reference at both sides of the threshold, because `test_svd` sets
`recursion_threshold: 8`.

Paired probe, same fixture and CPU sets, best of 7 (ms):

| n | T | values, divide and conquer | values, implicit QR |
| --- | --- | --- | --- |
| 64 | 8 | 0.526 | 0.525 (unchanged: already QR below the threshold) |
| 128 | 8 | 2.214 | 2.195 |
| 256 | 8 | 10.848 | 11.496 |
| 512 | 8 | 48.079 | 54.199 |
| 512 | 4 | 46.968 | 53.500 |
| 512 | 1 | 65.737 | 63.100 |

The QR route is 11% better at one thread (it drops the discarded vector work and the
recursive join that only helps the merge) and 13% worse at eight, where the
divide-and-conquer recursion's two-way joins are what made the remainder shrink with
threads at all. The campaign's cells are 4T/8T, so it was rejected on its own target
rather than kept for its single-thread number. Branch `perf/svd-values-qr`, commit
`9743497`; no part of it is on `main`.

## What this establishes

- LAPACK's values-only branch is a different algorithm (`dqds`), not a cheaper
  divide-and-conquer; faer's values path does pay the merge's vector work, and that
  work is *not* the gap: removing it costs more than it saves above 128 at 4T/8T.
- The values-only gap against MKL at 512/8T (48.1 ms against 22.4 ms in the campaign's
  own page) is dominated by the bidiagonalization (34.3 ms here), not by the
  bidiagonal SVD (13.9 ms). A blocked bidiagonalization in the `DGEBRD`/`DLATRD` shape
  (panel of reflectors, then the trailing update as one `gemm`) is the change that
  addresses it; it is a much larger diff than this iteration's scope and was not
  attempted here.
- No OpenBLAS, MKL or LAPACK source was ported. `DBDSDC`/`DLASDQ` were read as
  evidence for the split (BSD-3-Clause, Reference-LAPACK), not translated.
