# Removing measured slowdowns against upstream

## Scope and evidence

The objective is to remove reproducible fork slowdowns against upstream
`0539947ffb757a739d7e703a7d2fa0c792a909c1`, not to close vendor gaps or tune
specific host/shape exceptions. The starting measured code is `2240b07`
(`7364be5` has identical measured paths).

The final MKL full-suite page at that revision flags slow cells in LU/solve,
Householder apply, full-pivot LU, triangular solve and Cholesky. A family whose
own source was not modified can still change through shared helpers. The earlier
claim that this establishes historical causes or proves no newly introduced
regression was not justified. The OpenBLAS arm has no upstream row.

Diagnostic evidence lives in `faer-benchmark/data/probes/20261018-upstream-regression-triage/`.
It compares existing binaries at `0627828` and `2240b07` under the same prepared
public-call boundary, with full engine rotations and an explicit observed 1T
baseline. LU/solve c64 256-square F-order remains about 1.3 times upstream at
4T/8T on both revisions, but not at 1T. Thus the newest GEMM floor is not the sole
cause. This does not identify the cause of the historical gap. Householder's
small F-order 4T gap and tall C-order gaps reproduce too; their mechanisms remain
unverified. The direct small-Householder measurements do not justify the earlier
claim that its timings are unchanged by every recent change.

## First bounded candidate: remove LU's extra area gate

The fork added an `A11.nrows() * A11.ncols()` gate to both recursive panel solves
and trailing GEMMs. This ignores reduction length and scalar arithmetic cost.
The experiment restores upstream's forwarding of `par` at these two sites,
leaving the swap gate, row-major adaptation, numerical algorithm, public
parameters and scratch contract unchanged. GEMM retains its own work floor;
triangular solve retains its own splitting policy. No new constants or
machine-specific/dtype-specific thresholds are added.

The candidate measures the combined effect of restoring both sites on the
current fork. It is not an isolated proof about either site individually.
Rejection is appropriate if the previous f64 improvements cannot be retained
without measured guard losses.

### Predeclared validation and acceptance

- Focused direct-buffer LU test: f64 and genuinely complex c64, C/F buffers,
  square 64/256 and rectangular 129x65/65x129, sequential and owned 4/8-worker
  pools. Check reconstruction residual, permutation validity and inverse
  consistency; do not demand identical pivots across schedules.
- Run the crate library suite before adoption, plus fbench verification before
  timing. A mismatch or undeclared skip blocks acceptance.
- Formal pair: current-fork baseline `2240b07`, candidate clean commit recorded
  by the run. Same harness/toolkit, flags and measured checkouts throughout each
  recording. MKL rows fork/upstream/vendor, 250 ms priming, median of five,
  three complete process starts/rotations, guarded profile CPU sets 1/4/8T.
- Affected families: `lu`, `solve`, `full_piv_lu`, both dtypes/orders, every
  `faer-kernels` linalg shape: 32,64,96,128,256,512,1024x64,4096x256,64x1024,
  256x4096. Rectangular solve skips are declaration-owned. Measure all at
  1/4/8T to include neighbouring/small-cell guards as well as c64 256 targets.
- Collateral controls in the same pair: `cholesky` on its declared square
  shapes and `gemm/256x256x256`, f64/c64 C/F 1/4/8T.
- Invoke `candidate_compare.py`, retain its per-cell JSON, and inspect guard
  ratios against baseline spread. Its zero exit status alone is not an
  acceptance decision: any guard outside the recorded spread rejects adoption.
- `faer-kernels-large` is not part of this first LU isolation experiment;
  there is no inference about its unmeasured cells. Final suite work is separate
  and only starts after candidate selection.

## Outcome

Candidate `a537746` passes the focused direct-buffer test and the crate library
suite (211 passed, 2 ignored). The completed bounded independent code review found
no correctness, numerical or safety blocker; this is not performance approval.

The initial formal pair is recorded as
`20261018T070736Z-upstream-lu-baseline` and
`20261018T073559Z-upstream-lu-restored-budget` in the benchmark campaign. The
official comparator **refused** it: upstream Cholesky c64 F128 4T moved from
0.190123 to 0.178817 ms, outside the baseline band. It is the one failed control
out of 396, with median control ratio 0.999747. There is also an untrimmed slow
start for full-pivot LU affecting both engines; it must not be silently removed.
The apparent target gains in this pair are therefore diagnostic, not acceptance.

The one permitted complete-pair repeat retained the entire grid and reversed
revision order (candidate then baseline). Its runs are
`20261018T081053Z-upstream-lu-restored-budget-repeat` and
`20261018T084143Z-upstream-lu-baseline-repeat`; the predeclaration and complete
per-cell comparison JSON are under
`faer-benchmark/data/probes/20261018-upstream-lu-pair-repeat/`.

**Outcome: INCONCLUSIVE / NOT ADOPTED.** The official comparator refused the
repeat too: 12 of 396 upstream controls leave the baseline band, with median
control ratio 0.998988. Most failures are Cholesky cells; their cause is not
established as host variation versus a difference between linked binaries.
No third pair is attempted and neither cells nor thresholds are relaxed.
The target ratios are consistent across the two pairs, but that does not waive
the failed control gate or constitute acceptance evidence. All four manifests
and reports are retained, including the slow starts; no result page replaces
the published baseline.

The smallest candidate remains on `perf/upstream-lu-scheduling`, not on main.
Adoption requires a comparison whose unchanged controls reproduce the baseline;
the current two pairs cannot supply that evidence. Householder and shared-copy
investigations remain separate, unresolved work. No regression is marked
resolved.
