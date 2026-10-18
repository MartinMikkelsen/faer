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

Pending measurement. No regression is marked resolved and no new candidate is
adopted yet. Householder and shared-copy investigations remain separate work.
