# The LU recursion's work floor is counted in flops, not elements (2026-10-19)

**What was measured, by someone else.** The 2026-10-18 full `faer-kernels` page at the
fork pin `2240b07` against upstream `0539947` (MKL arm, the two rows in one binary,
three rotated starts) flags 92 cells slower. Classified by evidence rather than
magnitude, most of them are not evidence: 23 `householder_apply` cells all carry `!`
(their between-start range exceeds 5%), 30 are sub-20 us and dispatch-dominated
(including 11 of `cholesky`'s 12, and the fork's `cholesky` source is byte-identical to
upstream), and the clean remainder is the `c64 n256` `lu`/`solve` pair at 1.26-1.51 and
small `1T` C-order `lu`/`solve` at 1.10-1.17.

**The `c64 n256` gap is this experiment's floor, counted in the wrong unit.**
`28bc13b` compares `A11.nrows() * A11.ncols()` against `par_threshold`. The crossover
it picked (16384) came from a sweep on `f64`, so it is a statement about arithmetic,
but the comparison counts elements. A `c64` element is four real MACs: at `n256` the
`128x128` trailing update has the work of an `f64` `256x256` and stayed `Par::Seq`
where `f64` threads the same area. The page shows the sign flip this predicts -- the
same code is 1.26-1.51 slower at `c64 n256` and 0.49-0.66 faster at `f64` 64/96/128.

**What upstream does, and what the fork changed.** At `0539947`,
`lu/partial_pivoting/factor.rs` reads `par_threshold` in exactly one place, gating the
**swap stage**, and hands the caller's `par` to the panel TRSM and the trailing product
unconditionally. The floor is the fork's own addition; it gave an existing public field
a second meaning. That is why the campaign's independent PRE review called the
"trailing GEMM only" variant a public-contract change.

**The change.** Keep the field's meaning and fix the unit:

```rust
let work = A11.nrows().saturating_mul(A11.ncols())
    .saturating_mul(if T::IS_REAL { 2 } else { 8 });
let par_here = if work > params.par_threshold.saturating_mul(2) { par } else { Par::Seq };
```

This is the principle `2000dd9` already adopted for the dense GEMM floor -- decide in
flops, not in bytes or elements -- applied to the one other floor in the fork that
counted elements.

- `f64` decisions are unchanged (`area * 2 > 16384 * 2` iff `area > 16384`), so the
  sweep that chose 16384 keeps its meaning and its 1.68x-1.02x results.
- The complex crossover moves from `area > 16384` to `area > 4096`: `c64` shapes of 192
  and above now thread the trailing update, `n256`'s `128x128` included.
- The comparison saturates, so `par_threshold = usize::MAX` still returns `Par::Seq`
  everywhere. The "disable parallelism" meaning survives, which is what the
  trailing-only variant lost.

**What this is not.** Not a measurement. The recovery at `c64 n256` is predicted from the
`a537746` pair (which restores a superset of this scheduling and measured
`c64 F256` at 0.740 / 0.696 for 4T / 8T) and from the `f64` side being untouched;
neither is a timing run of this commit. Those pairs also failed the unchanged-upstream
control gate (1/396 and 12/396), so even the superset figure is diagnostic.

**Gate.** `cargo test -p t4a-faer --lib`: 210 passed / 2 ignored, the same count as main
at `7364be5`. `threaded_lu_matches_the_sequential_one` now covers `c64` at
96/128/192/256/512 against the sequential factorization, in addition to the `f64`
shapes it already covered.

**Open.** Adoption needs a pair that passes the unchanged-upstream control gate, and
that gate is now the blocker -- two complete pairs on identical source moved an
unchanged upstream cell with median control ratios of 0.99975 and 0.99899. The existing
same-source (rename-only, then swapped) control is the design that can say whether the
movement follows the position or the binary.
