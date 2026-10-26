# `Par<'team>`: the parallel authority is carried by the value

Design of the execution authority in this fork (tensor4all-meta#48, section "Scoped execution
authority"). `Par<'team>` replaces a copyable width plus a thread-local team: whoever holds a
`Par` holds the right to run work on exactly the threads it stands for.

## The type

```rust
pub enum Par<'team> {
    Seq,
    Rayon(NonZeroUsize),
    #[cfg(feature = "threadteam")]
    Team(threadteam::TeamExec<'team>),
    // with the feature off, an uninhabited variant owns the lifetime
    #[cfg(not(feature = "threadteam"))]
    #[doc(hidden)]
    __Lifetime(Infallible, PhantomData<&'team ()>),
}

pub enum ParShape { Seq, Rayon(NonZeroUsize), Team(NonZeroUsize, DispatchCosts) } // Copy, no authority
```

* `Par` is not `Copy`, not `Clone`, not `PartialEq`, for every variant. The borrow checker must
  reject a duplicated `par` whichever variant is present at run time, otherwise code that is
  correct for `Rayon` becomes a budget violation when a `Team` is passed.
* A `Team` is built from a `threadteam::TeamExec`, which only a team token (`Team::exec`, then
  `Exec::team`) can mint.
* Functions that run work take `par: Par` by value, so a parallel entry consumes the authority.
  The lifetime is elided in signatures (`par: Par`); callbacks written `impl FnOnce(Par)` are
  higher-ranked over it.
* Successive phases lend: `par.reborrow()` takes `&mut self`, returns `Par<'_>`, and locks the
  parent while the lent value lives. One value holds the team at any moment.
* Concurrent children are handed out by splitting, never copied. The two halves of `join_raw`,
  the two sides of the LU swap stage and the jobs of a `for_each` receive sub-teams as `Par<'c>`
  values owned by the child.
* There is no global parallelism: a team is exclusive authority over reserved threads, so it
  cannot be a process wide setting, and the setting is removed (see "Global parallelism").

## Queries without authority

Workspace sizes and thresholds need the size of a parallelism, not the right to run. They use
`ParShape` (`Copy`): every `*_scratch` function and every scratch method of `LinOp`, `BiLinOp`,
`Precond` and `BiPrecond` takes `par: ParShape`, so sizing needs no team and an engine can size a
workspace from a width before any team exists. `Par::shape(&self)` converts, `ParShape::{seq,
rayon, team}` construct from a width, and `degree()` is available on both (`Par::degree(&self)`).
`ParShape` cannot be passed to any function that executes work, so it conserves the budget
trivially. A `Copy` view that can run work is deliberately absent: two copies could be handed to
two concurrent children, each owning the whole team.

## Entry points (`utils::thread`, hidden from the public docs)

| function | authority | children |
| --- | --- | --- |
| `join_raw(op_a, op_b, par)` | consumed | halves of widths `n - n/2` and `n/2`, each a `Par` |
| `join_sized(par, size_a, op_a, op_b)` | consumed | widths `size_a` (clamped to leave one thread to each side) and the rest |
| `split(par, sizes, f)` | consumed | one `Par` per size; the widths add up to at most the degree |
| `for_each(par, n_jobs, iter, f)` | consumed | none: bodies are sequential |
| `for_each_with_par(par, n_jobs, iter, f)` | consumed | each job receives its share as a `Par` |
| `par_for_each(par, iter, f)` | consumed | none |

On a `Team`, `split` and the jobs of `for_each_with_par` are `TeamExec::split` and `fork`: the
child leader runs the callback on its own members, concurrently with its siblings, and members a
split leaves over stay parked. GEMM tiling (`utils/team.rs`, `gemm_tiled`, the route without the x86 kernel) consumes the team and
runs one sequential `gemm` per member on its tile of the output; the depth is never split. On x86_64 the kernel runs on the team instead (`gemm_team`, last section).
`Rayon` splits give each child `Par::Rayon(width)` (`Seq` for a width of one); `Seq` runs the
parts in turn.

Every `match par` is exhaustive over the variants; no arm treats a `Team` as `Rayon`. No code
path dispatches to the global Rayon pool with a `Team` in hand.

## Writing code against it

* A kernel with phases lends: `f(.., par.reborrow())` for every use but the last, `f(.., par)`
  for the last.
* A value derived from `par` for part of the work (`let par_here = if big { par.reborrow() } else
  { Par::Seq };`) is a lend, and is dropped before the next use of `par`.
* `par.shape()` and `par.degree()` are evaluated before a lend that is alive in the same
  expression (`let par_shape = par.shape();` ahead of the call that takes `par.reborrow()`).
* A closure that is called more than once and needs a parallelism takes it as an argument; one
  that captures `par` by value is `FnOnce`.
* Concurrent regions use `join_raw`, `join_sized`, `split` or `for_each_with_par` and pass the
  child they receive to everything they call. A closure that names the parent instead does not
  compile.

## API break list for users

| change | who sees it |
| --- | --- |
| `Par` is not `Copy`, `Clone` or `PartialEq` | code that reuses one `par`: loops calling faer once per batch item (tenferro-cpu `provider.rs`, `gemm/faer_gemm.rs`), tlinalg's `with_parallel` and its per-algorithm `par: faer::Par` arguments, tcontract `strategy/faer.rs` (its per-item closure already receives a child `Par`) |
| `*_scratch(.., par)` take `ParShape` | every caller of a scratch function: pass `par.shape()` or `ParShape::{Seq, rayon(n), team(n)}` |
| `LinOp`, `BiLinOp`, `Precond`, `BiPrecond` | implementors keep `par: Par` in `apply` and friends (lifetime elided) and take `ParShape` in the scratch methods; an `apply` that uses `par` twice adds `reborrow()` |
| `Par::Team(TeamExec<'team>)` | `exec.team(\|team\| faer_fn(.., Par::Team(team)))`; the width is the team size |
| `set_global_parallelism`, `get_global_parallelism`, `GlobalPar`, `disable_global_parallelism` | removed; pass `par`. The high level methods and solvers take a trailing `par: Par` (`mat.svd(par)`, `mat.llt(side, par)`, `solver.solve(rhs, par)`, `solver.reconstruct(par)`, `mat.determinant(par)`, `sp_lu(par)`, ...); operators (`&a * &b`) run sequentially |

## Global parallelism

Done (acceptance condition 4). The process-wide parallelism setting is removed entirely:

* `set_global_parallelism`, `get_global_parallelism`, `GlobalPar`, `disable_global_parallelism`
  and the faer-ffi get and set functions are removed (the C++ wrapper loses the `get_global_par()`
  default argument, so `Par` is explicit there too).
* High-level methods and solvers that read the global setting take an explicit `par: Par`
  argument, last in the list (the faer convention), for example `mat.llt(side, par)`. 97 call
  sites: 65 in `linalg/solvers.rs`, 17 in `sparse/solvers.rs`, 14 in `linalg/mat_ops.rs` and 1 in
  `linalg/reductions/determinant.rs`. A solver built once and then used takes the `par` where
  the work happens: the constructor, `solve*`, `reconstruct`, `inverse`, `pseudoinverse`,
  `compute_Q`, `compute_thin_Q`; the `SolveCore`, `SolveLstsqCore` and `DenseSolveCore` trait
  methods take it too.
* Operator overloads (`&a * &b`, `&a + &b` in `linalg/mat_ops.rs`) cannot take a `par`, so they
  always run sequentially (`Par::Seq`) as a convenience; so does `Distribution::sample` for
  `UnitaryMat` (stats). Parallel callers use the explicit functions, for example
  `matmul(.., par)`.
* Test: `utils::team::tests::high_level_*` run `singular_values`, `partial_piv_lu().solve` and
  `qr().solve` on a team and compare with `Par::Seq`.

## Parallel floors: one rule

Every cost gate of the library is the same decision, `ParShape::pays(estimate, sections, width)`:
a region runs in parallel when `estimate >= PAYS * sections * dispatch_cost(width)` with `PAYS = 3`.

* The **dispatch cost** belongs to the token. `ParShape::Team(n, costs)` carries the costs of its
  team token (threadteam `DispatchCosts`, steady state: team 1.4 us at width 6 on the M5 table, 0.9 us on the
  `x86_64-avx512` row);
  `Rayon(n)` is entered from outside the pool and uses the `Outside` cost of the host class (19 us at width 6
  on the M5, 8.7 us on `x86_64-avx512`, 40 us generic); `Seq` never pays. `ParShape::dispatch_cost(width)` and `pays_full` (the full
  degree) are the queries. A build without the `threadteam` feature keeps the same numbers in
  `utils::cost`.
* The **estimate** is units of work times the serial throughput of the kernel family
  (`utils::cost::rates()`, `Rate { real, cplx }` in ns per unit, read through `rate::<family>()`: GEMM per
  multiply-add, the reductions per element of the trailing block, matvec per matrix element, the rank-one
  Householder apply per target element, the divide and conquer solves per `m^2`, the triangular inverse per
  `m^3`, the sparse product per multiply-add; `f32` and `c32` are half of `f64` and `c64`, the GEMM measured
  where a row has it). One constant per family and **per host class**, measured with one thread.
* **Host classes.** Both tables, the rates and the Rayon dispatch costs, have one row per `HostClass`,
  keyed by the kernel dispatch the machine takes, not by the CPU model: `x86_64-avx512` (x86_64 with runtime
  `avx512f`), `x86_64-avx2` (avx2 and fma), `x86_64-scalar`, `aarch64-apple`, `aarch64`, `generic`.
  `HostClass::detect()` runs once (`is_x86_feature_detected!` under `std`; without `std` the target features
  of the build decide on x86_64). It is kept in sync by hand with `threadteam::HostClass` so that faer builds
  without the `threadteam` feature; a test compares the two when the feature is on. `Rates::for_class(class)`
  is a row, `rates()` the row of the host.

  | class | rates | Rayon outside (us at width 2, 4, 6) |
  | --- | --- | --- |
  | `x86_64-avx512` | Zen 5 (Ryzen AI 9 HX 470), 1 thread, 2.0 GHz, boost off (`faer-benchmark` `data/probes/20261026-awake-team-linux`, `tables-consts-exec3.md`) | 9.7, 10.1, 8.7 |
  | `aarch64-apple` | Apple M5 Max (the first calibration) | 14, 18, 19 |
  | every other class | `generic`: per entry the smaller of the measured rows | 40 flat |

  The `generic` row is the per entry minimum of the measured rows (fastest known time per unit, so an
  unmeasured host keeps work serial a little longer instead of dispatching early); a class that gets a
  measurement replaces its alias. **A row's rates and its dispatch costs must come from the same machine and
  clock state:** the boost moves both by about 1.66 times with a stable ratio (within a few percent,
  `tables-*-exec3-boost.md`), so mixing a boosted rate with an unboosted cost shifts every floor. The Rayon
  table of a build without the feature is the model of threadteam (linear between the anchors at widths 2, 4,
  6, flat above) with the same values. Region constants of the `x86_64-avx512` row: the block apply streaming
  per multiply-add, the divide and conquer solves (median over `m` 64 to 512; per `m^2` 152.9 and 97.4 ns, size
  dependent, 145 to 178 and 88 to 102), the sparse product (median of the measured densities) and the
  triangular inverse (median over `m`; 0.047 per `m^3` at 64 falling to 0.0145 at 512, as on the M5 row).
  Matvec is the mean of the column major and row major products.
* **`sections`** is the real number of parallel dispatches the region performs, read from the
  code of the gate and never tuned: one for a fused step, two for a step that runs two products
  (bidiagonalization, Hessenberg), two for the trailing update of an LU level, four for its swap
  stage (memory bound, provisional), one per join level of the triangular product into a full
  matrix and two per level for the product into a triangle.
* **`ParThreshold { Derived, Fixed(n), Never }`** is the type of every `par_threshold` field
  (partial and full pivoting LU, Bunch-Kaufman with full pivoting, QR with column pivoting,
  bidiagonalization, Hessenberg, tridiagonalization). `Derived` (the default) is the rule; `Fixed(n)`
  is an explicit floor in the unit the field documents (`units >= n`; for the partial pivoting LU
  the earlier area and flop floors); `Never` stays serial. The integer form of the C API maps `0`
  to `Derived` and `usize::MAX` to `Never`. `QrParams` (no pivoting) has no such field: its
  factorization delegates to the Householder and product gates.
* **Shape conditions stay when they state another concern than cost** and are documented as such
  at the gate: the reduction depth `rows >= 2 * n` and the rows per task `rows >= 4 * threads` of
  the Householder reductions, the chunk that must cover the unit block of the row split, and the
  split of the right hand sides of a triangular solve (`k > 64 && n <= 128`, taken for any
  parallelism; only its join is a cost).

* **A join is a section of width 2.** `join_raw` splits the budget into two halves; the dispatch of that split
  is the cost of the shape at width 2 (team 3.0 us, plain outside 14 us on the M5 table), not at the full
  degree. `utils::cost::join_pays(par, units, rate, sections)` decides every join of the library this way: the
  recursion and merge joins of the divide and conquer SVD and EVD, the triangular inverse, the triangular
  products and the split triangular solve. A flat section over all of the members (a column split, the sparse
  numeric product, the fused steps) keeps the full degree.
* **Recursions** (`halves_pays`): the work of a join of size `n` is the two subproblems, `2 (n/2)^power` units of
  the rate of a whole serial solve of size `n / 2` (`power` 2 for the divide and conquer solves, 3 for the
  triangular inverse); the merge after the join is not in it.
* **The block apply column split** is estimated as `products * M * N * K` multiply-adds at the GEMM rate plus
  `M * K` target elements at the rank one apply rate (`block_apply_estimate`).
* **A sequential token never pays**: `par_pays`, `join_pays` and `ParThreshold::region_pays` (not `Fixed`)
  return false before computing the estimate.
* **Norms, `zip`, elementwise kernels** have no parallel region, so they have no gate.
* **Probe feature `par-trace`** (`utils::cost::trace`): records site, dimensions, units, sections, estimate,
  dispatch cost and decision of every gate; off by default.

## API break list entries for the floors

| change | who sees it |
| --- | --- |
| `par_threshold: usize` of `FullPivLuParams`, `LbltParams`, `ColPivQrParams`, `BidiagParams`, `HessenbergParams`, `TridiagParams` and `PartialPivLuParams` is `ParThreshold` | callers that set it: `usize::MAX` is `Never`, an old floor `n` is `Fixed(n)` (in the unit the field documents), and the removed `0` ("always parallel") is `Fixed(1)`; callers that leave it at `auto!` get `Derived` |
| `QrParams::par_threshold` is removed (it was never read) | callers that named it; the `faer-ffi` mirror drops it too |

## Verification

* Library tests under `--features threadteam` (release): everything passes except the three
  `linalg::matmul::internal::tests::*strict_triangle*` cases recorded in `PATCHSET.md`, which are
  in files this work does not touch. The five integration test targets and the 55 doctests pass.
* Compile-fail doctests on `Par` reject `Copy`, `Clone`, a second use of a consumed `par`, and a
  `join_raw` whose two children both use the parent; the global setting that a
  `GlobalPar` compile-fail doctest guarded no longer exists.
* `utils::team` tests: GEMM, LU, QR, Cholesky on a team of six match the sequential result;
  teams of one, three and five; `join_raw` widths; `for_each_with_par` deals every item once;
  `join_sized` on a team of six hands out exactly six threads and never overlaps beyond six;
  the sparse numeric matmul forks the team; the LU swap stage splits it in two.
* Build matrix: default features, `--no-default-features --features std,linalg`, and
  `--features threadteam`, each with `--all-targets`.
* x86_64: a `Team` never hands its size to an outside runtime; the kernel forks on the team itself (`utils::team::kernel`); tests in `utils::team::tests` (records: `docs/worklogs/2026-10-26-team-authority-x86.md`, `docs/worklogs/2026-10-26-team-gemm-executor.md`).
* x86_64, `exp/team-gemm-executor`: a `Team` runs the `private_gemm_x86` kernel itself. The kernel forks its packing and its job queue through a `Fork` executor (`private_gemm_x86::gemm_with`), which `utils::team::kernel` implements over the team (`TeamExec::run` on a sub-team of the gate's width). Plain GEMM, `spicy_matmul` and the triangular product use it, gated by `pays(estimate, 2 sections, width)`; `thread::outside_threads` is removed because no team size reaches an outside runtime any more (record: `docs/worklogs/2026-10-26-team-gemm-executor.md`).
