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

pub enum ParShape { Seq, Rayon(NonZeroUsize), Team(NonZeroUsize) } // Copy, no authority
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
* The global parallelism is a separate `Copy` type, `GlobalPar { Seq, Rayon(NonZeroUsize) }`,
  with no `Team` variant: `set_global_parallelism(GlobalPar)` cannot be given a `Par`, so a team
  as a process wide setting is a compile error (compile-fail doctest on `GlobalPar`), since a
  team is exclusive authority over reserved threads. `get_global_parallelism()` returns
  `GlobalPar`; `GlobalPar::par()` (or `Par::from`) gives the owned `Par<'static>` that the high
  level `Mat::svd()` family runs with, and `ParShape::from` the shape. `ParShape` is not reused
  because it has a `Team` variant.

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
split leaves over stay parked. GEMM tiling (`utils/team.rs`, `gemm_tiled`) consumes the team and
runs one sequential `gemm` per member on its tile of the output; the depth is never split.
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
| `set_global_parallelism` takes `GlobalPar`, `get_global_parallelism()` returns it | callers pass `GlobalPar::Seq` / `GlobalPar::rayon(n)` instead of `Par`; read the setting as a `Par` with `.par()`; none for the high level `Mat::svd()` family |

## Global parallelism

The planned end state lands in the same breaking release as `Par<'team>`. The process-wide
parallelism setting is removed entirely:

* `set_global_parallelism`, `get_global_parallelism`, `GlobalPar`, `disable_global_parallelism`
  and the faer-ffi get and set functions are removed.
* High-level methods and solvers that read the global setting take an explicit `par: Par`
  argument instead, for example `mat.llt(par)`. There are 97 call sites: 65 in
  `linalg/solvers.rs`, 17 in `sparse/solvers.rs`, 14 in `linalg/mat_ops.rs` and 1 in
  `linalg/reductions/determinant.rs`.
* Operator overloads (`&a * &b`, `&a + &b` in `linalg/mat_ops.rs`) cannot take a `par`, so they
  always run sequentially as a convenience. Parallel callers use the explicit functions, for
  example `matmul(.., par)`.
* `GlobalPar` is the interim state until the explicit `par` arguments land.

## Verification

* Library tests under `--features threadteam` (release): everything passes except the three
  `linalg::matmul::internal::tests::*strict_triangle*` cases recorded in `PATCHSET.md`, which are
  in files this work does not touch. The five integration test targets and the 55 doctests pass.
* Compile-fail doctests on `Par` reject `Copy`, `Clone`, a second use of a consumed `par`, and a
  `join_raw` whose two children both use the parent; one on `GlobalPar` rejects a `Par`
  (so a team) in `set_global_parallelism`.
* `utils::team` tests: GEMM, LU, QR, Cholesky on a team of six match the sequential result;
  teams of one, three and five; `join_raw` widths; `for_each_with_par` deals every item once;
  `join_sized` on a team of six hands out exactly six threads and never overlaps beyond six;
  the sparse numeric matmul forks the team; the LU swap stage splits it in two.
* Build matrix: default features, `--no-default-features --features std,linalg`, and
  `--features threadteam`, each with `--all-targets`.
