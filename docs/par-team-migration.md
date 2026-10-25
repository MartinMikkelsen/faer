# `Par<'team>`: carrying the parallel authority by value

Design spike for the awake-team acceptance condition (tensor4all-meta#48, section "Scoped
execution authority"). It inventories every use of `Par` in the fork, names the blockers to a
non-`Copy` `Par`, and fixes the shape of the type and the order of the migration. Counts come
from `exp/team-par` at `e88599a` (`faer/src`, `faer/tests`, `faer/examples`; `faer-traits` and
`faer-macros` do not mention `Par`).

## Target type

```rust
pub enum Par<'team> {
    Seq,
    Rayon(NonZeroUsize),
    #[cfg(feature = "threadteam")]
    Team(threadteam::TeamExec<'team>),
    // with the feature off the lifetime needs a user: an uninhabited, hidden variant
    #[cfg(not(feature = "threadteam"))]
    #[doc(hidden)]
    __Lifetime(core::convert::Infallible, core::marker::PhantomData<&'team ()>),
}
```

* No `Copy`, no `Clone`, no `PartialEq`. This holds for every variant, including `Seq` and
  `Rayon`: the borrow checker must reject a duplicated `par` whichever variant happens to be
  present at run time, otherwise code that is correct for `Rayon` becomes a budget violation the
  moment a `Team` is passed.
* Parallel entries consume the authority: every public function takes `par: Par<'_>` by value.
  The lifetime is elided in signatures (`par: Par`), which is legal in edition 2021 and keeps
  the 470 signatures textually unchanged. Callbacks written as `impl FnOnce(Par)` or
  `dyn FnMut(Par)` are higher-ranked over the lifetime by elision.
* Successive phases borrow: `par.reborrow()` takes `&mut self` and returns `Par<'_>`. For
  `Seq` and `Rayon` it returns the same variant; for `Team` it is `TeamExec::reborrow`. The
  parent is locked while the child lives, so one value holds the team at a time.
* Concurrent children get a split, never a copy. The two halves of `join_raw`, the column split
  of the block Householder apply and the row blocks of a `for_each` are `TeamExec::split` or
  `fork` children, handed to the callback as `Par<'c>`.
* Read-only queries do not need authority: `par.degree(&self)` and a `Copy` value
  `ParShape` (`Seq`, `Rayon(n)`, `Team(width)`, no execution rights) from `par.shape()`.
* `Par<'static>` is constructible for `Seq` and `Rayon` without any token. `Team` can only
  be built from a `TeamExec` minted by `threadteam::Team::exec` (through `Exec::team`), so a
  global setting can never carry a team.
* No lifetime erasure anywhere: the thread-local installed team, `with_team`, `CURRENT`,
  `install`, `available` and the `TeamExec<'static>` pointer casts of `utils/team.rs` do not
  exist in the design.

### Entry points on `Par`

| method | consumes | children | replaces |
| --- | --- | --- | --- |
| `reborrow(&mut self)` | no | the same authority for one phase | repeated use of a `Copy` value |
| `run(self, f: Fn(i, n) + Sync)` | yes | none, flat section | `team::run`, the inner `for_each` of GEMM tiles |
| `split(self, sizes, f: for<'c> Fn(c, Par<'c>) + Sync)` | yes | one `Par` per size | `join_raw` internals, `team::split` |
| `fork(self, jobs, f: for<'c> Fn(job, Par<'c>) + Sync)` | yes | even shares | `thread::for_each` with nested parallelism, `team::fork` |
| `narrow(self, n, f: FnOnce(Par<'_>) -> R) -> R` | yes | at most `n` participants | `Par::with_degree`, `Par::team(par / n_tasks)` |

`join_raw(op_a, op_b, par)` keeps its signature (`FnOnce(Par)`); `thread::for_each` gains the
child `Par` as a second closure argument. A closure that wants the parent's `par` inside a
concurrent child cannot compile.

## Inventory

### Where `Par` appears

| kind | count |
| --- | --- |
| lines mentioning the identifier `Par` (src, tests, examples) | 1172 |
| `par: Par` parameters (`faer/src` 482, tests and examples 12 more files) | 494 |
| functions with a `Par` parameter and a body (`faer/src`) | 463 |
| of which `pub` / `pub(crate)` / private (trait methods count as private) | 243 / 16 / 212 |
| of which `*_scratch` functions | 126 |
| non-scratch functions | 337 |
| `Par::Seq` literals, `faer/src` / tests, examples, benches | 487 / 74 |
| `Par::Rayon(..)` and `Par::rayon(..)` literals, src / tests, examples | 49 / 18 |
| `Par::Team` and `Par::team` (the thread-local route) | 27 |
| `get_global_parallelism()` call sites | 102 (`linalg/solvers.rs` 66, `sparse/solvers.rs` 18, `linalg/mat_ops.rs` 15, `reductions/determinant.rs` 2, `lib.rs`) |
| doc comment lines that name `Par` | 66 |
| structs or enums that store a `Par` | 0 |

By module (functions with a `Par` parameter; scratch functions in parentheses):

| module | fns | module | fns |
| --- | --- | --- | --- |
| `sparse/linalg` | 82 (23) | `linalg` (top level) | 32 (0) |
| `operator/operator_impl` | 78 (26) | `linalg/lu` | 28 (10) |
| `linalg/cholesky` | 47 (17) | `linalg/gevd` | 22 (6) |
| `linalg/evd` | 43 (8) | `linalg/matmul` | 19 (0) |
| `operator` (traits, solvers) | 43 (13) | `linalg/svd` | 18 (8) |
| `linalg/qr` | 35 (12) | `operator/eigen`, `svd`, `self_adjoint_eigen` | 13 (3) |

Representative public signatures: `linalg::matmul::matmul` (`linalg/matmul/mod.rs:1180` for
the implementation), `linalg::lu::partial_pivoting::factor::lu_in_place`,
`linalg::svd::svd`, `linalg::evd::evd`, `operator::conjugate_gradient::conjugate_gradient`
(`operator/conjugate_gradient.rs:108`), `sparse::linalg::cholesky::*`.

### Scratch functions need a shape, not an authority

126 of 463 functions are `*_scratch(.., par: Par)`. 122 of them only pass `par` on to other
scratch functions or read its degree; 4 do not use it; 1 inspects the variant
(`linalg/gevd/qz_real/mod.rs:1479`). They never run anything, so they take `par: &Par<'_>`:
a shared reference cannot be turned into a runnable value because `Par` is not `Clone`. The
bodies do not change (a `&Par` passes through as `par`). Plan time sizing for a team that does
not exist yet passes a stand-in of the same degree (`&Par::rayon(width)`); `ParShape` is the
Copy value for callers that want to hold a shape (see "Decision").

### Internal uses of the same `par` more than once

Over the 337 non-scratch functions the identifier `par` occurs 1013 times:

| uses of `par` in the body | functions |
| --- | --- |
| 0 (`_par`, kept for the signature) | 12 |
| 1 | 174 |
| 2 | 56 |
| 3 | 26 |
| 4 | 13 |
| 5 or more | 56 |

151 functions use it at least twice, giving 688 later uses as an upper bound for inserted
`.reborrow()` calls; uses in disjoint branches of an `if`/`match` need none, so the real count
is between about 300 and 690 (the compiler enumerates them). 160 `let` bindings derive a new
`par` value (`par_here`, `inner_parallelism`, `let par = if .. { par } else { Par::Seq }`).

Heaviest: `operator/lsmr.rs:142` (32 uses, a loop that applies `A`, `M` and `qr` each
iteration), `linalg/svd/bidiag_svd.rs:1024` (29), `linalg/gevd/gen_hessenberg/mod.rs:223` (22),
`operator/bicgstab.rs:111` (21), `operator/conjugate_gradient.rs:108` (17),
`operator/eigen/mod.rs:270` and `:722` (16 each), `linalg/evd/hessenberg.rs:568` (15).
The sequential pattern, for example in `linalg/lu/partial_pivoting/factor.rs:86`:
`par_here` is used for the panel solve and the trailing product, then the recursion takes the
caller's `par`. That becomes `par_here = if work > floor { par.reborrow() } else { Par::Seq }`
followed by two uses of `par_here.reborrow()` and a final `par`.

### Places where `par` is split for concurrent work

* `join_raw`: 22 call sites in 8 files (`linalg/matmul/triangular.rs` 6,
  `linalg/svd/bidiag_svd.rs` 4, `linalg/householder.rs` 3, `linalg/triangular_inverse.rs` 2,
  `linalg/triangular_solve.rs` 2, `linalg/gevd/gen_hessenberg/mod.rs` 2,
  `linalg/evd/tridiag_evd.rs` 2, `linalg/lu/partial_pivoting/factor.rs` 1), 44 closures that
  receive the child `Par`. The two closures at `linalg/lu/partial_pivoting/factor.rs:203` ignore
  it (`|_|`) and use the outer `par` for a `for_each` inside both halves, which duplicates
  the budget today and is rejected by the new type; they use the child value.
* `thread::for_each` (spindle or team): 16 sites (`linalg/matmul/mod.rs` 6, `bunch_kaufman/factor.rs`
  3, `lu/partial_pivoting/factor.rs` 2, and one each in `householder.rs:815`, `evd/tridiag.rs:394`,
  `lu/full_pivoting/factor.rs:397`, `qr/col_pivoting/factor.rs:268`, `svd/bidiag.rs:130`), and
  `thread::par_for_each`: 3 sites (`linalg/householder.rs:233` and `:679`, `linalg/mod.rs:91`).
  In all 19 the `par` is only the dispatch argument; the closure bodies call sequential kernels
  with `Par::Seq`. Under `Team` these become `Par::fork`, and the closure receives the child.
* Budget arithmetic that builds a smaller `Par` from the degree of the big one:
  `linalg/householder.rs:636` (`Par::rayon(par / n_tasks)`, the inner parallelism of a column
  split). It becomes `fork(n_tasks, |c, child| ..)` with the child passed to the inner product.
* `match par { Par::Seq => .., Par::Rayon(n) => .. }`: 21 sites (20 functions). 15 of them
  are `_ =>` arms that treat every non-`Seq` variant as Rayon (the matmul and matvec kernels, six
  in `linalg/matmul/mod.rs`, three in `bunch_kaufman/factor.rs`, the LU swap stage, and so on).
  14 reach `thread::for_each`, which routes by variant; the one in `sparse/linalg/matmul.rs:166`
  calls raw Rayon, so a `Team` there silently runs on the global Rayon pool. After the change
  every `match` names all variants and the `_` arms disappear.
* GEMM tiling (`linalg/matmul/mod.rs:1357`, `utils/team.rs:gemm_tiled`): consumes the `Team`
  and runs `team.run(..)`.

### Stored in structs or captured in closures

* No struct stores a `Par`. The iterative solver implementations are inner functions that take
  `par` as an argument (`operator/lsmr.rs:153`, `bicgstab.rs:123`, `conjugate_gradient.rs:119`).
* Closures that capture `par` by reference and are called many times: none that call a
  parallel entry. `lu_in_place_recursion` captures `swap` (no `par`); the `for_each` bodies take
  items, not `par`.

### Traits and callbacks

* `LinOp`, `BiLinOp`, `Precond`, `BiPrecond` (`operator/mod.rs:30`, `:177`, `:201`, `:243`):
  36 method signatures in the trait file (`apply`, `conj_apply`, `transpose_apply`,
  `adjoint_apply`, `apply_in_place` and the matching scratch methods), plus 78 in
  `operator/operator_impl/*` (13 files, 6 per file), the `&M` blanket impls, and four
  `IdentityPrecond` impls. They are object safe today and stay so: `par: Par<'_>` is a
  per-method higher-ranked lifetime. The scratch methods take `&Par<'_>`.
* `&dyn LinOp<T>` and `&dyn Precond<T>` appear in the solvers
  (`operator/conjugate_gradient.rs:67`, `faer/examples/linop.rs:286`); each iteration calls
  `A.apply(.., par.reborrow(), stack)`.
* `join_raw(impl Send + FnOnce(Par), impl Send + FnOnce(Par), Par)` has an inner
  `dyn Send + FnMut(Par)` for monomorphisation (`utils/mod.rs:19`); it stays, with the
  higher-ranked `Par<'_>` argument.

### Tests, benches, examples

* In-crate `#[cfg(test)]` modules: about 400 `Par::` literals, mostly `Par::Seq`
  (`sparse/linalg/lu.rs` 27, `linalg/svd/mod.rs` 22, `linalg/qr/no_pivoting/factor.rs` 22,
  `sparse/linalg/cholesky.rs` 20, `utils/team.rs` 16; 13 more files carry test modules). A `Par::Seq` literal stays valid.
  A test that reuses one `par` for `want` and `got` needs `par.reborrow()` or a fresh literal.
* `faer/tests`: 5 files, 51 mentions (`public_bidiagonal_svd.rs` 13, `public_schur_eigenvectors.rs`
  12, `public_balance.rs` 10) plus `tests/common/mod.rs`.
* `faer/examples`: 5 files (`bench.rs` 33, `linop.rs` 14, `bench_iterative.rs`, `dynstack.rs`,
  `lblt.rs`); `bench.rs:1341` has `lapack_set_num_threads(parallel: Par)`, which reads the
  degree and takes `&Par` (or `ParShape`). There is no `benches/` directory.

## Blockers to a non-`Copy` `Par`

1. **Sequential reuse** (151 functions, up to 688 sites). Mechanical: rustc's E0382 reports
   the "value moved here" span; the edit inserts `.reborrow()` at exactly that span and
   repeats until the build is clean. Loops report the same span.
2. **`match par`** (21 sites). Moves the value; arms that bind `Rayon(n)` are fine, arms that
   keep using `par` need `match &par` or an explicit `reborrow`. Hand edit.
3. **Scratch functions** (126). Take `&Par<'_>` (or `ParShape`), see above. Pure signature
   edit plus `&` at about 150 external call sites (tests, examples, doc tests, downstream).
4. **The `_ =>` rayon arms** (15) treat `Team` as Rayon. They must become explicit; one of
   them (`sparse/linalg/matmul.rs:166`) is unrouted today and needs real work, the rest only
   change their pattern.
5. **`get_global_parallelism()`** (102 sites) returns `Par<'static>` of `Seq` or `Rayon`
   only. The global `AtomicUsize` encoding loses the `TEAM_GLOBAL` bit. A team is never global.
6. **The feature-gated variant owns the lifetime.** With `threadteam` off, `'team` would be
   unused and `Par<'team>` would not compile; the hidden uninhabited variant above fixes it
   without a `PhantomData` field in the live variants.
7. **`derive`s.** `Copy`, `Clone`, `PartialEq`, `Eq` go. `Debug` is kept (`TeamExec` has one).
   No test or source compares two `Par` values; `par.shape() == ParShape::Seq` replaces it.
8. **Closures that ignore the child** (3 `join_raw` sites, see above). A genuine fix: the type
   system exposes them.
9. **`join_raw` Rayon arm** puts `(op, Par)` pairs into an array handed to `spindle::for_each`,
   which requires the element type to be `Send`. `TeamExec` is made of shared references and
   plain data, so `Par` is `Send` in practice, but the arm builds `Par::Rayon(n)` inside the
   closure from a plain `usize` instead, so it does not rely on that.
10. **serde and `Default`**: `Par` implements neither; nothing to do.

None of these needs a trait change in `faer-traits`. There is no storage of `Par` and no
`Fn` closure that must call a parallel entry more than once.

## Migration route (each step builds and passes the focused test)

All steps are commits on `exp/par-team-lifetime`; the build matrix of a step is
`--no-default-features`, `--features rayon`, `--features threadteam`, and
`cargo test -p t4a-faer --features threadteam --lib utils::team` plus `cargo check --tests
--examples`. The pinned threadteam revision is `b7d094d` (it already has
`TeamExec::{run, run_with, split, fork, reborrow, size}` and `Exec::team`); it is not edited and
not bumped while the park and wake work continues on the same branch.

| step | change | size | `Par` still `Copy`? |
| --- | --- | --- | --- |
| 1. shape and scratch | add `Par::shape`, `ParShape`, `degree(&self)`; `*_scratch` take `&Par<'_>`; call sites add `&` | 126 signatures, about 150 call sites, scripted | yes |
| 2. child-passing callbacks | `thread::for_each` and `par_for_each` give the child `Par` to the closure; the 3 `|_|` join sites use the child; `with_degree` and `Par::team(par / n)` become `narrow` and `fork` shaped helpers behind the same Rayon behaviour | 22 `join_raw` sites checked, 19 `for_each` sites, 1 narrow site | yes |
| 3. lifetime parameter | `Par` becomes `Par<'team>` with `Seq`, `Rayon` and the hidden variant; the Team variant is not there yet, the thread-local route is deleted together with `Par::Team(NonZeroUsize)`; `Team` tests move to step 5 | enum plus about 60 lines; the 27 `Team` lines removed | yes |
| 4. drop `Copy` | remove `Copy`, `Clone`, `PartialEq`; compiler-guided `.reborrow()` insertion, manual `match par` | about 300 to 690 insertions in about 60 files, 21 matches | no |
| 5. `Team(TeamExec<'team>)` | add the variant, `run`/`split`/`fork`/`narrow` on `Par`, exhaustive matches replace the `_` rayon arms, `gemm_tiled` takes the `TeamExec`, global parallelism is `Seq` or `Rayon` only; `utils/team.rs` shrinks (no thread local, no pointer casts) | `team.rs` 644 lines to about 450; 15 `_` arms; about 12 files | no |
| 6. tests and docs | compile-fail doctests (`par` used twice, `Par: Copy`, `Par: Clone`, a concurrent child using the parent), a budget test that counts distinct threads inside a `join_raw` of a team of 8, `PATCHSET.md` row, worklog | tests only | no |

Why step 1 and 2 keep `Copy`: they carry no exclusive authority yet, so nothing is erased;
they only move signatures onto the shape the later steps need, and they can be reviewed apart
from the 700-site mechanical edit. Step 3 is also safe with `Copy` because the enum has no
variant that owns anything. Step 4 is the one large mechanical diff and is isolated in its own
commit, so a reviewer can verify it as "only `.reborrow()` and `match` edits" (`git diff
--word-diff-regex` on the commit shows no other tokens). Step 5 is where the team enters.

Estimate in agent working hours: step 1 about 1, step 2 about 1.5, step 3 about 1, step 4
about 3 including the 21 matches and a second pass for loops, step 5 about 3, step 6 about 2.
Benchmarks are run once at the end (blast radius is every family because `matmul` and the
shared helpers are touched; use the grid of `docs/adoption-protocol.md`), not per step.

## API break list for users

All of tenferro, tlinalg, tcontract and tprims are affected; none of them stores a `Par`.

| change | who sees it |
| --- | --- |
| `Par` is not `Copy`, `Clone`, `PartialEq` | any code that reuses one `par`: loops that call faer once per batch item (tenferro-cpu `provider.rs:357`, `gemm/faer_gemm.rs:79`), tlinalg's `with_parallel(.., FnOnce(faer::Par))` (`lib.rs:90`) and the per-algorithm `par: faer::Par` arguments (`svd.rs:105`, `:155`, `:187`, `:222`, plus `eig.rs`, `qr.rs`, `lu.rs`, `full_piv_lu.rs`), tcontract `strategy/faer.rs:256` (a per-item closure that already receives a child `Par`) |
| `*_scratch(.., par)` take `&Par<'_>` | every caller of a scratch function: pass `&par`, or `&Par::rayon(width)` before a team exists |
| `Par::Team(NonZeroUsize)`, `Par::team(n)`, `utils::team::with_team` removed | only the benchmark harness branches (`faer-benchmark-team-stage2`, `faer-benchmark-awake-stage1`) |
| `Par::Team(TeamExec<'_>)` | new: `exec.team(|team| faer_fn(.., Par::Team(team)))`; the width is the team size and cannot exceed it |
| `Par::with_degree` removed | none outside the fork |
| `LinOp`, `BiLinOp`, `Precond`, `BiPrecond` | implementors of these traits keep `par: Par` in their methods (lifetime elided, source compatible) and take `&Par` in the scratch methods; a type that calls `par` twice adds `reborrow()` |
| `get_global_parallelism()` returns `Par<'static>` | the high level `Mat::svd()` family: no source change for callers |
| `set_global_parallelism(Par::Team(_))` | rejected at compile time by `'static`; with the feature on it panics in no case because the variant cannot be built from a global |

Downstream edits are small: tenferro `faer_parallelism()` returns a `Par` built from the
provider's budget and each use inside a loop gains `.reborrow()`; tcontract's `to_faer(width)`
keeps `Seq`/`rayon(w)`; tlinalg's `with_parallel` hands the `Par` to the callback once.
None of them needs to hold a `Par` across threads.

## Decision: non-`Copy` everywhere, plus a `Copy` shape that cannot run anything

A `Copy` view that can start parallel work for "sequential phases" is the wrong shape. A
`Copy` runnable value is exactly the duplication the design forbids: two copies can be handed
to two concurrent children, each believing it owns the whole team, and nothing in the type says
no. Sequential phases are already expressed without it: `reborrow(&mut self)` holds the parent
exclusively, which conserves the budget as exactly as moving it.

What the code legitimately needs without authority is information: the degree for thresholds
and scratch sizes. That is `degree(&self)` and the `Copy` `ParShape` (and `&Par` in scratch
signatures). It cannot be passed to any entry point, so it conserves the budget trivially.
`ParRef<'a>` as a `Copy` view that runs work is rejected; `ParShape` as a `Copy` value that
carries no execution rights is accepted.

Open choice for the maintainer: the scratch signatures take `&Par<'_>` (zero new public types,
bodies unchanged) or `ParShape` (one new public type, callers can compute a size from a
width alone without constructing a `Par`). The plan above uses `&Par<'_>` and still adds
`ParShape` for the degree and variant queries (`par.shape()`), so either choice stays open.
