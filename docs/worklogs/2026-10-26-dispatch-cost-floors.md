# Parallel floors from the dispatch cost of the token

## Decisions

- **One rule.** A region runs in parallel when its serial time is at least `PAYS` (3) times
  `sections` times the cost of one section of the shape (`ParShape::dispatch_cost`, `ParShape::pays`).
  The rule and the team costs are threadteam's (`pays_at`, `DispatchCosts`); `utils::cost` keeps the same
  rule and the Rayon cost for a build without the `threadteam` feature. Design and data:
  `tensor4all/threadteam-rs` `docs/design/thresholds.md`.
- **`ParShape::Team(n, DispatchCosts)`** carries the costs of its token (copied by `Par::shape`), so a
  `PoolBuilder::dispatch_costs` override reaches the team decisions. `Rayon(n)` is entered from outside the pool
  and uses the `Outside` cost of the host default table; `Seq` is `Duration::MAX`.
- **`sections` is the real number of dispatches**: GEMM 1; LU trailing update 2 (solve and product, the second
  on cold caches); LU swap stage 4 (provisional, from the design, memory bound).
- **Throughput per dtype** (`utils::cost::ns_per_mac`): f64 0.037, c64 0.165 per complex MAC (design data);
  f32 0.0185 and c32 0.0825 by halving (measured on one thread: 0.019 and 0.084). Swap: 0.55 ns per row swap of
  one column (random pivots, columns of 64 to 1024, one thread, range 0.31 to 1.0).
- **`PartialPivLuParams::par_threshold`** is `ParThreshold { Derived, Fixed(usize), Never }` (default `Derived`),
  replacing the `usize` sentinels (0 derived, 1 always, `usize::MAX` never). `Derived` derives the two LU regions from
  the rule; `Fixed(n)` restores the old fixed floors (`Fixed(0)` and `Fixed(1)` are always parallel); `Never` disables
  parallelism. API break: a caller that wrote `par_threshold: usize::MAX` writes `ParThreshold::Never`, one that wrote
  `1` writes `Fixed(1)`, and one that wrote `0` to remove the gate now gets the derived rule. The excluded `faer-ffi`
  keeps its `size_t` field through `From<usize>`/`From<ParThreshold>` (0 derived, `usize::MAX` never).

## Fixed bug

The team route (`utils/team.rs` `gemm_tiled`) compared `m*n*k` MACs against the real-valued floor for every dtype,
so c64 `64^3` (43 us serial) stayed sequential on a team. The floor is now a time, so a complex product
reaches it at 4.5 times fewer MACs. Test: `utils::team::tests::the_team_floor_is_per_dtype`.

## Changes

- `utils/cost.rs` (new): rule, rates, `ParShape::{dispatch_cost, pays, pays_full}`.
- `matmul/mod.rs`: the x86_64 private kernel width and the aarch64/generic `gemm` crate `Rayon` decision use
  `pays_full`; the team call passes the dtype's rate to `gemm_tiled`.
- `lu/partial_pivoting/factor.rs`: `trailing_update_pays`, `swap_stage_pays`.
- `qr/no_pivoting/factor.rs`: the unused `QrParams::par_threshold` is removed (`faer-ffi`, outside the
  workspace, still mirrors it).
- threadteam pinned at `5afe0b4` (`feat/dispatch-cost` HEAD; adds `Exec::carries_team` to `b99e913`), so that downstream crates that patch `threadteam` to the same rev share one `TeamExec` type.

## Remaining (done in [2026-10-26-remaining-gates.md](2026-10-26-remaining-gates.md))

Full pivoting LU, Bunch-Kaufman, column-pivoting QR, bidiagonal SVD, Hessenberg and tridiagonal EVD
`par_threshold`s; `householder.rs` (`8192` and `gemm::get_threading_threshold`); matvec, triangular and internal
matmul floors; triangular solve splits. Listed with locations in the threadteam `thresholds.md` section 3.4.

## Verification

- `cargo test --release -p t4a-faer --lib --features "std rayon threadteam"`: 261 passed, 3 failed (the known
  `strict_triangle` failures), 2 ignored; with `std rayon` only: 250 passed, same 3 failures. `cargo check`
  for `x86_64-apple-darwin` passes (the x86 kernel call site compiles; not run).
- New tests: `utils::cost::tests` (decision functions), `the_team_floor_is_per_dtype`,
  `lu::partial_pivoting::factor::tests::{region_rules_follow_the_serial_time, derived_gates_keep_the_factorization}`.
- Rough M5 confirmation (6 threads, not an acceptance grid): faer-benchmark `bench/par-team-lifetime`,
  `data/probes/20261026-dispatch-cost-floors-m5-6t/`, issue #24.
