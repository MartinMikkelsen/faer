# The x86 kernel on the team: `Fork`, not tiles

Continuation of [2026-10-26-team-authority-x86.md](2026-10-26-team-authority-x86.md), which stopped the team size
from reaching spindle but left two costs: `spicy_matmul` and the triangular product ran the kernel on one thread per
participant (no team parallelism at all for LDLT and the supernodal Cholesky), and the plain product was cut into
tiles that each call the kernel, so the kernel's shared packing and job queue were never used on a team.

## Decisions

- The kernel owns its decomposition (right-hand side packed once and shared, a dynamic job queue, triangular and
  indexed destinations), so the fix is in the kernel, not around it: `t4a-private-gemm-x86` 0.1.25
  (tensor4all/t4a-private-gemm-x86#1, draft) adds `trait Fork { width; run(f) }` and `gemm_with(.., &dyn Fork)`.
  `run` is a fork-join of `f(0..width)`; `gemm(.., n_threads)` is unchanged (spindle).
- faer implements `Fork` for a team token (`utils::team::kernel`): `run` is `TeamExec::reborrow().run`, member 0 is the
  caller. The adapter holds the token in a `RefCell`; `Fork` does not require `Sync`.
- Width: `TeamExec::width_for(estimate, 2)`, accepted when `pays(estimate, 2, width)` (two sections per call:
  packing and compute; the unit is the MACs at the dtype's GEMM rate, halved for a triangular destination). A width
  narrower than the team is a `split(&[width])` sub-team; the members it leaves out stay parked. Below the gate the
  same kernel runs on the calling thread (`gemm`, one thread).
- Routes (all `cfg(x86_64, std)`, AVX2+FMA, native dtypes; the same detection as before):
  plain GEMM `linalg/matmul/mod.rs` -> `utils::team::gemm_team`; `spicy_matmul`
  (`linalg/matmul/internal/mod.rs`) and the triangular product (`linalg/matmul/triangular.rs`,
  `mat_x_mat_into_lower_impl_unchecked`) call `gemm_with` themselves. `gemm_tiled` is now only the route without the
  x86 kernel (other targets, other dtypes); its tile no longer calls the kernel.
- `spicy_matmul`'s one-section gate at the full degree is skipped for a team that can use the kernel (it would deny
  widths the two-section gate accepts at a cheaper `width_for`); it is applied after the x86 block for a team that
  cannot. The triangular team recursion remains for the generic route only; `thread::outside_threads` is removed.
- `par_pays`, `join_pays` and `ParThreshold::region_pays` (not `Fixed`) return false for `degree() < 2`, so a team of
  one costs nothing and behaves exactly as `Par::Seq` (the QR 128 probe evaluated 346 gates per item on such a team).
- Counter `stats::Stats::gemm_team_kernel`: products run by the kernel on a team fork. `gemm_below_floor` also counts
  the kernel gate's refusals.
- Pins: threadteam `709589322fe50a9bcd6f33ef81aec0ba3e6042ed` (Linux row, per-member parkers, `min_cost`; `faer/Cargo.toml` and `Cargo.lock`);
  `t4a-private-gemm-x86` as a git dependency at the commit of the 0.1.25 branch (the registry release has no `Fork`).
  `Cargo.lock` was edited by hand for those two entries: `cargo update` also rewrote unrelated `windows-sys`
  entries.

- `utils::cost::tests::rayon_uses_the_outside_cost` asserted the M5 number (`>= 19 us`) of the outside cost; with the
  `586738b` pin the cost comes from threadteam's Linux row and the number is smaller, so the test now states the shape
  (a finite, positive cost). No gate changed.

## Serial path overhead and the cheapest-width check (tensor4all-meta#51 follow-up)

- `e34d99f` (from `fix/serial-path-overhead`): a QR 128 on a `Par::Team` that cannot pay cost +11 % over `Par::Seq`
  because every product and triangular step of the 59 block applies asked its gate. The block apply now asks one
  floor for the whole apply and runs the inner steps with `Par::Seq`; gates read the token's own table
  (`utils::cost::par_pays_time`, no `ParShape` copy); the Rayon host table is built once (`OnceLock`).
  Tests `small_qr_on_a_team_is_the_sequential_one`, `par_pays_time_is_the_shape_gate`.
- The rejection no longer assumes the tables are non-increasing in width. `kernel_width` and the block apply floor
  ask `pays_at(estimate, sections, min_cost)` (`utils::cost::par_can_pay_time`, `par_can_pay_time_team`,
  `ParShape::min_dispatch_cost`; threadteam `DispatchCosts::min_cost`): does not pay at the cheapest width means
  serial, for any table shape. `gemm_team` keeps its full-width gate: it runs at the full team width, so that is
  the width it asks about. Tests `kernel_width_rejects_at_the_cheapest_width` (rising table),
  `min_dispatch_cost_is_the_cheapest_width`.
- Pin threadteam `7095893` (per-member parkers: a narrow section no longer wakes every parked member; table lookups).
  Not measured.

## Verification conclusions and constraints

- Tests (`utils::team::tests`, release): the kernel forks only on members of the team token (thread ids recorded by a
  wrapping `Fork`, whole team and sub-teams of width 3 and 2, each participant once per `run`); plain GEMM for f32, f64,
  c32, c64 and both accumulation modes; `spicy_matmul` lower, upper and with gathered rows and columns; the triangular
  product; Cholesky, LDLT, LU (n 640) and QR (n 1400, the products of n 640 are below the gate) against `Par::Seq`; a
  team of one equals `Seq`, with no gate pays and no team section.
- Not measured: no timing at all; the gate constants (`PAYS`, the GEMM rate, the dispatch costs) are the existing
  ones and were not re-fitted for a two-section kernel call. Whether `width_for` picks a good width for a kernel that
  packs once and shares it is the question for the Linux re-run.
- The thread census (no OS thread outside the team during a Cholesky probe) is by construction here (no call
  to spindle on the team routes, ids recorded inside the fork), not observed from outside.
- The kernel crate's own unit suite was not run (CI runs it on demand); its `fork` integration tests are in the PR.
