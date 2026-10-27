# The remaining parallel gates: one rule, one serial rate per kernel family

Continuation of [2026-10-26-dispatch-cost-floors.md](2026-10-26-dispatch-cost-floors.md). Design and data:
tensor4all/threadteam-rs `docs/design/thresholds.md` (section 3.4). Code: commit `6788b20`.

## Decisions

- **One rule decides**: `ParShape::pays(estimate, sections, width)`, `PAYS = 3`, steady state costs. The estimate
  is units of work times the serial rate of the family (`utils::cost::rate`); `sections` is counted in the code of
  the gate, not tuned.
- **One type**: `ParThreshold { Derived, Fixed, Never }` moved to `utils::cost` (re-exported from the LU path) and
  is the type of every `par_threshold` field. `Derived` is the default; `Fixed(n)` keeps the old floor in the old
  unit of that field; the old defaults (`192*256`, `256*512`, `128*128`) are gone. `ParThreshold::region_pays`
  is the shared decision, `par_pays` the one for a gate that holds a `Par` and no field.
- **Shape conditions stay when they state another concern**: `2 * n` (reduction depth against the `n x n`
  output) and `4 * threads` (rows per task) in `upgrade_householder_factor` and the row split, `M >= 2K` and the
  chunk covering the unit block in the row split, the `k > 64 && n <= 128` split of a triangular solve (taken for
  every parallelism, only its join is a cost). Each is commented at the gate.

## Regions

| region | old gate | estimate and sections |
| --- | --- | --- |
| full pivoting LU step | `(m-k-1)*(n-k-1) < 256*512` | elements of the trailing block at 0.216 / 0.600 ns, 1 section |
| Bunch-Kaufman (full pivoting) step | `rem*rem < 128*128` | `rem^2` at 0.238 / 0.380, 1 |
| QR with column pivoting step | `(m-k-1)*(n-k-1) < 192*256` | same unit at 0.202 / 0.713, 1 |
| QR without pivoting | no gate (a declared field was removed earlier) | delegates to the Householder and product gates |
| bidiagonalization step | `(m-k-1)*(n-k-1) < 192*256` | same unit at 0.480 / 1.36, 2 (fused update or `y2`, and `z2`) |
| Hessenberg step (unblocked) | `n*n < 192*256` | trailing `n^2` at 0.451 / 1.57, 2 (two products) |
| tridiagonalization step | `rem*rem/2 < 192*256` | triangle at 0.339 / 1.16, 1 |
| `upgrade_householder_factor` (two sites) | `rows*n*n >= 8192` | `deep_rows * n^2` multiply-adds at the GEMM rate, 1 |
| rank one Householder apply | `M*K >= 8192` | `M*K` at 0.212 / 0.725, 1 |
| Householder row split | `M*K >= 8192` | `M*N*K` multiply-adds, 1 |
| Householder column split | `M*K*4M < gemm threshold` | `(1 or 2) * M*N*K` multiply-adds at width `n_tasks`, 1 |
| matvec, both layouts | `nrows*ncols < 256*256` | elements at 0.100 / 0.340, 1 |
| `spicy_matmul` | `m*n*k > 32^3` | multiply-adds at the GEMM rate, 1 |
| triangular product, `mat_x_lower` join | `n*n*m < 128*128*64` | `m*n*n/2`, 1 per level |
| triangular product into a triangle | `n*n*k < 128^3` | `n*n*k/2`, 2 per level (outer join and the join in its second half) |
| triangular solve join (two sites) | no floor (split for `k > 64 && n <= 128` always) | `n*n*k/2`, 1 |
| LLT, LDLT, Bunch-Kaufman with partial pivoting | no gate of their own | the product and solve gates above |

Left without a work gate at `6788b20` (gated afterwards, see [2026-10-26-qr128-and-join-gates.md](2026-10-26-qr128-and-join-gates.md); the norms and `zip` have no parallel region): the divide and conquer joins of `bidiag_svd.rs` and
`tridiag_evd.rs`, `triangular_inverse.rs`, the norm reductions and `zip`, `sparse/linalg/matmul.rs`.

## Constants

One thread, M5, QoS `UserInitiated`, f64 and c64, ns per unit (`constants/` of the faer-benchmark probe
`20261026-threshold-recalibration`; the table with ranges and the host log are in its README): LU full pivoting
0.216 / 0.600, Bunch-Kaufman 0.238 / 0.380, QR column pivoting 0.202 / 0.713, bidiagonalization 0.480 / 1.36,
Hessenberg 0.451 / 1.57 (processes marked busy), tridiagonalization 0.339 / 1.16, matvec 0.100 / 0.340, rank one
apply 0.212 / 0.725. Block apply, triangular product and solve use the GEMM rate (measured 0.037 / 0.039 / 0.051
f64 per multiply-add). Derived floors at width 6 (`PAYS * sections * D / rate`): LU full pivoting 19 thousand
elements on a team (4.2 us), 264 thousand outside (57 us); bidiagonalization 17.5 thousand and 237 thousand
(two sections).

## Tests

`cargo test --release --lib` with `std rayon threadteam` and with the default features: 274 and 262 passed, the
3 known `strict_triangle` failures only, 2 ignored. New: `utils::cost::tests` (derived floors per family against
the rule on the M5 table, outside the pool larger than on a team, representative sizes, `Fixed` and `Never`,
the integer form, sequential never pays, per dtype rates, `forced_parallel_steps_equal_serial_{f64,c64}` for the
six reductions with `Fixed(1)`), `householder::gate_tests` (the Householder, matvec, triangular product and
solve parallel routes equal the serial ones above the floors). The hessenberg tests that wrote
`par_threshold: 0` write `Fixed(1)`.

## Rough M5 confirmation

faer-benchmark `bench/par-team-lifetime`, `data/probes/20261026-threshold-recalibration/` (README has the host
protocol, no busy process in the before and after set), f64, 6 threads, ratios to one thread serial, before
(`9cd64f0`) and after (`6788b20`): plain token QR 128, 256, 512 2.49, 1.99, 1.22 to 1.01, 0.80, 0.48; team QR
0.72, 0.61, 0.41 to 0.85, 0.58, 0.41; SVD (values) 256 and 512 1.05 and 1.09 to 0.87 and 0.88 on both tokens;
EVD and Cholesky unchanged within 4 percent. **QR 128 on a team is worse than before** (86 us to 102 us, serial
120 us); the cause is not established. Nothing was tuned for it.

## Left

- Divide and conquer joins, triangular inverse, sparse product: gated in [2026-10-26-qr128-and-join-gates.md](2026-10-26-qr128-and-join-gates.md); norm reductions and `zip`: no parallel region.
- QR 128 on a team (above); the f32 and c32 rates of the families other than GEMM are halved, not measured.
- The Hessenberg constant comes from busy processes. Acceptance is on a quiet Linux host, including x86_64
  (the x86_64 private GEMM route compiles but is not measured here).
