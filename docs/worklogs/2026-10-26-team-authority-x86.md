# A team token on x86_64: no threads outside the team, and the x86 kernel per tile

Continuation of [2026-10-26-par-team-lifetime.md](2026-10-26-par-team-lifetime.md). Found by the Linux acceptance
run (tensor4all-meta#51, summary in tensor4all-meta#48, data in `faer-benchmark/data/probes/20261026-awake-team-linux/`).

## Defect 1: the team size reached an outside runtime

`spicy_matmul` (`linalg/matmul/internal/mod.rs`) and `mat_x_mat_into_lower_impl_unchecked`
(`linalg/matmul/triangular.rs`) call `private_gemm_x86::gemm(.., par.degree())`. For `Par::Team`,
`parallelism_degree` is the team size, so the kernel threaded the product through spindle, outside the team, while
the members spun. Measured on the acceptance host: Cholesky 512/1024 under a team token had 8 busy non-team threads
and ran 1.1 to 5.8 times slower than the held Rayon row.

Fix:

- `utils::thread::outside_threads(&par)` is the thread count to hand to a runtime that lives outside `par`: 1 for a
  `Team`, `parallelism_degree` otherwise. `utils::thread::is_team` is the matching test.
- `triangular.rs`: a team whose join pays (`join_pays`, decided once, before the x86 block) skips the kernel and runs
  the recursion of the generic path on the team: the off-diagonal blocks are plain `matmul` calls that reach
  `gemm_tiled`. A team that does not pay takes the `DstKind::Lower` kernel with one thread. Rayon and Seq are
  unchanged (the Rayon x86 path keeps its ungated `par.degree()`).
- `spicy_matmul`: the kernel takes one thread for a team. It does not use the team: the blocks are triangular or
  indexed, and no tiling of them was written. This is the open lead: LDLT and the supernodal Cholesky on a team
  are serial per participant until a team split of those blocks exists (not measured here).

Other places that pass a degree to an outside runtime, all checked: the remaining `par.degree()` uses
(`matmul/mod.rs` matvec, rank update and the like, QR/LU/EVD/SVD/Bunch-Kaufman regions, sparse matmul, Hessenberg)
only size job counts that go through `utils::thread::for_each`/`join_*`, which route a team through the team; the
plain GEMM entry gives `Par::Team` `gemm::Parallelism::None` (and intercepts it earlier). No other call reaches
`private_gemm_x86`, `gemm::Parallelism::Rayon`, spindle or rayon with a team size.

## Defect 2: the team tile never used the x86 kernel

`gemm_tiled` called `gemm::gemm(.., Parallelism::None)` per tile, so on x86_64 a team never ran `private-gemm-x86`,
unlike the `Seq` and `Rayon` rows (1.05 to 1.11 times slower at square 512 to 2048). Each tile now calls
`private_gemm_x86::gemm` with one thread on x86_64 with `std` and AVX2+FMA (AVX-512 when present), the same feature
detection as the plain entry; dtype by `TypeId` of `E` (f32, f64, c32, c64), else the `gemm` crate. The `gemm` crate
convention `dst := alpha dst + beta lhs rhs` maps to the kernel's `Accum` and `alpha` (the plain entry passes
`alpha` of 0 or 1 only).

## Tests

`utils::team::tests`: `a_team_token_hands_no_threads_to_an_outside_runtime`,
`triangular_products_on_the_team_match_sequential_and_stay_on_it`, `spicy_matmul_on_a_team_matches_sequential`,
`gemm_tiled_matches_the_gemm_crate_for_every_dtype` (four dtypes, both accumulation modes, conjugation, and the tiled
counter proves the grid ran).

## What remains unverified

- No timing: the effect on the acceptance rows is for the Linux re-run.
- The test cannot observe OS threads of spindle directly; it asserts the thread count handed out (`outside_threads`),
  equality with sequential, and that the work ran as team sections. A thread census under a Cholesky probe is the
  measurement that closes it.
