# PATCHSET

The ledger of the experiments this fork runs. It records the diff against
upstream `faer`.

## Status legend

- **TODO** — not started. The issue column names the driving issue, or `—`.
- **EXPERIMENT** — being implemented on a branch. Breaking things is fine.
- **KEEP** — tensor4all keeps using it. Whether it goes upstream is a separate
  decision.
- **DROPPED** — tried and thrown away. The reason stays here.

## Fork infrastructure (not candidates for upstream)

| Change | What |
| --- | --- |
| Crate names | `faer` → `t4a-faer`, `faer-traits` → `t4a-faer-traits`. `[lib] name` stays `faer` / `faer_traits`, so downstream `use faer::...` is unchanged. |
| Workspace | `faer-ffi` is excluded (the C API and its cbindgen build are unused by tensor4all). |
| Publication | crates.io, in the order `t4a-faer-traits` then `t4a-faer`. The tag is `v<faer version>`. A version that is already on crates.io is skipped, so a release that bumps only one crate still succeeds. |
| Released | `t4a-faer-traits 0.24.0` and `t4a-faer 0.24.4`, published from commit `3a6a9d6`. The published `src/**` is byte-identical to that commit, so `3a6a9d6` is the revision a consumer pins. A later commit needs a new version. |
| LICENSE | A `LICENSE` copy lives in each crate directory so the published archive ships the MIT text (`license-file` next to `license` makes Cargo warn). |

## Experiments

| Experiment | What | Status | Issues | Notes |
| --- | --- | --- | --- | --- |
| Unpivoted QR column skip | The skip lived only in `qr_in_place_unblocked`; `qr_in_place_blocked` has no threshold of its own and recurses into that core, so removing it there covers the tall and wide paths at once. The reflector is applied and the row advances for every nonzero column, matching LAPACK; the tall/wide `svd` path goes through it too. | KEEP | tlinalg-rs#28, tensor4all-rs#836 | Adopted. Tests: `test_near_dependent_columns_are_not_skipped` (tall `m x 3`, m=100/1000/8192) and `test_near_dependent_rows_are_not_skipped` (its transpose, wide `3 x n`, n=100/1000/8192), both at `<=1e-13`; measured 1e-16 to 5e-15. The wide fixture was already accurate before the change: at m=3 the removed threshold `16 (m-k) eps ||column||` is ~1e-14 and never tripped it (a wide skip's relative reconstruction error is ~`16 eps sqrt(m) sqrt(m/n)`), so the wide path needed no separate treatment — the same core change covers it. tlinalg's reported wide `3.33e-7` is a reconstruction-indexing bug in that test (entry `(i,j)` written to flat `j+3i` while the factor is column-major `i+3j`); faer's factors for that matrix are accurate to ~4e-16. Existing columns are unchanged; formerly skipped nonzero columns now pay reflector/update flops (the threshold norm bookkeeping is removed). No existing benchmark suite was run. Release smoke, 3-run mean, before→after: 100000×8 2.984→2.986 ms; 1000×1000 19.793→19.921 ms. |
| Row-split reduction in the block Householder apply | `tmp = V^H C` has an `N x K` output and reduces over every row of the target. With `N` small and the target long, the products inside the generic path were one serial reduction inside one task, so the column split (capped at `min(par, K, 4)`) could not help: 4096x16 basis against a 4096x240 target measured 2.03 ms at 8T where OpenBLAS's `?gemm` on the same shape is 0.53-0.71. When the call is threaded and the shape is tall (`rows >= 4 * threads`, `N * 4 * threads <= rows`, `rows * K >= 8192`), the reduction now runs one `N x K` partial per thread and the partials are added -- they are tiny against the target. Measured in isolation the same shape goes 2.03 -> 0.55 ms at 8T with 1T unchanged (3.46 -> 3.47), and end to end through tlinalg's kernels on `zen5-cpu`, 8T pool: `householder_factor` 4096x256 b1 c64 17.44 -> 15.68 ms (f64 5.83 -> 5.58), plus `qr` c64 46.75 -> 42.87 and `svd_thin` c64 77.11 -> 67.79 through the same path, with every 1T row unchanged. Tests: `threaded_panel_matches_the_sequential_one` (2048x4, 4096x16 and a shape below the threshold) and the two existing cases. | KEEP | tlinalg-rs#46 | Costs one `N * K * threads` buffer per call on the new path; the public scratch function is unchanged, so if this goes upstream the scratch should carry it. Own code: no OpenBLAS or MKL source is used or adapted. |
| Threaded rank-one Householder apply | The `N == 1` arm of `apply_block_householder_on_the_left_in_place_generic` is a sequential rank-one update per target column and it never read `par`. A tall blocked QR's panel is made almost entirely of those updates, so the panel was the serial part of it: at 4096x256 c64 the panel is 5.96 ms at 1T and 6.31 ms at 8T, against OpenBLAS's `?gemv`/`?geru` panel at 5.97 -> 2.58 ms on the same eight cores. When the caller asks for threads and the update is at least 8192 elements, the arm now falls through to the generic path, which partitions the target's columns and threads its two products. Through tlinalg's compact factor (`householder_factor`), pool row, 4T / 8T: 4096x256 f64 6.73 -> 5.77 and 6.67 -> 5.83 ms, 4096x256 c64 21.42 -> 18.75 and 19.50 -> 17.44 ms (11-14%), with 512x512, 128x128, `qr`, `svd_thin` and every 1T row unchanged within noise. Tests: `threaded_rank_one_matches_the_sequential_one` (f64 and c64; 2048x8, 16384x1, 64x8) and `rank_one_negates_its_own_reflector_column`. | KEEP | tlinalg-rs#46 | Own code: no OpenBLAS or MKL source is used or adapted, their timings are only the target. |
| Tall compact-WY trailing update scaling | The blocked compact QR applies its panel through `apply_block_householder_transpose_on_the_left_in_place_with_conj`, whose generic path partitions the target's columns with `n_tasks = min(parallelism_degree(par), K, 4)` and `rayon(par / n_tasks)` inside each. On a tall DRAM-bound panel (4096x16 basis, 4096x240 target, eight physical Zen 5c cores sharing 8 MiB of L3, 16.8 MB target) the update is 8.72 ms at `Par::Seq` and 2.45 ms at `Par::rayon(8)` = 3.56x, where OpenBLAS reaches 5.5x on the same cores for the same factorization; splitting the columns by hand saturates at 3.87x, so the ceiling is the kernel, not the decomposition. Candidates: partition rows for the `C -= V tmp` half, which is row-parallel and has the long dimension (rows >> columns, while the column partition caps at 4), and the DRAM-bound path of the inner matmuls. | TODO | tlinalg-rs#46 | Reproducer and the phase split: tlinalg-benchmark `docs/worklog.md`, 2026-10-12. Drives no code change in tlinalg until the kernel moves. |
| Typed uninitialized overwrite destination | A `MatUninitMut<T>`-style destination that never reads the prior output, accepted by `matmul_with_conj`. | TODO | strided-rs#198 | strided-rs currently returns a Faer-only `Unsupported`. |
| Native SVD / eigh direct output | At nbatch=1, stop writing U and the eigenvectors into an intermediate `Mat` and copying them out; write into the caller's output region instead. | TODO | tlinalg-rs#25 | Preserve output shape, initialization, and failure cleanup. |
| Borrowed reflector columns | Stop copying reflectors out of the compact QR buffer into `basis` scratch; borrow them through a column split. | TODO | tlinalg-rs#17 | Numerics, the tau convention, and the restored diagonal beta stay identical. |
| Thread policy | A path that maps the caller's rayon pool / work model (lanes × item threads) onto `faer::Par` without relying on ambient global state. | TODO | tenferro-rs#2000, strided-rs#6 | Nested parallelism and the thread budget are the subject. |
| Scratch / plan reuse | Hold workspaces, scratch, and plans across calls. | TODO | — | Decide whether to file an issue when work starts. |
| Strided / batched views | Accept general strides and batches directly, without packing. | TODO | — | — |
| c64 planar | Accept a split real/imaginary complex layout directly. | TODO | — | — |

## Ground rules

- One experiment per branch. Keep the diff small enough to rebase onto an
  upstream release tag.
- No unrelated formatting churn or dependency bumps; every extra diff is a
  rebase cost.
- Every experiment carries its own test, run locally by whoever wrote it. The PR
  gate (build) is not a substitute for recording the result.
- Anything intended for upstream is cleaned up and reviewed separately before it
  is proposed.
