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
| Released | `t4a-faer-traits 0.24.0` and `t4a-faer 0.24.4`, published from commit `3a6a9d6`. The published `src/**` is byte-identical to that commit, so `3a6a9d6` is the revision a consumer pins. A later commit needs a new version. `t4a-faer 0.24.5` carries the unpivoted-QR fix for tlinalg-rs#28; it is `d16929bc` plus its version bump, so its `src/**` matches `d16929bc`, the revision `tlinalg` resolves once it moves off a git pin. |
| LICENSE | A `LICENSE` copy lives in each crate directory so the published archive ships the MIT text (`license-file` next to `license` makes Cargo warn). |

## Experiments

| Experiment | What | Status | Issues | Notes |
| --- | --- | --- | --- | --- |
| Unpivoted QR column skip | The skip lived only in `qr_in_place_unblocked`; `qr_in_place_blocked` has no threshold of its own and recurses into that core, so removing it there covers the tall and wide paths at once. The reflector is applied and the row advances for every nonzero column, matching LAPACK; the tall/wide `svd` path goes through it too. | KEEP | tlinalg-rs#28, tensor4all-rs#836 | Adopted. Tests: `test_near_dependent_columns_are_not_skipped` (tall `m x 3`, m=100/1000/8192) and `test_near_dependent_rows_are_not_skipped` (its transpose, wide `3 x n`, n=100/1000/8192), both at `<=1e-13`; measured 1e-16 to 5e-15. The wide fixture was already accurate before the change: at m=3 the removed threshold `16 (m-k) eps ||column||` is ~1e-14 and never tripped it (a wide skip's relative reconstruction error is ~`16 eps sqrt(m) sqrt(m/n)`), so the wide path needed no separate treatment — the same core change covers it. tlinalg's reported wide `3.33e-7` is a reconstruction-indexing bug in that test (entry `(i,j)` written to flat `j+3i` while the factor is column-major `i+3j`); faer's factors for that matrix are accurate to ~4e-16. Existing columns are unchanged; formerly skipped nonzero columns now pay reflector/update flops (the threshold norm bookkeeping is removed). No existing benchmark suite was run. Release smoke, 3-run mean, before→after: 100000×8 2.984→2.986 ms; 1000×1000 19.793→19.921 ms. |
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
