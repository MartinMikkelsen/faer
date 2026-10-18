# Small-matrix GEMM: the thread budget the kernel never used

The issue's third item: find where a small matrix's parallel fixed cost exceeds the
work, and check the boundary on tall, wide and short-contraction shapes rather than
on squares alone. This record has the probe measurement, the one-line condition it
produced, and the campaign pair that adopted it.

## What was measured first (probe at fork `0627828`, best and median of 21 calls)

`faer::linalg::matmul::matmul` into a prepared destination, CPUs `4` / `4-7` /
`4-11` for 1T / 4T / 8T, observer off that L3 domain, us:

| dtype | shape | 1T | 4T best (median) | 8T best (median) |
| --- | --- | --- | --- | --- |
| f64 | `32^3` | 2.31 | 9.23 (10.87) | 12.02 (16.93) |
| f64 | `64^3` | 17.12 | 17.50 (24.44) | 18.31 (21.10) |
| f64 | `96^3` | 56.59 | 31.56 (60.74) | 28.19 (30.68) |
| f64 | `128^3` | 133.4 | 52.10 | 41.04 |
| f64 | `192^3` | 449.7 | 125.6 | 98.46 |
| c64 | `32^3` | 8.75 | 8.75 | 8.77 |
| c64 | `64^3` | 67.49 | 35.47 (39.01) | 37.23 (46.30) |
| c64 | `96^3` | 226.1 | 83.94 (90.51) | 48.39 (54.66) |

The f64 `32^3` call is five times slower at 4T/8T than at 1T, and f64 `64^3` keeps
its best time but loses 1.2-1.4x in the median, while `96^3` is already twice as
fast at eight threads. c64 `32^3` is unaffected and c64 `64^3` is twice as fast, so
the boundary is not a MAC count.

## The mechanism

`matmul_imp`'s native path hands `private_gemm_x86::gemm` `par.degree()` as soon as
`M * N * K >= 4096 * size_of::<T>()` (32 ki MACs for f64, 64 ki for c64). The
generic `gemm` crate's kernels refuse to thread below `48 * 48 * 256` MACs
(`gemm::get_threading_threshold()`), but `private_gemm_x86` has no such floor: it
runs `spindle::for_each_raw(self.n_threads, ..)` whenever it is handed more than
one thread. So f64 `32^3` (65 ki flops) was split eight ways and paid a region
(~10 us) for 2 us of work, and f64 `64^3` (524 ki flops) sat right on the boundary
where the region costs as much as it buys.

## The change

`faer/src/linalg/matmul/mod.rs`: the budget is handed over only when the work
clears the `gemm` crate's own threading floor expressed in flops
(`2 * gemm::get_threading_threshold()`), and the work is counted in flops (`2` per
real MAC, `8` per complex one) instead of the dtype's byte width. That is the
condition the measurements ask for: f64 `64^3` is half the floor and stays
sequential, c64 `64^3` is twice it and keeps its two-times gain. The floor is a
property of the *work*, not of a shape or a square size: the tall and wide cells
(`16x4096x240`, `4096x16x240`, `64x64x4096`) are 15-17 Mflops and keep their budget
whatever their aspect. `PAR_THRESHOLD_MNK` is gone; `NANO_GEMM_THRESHOLD` still
routes `16^3` and below to the planless kernel.

After the change, same probe: f64 `32^3` 2.30 us at 4T and 8T, `64^3` 16.98 / 17.01
(median 17.00 / 17.03), `96^3` 37.0 / 27.5 best (66.3 / 32.3 median, inside the
baseline's own spread), `128^3` inside noise; c64 unchanged (`64^3` 34.2 / 35.4
best, 44.5 / 40.7 median; `96^3` 84.4 / 49.3). Test:
`thread_budget_does_not_change_the_product` compares the sequential and eight-thread
products at `32^3`, `64^3`, a non-square small shape and `128^3`, so both sides of
the floor are covered. `cargo test -p t4a-faer --lib`: 210 passed, 2 ignored.

## Campaign pair (candidate `2000dd9` against baseline `0627828`)

Cells in `20261018T014801Z-baseline-gemm-guard-mkl` (fork `0627828`) and
`20261018T015622Z-candidate-gemm-thread-floor-mkl` (fork `2000dd9`), MKL arm,
rotation-averaged ratio candidate/baseline (below 1 is faster):

| dtype | shape | C 4T | C 8T | F 4T | F 8T |
| --- | --- | --- | --- | --- | --- |
| f64 | `32^3` | **0.233** | **0.178** | **0.222** | **0.163** |
| f64 | `64^3` | **0.702** | **0.869** | **0.766** | **0.824** |
| f64 | `96^3` | 1.119 | 0.996 | 1.072 | 1.046 |
| f64 | `128^3` | 0.932 | **0.879** | 1.054 | 0.901 |
| f64 | `192^3` | 1.019 | 1.121 | 0.964 | 0.925 |
| c64 | `32^3` | 0.997 | 0.992 | 0.999 | 0.992 |
| c64 | `64^3` | 0.992 | 0.947 | 1.043 | 1.003 |
| c64 | `96^3` | 0.976 | 0.980 | 1.001 | 1.005 |
| c64 | `128^3` | 1.005 | 0.924 | 0.995 | 1.041 |
| c64 | `192^3` | 0.996 | 1.040 | 0.998 | 1.013 |

The target cells are the two smallest f64 squares: `32^3` is 4.3-5.6x faster and
`64^3` 15-30% at 4T/8T, and f64 `128^3` at eight threads gains 10-12%. The `96^3`
4T cells read 1.07-1.12, but their own per-start ratios are 0.79-1.35 and the
baseline's medians there are flagged `!` (between-start range over 5%); the 8T
`96^3` cells are 0.996 / 1.046. `192^3` is inside its own spread (baseline C 8T
`!`). Every c64 cell is unchanged, which is the point of counting flops: c64 `64^3`
keeps the two-times gain the kernel's own threshold already gave it. The
`faer-upstream` control is 0.94-1.06 except the sub-3 us f64 `32^3` row, which is
timer-dominated at both revisions (its 8T F ratio is 4.33 from a 2.2 us cell, and
the same row at 1T is 1.00).


## What this establishes

The small-shape "parallel fixed cost" the earlier probe attributed to a thread
region is real, and it was not the vendor kernel refusing the work: faer's own
kernel has no floor and threads anything it is handed. The floor belongs to faer,
and expressing it in flops keeps one rule for real and complex dtypes. No vendor
source was adapted; the `gemm` crate's published threshold is reused as the
reference, not a machine-specific constant.
