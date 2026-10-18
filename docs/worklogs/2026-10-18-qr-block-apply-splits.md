# QR factor: where the eight-thread time goes, and the split sizes that stop it

The issue's second item: check the panel factorization, the compact-WY coefficient
build and the trailing update, plus the number of synchronisations, for the QR
factor. This record has the phase split at the pinned revision, the two changes the
split pointed at, and the campaign measurements that adopted the first and rejected
the second.

## The phase split (probe at fork `5578ccc`, f64, column-major, best of 7)

The probe times `qr_in_place` and accumulates, from inside the blocked recursion,
the panel (`qr_in_place_unblocked`), the compact-WY upgrade
(`upgrade_householder_factor`) and the trailing block apply
(`apply_block_householder_transpose_on_the_left_in_place_with_conj`), with the
number of calls of each. CPUs `4` / `4-7` / `4-11`; the observer was off that L3
domain. Times in ms; the phase times are summed over the recursion, so they are the
call's real serial contribution when the phase is sequential.

| n | T | total | panel (calls) | upgrade (calls) | apply (calls) |
| --- | --- | --- | --- | --- | --- |
| 256 | 1 | 1.42 | 0.08 (244) | 0.59 (244) | 0.70 (125) |
| 256 | 8 | 1.56 | 0.09 | 0.66 | 0.77 |
| 512 | 1 | 7.94 | 0.22 (496) | 0.84 (496) | 6.79 (255) |
| 512 | 8 | 5.29 | 0.25 | 1.75 | 3.19 |
| 1024 | 4 | 21.67 | 1.00 (667) | 4.61 (667) | 15.88 (343) |
| 1024 | 8 | 20.10 | 1.02 | 5.50 | 13.40 |

The panel is 3-5% of the call and already sequential; the trailing update is
60-85%; the upgrade is 10-25% but *grows* with the thread budget.

Two measurements inside the same probe isolated the two costs:

- Forcing the trailing apply sequential (eight threads), `512^2`: total 9.23 ms, so
  the threaded apply (3.19 ms summed) is 2.1x its sequential time and gains nothing
  from 4T to 8T. The column split was capped at four tasks, and with
  `threads >= 2 * n_tasks` each task then nested `threads / 4 = 2` inner threads.
  Raising the task cap to the budget made `512^2` 5.27 -> 4.74 ms and `1024^2`
  20.5 -> 16.9 ms. The isolated apply shows where it comes from: a `512 x 48` basis
  against a 464-column target (the shape a blocked QR's trailing update takes) is
  0.318 ms capped and 0.215 ms uncapped at eight threads; a tall `4096 x 16` basis
  against a 240-column target, which takes the row split instead, is unchanged
  (0.397 vs 0.405).
- Forcing the upgrade sequential was faster than leaving it threaded at every size
  (`256^2`: 1.56 -> 1.31 ms; `512^2`: 5.29 -> 4.67; `1024^2`: 20.1 -> 18.9 at eight
  threads). Its region gate was an unscaled `8 * 1024` work floor, and a blocked QR
  calls it once per block per recursion level — 667 times at `1024^2`, most at
  `n = 6..24` — so the call opened regions for reductions of a few microseconds.
  The isolated kernel sweep (flat `V^H V`, threaded against sequential, eight
  threads) gives the shape of the floor: `4096 x 16` 0.078 -> 0.032 ms, `4096 x 48`
  0.49 -> 0.13, `1024 x 48` 0.123 -> 0.070, `1024 x 16` 0.021 -> 0.019 (break-even),
  `512 x 32` 0.031 -> 0.042, `256 x 64` 0.055 -> 0.068. Scaling the existing floor
  by the thread count (`8 * 1024 * threads`) keeps every reduction whose per-thread
  work is real, including the tall panel's, and leaves the rest serial.

## The change

`faer/src/linalg/householder.rs`: when the rows are not split, the block apply's
column split now takes the whole thread budget (`n_tasks = min(threads, K)`) and
leaves one thread inside; the four-task cap with its nested `threads / 4` inner
threads stays only for the row-split path, where that split already owns the
budget. `threads` and the row-split predicate moved above `n_tasks` to state the
decision in one place. It is a property of the shape and the budget, not of one
size, host or vendor constant.

Two focused tests extend the existing threaded-vs-sequential pair:
`check_panel_with(256, 32, 224, 8)` and `check_panel_with(512, 48, 464, 8)` exercise
the eight-task column split. `cargo test -p t4a-faer --lib` (f64 and c64, both
paths): 209 passed, 2 ignored.

## Campaign measurement (candidate `0627828` against baseline `5578ccc`)

Same harness (`976df2c`), same host block, same timing policy (250 ms priming, 5
calls, three process sets, boost off, observer off the measured L3, whole-domain
idle guard), MKL arm, f64, C and F, 1T/4T/8T. Cells in the baseline run
`20261018T000948Z-baseline-v2-diagnostic-mkl` and the candidate run
`20261018T011734Z-candidate-qr-apply-split-only-mkl`; the number is the
rotation-averaged ratio candidate/baseline (below 1 is faster).

| family | n | C 4T | C 8T | F 4T | F 8T | 1T (C / F) |
| --- | --- | --- | --- | --- | --- | --- |
| `qr_factor` | 256 | 0.981 | **0.916** | 1.000 | **0.910** | 0.987 / 0.994 |
| `qr_factor` | 512 | 0.990 | **0.881** | 0.996 | **0.879** | 0.992 / 0.994 |
| `qr` | 256 | 1.002 | **0.862** | 0.990 | **0.849** | 0.995 / 0.996 |
| `qr` | 512 | 0.934 | **0.862** | 0.935 | **0.856** | 1.002 / 0.997 |

The improvement is 12-15% at 256^2 and 512^2 at eight threads, inside noise at
four and zero at one thread, where the split sizes do not change (`n_tasks` was
already `min(threads, K, 4)`). The guard families in the same runs are unchanged or
better: `svd_values` (does not call the apply: U and V are absent) 0.992-1.002,
`svd_thin` (calls it to form U and V) 0.965-1.008. The `faer-upstream` row, which is
the machine-state control, reproduced the baseline (median 1.00, worst 1.045) and the
MKL row reproduced it as well (0.97-1.04).

## The tall shapes (the row split's own targets)

The change leaves the row split's task cap and inner parallelism alone, and the
tall shapes are the guard, not the target. Measured on a separate pair of raw runs (`--no-publish`, so the published mid-size
page for each revision is not replaced; `20261018T005438Z-baseline-tall-guard-mkl`
and `20261018T013320Z-candidate-split-only-tall-guard-mkl`):
`qr_factor`/`qr`/`householder_apply`/`upgrade_factor` at `4096x256` and `1024x64`,
4T/8T, f64, both orders, same settings.

| family | shape | C 4T | C 8T | F 4T | F 8T |
| --- | --- | --- | --- | --- | --- |
| `qr_factor` | `4096x256` | 0.990 | 0.994 | 0.984 | 0.996 |
| `qr` | `4096x256` | 0.850 | 0.988 | 0.850 | 1.010 |
| `householder_apply` block | `4096x256` | 1.009 | 0.986 | 1.004 | 0.976 |
| `householder_apply` sequence | `4096x256` | 1.021 | 1.003 | 1.000 | 0.993 |
| `qr_factor` | `1024x64` | 1.008 | 0.966 | 1.016 | 0.989 |
| `qr` | `1024x64` | 0.999 | 0.986 | 1.000 | 0.985 |
| `householder_apply` block | `1024x64` | 0.985 | 0.991 | 0.963 | 1.028 |
| `householder_apply` sequence | `1024x64` | 1.041 | 1.021 | 0.992 | 1.008 |

Every tall cell is inside the recorded between-start spread (a few per cent). The
`qr` `4096x256` 4T cell reads 0.850 in both orders, consistently across the three
starts, but the baseline's own 4T value for that cell is flagged `!` (its
between-start range is over 5%), and the change does not alter the 4T split sizes,
so it is recorded as a baseline-side outlier rather than a gain. The
`upgrade_factor` family is declared by the suite but the harness emits it as
`skipped`, so the tall guard covers the apply directly and the upgrade through
`qr_factor`.

## The upgrade floor that was measured and dropped

The second half of the same probe was an upgrade floor problem: its region gate was
an unscaled `8 * 1024` work floor, and a blocked QR calls the upgrade once per block
per recursion level (667 times at `1024^2`, most at `n = 6..24`), so it opened
regions for reductions of a few microseconds. Scaling the floor by the thread count
(`8 * 1024 * threads`) fixed that and was faster on the whole call (`256^2`
1.41 -> 1.28 ms, `512^2` 4.69 -> 4.41, `1024^2` 17.0 -> 16.1 at eight threads), and
it kept the target: the same campaign grid gave `qr_factor` 512^2 0.820 / 0.811 and
`qr` 0.841 / 0.825 at eight threads. It was **rejected** on its guard set: the
`householder_apply` `1024x64` cell at eight threads went from 0.156-0.164 ms to
0.180-0.189 ms (+18%, three consistent starts, both the block and the sequence
form), and a build carrying only the split change reproduces the baseline on that
cell (0.154-0.159 ms), so the regression belongs to the floor change.

The timed call there executes the same code either way -- `1024x64` takes the row
split at both revisions -- and the only difference is what the *untimed* `prepare`
does before it: the prepare's `qr_in_place` opens far fewer regions once the
upgrade floor is per thread, so the pool is parked when the timed apply starts and
the apply pays the wake-up. That is a boundary effect of the harness's
prepare/run split, not a kernel regression, and it is not a reason to keep a
candidate whose guard cell moves outside the recorded spread; the smaller change
keeps 12-15% of the 8T gain and leaves every guard cell where it was.

## What is left

The apply is still the largest term (60-80%), and at eight threads the blocked QR
still opens one region per block apply (~255 at 512^2, ~343 at 1024^2) and one per
threaded upgrade. Aggregating several blocks' applies into fewer regions, or moving
the whole trailing update to one `gemm`-shaped apply per block row (the LAPACK
`DGEQRF`/`DLARFB` blocking) is the next step, not this one: it changes the
factorization's structure, not its split sizes.

No OpenBLAS, MKL or LAPACK source is used or adapted. The vendor timings in the
campaign pages are the target, and BLIS's multithreading notes
(https://github.com/flame/blis/blob/master/docs/Multithreading.md) were read as
background for splitting the reduction along the dimension that is deep enough,
which is what the existing row split already does.
