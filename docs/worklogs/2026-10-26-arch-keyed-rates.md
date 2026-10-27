# Architecture keyed serial rates and Rayon costs

tensor4all-meta#51 follow-up, maintainer approved design. No timing was taken: the values are the
recorded ones.

## What changed

- `utils::cost::HostClass` (`x86_64-avx512`, `x86_64-avx2`, `x86_64-scalar`, `aarch64-apple`,
  `aarch64`, `generic`) with `detect()` (cached in an atomic, `is_x86_feature_detected!` under
  `std`, the build's target features without it) and `name()`. It is a copy of
  `threadteam::HostClass` (threadteam `2024882`); a test (`host_class_and_outside_rows_agree_with_threadteam`)
  compares the detected class, the names and the Rayon outside row of every class when the
  `threadteam` feature is on.
- `Rates` (one `Rate { real, cplx }` per kernel family) with `Rates::for_class`, `rates()`;
  `rate::<family>()` functions replace the `rate::FAMILY` constants (call sites changed
  mechanically), `ns_per_mac::{f64,c64,f32,c32}()` replace the constants. New entries:
  `gemm_single` (f32/c32 measured), `block_apply_mac` (the block apply's multiply-adds).
- The Rayon outside cost of a build without `threadteam` is `outside_cost_of(class, width)`, the
  threadteam model (linear between widths 2, 4, 6, flat above; the former proportional growth
  above 6 is gone, as in threadteam) with the class rows.
- The threadteam pin moves to `2024882b4ee759873a0b403a9c0a270361ae527a` (feat/dispatch-cost,
  `HostClass`, `DispatchCosts::for_class`; `generic()` and `apple_m5()` no longer exist).

## Values

`x86_64-avx512`: `faer-benchmark` `data/probes/20261026-awake-team-linux/tables-consts-exec3.md`
(1 thread, Zen 5 at 2.0 GHz, boost off), family medians; the region constants from the same
file. Divide and conquer per `m^2` is the median of `ns / n^2` over `raw-exec3/rconsts-{bsvd,tevd}-r{0,1}.csv`
(`m` 64 to 512; the file's `ns_per_unit` is per `m^3`): 152.9 (range 145 to 178) and 97.4 (88 to 102);
size dependent, as on the M5 (48 to 62). Sparse product: median of the 12 group medians per dtype.
Triangular inverse: the group median (falls with size, 0.047 at 64 to 0.0145 at 512 for f64).
Matvec: mean of the column major and row major medians (the M5 value 0.1 / 0.34 is the same mean
within rounding). `NS_PER_SWAP` stays 0.55 (M5, not measured on x86, memory bound).

Rayon outside, us at widths 2, 4, 6: avx512 9.7, 10.1, 8.7; M5 14, 18, 19; generic 40 flat
(threadteam's rows).

`generic` is the per entry minimum of the measured rows (const `Rates::min_of`), so it is the M5
row today: every measured Zen 5 entry is slower than its M5 entry.

## Consequence on this host

The honest Zen 5 rates are about 1.7 to 2.3 times the M5 ones, so floors are lower: a 48 x 48
unpivoted QR on a team of 4 now splits one block apply (the estimate reaches 3 times the width 4
cost, 0.7 us); `small_qr_on_a_team_is_the_sequential_one` uses 32. Rates and costs of one row must
come from one machine and clock state: boost moves both by about 1.66 times with a stable ratio
(`tables-*-exec3-boost.md`).

## Verified

`cargo test -p t4a-faer --features threadteam` (303 lib tests) and `cargo test -p t4a-faer`
(274) pass; `cargo check --no-default-features` and with `std` only build.
