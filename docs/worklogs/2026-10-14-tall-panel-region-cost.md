# The tall panel's per-region cost

Ledger row: `Tall panel per-call region cost`. Consumer: `tlinalg-rs#46`'s open item,
the compact-WY cell at `4096 x 256` `b1 c64` eight threads, whose serial part is said to
be the level-2 panel.

## Decisions

- **No code change. The panel's parallelism is bounded by its own dispatch, and the
  dispatch is a work-model question, not a gate.** The level-2 panel of a tall blocked QR
  is a chain of reflectors, each of which depends on the previous one: batching them into
  a compact WY factor is what the *blocked* level already does one level up, and the panel
  is what is left after that. What is left to parallelise is inside one reflector -- the
  fork's rank-one arm already falls through to the generic, column-split path above
  `8192` elements (`fa22a69`) -- and the cost of *asking* for that split is a fresh rayon
  region per reflector, with the pool parked in between because
  `make_householder_in_place` and the column norm are serial. Reusing one region across
  the chain, or batching the reflector applications, needs a caller-owned work model
  rather than `Par::rayon` per call. That is the ledger's `Thread policy` row
  (`tenferro-rs#2000`, `strided-rs#6`), and this record says so with the measurements
  rather than half-building it.
- **The panel does not reach the threads, and it is not four threads that stops it.** The
  unblocked `4096 x 16` QR -- 16 reflectors, each a `(4096 - j) x (16 - j)` rank-one
  update, which is the level-2 panel shape -- measures 0.1957 ms at one thread, 0.1788 at
  two, 0.1731 at four and 0.1968 at eight. The whole gain is 1.13x, and it is already
  three quarters spent at two threads. The eight-thread column is *worse* than one
  thread. So the premise that the panel "stops scaling at four threads" is optimistic:
  on these four cores it barely starts.
- **The per-call region cost, measured three ways.** Same driver, `f64`, four-thread pool
  on CPUs 0-3, 300 repetitions of one `64 x 64 x 64` product, best of three calls,
  smallest over five interleaved repetitions, µs per repetition:
  - the product sequential: 6.44
  - the product threaded, back to back with the next: 8.55 -- the region costs 2.11 µs
    *more* than the sequential call for the same work, and buys nothing, because `64^3`
    is below any useful split
  - the product threaded, separated from the next region by a second (sequential)
    product: 16.71 for the pair, so 10.27 µs for the threaded half, against 6.44 for the
    sequential one -- a serial gap of one product turns 2.11 µs of overhead into 3.83 µs.

  Fifteen reflectors per panel is therefore 32-57 µs of region dispatch, against the
  173 µs the panel takes. It is a fifth to a third of the panel and it is not the whole
  story: even with no dispatch at all the panel would not scale far, because each
  reflector is one `4096 x 15` pass over 0.5 MB and the four cores share one memory
  system.
- **Recorded, not taken: the rank-one arm's `8192`-element floor is below the crossover
  for this shape class.** The same unblocked `m x 16` QR at one and four threads (ms,
  four-thread pool, CPUs 0-3): `1024` 0.0535 -> 0.0719 (34% *slower* threaded), `2048`
  0.1004 -> 0.1171 (17% slower), `3072` 0.1483 -> 0.1423 (4% faster), `4096` 0.1958 ->
  0.1733 (11% faster). The rank-one arm is active from `8192` elements, i.e. `m` about
  546 at this width, so it is threading throughout the losing range. Whether the loss is
  the region or the arm's column split is not separated here, and the arm's floor is
  load-bearing where it was measured (`fa22a69`, the ledger's threaded-rank-one row), so
  it is left alone.

## Verification conclusions and constraints

- Establishes: inside this fork the level-2 panel's problem is not a threshold. The
  threads are asked for on every reflector and the answer is 1.13x; the cost being paid
  for that is 2.1-3.8 µs per region, measured on a product whose arithmetic is small
  enough that the region cannot pay for itself.
- Context: the blocked `4096 x 256` `f64` QR through faer's own API is 12.9014 ms at one
  thread and 8.2981 ms at four (1.55x), so the panel above is 1.3% of that call. The
  campaign's `4096 x 256 b1 c64` cell is `c64` through tlinalg's provider and is not
  reproduced here; its 13.4 ms and its 5 ms panel are not this record's numbers, and the
  phase shares are not comparable across the two.
- Limits: **four physical cores.** Everything is pinned to CPUs 0-3; the campaign holds
  4-11 and their siblings. The eight-thread columns are eight workers on four cores,
  which is why the panel's `8` row is worse than its `1` row, and why the per-region cost
  measured here is a four-core cost. A wider pool can only reduce the wake-up share.
- Limits: the region-cost probe uses `linalg::matmul` on a `64 x 64 x 64` product, not a
  reflector application, so it measures rayon's fork-and-join and wake-up inside faer's
  dispatch and not the arm's own bookkeeping. The panel's own dispatch also allocates its
  column partition, which the probe does not.
- No test and no gate: no code changed.
