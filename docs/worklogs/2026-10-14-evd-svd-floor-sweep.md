# The tridiagonalisation and bidiagonalisation floors, swept

Ledger row: `EVD and SVD factorisation floor`. Extends the DROPPED row
`Bidiagonalisation parallel hint`, whose record asked the question this one answers.
Consumer: the campaign's `eigh` and `svd_values` rows at `128 x 128`, flat from one
thread to eight (1.26 -> 1.24 and 1.41 -> 1.43 ms) where MKL's `?syevd` and `?gesdd` do
thread that size (1.08 -> 0.586 and 0.704 -> 0.678).

## Decisions

- **No code change; the experiment is dropped.** `tridiag_in_place` and
  `bidiag_in_place` both latch to `Par::Seq` for the rest of the factorisation once the
  trailing area drops below `192 * 256`, and at `128 x 128` that latch closes at the
  first column (`127^2 / 2 = 8064` and `127^2 = 16129`). The question was whether
  threading *that size* pays on this fork. It does not, and it does not by a wide
  margin: forcing the gate open costs 1.5x to 4x, so the floor is not merely harmless
  but load-bearing, and the vendors are ahead at `128 x 128` for a reason this
  implementation cannot copy by threading harder.
- **Measured with the gate forced open and forced closed, not by reading the code.** The
  threshold is a parameter (`TridiagParams::par_threshold`, `BidiagParams::par_threshold`),
  so the floor was moved without touching the library: `0` is the gate open, `usize::MAX`
  is the latch closed at `k = 0` (a fully sequential call), and the default is the floor
  as shipped. A fully sequential call is also the one-thread reference, which is why the
  `max` column below carries both meanings.
- **Rejected: removing or lowering the floor.** Every value below the default in the
  sweep is worse than the default at every shape, and `0` is much worse. The floor is
  not closing a window that would otherwise pay.
- **Rejected: raising the floor, which is the one direction the sweep left open.** See
  the lead at the end of the next section; it is real at `256 x 256` and noise
  elsewhere, so it is recorded rather than taken.
- **The two routines' floors stay where they are, and stay equal.** They are separate
  constants on separate parameter types, so they could move independently; the sweep
  says they should not, at these shapes and on this core count.

## Verification conclusions and constraints

- Isolation driver on `tridiag_in_place` and `bidiag_in_place` directly, `f64`, square
  `n`, four threads asked of a four-thread pool, CPUs 0-3, best of ten calls, smallest
  such value over six interleaved repetitions, ms (`512` and `1024` use best of four to
  six over four, their calls being 3-60 ms). Columns: gate open (`0`), the shipped
  floor, latch closed at the first column (`max`). `128` and below is fully sequential in
  the last two columns, so those columns are the same call.

  | n | tridiag open | tridiag shipped | tridiag max | bidiag open | bidiag shipped | bidiag max |
  |---|---|---|---|---|---|---|
  | 96 | 0.2662 | 0.0682 | 0.0677 | 0.3433 | 0.1516 | 0.1515 |
  | 128 | 0.3890 | 0.1266 | 0.1264 | 0.6024 | 0.3098 | 0.3115 |
  | 160 | 0.5368 | 0.2221 | 0.2213 | 0.9004 | 0.5069 | 0.5096 |
  | 192 | 0.6927 | 0.3393 | 0.3418 | 1.3596 | 0.8145 | 0.8169 |
  | 256 | 1.0405 | 0.7027 | 0.6973 | 3.0776 | 2.3277 | 2.1858 |
  | 512 | 3.3200 | 3.0730 | 4.4959 | 13.7489 | 13.0142 | 16.0337 |
  | 1024 | 15.7921 | 15.7164 | 33.8030 | 61.9936 | 61.2500 | 105.7106 |

  At `96` to `256` the shipped column equals the fully sequential one to within 1%: the
  floor costs nothing there, and it is the only thing standing between the implementation
  and the 1.5-4x the open column pays. At `512` and `1024` the floor is the best of the
  three and the fully sequential call is 1.5-1.7x worse, so the window it leaves open is
  worth having.
- The lower floors in the sweep gain nothing. `8192`, `16384` and `32768` against the
  shipped `49152`, same driver, `f64`, four threads (ms): tridiag `160` 0.2696 / 0.2224 /
  0.2208 against 0.2221; `192` 0.4198 / 0.3558 / 0.3413 against 0.3393; `256` 0.7613 /
  0.7042 / 0.6925 against 0.7027; bidiag `160` 0.6896 / 0.6078 / 0.5086 against 0.5069;
  `192` 1.1248 / 1.0282 / 0.8455 against 0.8145; `256` 2.7997 / 2.6483 / 2.4517 against
  2.3277. The only cell that beats the shipped floor is `tridiag 256` at `32768`
  (0.6925 against 0.7027, 1.5%), against a `bidiag` counterpart moving the other way.
- **Eight threads on four cores is the same measurement.** Repeating the `128` and `256`
  rows with eight workers asked of an eight-thread pool on the same four cores changes
  the shipped column by under 1% for both routines (tridiag `128` 0.1266 -> 0.1298,
  `256` 0.7027 -> 0.7102; bidiag `128` 0.3098 -> 0.3136, `256` 2.3277 -> 2.3300), and the
  open column stays 2.6-4.6x worse. Unlike the partial-pivoting LU, these two do not
  degrade when the pool is larger than the core count.
- Limits: **four physical cores.** Every number is pinned to CPUs 0-3, so this measures
  the four-thread crossover and not the eight-core one the campaign's page is drawn on.
  More idle cores move such a crossover *down* -- threading pays earlier -- which is the
  direction that argues against the floor being too high, and it is the direction this
  driver cannot reach.
- **A measured lead, not taken: `256 x 256` `f64` bidiagonalisation.** A floor above the
  first column's area (`256 * 256 = 65536`, and `100000`, and `max`) beats the shipped
  floor there by 5.6% (`2.3092 -> 2.1805`), and `65536` also beats it at `512`
  (`12.9033 -> 12.8248`, 0.6%). It is unchanged at `96` to `192`, already fully latched,
  and at `768` and `1024` it is within 0.5% either way (`27.1957 -> 27.0848`;
  `61.2922 -> 61.6008`, the one that moves the other way). So the move is a 5.6% win at
  one shape and noise at five, which is not a constant this sweep pins; it is left for a
  measurement on eight idle cores, where the optimum is expected to move the other way.
  The same sweep on `tridiag 256` gives `0.6938 / 0.7013 / 0.6836` for
  shipped / `65536` / `max`, i.e. 1.5% for the fully sequential call only.
