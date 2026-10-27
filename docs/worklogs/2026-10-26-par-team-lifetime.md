# `Par<'team>`: parallel authority carried by the value

## Decisions

- **`Par<'team>` is non-`Copy` for every variant.** `Seq` and `Rayon` carry no exclusive
  resource, but a `Copy` `Seq` would let code duplicate a `par` that the type system could not
  reject once the same code receives a `Team`. A `Copy` view that can run work (`ParRef`) was
  rejected for the same reason: two copies handed to two concurrent children each own the whole
  team. Sequential phases lend (`par.reborrow()`), concurrent children split.
- **Sizing is separate from running.** Workspace queries take `ParShape` (`Copy`, no authority)
  instead of `&Par`, so an engine can size a workspace from a width before any team exists.
  `Par::shape()` converts; `ParShape::{seq, rayon, team}` construct from a width.
- **The thread-local team of the prototype is gone.** `Par::Team(TeamExec<'team>)` holds the team
  itself. `join_raw`, `join_sized`, `split`, `for_each_with_par` and the GEMM tiling hand each
  concurrent part its own sub-team as a `Par` or consume the team, so no `TeamExec<'static>`
  pointer cast or installed pointer remains.
- **No global parallelism.** A team cannot be a process wide setting, so the setting is removed; the high level methods take `par` (record: `2026-10-26-remove-global-parallelism.md`).
- **Every `match par` names all variants.** The `_ =>` arms that treated a `Team` as Rayon are
  explicit; no path starts a Rayon job with a `Team` in hand.
- **Signatures keep `par: Par`.** The lifetime is elided (edition 2021), so about 470 signatures
  did not change; the edit is in the bodies. About 750 uses became `par.reborrow()` where another
  use follows, by following the borrow checker's reported move sites; the rest is hand edits:
  three closures that captured `par` take it as an argument, `par.shape()` and
  `parallelism_degree(&par)` are evaluated before a lend in the same call, the examples build a
  `Par` per use.
- **threadteam stays a pinned dependency** (`6e87862` on `feat/team`, with no API change from the previous pin);
  nothing in threadteam was edited.

## Verification conclusions and constraints

- Behaviour with a team matches the sequential results for GEMM, LU, QR and Cholesky on teams of
  one, three, five and six; the width accounting of `join_raw` and `join_sized` is exact, and the
  concurrent peak never exceeds the team. Full list in `docs/par-team-migration.md`.
- The Rayon and sequential paths keep their arithmetic and their widths, with one exception: the
  LU swap stage and the two triangular solves split their budget between the two sides as
  described in `2026-10-26-join-raw-parent-par-and-sparse-team-route.md`.
- Timing is a rough check on the M5 host only (`faer-benchmark` smoke families); no acceptance
  grid was run, and the three pre-existing `strict_triangle` failures are unchanged.
