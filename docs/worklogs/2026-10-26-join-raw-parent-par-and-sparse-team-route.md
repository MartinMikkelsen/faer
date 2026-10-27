# Two parallelism defects found by the non-`Copy` `Par`

Both were found while making `Par` non-`Copy` (`docs/par-team-migration.md`): the borrow checker
refused a `par` used by two concurrent closures, and an audit of every `match par` found an arm
that sent a `Team` to Rayon. Base revision for the line numbers: `752a0a6` (the fork's `main`).

## Decisions

- **Triangular solves: both `join_raw` children used the parent `par`.**
  `linalg/triangular_solve.rs:444` (`solve_unit_lower_triangular_in_place_imp`) and `:525`
  (`solve_lower_triangular_in_place_imp`) split the right-hand side into two column blocks and
  ran them in `join_raw(|_| solve(.., par), |_| solve(.., par), par)`: `join_raw` hands each
  half its own width, the closures ignore it, and both halves run with the full parent `par`.
  With `Par::Rayon(n)` the two concurrent halves each ask for `n` threads, so the budget is used
  twice at every level of the recursion; with a team it would be two owners of one team. Fixed
  in `8fe65b3`: the closures take and use their child `par`.
- **LU swap stage: the children ignored the `par` they were handed.**
  `linalg/lu/partial_pivoting/factor.rs:202` ran `join_raw(|_| for_each(left_threads, ..),
  |_| for_each(right_threads, ..), par)`. The two widths are computed from the column counts and
  add up to `n`, so the Rayon budget was not exceeded, but the sides did not run on the widths
  `join_raw` gave them (an even `n - n/2`, `n/2` split) and the child `par` was unused, which a
  non-`Copy` `Par` cannot express. Fixed in `8a6b555`: `thread::join_sized` splits `par` into
  the proportional widths (clamped to leave one thread to each side) and each side runs
  `for_each` on its child; a side with no columns leaves the whole `par` to the other.
- **Sparse numeric matmul sent a `Team` to the global Rayon pool.**
  On `exp/team-par` (`e88599a`) `sparse/linalg/matmul.rs:170` was `_ => { .. (0..n).into_par_iter()
  .for_each(job) }` after the prototype widened the `Par::Rayon` arm to catch the new `Team`
  variant. A `Team` there ran on the global Rayon pool with `par.degree()` jobs, outside the
  team's reservation. Fixed in `8a6b555`: the arm calls `thread::for_each`, which routes a
  `Team` through the team. The prototype arm is the only one of fifteen such `_ =>` arms that
  reached raw Rayon; the other fourteen already went through `thread::for_each`.
- **Origin.** The two triangular-solve sites and the LU swap stage are upstream faer's code,
  identical at the fork's upstream base `0ecd846` (upstream `linalg/triangular_solve.rs:435`,
  `:516`; upstream `linalg/lu/partial_pivoting/factor.rs:163`), so upstream is affected. The
  sparse matmul arm exists only on this fork's `exp/team-par` branch (upstream has no `Team`
  variant), so upstream is not affected by it. Upstream feedback is deferred: nothing was
  reported upstream, and this record is the trace.

## Verification conclusions and constraints

- Sparse matmul: `utils::team::tests::sparse_matmul_on_a_team_runs_on_the_team` counts the
  team's dispatches and requires a four-way fork during the numeric phase; with the raw Rayon arm
  restored it fails, with the fix it passes, and the product equals the sequential one.
- LU: `utils::team::tests::lu_swap_stage_splits_the_team_it_is_handed` checks the result against
  the sequential factorization and that the swap stage split the team in two;
  `utils::thread::tests::join_sized_children_stay_within_their_share` (Rayon) and
  `utils::team::tests::join_sized_on_a_team_hands_out_exactly_the_team` check that the widths the
  two sides see add up to the budget and that their concurrent peak does not exceed it.
- The double use itself is not observable as a count (a doubled Rayon budget only oversubscribes
  threads); it is excluded by the type: the compile-fail doctests on `Par` reject a `join_raw`
  whose two closures both use the parent.
- No timing is claimed. A triangular solve that took both halves at full width now takes `n -
  n/2` and `n/2`; whether that moves any solve's time was not measured.
