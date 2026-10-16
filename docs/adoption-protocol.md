# Verification and adoption protocol (reproducible)

This repository is the measured library. It deliberately has **no benchmark harness of its own**:
`benchmarks/` is forbidden here (see `AGENTS.md`), and the campaign that measures this crate lives in
[`faer-benchmark`](https://github.com/tensor4all/faer-benchmark), which owns the harness, the declarations
and the published pages. This file is the short version a change author needs.

## Two jobs, two machines

| job | where it runs | what it covers | published |
| --- | --- | --- | --- |
| **Acceptance** - may this change be merged? | the machine the change is developed on | the change's blast radius (below), about an hour, once per code change | no: the evidence lives in the pull request, the campaign worklog and `PATCHSET.md` |
| **Public dataset** - the benchmark pages | a dedicated quiet measurement host, re-recorded periodically and published automatically | the whole declared suite | yes |

Both use the same frozen contract, so an acceptance run and a published page mean the same thing:
`--prime-ms 250 --reps 5 --aa 3 --statistic median`, CPU boost off, the observing side pinned off the
measured L3 domains, an idle guard that watches every CPU sharing the measured cache, and
`PINNED_RETRIES=8`. The full text is `faer-benchmark/docs/protocol-v2-baseline.md`; the page-level rules
are `faer-benchmark/docs/timing-policy.md` and `docs/provisional-and-clean-datasets.md`.

## The acceptance grid comes from the change's blast radius

Declare, *before* running anything, which families the change can reach by reading the changed paths:

| changed path | families to measure |
| --- | --- |
| `faer/src/linalg/qr/no_pivoting/**` | `qr_factor`, `qr` |
| `faer/src/linalg/lu/partial_pivoting/**` | `lu`, `full_piv_lu`, `solve` |
| `faer/src/linalg/evd/**`, `.../svd/**`, `.../cholesky/**` | the matching decomposition family |
| a shared helper such as `faer/src/mat/matref.rs` | every family that copies or owns a matrix |

Then measure the affected families at every declared shape, dtype and order at 4T and 8T, the target
cells at 1T, and two or three cells of untouched families as a collateral control - one hour for a
change like the row-major QR/LU work, more only when a shared helper is involved.

## Measure once, merge without measuring again

1. Measure the candidate at the revision that will be merged (branch head, clean tree).
2. Merge it. The merge must leave the measured paths' tree hashes unchanged.

   ```sh
   cd ../faer-benchmark
   python3 scripts/candidate_compare.py --baseline <recorded baseline run> --candidate <candidate run> \
       --checkout ../faer --merged-commit <merge commit>
   ```

   The command refuses when the merge changed a measured path (`faer/src`, `faer/Cargo.toml`,
   `faer-traits/**`, `Cargo.lock`), so the same code is never measured before and after the merge. A
   merge that does change one of them invalidates only the cells that depend on it; re-measure those
   with the same blast-radius rule, never the whole suite.
3. Record the merge commit next to the measured commit in `PATCHSET.md` and in the campaign worklog.

## What must be true before a merge

- `cargo test` in this repository passes (the crate's own suite), plus the regression test the change
  adds.
- The campaign's `fbench verify` passes for the affected populations and budgets; a mismatch or an
  undeclared skip fails acceptance.
- The target cells improve beyond the between-start spread the baseline recorded, and no measured
  guard cell regresses beyond it. The campaign's `candidate_compare.py` prints exactly that.
- The change carries its own test and its own record; the fork's CI (a build) is not a substitute for a
  recorded result.

Upstream (`sarah-quinones/faer`) is a separate matter: nothing here is proposed upstream automatically.
