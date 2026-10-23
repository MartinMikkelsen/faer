---
name: faer-optimization
description: Optimise a kernel or decomposition in the tensor4all faer fork (t4a-faer) against a vendor yardstick. Use when asked to make an SVD, QR, LU, eigen or matmul path faster, to close a gap to MKL/OpenBLAS, or to investigate why a benchmark cell is slow. Covers the order of work (breakdown first, hypothesis probes, one change per commit, campaign measurement), the lessons that carry over (PERFORMANCE_TIPS.md), the provenance rule for proprietary yardsticks, and where the evidence goes.
---

# Optimising in the faer fork

This is the workflow layer. The knowledge lives in two places this skill points at, and it is
the same for every harness (Claude Code, Codex, opencode, pi, DSH): this file is the only copy,
reached through `skills/faer-optimization/SKILL.md` and the per-harness links beside it.

- **Lessons that carry over:** [`PERFORMANCE_TIPS.md`](../../PERFORMANCE_TIPS.md) at the repo root.
  Read it whole before touching a hot path. Each entry names the worklog with the numbers.
- **Rules:** the shared tensor4all agent rules (`rules/rust/performance.md`,
  `rules/common/performance.md`, `rules/benchmark/measurement.md`), the fork's `AGENTS.md`, and the
  maintainer's decision profile (`shinaoka-advisor/profile/rules.md`, `## performance`).
- **Protocol:** `docs/adoption-protocol.md` here and `faer-benchmark/docs/rules.md` (the campaign's
  acceptance arm, the observer pinning, the lock-file trap).

## Order of work

1. **Where is the time.** Time the stages of the public call you were asked about (the
   `faer-benchmark/probes/` drivers are the pattern: `svd-phases` splits `svd_values` into
   bidiagonalisation and the rest; `qr-phases` splits `qr_in_place` by recursion depth). Then put
   temporary timers around the steps of the loop you are about to change, at 1T and 8T, with the
   caller holding the worker group (`faer::spindle::with_lock`) as the harness does. Two design
   reviews did not predict what one such pass showed. Remove the timers before committing.
2. **Hypotheses, each with a probe.** Write them down with the measurement that would decide each
   (a 500-vs-512 run for cache aliasing, an isolated call for overhead, a region microbenchmark for
   synchronisation), run the probes, and keep the one that moved. Probes are untracked diagnostics
   under `faer-benchmark/probes/<name>/`; do not overwrite one that already exists there.
3. **One change per commit**, with its own numbers in the message: the stage it moved, at which
   sizes and budgets, and what did not move. Constants (block width, floor, threshold, back-off
   factor) are scanned, and the scan's table goes next to the constant and into the worklog.
4. **Tests before timing.** `cargo test -p t4a-faer --lib` (all of it, every time: a numerical
   corner shows up in a special-matrix test far from the file you edited) plus one focused test
   that crosses the new code's boundaries (a shape that takes two panels and the tail; a graded
   spectrum).
5. **Campaign measurement.** The acceptance arm in `faer-benchmark/docs/rules.md`: affected
   families at every declared shape, four dtypes, 1T/4T/8T, the MKL arm, paired against the
   same-harness pin baseline, `scripts/candidate_compare.py --cells <families>`. Before the run:
   pin your own process and every other agent off the measured domain, then touch no measured
   repository until it ends. A family you changed without a baseline under the current harness
   gets its baseline recorded first.
6. **Record.** Worklog under `docs/worklogs/YYYY-MM-DD-<slug>.md` (decisions, rejected candidates
   with the number that rejected them, verification, campaign), the `PATCHSET.md` row, and a
   lesson that will outlive the change into `PERFORMANCE_TIPS.md`.

## The yardstick is not a source

The vendor is timed through its public API only. No disassembly, no profiler into its internals,
no vendor source, no constants learned from it. Algorithms come from published papers or
licence-compatible reference implementations, cited in the module header; design choices trace to
the breakdowns in step 1. Say so in the worklog and the ledger row
(`rules/common/performance.md`, *Optimising Against Proprietary Software*).

## When to stop

The predeclared target and guard decide. A change that meets neither is recorded as a dropped
experiment with its measurement, not tuned until it passes. A result inside the recorded spread is
"did not move", not "slightly better".
