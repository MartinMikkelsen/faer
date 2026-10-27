# AGENTS.md

Guidance for agents working in this repository.

This is a temporary tensor4all fork of
[faer](https://codeberg.org/sarah-quinones/faer). Read the shared tensor4all
agent rules before acting:
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`
(offline fallback: `../tensor4all-agent-rules/rules/index.md`). Load only the
rule files the task needs.

Verification and adoption of a change: [docs/adoption-protocol.md](docs/adoption-protocol.md) — the
blast-radius acceptance grid, the measure-once rule and the exact commands.

Read [the fork's review rules](docs/review-rules.md) before opening an issue,
preparing a change, or reviewing a PR. They define the faer-specific naming,
file-layout and test-placement overrides to the shared rules.


## Repository-specific rules

* This fork is an AI-driven experiment bench. Experiments are allowed to be
  disruptive. API compatibility with upstream faer is not promised.
* `PATCHSET.md` is the experiment ledger. Record what each experiment does, its
  status, and the tensor4all issue it serves. Keep it current in the same change
  that adds or drops an experiment.
* Keep the rename mechanical so upstream rebases stay cheap:
  * package name `t4a-faer` / `t4a-faer-traits`;
  * `[lib] name = "faer"` / `"faer_traits"`, and dependency keys unchanged, so
    downstream `use faer::...` and `use faer_traits::...` are untouched;
  * `faer-ffi` stays out of the workspace (unused, cbindgen build);
  * each crate directory keeps a `LICENSE` copy so the published archive ships
    the MIT text (`license-file` plus `license` makes Cargo warn).
* Do not carry unrelated formatting churn or dependency bumps. Every extra diff
  is a rebase cost against upstream.
* Upstream remote is Codeberg (`git remote add upstream
  https://codeberg.org/sarah-quinones/faer.git`). The GitHub parent
  `sarah-quinones/faer-rs` is a mirror; do not treat it as the source of truth.
* **Direction of an upstream change is a fact to verify, not to infer.** Content
  that is byte-identical to upstream is upstream's work unless its commit message
  says otherwise; read the messages before writing who wrote what. A tree-hash
  comparison proves a merge changed nothing, not who authored the code.
* Upstream updates arrive as a merge commit from `upstream/main` (fetch with
  `git fetch upstream +refs/heads/main:refs/remotes/upstream/main`), landing on a
  branch that already contains the fork's own work, and the sync records the new
  base revision in the `Upstream base` row of `PATCHSET.md`. If the imported work
  was already cherry-picked here, the sync is ancestry-only: say so and show the
  tree hashes match, rather than presenting the imported fixes as the fork's.
* An imported upstream fix is upstream-to-fork. Its commit message names the
  upstream commit SHAs or Codeberg PR numbers **and the original authors**, and
  carries the upstream tests: `Taken from upstream faer (Codeberg #NNN): <shas>,
  by <authors>`. Never describe such a fix as the fork's own work; feeding a fork
  change back upstream stays under the deferred-feedback rule below.
* Every experiment carries one focused test. Run it locally; the fork CI gate is
  a build, not the full upstream suite (faer's test dependencies are heavy).
  The inherited upstream workflows still run their own heavier jobs.
* Upstream feedback is **deferred at this stage**: do not prepare or open an
  upstream issue or PR and do not write to an upstream tracker, even for a defect
  that is clearly upstream's. Measure it, record it in `docs/worklogs/` with the
  upstream revision that was measured, and leave the decision to the maintainer.
  Explicit permission is required before anything more than a record here.

## Work Logs And Review Intent

Follow the shared [Work Logs And Design
Records](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/repository.md#work-logs-and-design-records)
policy and the [local format](docs/worklogs/README.md). Keep a record for a
nontrivial multi-phase change, a non-obvious design choice, or a performance
experiment; small fixes and AI assistance alone do not require one. One file per
change theme under `docs/worklogs/`, linked from its PR, and read the linked
record when reviewing the decisions it explains.

Every performance experiment in this fork needs one, because the fork's whole
purpose is to be a measurable delta against upstream. `PATCHSET.md` says what the
fork carries today; the record says why — the isolation, the candidates that were
tried and dropped *with the measurement that dropped them*, and what the result
establishes. It is how the difference from upstream and the course of an
optimisation stay followable by someone who was not in the session, and how a
dead end that cost a day is not tried twice.

Keep complete performance results in the linked evidence (the campaigns'
published pages) rather than copying them into the record, and say plainly when a
number is a scan rather than a paired comparison, or when something is a
hypothesis rather than a measurement. A lead nobody has measured yet goes under
"what remains unverified" in the record of the experiment that uncovered it — the
ledger lists what the fork *carries*, not what it suspects.
