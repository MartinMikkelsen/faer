# Shared native GEMM work floor — isolated experiment

Status: EXPERIMENT, unintegrated and not performance accepted. Base b7715c5
contains the separately reviewed native empty-extent/K0 correctness repair;
neither LU-budget restoration, copy-tile nor Householder-merge candidates enter
this branch. Main/origin-main remains unchanged.

One private x86_64+std helper reuses the adopted dense policy at all three native
entries. Mathematical MAC work W requires W≥threshold for real and W≥ceil(threshold/4)
for complex. Saturating MAC multiplication and div_ceil avoid overflowing FLOP
arithmetic and twice-threshold. Native triangular/internal calls use exactly
post-shift effective dimensions; the internal32³ gate is moved unchanged to the
fallback after native return, with original outer dimensions. Nano/generic
policies, LU ancestor gate, pointers, accumulation, conjugation, mappings and
scratch are unchanged. No new threshold, API, unsafe or dependency.

PRE: COMPLETE, openai/gpt-6.1-sol, no mandatory finding. Plan/report in campaign
`docs/reviews/20261018-native-gemm-budget-{plan,pre-review}.md`.

## Checks

From this worktree, CARGO_TARGET_DIR=/home/shinaoka/projects/tensor4all/faer/target:

- `cargo test -p t4a-faer --lib native_ -- --nocapture`:2 passed initially.
- Added hypothetical extreme-cutoff algebra checks without changing global
  threshold; final `cargo test -p t4a-faer --lib`:217 passed,2 existing ignored.
- `cargo check -p t4a-faer --no-default-features --features std,linalg`:passed,
  13 existing warnings on unchanged declarations (including imports in changed
  parent mod.rs); no new helper warning. Not forced-generic numerical execution.
- `kache doctor --verify`:33,981/33,981 valid; no corruption/orphans/missing/checksum
  failure/index drift. New test file formatted only; stable rustfmt reports
  ignored nightly-only config options. `git diff --check` passes.

Module-owned tests: configured cutoff below/at and Seq for f32/f64/c32/c64,
wide FLOP oracle including saturated products followed by zero; hypothetical
threshold0/near-MAX/MAX algebra. Public triangular f64/genuine-c64 scalar oracle
on1/32/64/128 square, C/F input+output views, lower/upper/strict/unit representative
masks, Add/Replace, left conjugation and parent guards; owned8-worker pool and
Seq budget. Mapped rectangular/strict-lower128×96×64 Add with real diagonal
present/absent checks the above-floor route. Existing exhaustive small internal
map/mask/depth0/2 oracle is repeated with owned8 budget. All outputs checked finite.
No f32/c32 numerical structured or forced non-native execution claim.

POST COMPLETE on gpt-6.1-sol for7f713a1; no findings or deletion request. All
requested bounded areas traced; no reviewer builds/tests/timing and no native
worker-utilization claim. Report in campaign
`docs/reviews/20261018-native-gemm-budget-post-review.{md,json}`. Source hashes
still match reviewed snapshot; no production/test corrective edit.

Before timing: comparison-completeness repair verified,
exact targets/guard grids/counts frozen, fresh immutable A=b7715c5/B candidate
builds/linkage/worker census and topology-based pins. This cannot restore LU's
already-lowered caller budget or explain1T binary shifts. No timing yet; square
2048/3072/4096 remain final-only. Raw checks/diff/identity are retained in campaign
`data/probes/20261018-native-gemm-budget/`.

Provenance: own code/tests in MIT fork; existing gemm threshold getter reused,
not vendor implementation. Prior inherited unsafe code/notice left unchanged.
