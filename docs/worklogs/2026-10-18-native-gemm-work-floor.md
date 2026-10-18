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

## Completed stage1 — INCONCLUSIVE / NOT ADOPTED

Fresh A=b7715c53ba75946d68a97d5655f481b198332447 versus
B=7f713a1401f18546d1ef0c134df982be80626b2e, upstream
0539947ffb757a739d7e703a7d2fa0c792a909c1, harness
651807301c4fd0d485a66038346388dffe289073, toolkit9daa5bc69a13b36c2a31680e20dbb53e2e42c9d5,
MKL2026.1. Four existing-recorder groups/side, each three starts/rotations,
median5/250ms, exactly648 full identity/status records per start per side (3888
timed-table rows including declared skips). All verification/manifest/source/
binary checks passed; explicit1/4/8 budgets read back. Binaries SHA256 A=
2decf425848a5b0706f9c20de65084822a23d659071b38b6bf7c9dea9d347e4a,
B=f8e8e3337463fe7219395de47c8161bba8c95fb604e743ca89266f587bcfce2a.

Official controls failed6/60 Cholesky,1/48 GEMM,4/48 triangular-solve; HH72/72
reproduced. Thus NO adoption, no stage2. All228 fork cells retained descriptively:
Cholesky target24 ratios .529–.999 are NOT acceptance; one accepted A c64C256/8T
start is5.143332ms versus .822805/.849886ms later, retained without filtering.
Four dense GEMM rotation ratios exceed1.05; three medians exceed baseline process
max+5%. No source/worker/alignment explanation waives these or proves causality.

Measured CPUs4 /4-7 /4-11, watched full L3/SMT4-11,16-23; observer0-3,12-15,
agent1492359 all11 threads outside. Boost0, performance/EPP-performance/max2GHz.
Three post-rejected attempts retained completely: verification peer18 at10.3%,
verification peer21 at11.9%, HH timing peer16 at10.3%; normal declared retries,
not dropped outliers. Initial offline-preflight missing crate was recovered by
locked dependency fetch, no lockfile/source change, before any build/timing.

Raw8 runs remain in campaign data/results with no published page; exact inventory,
full descriptive output and CLI logs in `data/probes/20261018-native-gemm-budget/`.
Source hashes still match reviewed7f713a1. No code correction, main integration,
large-square or final campaign. This cannot restore LU's ancestor-reduced budget.
Completed upstream-only control isolation is DIAGNOSTIC ONLY, not another
adoption pair or a replacement gate: same binaries/compiled6518073, recording
harness311c2fb,96cells×8ABBABAAB blocks/768records,4starts/binary;18verify/72timing/
6info processes. All budgets/guards/identity/census pass, two post-reject archives
retained, no source edit/build/checkout. Seven descriptive bands fail, including
c64Ftriangular256/1T medianB/A=.89646 with all4 B below A-min-5%; fork/vendor
operations not executed (setters/loading remain). Their execution history alone
is insufficient; placement/allocation/caller/temporal mechanisms not identified.
Canonical refusal unchanged, no adoption/stage2. Full raw/summary in campaign
runtime-isolation; no trimmed data or new thresholds.

Four dense matmul_imp symbols matched; two normalized bodies unchanged and two
changed230/634lines (3444→3437 and3368→3342bytes), RIP normalized/no engine or
whole-program attribution. Equivalent dense policy did not preserve all generated
code. Next distinct proposal keeps dense body untouched and changes only the two
structured routes, with a pre-timing static-preservation filter; PRE pending,
not implemented. Full goal and square2048/3072/4096 remain pending.

Provenance: own code/tests in MIT fork; existing gemm threshold getter reused,
not vendor implementation. Prior inherited unsafe code/notice left unchanged.
