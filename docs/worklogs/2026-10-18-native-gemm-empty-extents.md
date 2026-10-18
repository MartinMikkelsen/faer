# Native indexed/diagonal matmul: empty extents and empty products

## Status / source isolation

Local correctness fix on `fix/native-gemm-empty-extents`, based on main
7364be5ce22d950343c10d82b99e9fc79d537cd2. Not integrated, released, or performance-
accepted. No LU-budget, numeric-copy or Householder-merge experiment is included.
The original `matmul/internal/mod.rs` production source is byte-identical to
upstream0539947ffb757a739d7e703a7d2fa0c792a909c1. Only this fork's temporary worktree
was executed; do not call that an independently executed upstream reproducer.

## Defect, contract and smallest correction

`spicy_matmul` takes original logical output dimensions, then its native branch
shifts source/destination/index views for strict/unit lower or upper output. It
passed the stale original dimensions to the backend. A1×1 strict lower output is
empty, but changed a sentinel diagonal7→8. A3×3 view inside an allocated4×4 parent
changed an outside-view guard row7→20. Parent allocations keep the canary test
bounded; the wrapper's logical extent contract is nevertheless violated.

Recompute native-local dimensions from the adjusted A/B views, and return if
either is zero. Original dimensions remain available to the unchanged fallback.
Strides, mappings, conjugation, D, alpha, accumulation and scheduling are retained.
The current five production callers (three dense LLT/LDLT, two sparse Cholesky)
use Rectangular or inclusive TriangularLower, not this strict/unit branch. This
is not an explanation of the recorded performance regressions.

The required zero-depth check found a second existing defect: the native backend
leaves stored entries unchanged for K=0/Replace. The common wrapper now handles
that empty product after existing index validation: Add returns unchanged;
Replace writes zero to logically stored entries, then maps those coordinates to
the physical destination. Unit/strict diagonal and opposite triangle remain
untouched. This matches the dense matmul wrapper's K=0 convention; it is not an
unconditional zero-depth return. No new unsafe, public API, threshold, dependency
or scratch requirement. Positive-depth numerical paths are unchanged except for
the corrected native extents.

## Evidence / runnable checks

[Initial negative tests and source identity](../../../faer-benchmark/data/probes/20261018-native-gemm-callers/README.md)
(campaign7355918) retain both failures, the test-only patch and original plan.
The later K=0 failure and fixed checks are retained with the follow-up evidence.
No build overlapped a performance recording. No large-suite tier was run.

Module-owned tests in `matmul/internal/mod.rs` retain both original reproducers
and an independent scalar-product oracle over f64/c64, all seven structures,
N=0/1/3 and nonsquare1×3/3×1/2×4/4×2, Add/Replace, K=0/2, four index-presence
combinations with descending nonidentity maps, conjugated left input, real D,
complex alpha and unchanged parent guards. The oracle matches the enum variants
separately rather than sharing the production storage predicate. Self-review added
an explicit finite-output assertion before the max-norm comparison, so NaNs cannot
be hidden by max-reduction semantics. Focused/full library checks were rerun after
this test-only strengthening; production is unchanged from the POST snapshot.

Validation, sequentially, with `CARGO_TARGET_DIR=/home/shinaoka/projects/tensor4all/faer/target`:

- `cargo test -p t4a-faer --lib strict_triangle -- --nocapture`:4 passed.
- `cargo test -p t4a-faer --lib`:214 passed,2 ignored.
- `cargo check -p t4a-faer --no-default-features --features std,linalg`:passed;
  13 warnings, all in untouched files, none in the changed module.
- `git diff --check`:passed. Targeted rustfmt changed only the two added
  production regions and new tests, not existing formatting.

The validation host exposes AVX2/FMA/AVX512F; default features include std and the
closed f64/c64 native traits, so the tested dispatch uses the native path. No
forced-generic numerical run or separate f32/c32 fixture is claimed. No-Rayon's
compilation is not evidence of forced-generic execution. Tests request Par::Seq;
these are correctness checks, not timing results. kache doctor verified33,980
entries, zero corruption before this worktree was started.

## Independent review / remaining work

The first broad PRE attempt was INCOMPLETE (tool-call limit), not a review.
A reduced, self-contained gpt-6.1-sol read-only PRE completed and approved the
effective-dimension plan; its required K=0 test exposed the second defect. A
second bounded PRE completed and approved the common zero-product handling.
The optional deletion of old redundant pre-shift guards was not needed for this
small correction. No foreign implementation/tests were imported; existing MIT
notices remain. Upstream tracker feedback is deferred by AGENTS.md.

Independent POST must complete before handoff/integration. Performance claims,
shared GEMM-budget changes, adoption comparison and final all-suite recordings
remain separate and unresolved.
