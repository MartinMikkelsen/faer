# Structured native GEMM floor, dense body preserved

Status: EXPERIMENT, no performance acceptance/integration. Baselineb7715c5 includes
the separately reviewed strict-empty/K0 correctness fix. This branch does not
include LU/copy/HH experiments. Prior7f713a1 was NOT ADOPTED after canonical
controls/guards; upstream-only diagnosis did not waive that refusal. Full evidence
in faer-benchmark/data/probes/20261018-native-gemm-budget/.

Minimal new distinction: keep the adopted dense matmul_imp SOURCE body exactly
asb7715c5. Reuse7f713a1's saturating real-MAC/complex-ceil-quarter private helper
at TWO structured native entrances, effective dimensions/caller budget, and its
fallback-only32³ gate relocation. No new threshold, shape/CPU tuning, API, unsafe,
scratch/dependency or numerical algorithm. The old dense2*threshold arithmetic is
left unchanged; no claim that route gains the helper's overflow hardening.

Inherited7f713a1 tests are byte-identical: f32/f64/c32/c64 threshold math incl
extremes; genuinef64/c64 structured scalar products, masks/Add/Replace/conjugation/
maps/finite guards/Seq/owned8; small strict/K0 repeats. No f32/c32 structured
numeric or forced-generic execution claim.

PRE: first broad attempt INCOMPLETE/timeoutexit124, not approval. Narrow immutable
delta-only PRE COMPLETE on gpt-6.1-sol, no findings; approve plan implementation,
not code/performance. Full reports in campaign docs/reviews. Reviewer wording
"strict cases final-only" is not the plan: strict/unit correctness tests remain
development work; ONLY square2048/3072/4096 benchmark tiers are final-only.

Checks on this code change, shared CARGO_TARGET_DIR=/home/shinaoka/projects/tensor4all/faer/target,
run sequentially: `cargo test -p t4a-faer --lib native_ -- --nocapture`2pass;
`cargo test -p t4a-faer --lib`217pass/2existingignored (8.42s); `cargo check -p
t4a-faer --no-default-features --features std,linalg`passes/13existingwarnings,
no new helper warning. Kache33975/33975 valid/no corruption/index drift before work.
Three retained Rust files/tests byte-identical7f713a1; mod.rs suffix from
pub(crate)modinternal through EOF byte-identicalb7715c5, including dense function.
No broad formatting or dependency change. Implemented-diff POST pending;
no fresh performance build, generated-code comparison or timing yet.

After checks/POST, fresh A/B builds through existing full-checkout builder and
exact Cargo/source/binary evidence. Development static pre-filter requires all4
matching dense f64/c64 matmul_imp sizes/normalized bodies unchanged; otherwise
hold without timing or flag/alignment fiddling. This is NOT a new acceptance
gate, equivalence/reference proof or causal claim. Only then unchanged canonical
648recordstage1×3starts/side with all controls/guards, conditional2688recordall14
familystage2; no intersections/waivers/third LU pair. No final freeze/main merge.

Provenance: own reviewed helper/tests reused in MIT fork, existing gemm getter;
no vendor implementation or foreign tests imported. Preserve inherited notices.
