# Builtin copy tiles with unchanged generic Clone semantics

## Decisions

The cross-layout copy already has one Option tile per worker, not per tile. There
is no initialization-hoist fix to make. The earlier zero-valued numeric-tile
candidate d635 was not adopted: its extra zero/clone operations on arbitrary
ComplexField values were not an approved contract tradeoff, and an unchanged
F-order control shifted in its diagnostic. Those results are not acceptance.

This separate experiment starts at main7364be5 (runtime code2240b07). Its private
Clone-only entry selects only exact f32/f64/c32/c64 identities through the
existing safe nonstatic_typeid helper. It does not trust overridable native
flags or add a static bound. For every other type the entire original Option
body remains unchanged: one source clone, move into destination, no seed.

The four builtin types have pure bit-copy Clone and no Drop. Their value tile
is seeded from the first source only after an empty-extent return; every slot
read by writeout is overwritten by staging. No zero/default call is added.
The tile dimensions, both traversal orders, caller arguments, worker partitions,
public allocation boundaries and conjugated routing remain unchanged. Keeping
one small representation-specific loop avoids unsafe uninitialized storage,
new scratch/heap allocation, slot abstractions, macros or public APIs.

The existing identity helper's internal unsafe implementation and attribution
are unchanged; this experiment relies on its existing safe-interface contract,
not a new soundness proof. Alternate scalar implementations and borrowed/custom
values retain the generic route. Public cloned() is not accelerated. Existing
allocation-time zeros are not removed. F-order composite LU scheduling regressions
are not solved by this copy change.

## Verification conclusions and constraints

Two implementation-owned checks pass. Builtin copies preserve exact component
bits, including signed zeros, infinities and NaN payloads, with C/F directions,
negative source and destination strides, empty/one/tail/rectangular dimensions,
and whole-parent guards. Caller budgets Seq and owned4/8 pools have checked pool
sizes; worker utilization is not inferred. The actual private dispatcher also
preserves one clone and one overwritten-destination drop per logical element
for a heap-owning Clone-only value with non-static borrowed atomic counters.
Parent guards, independent payload allocation, both empty axes and total drops
after destroying matrices are checked. This is not a dummy customComplexField.

Library212passed/2existingignored; no-Rayonnarrowcheck passes with13existing
warnings. Original Option body, wrapper and other production paths are preserved;
only targeted formatting remains. Bounded finished-diff independent POST is
complete with no must-fix source findings. It did not execute checks, rehash the
checkout, or prove the existing identity helper sound; supplied diff/equality
records and the completed PRE support unchanged-callers preservation.
No new performance binary, verify or timing exists for this candidate, and no
upstream measurement of0539947 has been performed for it. Source correctness
cannot establish copy speedup, composite benefit or adoption.

Performance evidence will stay in faer-benchmark/data/probes/20261018-native-copy-tile/.
Any performance verify or timing, including the public-copy probe, requires
reviewed/frozen code, actual full-checkout linkage and binary proof, and the
existing four dense size/normalized-body equalities. A mismatch is HOLD without
flag/position tweaks or retries. With static qualification, explicit effective1T
public-copy boundary, all retained samples/rejections,
and the original formal control/guard/coverage/statistical acceptance conditions
on the declared blast radius. No large final-only squares during development.
The separate native-floor7cd/f013 HOLD and generated-code gate remain unchanged;
this is not permission to relax them. Main/pin/release/upstream submission stay
unchanged; the expanded regression goal and final full campaigns remain unfinished.
