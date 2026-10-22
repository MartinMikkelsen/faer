# Review rules for the tensor4all faer fork

Read these rules before opening an issue, preparing a change, or reviewing a PR.
They supplement [AGENTS.md](../AGENTS.md), the shared
[tensor4all rules](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md),
and the [adoption protocol](adoption-protocol.md). The specific overrides below
keep this fork aligned with faer and keep upstream rebases manageable.

## Follow faer's conventions

- Match the naming, signatures, module boundaries, file layout, and formatting
  of the corresponding faer code. This overrides the shared output-update naming
  and test-file organization rules where they differ from faer's conventions.
- Do not require `*_in_place` / `*_into` renaming merely to match the general
  tensor4all naming rule. Document mutation and output semantics clearly using
  the surrounding faer style.
- Do not split files, move existing tests, add abstraction layers, or reformat
  unrelated code solely to satisfy a cross-repository structural preference.
  Any structural change needs a concrete local reason and a focused diff.

## Test placement and scope

- Test a private function in a child test module of its owning module. Follow
  the surrounding faer pattern: an inline `#[cfg(test)] mod tests { ... }` is
  acceptable; an existing separately declared child test module is acceptable
  too. Child modules can access their parent's private items. Do not widen
  production visibility just to make an integration test reach private code.
- Use crate-root integration tests for public API contracts when appropriate.
- Keep ordinary unit tests lightweight: small representative inputs and minimal
  reproductions of defects. Include relevant algorithm/dispatch boundaries and
  failure paths without adding a costly exhaustive sweep to the default run.
- Recommend porting relevant tests from OpenBLAS
  [`test/`](https://github.com/OpenMathLib/OpenBLAS/tree/develop/test) and
  [`ctest/`](https://github.com/OpenMathLib/OpenBLAS/tree/develop/ctest) into
  lightweight unit tests, adapting them to faer's contracts and conventions.
  Keep each selected case bounded; place an expensive full combination sweep
  in the scheduled suite.
- Broader ports of OpenBLAS's bundled `lapack-netlib/TESTING/` are candidates
  for future nightly validation, tracked in
  [#31](https://github.com/tensor4all/faer/issues/31). This suite is planned;
  its absence is not an additional gate on current PRs. Retain small regression
  cases in ordinary tests when failures can be reduced.

## Scalar support and numerical contracts

- Check that a new implementation supports the scalar types of comparable faer
  operations. Use `f32` and `f64` for real-only algorithms; also support `c32`
  and `c64` where the operation applies to complex inputs. Explain intentional
  restrictions rather than silently narrowing the supported types.
- Check both generic signatures and executed test coverage. A generic function
  tested only with `f64` does not establish single-precision or complex
  correctness. Use precision-, dimension-, and scale-aware tolerances.
- Prefer reconstruction, residual, orthogonality/unitarity, ordering, and
  structural invariants over shape-only or finite-only assertions. Cover
  representative zero, singleton, repeated/clustered, ill-conditioned, and
  extreme-scale cases relevant to the algorithm; distinguish normal and
  subnormal values where scaling is involved.
- Check shape, layout/stride, scratch, accumulation, mutation, and error
  contracts. An operation reporting success must meet its documented numerical
  contract. Rejection/nonconvergence must leave outputs in the documented state.
- Exercise changed numerical branches and inspect their inner loops, scratch
  use, copies, and runtime dispatch even when no performance claim is made.

## Ported tests and provenance

- Recommend reusing relevant numerical validation from established open-source
  projects instead of inventing every test input and oracle independently.
- At the port site, record the source project, fixed commit/release, file and
  routine/test, source URL, fixture/generator provenance, and material changes.
  For tests taken from `lapack-netlib`, credit LAPACK as the original source and
  identify the OpenBLAS snapshot that distributed it.
- Preserve applicable copyright notices and license texts. Check the actual
  source file and directory: OpenBLAS, bundled LAPACK, and reference BLAS test
  directories do not all use the same license notice. Translation to Rust
  does not remove provenance or license obligations; follow the shared
  [provenance rules](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/provenance.md).
- Preserve the numerical intent while adapting API differences explicitly.
  Eigenvector phases, equivalent factorizations, valid balancing permutations
  and scales, or error interfaces need not match another implementation byte
  for byte. Do not copy an absolute tolerance that makes an extreme-value test
  vacuous, or weaken an oracle merely to make faer pass.

## Benchmark companions and evidence

- For a newly implemented numerical algorithm, recommend a companion PR in
  [faer-benchmark](https://github.com/tensor4all/faer-benchmark) covering
  representative sizes, scalar types, and input characteristics. Cross-link
  the implementation and benchmark PRs and distinguish checked cases from
  timed cases. The companion is a recommendation, not a new universal merge
  gate; existing performance-acceptance requirements still apply.
- Keep benchmark harnesses in faer-benchmark. Numerical correctness tests belong
  with the code they validate; timing results do not establish correctness.
- Check the experiment's PATCHSET entry, linked work log, applicable CI, and
  focused local validation under AGENTS.md and the adoption protocol. State
  what was verified and what remains unverified without inflating the evidence.

## Review findings

- Anchor findings to the reviewed commit and relevant source. State the input,
  observable failure, violated contract, and available reproduction evidence.
  Check documented local invariants and exceptions before requesting changes.
- State whether a defect is introduced by the diff, inherited from an existing
  kernel, or newly exposed through a public API. For an inherited defect,
  explain its effect on the proposed contract instead of calling it a new
  algorithmic regression.
- Treat correctness defects and violations of applicable requirements as
  findings. Label benchmark companions, test-port opportunities, and other
  recommendations as recommendations; do not turn planned infrastructure into
  an undocumented merge gate.
- Write GitHub review comments in English. Follow the maintainer's language
  preference in conversation. Keep upstream feedback subject to AGENTS.md's
  explicit-permission rule.
