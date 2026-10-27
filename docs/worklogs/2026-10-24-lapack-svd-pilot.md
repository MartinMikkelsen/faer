# Focused f32/f64/c32/c64 LAPACK SVD pilot (#31)

## Decisions

- Adapt selected LAPACK DCHKBD/DDRVBD matrix families and numerical properties
  in a private child of faer's existing SVD tests. Preserve the pinned
  OpenBLAS distribution revision, original LAPACK provenance, and BSD notices.
  A Rust port avoids requiring a Fortran compiler or external BLAS to run it.
- Use fixed orthogonal rotations and known diagonal spectra before undertaking
  the much larger DLATMS generator port. Keep residual products and reductions
  independent of faer's norms, normalize before multiplying extreme scales,
  and retain the source input file's threshold of 50.
- Exercise five output modes and both reduced/default recursion boundaries.
  Keep the validation groups explicitly selectable as required by #31; their
  checker self-tests run ordinarily. Known failures remain failures in the
  explicit suite, with no tolerance relaxation or expected-failure assertion.
- Test the bidiagonal adapter directly with DCHKBD checks 15–19, including
  the source's safe-minimum spectrum-comparison denominator without a dimension
  factor. Adapt faer's extra zero row explicitly while preserving the source's
  denominator n. Direct tests cover all four U/V-presence combinations.
- Put rank-one entries at the first, middle, and last diagonal positions,
  including the documented size-8 / threshold-4 case. The initial first-entry
  case did not expose the zero-child merge from #33; a passing first-entry
  case is not evidence that the split-position defect is fixed.
- Separate the test port from production fixes. The full-factor reused-buffer
  defect and the split-position output-mode defect are tracked by #41
  and #33; the initial port reproduced them on its recorded source state.
- Reuse the numerical checks in a sibling test module for layout and lifecycle
  coverage. Extend known spectra with repeated and tightly clustered values,
  verify the fixtures against literals, and exercise extreme scales through
  padded/reversed inputs and both output orientations. Border guards and
  input snapshots distinguish wrong numerics from unintended writes.
- Reuse output and scratch buffers across A and -A/2, rather than treating
  initial prefill alone as proof of buffer reuse. Use dedicated 2-/4-thread
  pools and default algorithm thresholds on sizes that reach the parallel
  branches. Restrict those larger cases to thin/omitted factors so independent
  scalar checks stay inexpensive; do not compare non-unique vector entries.
- Run all SVD/checker tests, including the ignored pilot, in a separate weekly
  Ubuntu workflow with a manual trigger. Local timing supports a two-minute
  execution limit; reserve 30 minutes for the job because clean hosted setup
  and dependency compilation are unmeasured. Keep the push/PR gate unchanged.
  Save dependencies even on failure, retain logs for 14 days, and use a separate
  two-minute job to create/update one labeled fork issue on failures or job
  timeouts. Known numerical failures keep the workflow red.
- Instantiate the same case files under concrete f32/f64 test modules, using
  a private Real alias to avoid duplicating cases or adding a numeric-trait
  abstraction. Fixtures, scalar products, epsilon, safe minimum, and scale
  limits use the native precision; the ratio threshold stays 50. Preserve the
  single-precision SCHKBD/SDRVBD/SBDT03 references at the same pinned revision.
  A calibration test prevents casting f64 extreme fixtures into f32.
- Record panics with their case context and continue to later calls. F32
  exposed a panic inside the matching-query scratch allocation; stopping the
  group there would leave later cases untested. The shared driver/direct/view
  call checker preserves errors and panic diagnostics without padding scratch
  or changing numerical code; its self-test proves continuation after failures.
- Extend the same files to c64 through separate Scalar/Real aliases. Use
  nonuniform diagonal phases and complex unitary rotations to preserve known
  real spectra, rather than casting real inputs to complex. Reconstruct with
  V's conjugate transpose and check Q's full conjugate Gram matrix using scalar
  products independent of faer's matrix operations. Reject nonreal singular
  values, including nonfinite imaginary components. Literal phased fixtures
  and checker calibrations establish these complex-specific requirements.
- Preserve ZCHKBD/ZDRVBD/ZBDT03/ZUNT01 provenance at the same pinned revision.
  ZCHKBD has only fourteen tests; the direct complex adapter coverage extends
  DCHKBD 15–19 properties to faer's phase-removal wrapper. Complex moduli use
  native standard-library hypot; some LAPACK helpers use component absolute
  sums, so these are bounded mathematical adaptations, not exact residual copies.
- Complete the scalar coverage with c32/f32 aliases and the same eight case
  groups. Share the complex fixture/checker calibrations between c32 and c64,
  keeping each type's real epsilon and scale limits. Preserve the single-complex
  CCHKBD/CDRVBD/CBDT03/CUNT01 references at the same revision and license; their
  direct adapter coverage has the same deliberate extension as c64. Native-f32
  scale calibration rejects accidental double-precision fixtures.

## Verification conclusions and constraints

- All 18,172 configured calls were attempted: 4,543 each for f32, f64, c32,
  and c64. Two single-precision rank-one calls panic; their diagnostics are
  retained and later cases continue. Twenty-nine opt-in groups pass and three
  fail: f32 has six passes/two failures, c32 seven/one, f64 and c64 eight/zero.
  All 45 ordinary SVD/checker tests pass, including 32 pilot checker/fixture
  tests. The combined scheduled selection has 74 passes and three failures,
  none ignored, and remains red.
- The current production source is `59a62189fdc7fad26dafe1a89ed84501d6118af7`,
  whose ledger records Codeberg upstream base `0ecd846`. The checkout already
  carries fixes for #33/#41. All split-position spectrum, prefilled full-factor,
  and guarded layout/reuse checks now pass, with no guard writes or input
  changes. The earlier failures in the coverage document measured an older
  source state. This test extension makes no production or dependency changes;
  no separate upstream or current remote revision was tested.
- Both single-precision rank-one panics occur at size 8, pivot `(0,0)`, recursion
  threshold 4, sequential V-only output. C32 has `A[0,0] = i`; f32 has 1. The
  matching scratch query leaves zero bytes when the f32 divide-and-conquer
  merge requests two usize entries (16 bytes on this host). A focused c32
  reproduction confirms the path through the complex phase adapter. No
  workspace padding, tolerance relaxation, or expected-failure assertion
  conceals these results.
- F32's size-127 logarithmic lower-bidiagonal fixture still fails V
  orthogonality with ratio 143.95331 against 50 for full/full, thin/thin, and
  V-only modes. The earlier independent f64 Gram calculation confirms the
  corresponding norm of about 2.179e-3 and ratio 143.95; this is not native-f32
  reference accumulation error. C32's phased fixture passes; the different
  inputs do not establish a cross-type accuracy comparison.
- All 288 parallel-group calls pass across the four types. Complex calibrations
  check conjugation, imaginary corruption, and literal phased entries at both
  precisions. C32's native-scale calibration rejects double-precision fixture
  limits and passes with f32 limits; the shared case loops and numerical
  checkers are unchanged by the c32 extension.
- The workspace test invocation reaches 279 passing library tests, the same
  three previously recorded triangular-matmul failures, and 34 ignored tests.
  Their full names and conditions are in the coverage document. The library
  failures stop the invocation before doctests; those were not separately
  repeated for this test-only extension.
- Native c32 test-binary compilation takes 45.60 seconds with dependencies
  cached. The subsequent workflow build step reuses it (0.717 seconds); the
  combined test invocation takes 23.029 seconds, including 22.58 seconds of
  execution. Retain the two-minute execution and 30-minute validation-job
  limits. These local observations are not cold hosted timings or performance
  comparisons. The workflow selects the fourth type automatically.
- All changed Rust test sources and the inserted module block pass formatting;
  the complete parent module has a pre-existing difference in
  full_factors_ignore_old_output_entries, left untouched. The diff has no
  whitespace errors. Cargo's offline package listing includes the three shared
  sources and licenses. CCHKBD/CDRVBD/CBDT03/CUNT01 have the same retained BSD
  terms; the independent complex calibrations remain MIT-licensed. The workflow
  and its previously checked failure-reporting behavior are unchanged. No
  commits or remote mutations were made.
- [Coverage, adaptations, reproduction instructions, and measured results](../lapack-svd-pilot.md)
  are the detailed reference. Normwise checks do not establish relative
  accuracy of tiny singular values. Full LAPACK generators, other LAPACK
  families, arbitrary stride/thread/shape combinations, production fixes,
  and clean hosted execution remain outside this pilot. The workflow has not
  been pushed or run on GitHub.
