# LAPACK numerical validation inventory

This is a bounded Rust adaptation of the candidates in [#31](https://github.com/tensor4all/faer/issues/31).
Port completion and numerical success are separate.

## Source and baseline

LAPACK 3.12.0, distributed by OpenBLAS revision
`791b02a1a00c996b66a894f6df93421c53b62d80`, under `lapack-netlib/TESTING`.
[Source identities](lapack-validation-sources.json) record the inspected files.
Source routines/fixtures and helper formulas are read before each port.
The Rust case construction is an adaptation, not LAPACK's random sequence or
a full LATMS/LATMR translation. Retain BSD notices in adapted files and
`faer/COPYING.LAPACK.BSD`; independent test wiring is MIT.

Development measurements were taken at `59a62189fdc7fad26dafe1a89ed84501d6118af7`,
which is `main` `5a27bce` plus one test in the `solvers.rs` test module; the library
code is identical. The final combined run below is repeated on this change's own
tree, based on `5a27bce`. Production source stays unchanged except cfg(test) wiring.
Rust 1.98.1, macOS arm64, default features (std/rayon), optimized dev profile,
`CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_INCREMENTAL=0`.
The [SVD record](lapack-svd-pilot.md) supplies the verified baseline: f32/c32
rank-one scratch panics, f32 logarithmic-bidiagonal V orthogonality, and three
pre-existing strict-triangle matmul failures. No upstream checkout is measured.

## Inventory

All applicable rows cover f64/f32/c64/c32; GET34 adjacent 2×2 blocks are real-only.
EPS follows each source helper: LAMCH(P)=native EPSILON, LAMCH(E)=EPSILON/2
on the measured binary IEEE types. Source ratios keep their dimension factors
and safe-minimum handling. Complex-modulus adaptations are identified explicitly.

| Filter suffix (`lapack_...`) | Port status | Source checks/helpers | Threshold | Owning test module | Deliberate adaptation/omission |
| --- | --- | --- | --- | --- | --- |
| svd | adapted; passing (f32 V orthogonality fixed by #75, rank-one scratch by #73) | S/D/C/ZCHKBD; D/S/C/ZDRVBD; BDT03, ORT01/UNT01 | 50 | svd::tests | Existing four-type pilot; see linked record |
| hessenberg | adapted; passing | CHKHS 1–2, HST01, ORT01/UNT01 | 20 | evd::lapack | Packed reflectors extracted to logical H |
| schur | adapted; passing | CHKHS 3–8; HST01 | 20 | evd::lapack | Equivalent Schur forms; compare invariant/residual/spectra |
| nonsymmetric_evd | adapted; residual failures recorded | CHKHS right/left residuals; GET22 | 20 | evd::lapack | Full EVD exposes both sides; Schur-vector helper exposes right only |
| schur_reorder | implemented; passing | S/DGET34, S/D/C/ZGET36; *ec.in selected records | 20 | evd::lapack | Real 1×1/2×2; complex 1×1; rejection contract |
| balance | implemented; passing | CHKBL/CHKBK; *bal.in/*bak.in | source checker | evd::lapack | Combined balance; right-vector back-transform; metadata adaptation |
| tridiagonal_evd | implemented; failures recorded | CHKST 9–12/22–26 where applicable | 60 | evd::lapack | QR/DC properties; no MRRR/range selection/Sturm certification |
| self_adjoint_evd | implemented; passing (spectrum accuracy fixed by #76) | CHKST reduction/full interface; SYT21/HET21 | 60 | evd::lapack | Lower-triangle ownership; no packed/banded interfaces |
| lu | implemented; passing | CHKGE, GET01/02/03 | 30 | lu::lapack | Full-pivoting labeled extension; omit RFS/CON |
| cholesky | implemented; failures recorded | CHKPO/DRVPO/CHKPS, POT01/02/03 | 30 | cholesky::lapack | LLT/pivoted LLT; omit RFS/CON |
| indefinite | implemented; failures recorded | S/DCHKSY, C/ZCHKHE, SYT01/HET01 | 30 | cholesky::lapack | Separate D; unpivoted LDL extension; no complex-symmetric contract |
| qr | implemented; passing | CHKQR/CHKQ3, QRT01/02/03 | 30 | qr::lapack | Free column pivoting; no fixed-column JPVT flags or QL/RQ/LQ |
| triangular | implemented; failures recorded | CHKTR, TRT02 | 30 | triangular_solve::lapack | TRS contract; no overflow-scaled LATRS/RFS/CON |
| lstsq | implemented; passing | DRVLS, QRT14/16/17 | 30 | solvers::lapack | Tall QR, supported SVD minimum-norm; no new driver algorithms |

## Original twelve reports

| Report | Property | Existing regression/evidence | Validation family |
| --- | --- | --- | --- |
| #33 | Zero-child split preserves SVD spectrum across modes | test_zero_child_at_split; test_zero_block_at_split | SVD, retained pilot |
| #34 | LDL reconstruction reads separate D | test_reconstruct_separate_diagonal | indefinite |
| #35 | Column-pivoted QR of zero input remains finite/reconstructs | zero_matrix_reconstructs | qr |
| #36 | Strided norm handles extreme range | test_norm_l2_strided_range | Out of LAPACK decomposition scope; independent scaled reference norms avoid this dependency |
| #37 | GEVD reused vector output initialization | reused_eigenvector_buffers | Generalized EVD outside #31 candidate list |
| #38 | Zero/subnormal LU pivot | zero_pivot_column_reconstructs | lu |
| #40 | Rank-deficient SVD least squares | svd_lstsq_rank_deficient | lstsq |
| #41 | Full SVD factors ignore prior output | full_factors_ignore_old_output_entries | SVD, retained pilot |
| #42 | Finite 2×2 EVD spectrum near range limit | two_by_two_finite_spectrum_at_the_range_limit; two_by_two_subnormal_diagonal | self_adjoint_evd |
| #43 | Subnormal scalar triangular pivot | tiny_pivot_scalar_solve | triangular |
| #44 | Empty EVD scratch/query | empty_matrix | self_adjoint_evd |
| #45 | Pivoted LLT early rank exit | test_early_rank_reconstruct | cholesky |

The mapping follows the two 2026-10-22 worklogs, not a contiguous issue range
(#39 is a fix PR). Verify current test paths/names while porting each owner.

## Results and execution

Each suite records planned/attempted API calls, numeric/call failures,
source state and elapsed time. Numerical failures remain failures. Missing APIs,
error estimation, packed/banded storage and unrelated algorithms are omitted
explicitly rather than implemented here. The combined runner executes every
family and preserve failure statuses. Ordinary checker/fixture calibrations
remain small; heavy numerical groups are ignored unless selected.

### Phase 1 — Hessenberg and Schur

All 8,712 configured calls pass: 2,904 Hessenberg and 5,808 Schur, including
sequential and dedicated 2-/4-thread groups. The calls reuse scratch/matrix
buffers, cover four guarded layouts, and separate packed Householder storage
from logical H. Schur accumulation starts from a valid identity/nonidentity Q,
as required by its contract. Cases include 0/1/2, source small sizes, 74/75/76
and a 257 parallel fixture. Execution takes 8.50s + 17.80s locally.
Independent scalar checks retain HST01's dimension/safe-minimum scaling with
a strict zero-input check; complex norms use Euclidean moduli. Condition-weighted
CHKHS spectrum comparisons lack a matching faer estimator: compare modes for
normal fixtures and retain backward residuals for Jordan/nonnormal cases.
[Per-type counts](lapack-validation-results.json) are the compact evidence.

### Phase 2 — Nonsymmetric eigenvectors

All 7,200 configured calls return successfully; eight ordinary native checker
calibrations pass, and eight numerical groups report GET22-style residual
violations. Full EVD covers left/right/both/no vectors, plus Schur-to-right-vector
extraction, recursion boundaries, layouts, accumulation and dedicated pools.
Left vectors are exposed by full EVD and are therefore included (the standalone
Schur helper is right-only). Faer's unit-2-norm contract replaces GET22's max-entry
normalization; the residual keeps the source's no-dimension-factor denominator.

An independent 80-decimal-digit calculation on returned f64 vectors confirms
a size-129 tightly clustered fixture: native residual ratio 39.8056954384,
reference ratio 39.7553910914 against threshold 20, with residual norm
3.5466793623e-14, A 1-norm 1.00000000000024 and X 1-norm 4.017775556.
This corroborates the measured residual rather than attributing it to native
reference accumulation. It is a finding on this checkout, not a claim about
current upstream. No production change or tolerance adjustment was made.
The four-type numerical run takes 5.85s locally; diagnostics retain all failures.

### Phase 4 — Balancing and back-transformation

All 1,624 configured calls and four literal calibrations pass (0.02s execution);
the 1,616 eigenvector back-transformations run as dependent calls.
CHKBL's unscaled relative invariant is adapted to eight native epsilons; CHKBK
retains entry-relative error divided by epsilon. A literal nonidentity permutation
and powers-of-two scales calibrate orientation. First BAL/BAK records are carried
as invariant fixtures; exact LAPACK metadata choices are not required. Other
records, separate job flags and left-vector back-transformation lack matching
faer interfaces and are omitted. The combined API is serial. Known eigenvector
round trips, isolation, nonfinite no-write errors, four layouts and reuse are covered.

### Phases 3, 5 and 6 — Reordering and self-adjoint EVD

Reordering passes all 3,132 calls (0.04s). A rejected GET34 swap must leave T
and Q bitwise unchanged (the DLAEXC contract); no current fixture is rejected,
so that check has not yet run on a rejection. GET34/36's normwise similarity contract
sets the full Schur matrix scale for moving-block identity; it does not promise
relative accuracy with respect to a tiny individual block. The initial local
relative oracle falsely rejected valid swaps; its normalization was corrected
without changing threshold 20. Cardinality matching rejects duplicate/lost values.
Selected real EC input line 3 and complex line 142 records retain literal fixtures.
Other input records, condition estimation and alternate reordering drivers are omitted.

Tridiagonal EVD attempts all 3,504 calls (11.62s). Real groups pass. Each complex
type has 36 empty, vector-producing calls that panic while accessing scale[0];
values-only and all nonempty/parallel cases pass. A minimal reproduction uses
n=0, empty diagonals and an empty complex Q with exactly the returned scratch.
This is retained as an opt-in failure. CHKST 11/26 retain their no-dimension
spectrum-comparison ratio; reconstruction/unitarity use dimension-scaled ratios.
Toeplitz, split blocks, signed/clustered diagonals, graded and range-scale cases
replace LATMS, with complex phase similarity preserving the real spectrum.
ST ERF/QR/DC properties are adapted; bisection, inverse iteration, MRRR, selected
ranges, Sturm counts and vendor-algorithm comparisons have no equivalent interface.

Dense self-adjoint EVD attempts all 5,064 calls (27.00s). Reduction, reconstruction,
unitarity, guards and poisoned-unused-upper-triangle checks pass. Spectrum/mode
comparisons exceed 60 in five groups (f64 and both single-precision types);
c64 groups pass. For example f64 n=127, kind=2, threshold=8 has values-only known
spectrum ratio 71.5 and vector/value mode ratio 73.5. Mode disagreement compares
returned spectra from the same stored input and does not rely on the known-spectrum
generator. These are numerical findings, not proven algorithmic root causes.
The subnormal 2×2 range fixture must return its exactly representable spectrum
bit for bit: the ratio's safe-minimum floor accepts any error below MIN_POSITIVE,
so it cannot judge that case. It passes at all types.
Packed/banded storage, alternate UPLO and additional LAPACK driver algorithms
are intentionally omitted. Dense calls include empty inputs, thresholds 8/128,
127/128/129, four layouts, reused storage and dedicated 2-/4-thread pools at 257.

### Phases 7–11 — Dense LIN adaptations

The initial LU run passes all 7,968 factor/reconstruction/solve/inverse calls
(43.82s). Partial/full pivoting use independently expanded rectangular L/U and
literal three-cycle permutations. Full pivoting is a faer extension of GE
properties. Modes include normal, transpose and adjoint; singular fixtures are
factored and reconstructed without singular solves. Zero columns and native
large/small scales are included. RFS/CON/equilibration and expert drivers are omitted.

Later LIN groups count each configured factorization as a principal call;
dependent reconstruction/solve/inverse/Q-application calls are reported separately
as `auxiliary_attempted`. A bad factor/metadata records a failure and skips unsafe
dependent calls while continuing later principal cases. The final LU run uses the
same accounting. This avoids misrepresenting a library failure as an incomplete
case loop. Scratch queries and fixture/oracle products are never counted.

Cholesky attempts all 1,712 principal calls plus 4,648 dependent calls (14.55s).
All positive-definite, non-positive-diagonal error, rank-truncated reconstruction,
normal and conjugated solves, inverse, poisoned-upper, view/reuse and parallel
checks pass. Pivoted LLT never compares the first pivot with its rank tolerance
(`if j > 0` in `llt_pivoting/factor.rs`), so a zero first pivot is not detected.
At every dtype a 1×1 zero matrix returns rank one (four diagnostics/type: zero and
rank-zero diagonal fixtures, two buffer passes), and every larger zero matrix
returns NonPositivePivot instead of the rank 0 that DPSTRF reports (22
diagnostics/type). Packed/banded storage, regularization and RFS/CON are omitted.

Indefinite LDL attempts all 3,376 principal calls plus 7,840 dependent calls
(25.06s). Initialized metadata, 1×1/2×2 D blocks, independent LDLH reconstruction,
separate-D reconstruction, nonsingular solves/inverses and parallel groups pass.
The retained prefilled subdiagonal cases each dtype have 178 invalid-metadata
diagnostics. A minimal reproduction is n=1, A=[1], subdiagonal prefilled nonzero:
the factorization leaves the value unchanged. The affected no-update branch is
also exercised on larger diagonal matrices and reused buffers. Once metadata is
invalid, dependent routines are not called. The unpivoted LDL path is an adaptation;
complex fixtures are Hermitian, not complex symmetric. Alternate pivot strategies,
singular solves, packed IPIV comparison and error estimates are omitted.

QR passes all 4,656 principal calls and 23,280 dependent calls (16.80s). A native
scalar product of elementary reflectors independently defines Q; block inverse-tau
storage is explicitly adapted and calibrated. It checks R-QH AP, full/thin Q,
reconstruction and Q/QH application from both sides, at block sizes 1/4 and selected
47/48/49 boundaries, guarded layouts and dedicated pools. Zero/rank-dependent,
controlled singular-value and range-scale fixtures are included. QRT12's
cross-decomposition singular-value comparison is omitted (SVD has its own pilot),
as are fixed JPVT columns and QL/RQ/LQ drivers.

Triangular and least-squares final results appear in the combined execution record.
TRT02 uses implicit unit diagonals with poisoned storage, normal/transpose/adjoint,
multiple RHS, native tiny pivots and guarded layouts. Overflow-scaled LATRS and
condition/error estimators are omitted. QR least squares stays in its documented
m>=n full-rank domain; known inconsistent residuals are orthogonal to A's columns.
QRT16 residuals, QRT17 conjugate normal residuals and condition-scaled literal
solutions are checked. QRT14's row-space property is checked by literal SVD
minimum-norm/nullspace solutions instead of factoring an augmented matrix. SVD's
current least-squares interface also requires m>=n: wide and epsilon-cutoff
pseudoinverse contracts are omitted, not inferred. Rank-deficient cases retain #40.

### Fixed case manifests and range/dispatch coverage

Each `*_cases` loop is the executable bounded manifest (literal dimensions,
kind/scale indices, modes, layouts and thresholds); every scalar instantiation
runs the same loop with native epsilon/range. Stateless seed 0x4c415041434b and
adjacent 0.6/0.8 unitary rotations replace LATMS/LATMR. Four layouts are padded
column-major, row-major and their doubly reversed views; logical fills preserve
the same input. Two passes reuse factor/output/scratch storage. Dtype aliases are
compile-time private modules, not runtime precision conversions.

LIN base sizes include 0/1/2/3/5/10/50, selected rectangular shapes, LU 15/16/17
and 63/64/65, Cholesky/LDL 63/64/65 and 127/128/129, QR 47/48/49 and 3/4/5,
TRS 63/64/65 and 127/128/129. Dedicated sequential/2-/4-thread cases use LU n=513,
other factors n=257, TRS (n,nrhs)=(257,65), and LS (m,n)=(257,193), crossing current
size dispatch floors. Balance/reorder are serial interfaces. Parallelism is
requested through the routine's actual API; no speedup claim is made.

LIN ratios retain LAMCH('E') = Rust EPSILON/2 for round-to-nearest, whereas EIG
ULP is Rust EPSILON. Complex matrix norms use moduli; BLAS ASUM-style RHS/error
sums use abs(real)+abs(imag), as in ZGET02/ZPOT02/ZTRT02. Products are scaled first
at extreme magnitudes; nonfinite intermediates fail. Strict zero checks supplement
source quick exits. This is numerical validation, not a benchmark campaign;
production implementations and dependency versions have no acceptance delta.

### Reproduction and scheduled operation

Run `scripts/run-lapack-validation.sh` from the repository root. It builds one
locked default-feature test executable, enumerates exact test names, rejects empty
or unclassified selections and runs every family sequentially. It preserves each
family's exit status, verifies test and configured-call counts, and returns failure
if any family fails or execution is incomplete. Logs and `summary.json` go to
`target/lapack-validation/`; `--output` selects another evidence directory.
Ordinary checker/fixture calibrations stay active in normal workspace tests;
heavy numerical groups stay ignored there. SVD's existing suite only gains explicit
attempt counters and call-failure diagnostics; its matrices, thresholds and calls
are preserved.

The replacement workflow runs weekly Monday 03:23 UTC / 12:23 JST and manually.
It caches dependencies on failure and uploads environment, build timing, exact
selection, family logs and summary for 14 days. Per-family timeouts sum to
20 minutes; the build limit is 20 minutes, the combined step 44 minutes and the
validation job 50 minutes. A separate two-minute reporting job creates/updates one
open issue titled “Scheduled LAPACK numerical validation is failing” with label
`scheduled-lapack-validation`, retaining the failed status and run link. These are
upper bounds, not expected runtimes. They allow every family timeout to expire
without the total step stopping later groups.

Default-branch workflow contents were inspected: the earlier uncommitted SVD
workflow is absent there. This change replaces its local file rather than adding
a second schedule. No workflow, issue, PR or commit was published/dispatched.
Hosted Linux execution, a cold hosted build and failure-report writes remain
unrun. Eight offline runner/reporter checks cover empty/unclassified selection,
Schur-prefix separation, SVD calibration-count exclusion, failure/timeout
continuation, create/update filtering, cancellation and API errors.

### Final combined execution and adoption checks

The completed runner selects 325 tests: **297 pass, 28 fail**, with all selections
executed. All 136 numerical records match their frozen principal counts:
**71,856 principal calls and 50,952 dependent calls**. All ordinary calibrations
pass. The runner returns failure for the numerical findings; none is marked
expected success or hidden by a continue-on-error workflow step.

Build: **58.85s**, third-party dependencies cached and the changed optimized test
executable rebuilt. Execution: **207.13s (3m27s)**. These are local runtime
observations on the recorded macOS host, not speedup or hosted-Linux measurements.
The first full run also completed every test; a count check wrongly included four
SVD calibration names as numerical groups. An offline regression now rejects that
misclassification, and this final run confirms complete counts.

| Family | Principal calls | Dependent calls | Seconds | Result |
| --- | ---: | ---: | ---: | --- |
| common | 0 | 0 | 0.02 | pass |
| svd | 18,172 | 0 | 23.64 | failures retained |
| hessenberg | 2,904 | 2,904 | 7.89 | pass |
| schur | 5,808 | 0 | 17.69 | pass |
| nonsymmetric_evd | 7,200 | 0 | 6.00 | failures retained |
| schur_reorder | 3,132 | 0 | 0.04 | pass |
| balance | 1,624 | 1,616 | 0.02 | pass |
| tridiagonal_evd | 3,504 | 0 | 15.32 | failures retained |
| self_adjoint_evd | 5,256 | 1,752 | 31.23 | failures retained |
| lu | 2,256 | 5,776 | 43.94 | pass |
| cholesky | 1,712 | 4,648 | 14.55 | failures retained |
| indefinite | 3,376 | 7,840 | 24.68 | failures retained |
| qr | 4,656 | 23,280 | 16.25 | pass |
| triangular | 11,472 | 0 | 6.12 | failures retained |
| lstsq | 784 | 3,136 | 0.83 | pass |

The failing families of that run, and the status of their findings:

- SVD: the f32/c32 rank-one scratch panics (#73) and the f32 V orthogonality (#75)
  are fixed; the family passes.
- Nonsymmetric EVD: residual/normalization findings. The six c32 values-only
  mode discrepancies at the n=129 clustered fixture (sequential/2T/4T, both passes)
  were a checker-budget defect (#78): each mode's spectrum lies within 20 epsilons of
  an exact-input double reference, but their mutual gap reached 21. Mode-vs-mode
  agreement now uses the composed budget 2×20 at all four types; residual and
  unit-norm thresholds stay at 20.
- Tridiagonal EVD: 72 empty-complex, vector-producing call panics across both types.
- Dense self-adjoint EVD: the spectrum/mode comparisons in f64/f32/c32 came from the
  tridiagonal QR sweep's diagonal update (#76) and are fixed; the family passes. All
  reduction, reconstruction, AV−VΛ and unitary checks pass, and the added 192
  near-MAX/subnormal 2×2 calls pass at all types.
- Pivoted LLT: the first pivot is not compared with the rank tolerance; the n=1
  zero matrix reports rank one (four diagnostics/type) and larger zero matrices
  return NonPositivePivot instead of rank 0 (22 diagnostics/type).
- Pivoted LDL: stale nonzero subdiagonal metadata on no-update 1×1 pivots
  (178 diagnostics/type); initialized-buffer and parallel checks pass.
- Triangular solves: complex subnormal scalar pivots yield nonfinite answers
  (48 diagnostics/complex type). Minimal input is a 1×1 purely imaginary pivot
  i*MIN_POSITIVE/2 with identical RHS, whose exact solution is 1. Real cases pass.

Ordinary workspace validation: **423 pass, 3 fail, 138 ignored**. The failures are
the same pre-existing strict-triangle matmul tests (`empty_strict_triangle_preserves_parent_guards`,
`strict_triangle_maps_preserve_stored_entries_and_parent_guards`, and
`strict_triangle_preserves_nonempty_parent_guards`). Their source/test file hashes
are unchanged from the execution baseline. Remaining workspace libraries pass;
all **55 doctests pass** when run separately. New test files pass formatting;
workflow YAML/Bash and eight offline reporting/runner checks pass. Offline Cargo
package listing confirms all 28 test-source files and both license texts are shipped.

The source audit checks 148 pre-existing source/lock files. Exactly six files
have appended private cfg(test) hooks; every original byte remains intact. All
other checked files, including Cargo.lock and faer-traits sources, retain their
starting identities. The existing SVD module diff predates this execution. No
production-body change, dependency change, commit or external tracker mutation
was made. The scheduled job is intentionally red until its numerical findings
are resolved separately.

Detailed per-type counts and timings are in [the results JSON](lapack-validation-results.json).
[Complete local evidence](worklogs/evidence/2026-10-26-lapack-validation.tar.gz)
contains the full final logs, selection, build/environment, ordinary/doctest checks,
source audit and the high-precision eigenvector reference calculation.
