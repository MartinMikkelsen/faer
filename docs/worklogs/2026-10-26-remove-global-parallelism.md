# Remove the global parallelism

Acceptance condition 4 of `Par<'team>` (PR #61): the process-wide setting goes, planned in
`docs/par-team-migration.md` ("Global parallelism").

## What changed

- Removed `set_global_parallelism`, `get_global_parallelism`, `GlobalPar`,
  `disable_global_parallelism`, the faer-ffi `get_global_par` / `set_global_par` and their
  header and C++ wrapper entries, and the `GlobalPar` compile-fail doctest.
- The 97 readers take `par: Par` last (these methods have no
  stack argument; the low-level functions put `par` before the stack and params): constructors
  (`Llt::new(A, side, par)`, `Svd::new`, `Qr::new`, ...), the `Mat` methods (`llt`, `ldlt`, `lblt`,
  `qr`, `col_piv_qr`, `partial_piv_lu`, `full_piv_lu`, `svd`, `thin_svd`, `self_adjoint_eigen`,
  `self_adjoint_eigenvalues`, `singular_values`, `eigen`, `eigenvalues`, `generalized_eigen`,
  `determinant`, the four `solve_*_triangular_in_place`), the solver traits (`SolveCore`,
  `SolveLstsqCore`, `DenseSolveCore` and the `Solve` / `SolveLstsq` extension methods),
  `reconstruct`, `inverse`, `pseudoinverse`, `compute_Q`, `compute_thin_Q`, and the sparse
  `sp_cholesky`, `sp_lu`, `sp_qr`, `sp_solve_*_triangular_in_place`, `try_new_with_symbolic`.
- Operators (`&a * &b` and friends, dense and sparse) run with `Par::Seq`; documented on the
  module and the crate docs. `UnitaryMat::sample` runs its QR sequentially.

## Decision

A team is exclusive authority, so the value must travel with the call; the only alternative
(`GlobalPar`, a `Copy` subset) kept a hidden input to every high level method. The cost is a
breaking signature on all of them, accepted with the rest of the `Par<'team>` break.

## Unverified

No timing claim; sequential defaults for operators equal the old behaviour only when the old
global setting was `Seq`: operator products used the global rayon pool before.
