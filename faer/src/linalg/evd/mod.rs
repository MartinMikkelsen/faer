//! low level implementation of the eigenvalue decomposition of a square
//! diagonalizable matrix.
//!
//! the eigenvalue decomposition of a square matrix $A$ of shape $(n, n)$ is a
//! decomposition into two components $U$, $S$:
//!
//! - $U$ has shape $(n, n)$ and is invertible
//! - $S$ has shape $(n, n)$ and is a diagonal matrix
//! - and finally:
//!
//! $$A = U S U^{-1}$$
//!
//! if $A$ is self-adjoint, then $U$ can be made unitary ($U^{-1} = U^H$), and
//! $S$ is real valued
mod balance;
/// hessenberg decomposition
pub mod hessenberg;
#[doc(hidden)]
pub mod schur;
/// self-adjoint tridiagonalization
pub mod tridiag;
pub(crate) mod tridiag_evd;
use crate::assert;
use crate::internal_prelude::*;
pub use balance::{
	BalanceError, balance_eigenvectors_in_place, balance_eigenvectors_scratch,
	balance_in_place, balance_scratch,
};
use hessenberg::HessenbergParams;
use linalg::matmul::triangular::BlockStructure;
pub use schur::SchurParams;
use tridiag::TridiagParams;
/// eigendecomposition error
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EvdError {
	/// reached max iterations
	NoConvergence,
}
/// schur to eigendecomposition conversion parameters
#[derive(Clone, Copy, Debug)]
pub struct EvdFromSchurParams {
	/// threshold at which the implementation should stop recursing
	pub recursion_threshold: usize,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}
/// eigendecomposition tuning parameters
#[derive(Clone, Copy, Debug)]
pub struct EvdParams {
	/// hessenberg parameters
	pub hessenberg: HessenbergParams,
	/// schur from hessenberg conversion parameters
	pub schur: SchurParams,
	/// eigendecomposition from schur conversion parameters
	pub evd_from_schur: EvdFromSchurParams,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}
/// self-adjoint eigendecomposition tuning parameters
#[derive(Clone, Copy, Debug)]
pub struct SelfAdjointEvdParams {
	/// tridiagonalization parameters
	pub tridiag: TridiagParams,
	/// threshold at which the implementation should stop recursing
	pub recursion_threshold: usize,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}
impl<T: ComplexField> Auto<T> for EvdParams {
	fn auto() -> Self {
		Self {
			hessenberg: auto!(T),
			schur: auto!(T),
			evd_from_schur: auto!(T),
			non_exhaustive: NonExhaustive(()),
		}
	}
}
impl<T: ComplexField> Auto<T> for EvdFromSchurParams {
	fn auto() -> Self {
		Self {
			recursion_threshold: 64,
			non_exhaustive: NonExhaustive(()),
		}
	}
}
impl<T: ComplexField> Auto<T> for SelfAdjointEvdParams {
	fn auto() -> Self {
		Self {
			tridiag: auto!(T),
			recursion_threshold: 128,
			non_exhaustive: NonExhaustive(()),
		}
	}
}
/// whether the eigenvectors should be computed
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComputeEigenvectors {
	/// do not compute eigenvectors
	No,
	/// compute eigenvectors
	Yes,
}
/// computes the layout of the workspace required to compute a self-adjoint
/// matrix's eigendecomposition
pub fn self_adjoint_evd_scratch<T: ComplexField>(
	dim: usize,
	compute_u: ComputeEigenvectors,
	par: ParShape,
	params: Spec<SelfAdjointEvdParams, T>,
) -> StackReq {
	let n = dim;
	let bs = linalg::qr::no_pivoting::factor::recommended_block_size::<T>(n, n);
	let prologue = StackReq::all_of(&[
		temp_mat_scratch::<T>(n, n),
		temp_mat_scratch::<T>(bs, n),
		StackReq::any_of(&[tridiag::tridiag_in_place_scratch::<T>(
			n,
			par,
			params.tridiag.into(),
		)]),
		temp_mat_scratch::<T::Real>(n, 1).array(2),
	]);
	if compute_u == ComputeEigenvectors::No {
		return prologue;
	}
	StackReq::all_of(&[
		prologue,
		temp_mat_scratch::<T::Real>(
			n,
			if const {
				T::IS_REAL
			} {
				0
			} else {
				n
			},
		)
		.array(2),
		StackReq::any_of(&[
			if n < params.recursion_threshold {
				StackReq::empty()
			} else {
				tridiag_evd::divide_and_conquer_scratch::<T>(n, par)
			},
			temp_mat_scratch::<T>(n, 1),
			linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
				n.saturating_sub(1),
				bs,
				n,
			),
		]),
	])
}
/// Workspace for [`tridiagonal_self_adjoint_evd`].
///
/// # Examples
/// ```
/// use faer::Par;
/// use faer::linalg::evd::*;
/// let _buffer =
/// 	dyn_stack::MemBuffer::new(tridiagonal_self_adjoint_evd_scratch::<f64>(
/// 		3,
/// 		ComputeEigenvectors::Yes,
/// 		faer::ParShape::Seq,
/// 		Default::default(),
/// 	));
/// ```
pub fn tridiagonal_self_adjoint_evd_scratch<T: ComplexField>(
	dim: usize,
	compute_u: ComputeEigenvectors,
	par: ParShape,
	params: Spec<SelfAdjointEvdParams, T>,
) -> StackReq {
	let n = dim;
	StackReq::all_of(&[
		temp_mat_scratch::<T::Real>(n, 1).array(2),
		if compute_u == ComputeEigenvectors::No {
			StackReq::EMPTY
		} else {
			temp_mat_scratch::<T::Real>(n, if T::IS_REAL { 0 } else { n }).and(
				temp_mat_scratch::<T>(n, 1).or(
					if n < params.recursion_threshold {
						StackReq::EMPTY
					} else {
						tridiag_evd::divide_and_conquer_scratch::<T::Real>(
							n, par,
						)
					},
				),
			)
		},
	])
}
/// computes the matrix $A$'s eigendecomposition, assuming it is tridiagonal and
/// self-adjoint
///
/// the eigenvalues are stored in $S$, and the eigenvectors in $U$ such that the
/// eigenvalues are sorted in nondecreasing order
pub fn tridiagonal_self_adjoint_evd<T: ComplexField>(
	diag: DiagRef<'_, T>,
	subdiag: DiagRef<'_, T>,
	s: DiagMut<'_, T>,
	u: Option<MatMut<'_, T>>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<SelfAdjointEvdParams, T>,
) -> Result<(), EvdError> {
	let n = diag.dim();
	if n == 0 {
		return Ok(());
	}
	let (mut real_diag, stack) =
		unsafe { temp_mat_uninit::<T::Real, _, _>(n, 1, stack) };
	let (mut real_offdiag, stack) =
		unsafe { temp_mat_uninit::<T::Real, _, _>(n, 1, stack) };
	let mut real_diag = real_diag
		.as_mat_mut()
		.col_mut(0)
		.try_as_col_major_mut()
		.unwrap();
	let mut real_offdiag = real_offdiag
		.as_mat_mut()
		.col_mut(0)
		.try_as_col_major_mut()
		.unwrap();
	for i in 0..n {
		real_diag[i] = diag[i].real();
		if i + 1 < n {
			if const { T::IS_REAL } {
				real_offdiag[i] = subdiag[i].real();
			} else {
				real_offdiag[i] = subdiag[i].abs();
			}
		} else {
			real_offdiag[i] = zero();
		}
	}
	let mut s = s;
	let mut u = match u {
		Some(u) => u,
		None => {
			tridiag_evd::qr_algorithm(
				real_diag.rb_mut(),
				real_offdiag.rb_mut(),
				None,
			)?;
			for i in 0..n {
				s[i] = real_diag[i].to_cplx();
			}
			return Ok(());
		},
	};
	let (mut u_real, stack) = unsafe {
		temp_mat_uninit::<T::Real, _, _>(
			n,
			if T::IS_REAL { 0 } else { n },
			stack,
		)
	};
	let mut u_real = u_real.as_mat_mut();
	let mut u_evd = if const { T::IS_REAL } {
		unsafe { core::mem::transmute(u.rb_mut()) }
	} else {
		u_real.rb_mut()
	};
	if n < params.recursion_threshold {
		tridiag_evd::qr_algorithm(
			real_diag.rb_mut(),
			real_offdiag.rb_mut(),
			Some(u_evd.rb_mut()),
		)?;
	} else {
		tridiag_evd::divide_and_conquer::<T::Real>(
			real_diag.rb_mut(),
			real_offdiag.rb_mut(),
			u_evd.rb_mut(),
			par,
			stack,
			params.recursion_threshold,
		)?;
	}
	if const { !T::IS_REAL } {
		let normalized = |x: T| {
			if x == zero() {
				one()
			} else {
				x.mul_real(x.abs().recip())
			}
		};
		let (mut scale, _) = unsafe { temp_mat_uninit::<T, _, _>(n, 1, stack) };
		let mut scale = scale
			.as_mat_mut()
			.col_mut(0)
			.try_as_col_major_mut()
			.unwrap();
		let mut x = one::<T>();
		scale[0] = one();
		for i in 1..n {
			x = normalized(&subdiag[i - 1] * x);
			scale[i] = x.copy();
		}
		for j in 0..n {
			z!(u.rb_mut().col_mut(j), u_real.rb().col(j), scale.rb()).for_each(
				|uz!(u, real, scale)| {
					*u = scale.mul_real(real);
				},
			);
		}
	}
	for i in 0..n {
		s[i] = real_diag[i].to_cplx();
	}
	Ok(())
}
/// computes the matrix $A$'s eigendecomposition, assuming it is self-adjoint
///
/// the eigenvalues are stored in $S$, and the eigenvectors in $U$ such that the
/// eigenvalues are sorted in nondecreasing order
///
/// only the lower triangular half of $A$ is accessed
pub fn self_adjoint_evd<T: ComplexField>(
	A: MatRef<'_, T>,
	s: DiagMut<'_, T>,
	u: Option<MatMut<'_, T>>,
	mut par: Par,
	stack: &mut MemStack,
	params: Spec<SelfAdjointEvdParams, T>,
) -> Result<(), EvdError> {
	let n = A.nrows();
	assert!(all(A.nrows() == A.ncols(), s.dim() == n));
	if let Some(u) = u.rb() {
		assert!(all(u.nrows() == n, u.ncols() == n));
	}
	let s = s.column_vector_mut();
	if n == 0 {
		return Ok(());
	}
	#[cfg(feature = "perf-warn")]
	if let Some(matrix) = u.rb() {
		if matrix.row_stride().unsigned_abs() != 1
			&& crate::__perf_warn!(QR_WARN)
		{
			if matrix.col_stride().unsigned_abs() == 1 {
				log::warn!(
					target : "faer_perf",
					"EVD prefers column-major eigenvector matrix. Found row-major matrix."
				);
			} else {
				log::warn!(
					target : "faer_perf",
					"EVD prefers column-major eigenvector matrix. Found matrix with generic strides."
				);
			}
		}
	}
	let (mut trid, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
	let mut trid = trid.as_mat_mut();
	trid.copy_from_triangular_lower(A);
	let bs = linalg::qr::no_pivoting::factor::recommended_block_size::<T>(n, n);
	let (mut householder, stack) =
		unsafe { temp_mat_uninit::<T, _, _>(bs, n - 1, stack) };
	let mut householder = householder.as_mat_mut();
	{
		tridiag::tridiag_in_place(
			trid.rb_mut(),
			householder.rb_mut(),
			par.reborrow(),
			stack,
			params.tridiag.into(),
		);
	}
	let trid = trid.rb();
	let (mut diag, stack) =
		unsafe { temp_mat_uninit::<T::Real, _, _>(n, 1, stack) };
	let (mut offdiag, stack) =
		unsafe { temp_mat_uninit::<T::Real, _, _>(n, 1, stack) };
	let mut diag = diag.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap();
	let mut offdiag = offdiag
		.as_mat_mut()
		.col_mut(0)
		.try_as_col_major_mut()
		.unwrap();
	for i in 0..n {
		diag[i] = trid[(i, i)].real();
		if i + 1 < n {
			if const { T::IS_REAL } {
				offdiag[i] = trid[(i + 1, i)].real();
			} else {
				offdiag[i] = trid[(i + 1, i)].abs();
			}
		} else {
			offdiag[i] = zero();
		}
	}
	let mut s = s;
	let mut u = match u {
		Some(u) => u,
		None => {
			tridiag_evd::qr_algorithm(diag.rb_mut(), offdiag.rb_mut(), None)?;
			for i in 0..n {
				s[i] = diag[i].to_cplx();
			}
			return Ok(());
		},
	};
	let (mut u_real, stack) = unsafe {
		temp_mat_uninit::<T::Real, _, _>(
			n,
			if T::IS_REAL { 0 } else { n },
			stack,
		)
	};
	let mut u_real = u_real.as_mat_mut();
	let mut u_evd = if const { T::IS_REAL } {
		unsafe { core::mem::transmute(u.rb_mut()) }
	} else {
		u_real.rb_mut()
	};
	if n < params.recursion_threshold {
		tridiag_evd::qr_algorithm(
			diag.rb_mut(),
			offdiag.rb_mut(),
			Some(u_evd.rb_mut()),
		)?;
	} else {
		tridiag_evd::divide_and_conquer::<T::Real>(
			diag.rb_mut(),
			offdiag.rb_mut(),
			u_evd.rb_mut(),
			par.reborrow(),
			stack,
			params.recursion_threshold,
		)?;
	}
	if const { !T::IS_REAL } {
		let normalized = |x: T| {
			if x == zero() {
				one()
			} else {
				x.mul_real(x.abs().recip())
			}
		};
		let (mut scale, _) = unsafe { temp_mat_uninit::<T, _, _>(n, 1, stack) };
		let mut scale = scale
			.as_mat_mut()
			.col_mut(0)
			.try_as_col_major_mut()
			.unwrap();
		let mut x = one::<T>();
		scale[0] = one();
		for i in 1..n {
			x = normalized(&trid[(i, i - 1)] * x);
			scale[i] = x.copy();
		}
		for j in 0..n {
			z!(u.rb_mut().col_mut(j), u_real.rb().col(j), scale.rb()).for_each(
				|uz!(u, real, scale)| {
					*u = scale.mul_real(real);
				},
			);
		}
	}
	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
		trid.submatrix(1, 0, n - 1, n - 1),
		householder.rb(),
		Conj::No,
		u.rb_mut().subrows_mut(1, n - 1),
		par,
		stack,
	);
	for i in 0..n {
		s[i] = diag[i].to_cplx();
	}
	Ok(())
}
/// computes the layout of the workspace required to compute a self-adjoint
/// matrix's pseudoinverse, given the eigendecomposition
pub fn pseudoinverse_from_self_adjoint_evd_scratch<T: ComplexField>(
	dim: usize,
	par: ParShape,
) -> StackReq {
	_ = par;
	temp_mat_scratch::<T>(dim, dim).array(2)
}
/// computes a self-adjoint matrix's pseudoinverse, given the eigendecomposition
/// factors $S$ and $U$
#[track_caller]
pub fn pseudoinverse_from_self_adjoint_evd<T: ComplexField>(
	pinv: MatMut<'_, T>,
	s: DiagRef<'_, T>,
	u: MatRef<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	pseudoinverse_from_self_adjoint_evd_with_tolerance(
		pinv,
		s,
		u,
		zero(),
		eps::<T::Real>() * from_f64::<T::Real>(u.ncols() as f64),
		par,
		stack,
	);
}
/// computes a self-adjoint matrix's pseudoinverse, given the eigendecomposition
/// factors $S$ and $U$, and tolerance parameters for determining zero
/// eigenvalues
#[track_caller]
pub fn pseudoinverse_from_self_adjoint_evd_with_tolerance<T: ComplexField>(
	pinv: MatMut<'_, T>,
	s: DiagRef<'_, T>,
	u: MatRef<'_, T>,
	abs_tol: T::Real,
	rel_tol: T::Real,
	par: Par,
	stack: &mut MemStack,
) {
	let s = s.column_vector();
	let mut pinv = pinv;
	let n = u.ncols();
	assert!(all(u.nrows() == n, u.ncols() == n, s.nrows() == n));
	let smax = s.norm_max();
	let tol = abs_tol.fmax(rel_tol * smax);
	let (mut u_trunc, stack) =
		unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
	let (mut up_trunc, _) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
	let mut u_trunc = u_trunc.as_mat_mut();
	let mut up_trunc = up_trunc.as_mat_mut();
	let mut len = 0;
	for j in 0..n {
		let x = s[j].absmax();
		if x > tol {
			let ref p = s[j].real().recip();
			u_trunc.rb_mut().col_mut(len).copy_from(u.col(j));
			z!(up_trunc.rb_mut().col_mut(len), u.col(j))
				.for_each(|uz!(dst, src)| *dst = src.mul_real(p));
			len += 1;
		}
	}
	let u_trunc = u_trunc.get(.., ..len);
	let up_trunc = up_trunc.get(.., ..len);
	linalg::matmul::triangular::matmul(
		pinv.rb_mut(),
		BlockStructure::TriangularLower,
		Accum::Replace,
		up_trunc.rb(),
		BlockStructure::Rectangular,
		u_trunc.rb().adjoint(),
		BlockStructure::Rectangular,
		one(),
		par,
	);
	for j in 0..n {
		for i in 0..j {
			pinv[(i, j)] = pinv[(j, i)].conj();
		}
	}
}
fn dot2x1<T: RealField>(
	lhs0: RowRef<'_, T>,
	lhs1: RowRef<'_, T>,
	rhs: ColRef<'_, T>,
) -> (T, T) {
	let n = rhs.nrows();
	assert!(all(lhs0.ncols() == n, lhs1.ncols() == n));
	let mut acc00 = zero::<T>();
	let mut acc01 = zero::<T>();
	let mut acc10 = zero::<T>();
	let mut acc11 = zero::<T>();
	let n2 = n / 2 * 2;
	let mut i = 0;
	while i < n2 {
		acc00 += &lhs0[i] * &rhs[i];
		acc10 += &lhs1[i] * &rhs[i];
		acc01 += &lhs0[i + 1] * &rhs[i + 1];
		acc11 += &lhs1[i + 1] * &rhs[i + 1];
		i += 2;
	}
	while i < n {
		acc00 += &lhs0[i] * &rhs[i];
		acc10 += &lhs1[i] * &rhs[i];
		i += 1;
	}
	(acc00 + acc01, acc10 + acc11)
}
fn dot2x2<T: RealField>(
	lhs0: RowRef<'_, T>,
	lhs1: RowRef<'_, T>,
	rhs0: ColRef<'_, T>,
	rhs1: ColRef<'_, T>,
) -> (T, T, T, T) {
	let n = rhs0.nrows();
	assert!(all(lhs0.ncols() == n, lhs1.ncols() == n));
	let mut acc0 = zero::<T>();
	let mut acc1 = zero::<T>();
	let mut acc2 = zero::<T>();
	let mut acc3 = zero::<T>();
	let mut i = 0;
	while i < n {
		acc0 += &lhs0[i] * &rhs0[i];
		acc1 += &lhs1[i] * &rhs0[i];
		acc2 += &lhs0[i] * &rhs1[i];
		acc3 += &lhs1[i] * &rhs1[i];
		i += 1;
	}
	(acc0, acc1, acc2, acc3)
}
/// Normalizes the eigenvector columns of `u` to unit 2-norm. For a real Schur
/// form `h`, a conjugate pair `j`, `j + 1` (`h[(j + 1, j)] != 0`) is scaled by a
/// common factor so the represented complex eigenvector `re + i * im` has unit
/// 2-norm. This is the normalization contract `xGEEV` provides.
fn normalize_eigenvectors<T: ComplexField>(
	mut u: MatMut<'_, T>,
	h: MatRef<'_, T>,
) {
	let n = u.nrows();
	let mut j = 0;
	while j < n {
		let paired = T::IS_REAL && j + 1 < n && h[(j + 1, j)] != zero::<T>();
		let nrm = if paired {
			u.rb().col(j).norm_l2().hypot(&u.rb().col(j + 1).norm_l2())
		} else {
			u.rb().col(j).norm_l2()
		};
		if nrm != zero::<T::Real>() {
			let inv: T = nrm.recip().to_cplx();
			for i in 0..n {
				u[(i, j)] *= inv.copy();
				if paired {
					u[(i, j + 1)] *= inv.copy();
				}
			}
		}
		j += if paired { 2 } else { 1 };
	}
}
/// Returns `false` if a returned eigenvector column is nonfinite or
/// degenerate, which means the normalization contract could not be met.
fn eigenvectors_ok<T: ComplexField>(u: MatRef<'_, T>) -> bool {
	u.is_all_finite()
		&& (0..u.ncols()).all(|j| u.col(j).norm_l2() > zero::<T::Real>())
}
/// The shift clamp `eps * sum |a_ij|` of the back substitution, summed over the
/// upper triangle and the `subdiag` subdiagonals below it (`1` for a real
/// Schur form, `0` for a triangular one).
///
/// The plain sum overflows to infinity for a finite matrix whose entries are
/// near the top of the range, which would clamp every shift to infinity and
/// return a wrong direction with `Ok`. `eps * |a_ij|` does not overflow there,
/// so the sum is then recomputed term by term. It is only used in that case,
/// because summing the scaled terms rounds them to zero for subnormal entries
/// where the plain sum still gives a nonzero clamp.
fn eps_norm<T: ComplexField>(A: MatRef<'_, T>, subdiag: usize) -> T::Real {
	let n = A.nrows();
	let eps = eps::<T::Real>();
	let rows = |j: usize| Ord::min(j + 1 + subdiag, n);
	let mut norm = zero::<T::Real>();
	for j in 0..n {
		for i in 0..rows(j) {
			norm += A[(i, j)].abs1();
		}
	}
	let mut eps_norm = &eps * &norm;
	if !eps_norm.is_finite() {
		let mut acc = zero::<T::Real>();
		for j in 0..n {
			for i in 0..rows(j) {
				let a = &A[(i, j)];
				// `abs1` is `|re| + |im|`, which overflows on its own for a
				// complex entry with both parts large, so scale them apart
				acc += &eps * &a.real().abs() + &eps * &a.imag().abs();
			}
		}
		eps_norm = acc;
	}
	eps_norm
}
pub(crate) fn evd_from_real_schur_imp<T: RealField>(
	A: MatRef<'_, T>,
	V: MatMut<'_, T>,
	mut par: Par,
	params: EvdFromSchurParams,
) {
	let one = one::<T>;
	let zero = zero::<T>;
	let mut V = V;
	let n = A.nrows();
	let eps_norm = eps_norm(A, 1);
	let mut k = n;
	loop {
		if k == 0 {
			break;
		}
		k -= 1;
		if k == 0 || A[(k, k - 1)] == zero() {
			let ref p = A[(k, k)].copy();
			V[(k, k)] = one();
			for i in 0..k {
				V[(i, k)] = -&A[(i, k)];
			}
			let scale = solve_real_shifted_upper_quasi_triangular_system(
				A.submatrix(0, 0, k, k),
				p.copy(),
				V.rb_mut().subrows_mut(0, k).col_mut(k),
				eps_norm.copy(),
				par.reborrow(),
				params,
			);
			// The pivot is not part of the back-substituted slice, so it must
			// carry the accumulated rescaling factor for the column direction
			// to be preserved.
			V[(k, k)] *= scale;
		} else {
			let ref p = A[(k, k)].copy();
			let ref q = A[(k, k - 1)].abs().sqrt() * A[(k - 1, k)].abs().sqrt();
			if A[(k - 1, k)].abs() >= A[(k, k - 1)].abs() {
				V[(k - 1, k - 1)] = one();
				V[(k, k)] = q / &A[(k - 1, k)];
			} else {
				V[(k - 1, k - 1)] = -q / &A[(k, k - 1)];
				V[(k, k)] = one();
			}
			V[(k - 1, k)] = zero();
			V[(k, k - 1)] = zero();
			for i in 0..k - 1 {
				V[(i, k - 1)] = -&V[(k - 1, k - 1)] * &A[(i, k - 1)];
				V[(i, k)] = -&V[(k, k)] * &A[(i, k)];
			}
			let scale = solve_cplx_shifted_upper_quasi_triangular_system(
				A.submatrix(0, 0, k - 1, k - 1),
				p.copy(),
				q.copy(),
				V.rb_mut().submatrix_mut(0, k - 1, k - 1, 2),
				eps_norm.copy(),
				par.reborrow(),
				params,
			);
			V[(k - 1, k - 1)] *= scale.copy();
			V[(k, k)] *= scale;
			k -= 1;
		}
	}
}
pub(crate) fn evd_from_cplx_schur_imp<T: ComplexField>(
	A: MatRef<'_, T>,
	conj_A: Conj,
	V: MatMut<'_, T>,
	mut par: Par,
	params: EvdFromSchurParams,
) {
	let one = one::<T>;
	let mut V = V;
	let n = A.nrows();
	let eps_norm = eps_norm(A, 0);
	let mut k = n;
	loop {
		if k == 0 {
			break;
		}
		k -= 1;
		let ref p = if conj_A == Conj::Yes {
			A[(k, k)].conj()
		} else {
			A[(k, k)].copy()
		};
		V[(k, k)] = one();
		if conj_A == Conj::Yes {
			for i in 0..k {
				V[(i, k)] = -A[(i, k)].conj();
			}
		} else {
			for i in 0..k {
				V[(i, k)] = -&A[(i, k)];
			}
		}
		let scale = solve_shifted_upper_triangular_system(
			A.submatrix(0, 0, k, k),
			conj_A,
			p.copy(),
			V.rb_mut().subrows_mut(0, k).col_mut(k),
			eps_norm.copy(),
			par.reborrow(),
			params,
		);
		let scale: T = scale.to_cplx();
		V[(k, k)] *= scale;
	}
}
fn solve_real_shifted_upper_quasi_triangular_system<T: RealField>(
	A: MatRef<'_, T>,
	p: T,
	x: ColMut<'_, T>,
	eps_norm: T,
	mut par: Par,
	params: EvdFromSchurParams,
) -> T {
	let n = A.nrows();
	let one = one::<T>;
	let zero = zero::<T>;
	let ref p = p;
	let mut x = x;
	// While the shift `A[i, i] - p` is clamped to `eps_norm`, back
	// substitution can grow the solution by `1 / eps_norm` per row and
	// overflow for finite input. Whenever a computed entry exceeds this bound
	// the whole slice is rescaled; the factor is also returned so the caller
	// can apply it to the pivot entry, keeping the eigenvector direction.
	let big = sqrt_max_positive::<T>();
	let mut scale = one();
	if par.degree() == 0 || n < params.recursion_threshold {
		let mut i = n;
		loop {
			if i == 0 {
				break;
			}
			i -= 1;
			if i == 0 || A[(i, i - 1)] == zero() {
				let dot = linalg::matmul::dot::inner_prod(
					A.row(i).subcols(i + 1, n - i - 1),
					Conj::No,
					x.rb().subrows(i + 1, n - i - 1),
					Conj::No,
				);
				x[i] -= dot;
				let mut z = &A[(i, i)] - p;
				if z.abs() < eps_norm {
					z = eps_norm.copy();
				}
				if x[i] != zero() {
					x[i] /= z;
				}
				let m = x[i].abs();
				if m > big {
					let r = m.recip();
					for j in 0..x.nrows() {
						x[j] *= r.copy();
					}
					scale *= r;
				}
			} else {
				let (dot0, dot1) = dot2x1(
					A.row(i - 1).subcols(i + 1, n - i - 1),
					A.row(i).subcols(i + 1, n - i - 1),
					x.rb().subrows(i + 1, n - i - 1),
				);
				x[i - 1] -= dot0;
				x[i] -= dot1;
				let ref a = &A[(i, i)] - p;
				let ref b = A[(i - 1, i)];
				let ref c = A[(i, i - 1)];
				let ref r0 = x[i - 1];
				let ref r1 = x[i];
				let ref inv_det = (a.abs2() - b * c).recip();
				(x[i - 1], x[i]) =
					((a * r0 - b * r1) * inv_det, (a * r1 - c * r0) * inv_det);
				let a0 = x[i - 1].abs();
				let a1 = x[i].abs();
				let m = if a0 > a1 { a0 } else { a1 };
				if m > big {
					let r = m.recip();
					for j in 0..x.nrows() {
						x[j] *= r.copy();
					}
					scale *= r;
				}
				i -= 1;
			}
		}
		scale
	} else {
		let mut mid = n / 2;
		if A[(mid, mid - 1)] != zero() {
			mid -= 1;
		}
		let (A00, A01, _, A11) = A.split_at(mid, mid);
		let (mut x0, mut x1) = x.rb_mut().split_at_row_mut(mid);
		let f1 = solve_real_shifted_upper_quasi_triangular_system(
			A11,
			p.copy(),
			x1.rb_mut(),
			eps_norm.copy(),
			par.reborrow(),
			params,
		);
		// The right block may have been rescaled; the left block's pending
		// right-hand side must carry the same factor for the coupled system to
		// stay consistent.
		for j in 0..x0.nrows() {
			x0[j] *= f1.copy();
		}
		linalg::matmul::matmul(
			x0.rb_mut().as_mat_mut(),
			Accum::Add,
			A01,
			x1.rb().as_mat(),
			-one(),
			par.reborrow(),
		);
		let f0 = solve_real_shifted_upper_quasi_triangular_system(
			A00,
			p.copy(),
			x0.rb_mut(),
			eps_norm,
			par,
			params,
		);
		for j in 0..x1.nrows() {
			x1[j] *= f0.copy();
		}
		f0 * f1
	}
}
fn solve_cplx_shifted_upper_quasi_triangular_system<T: RealField>(
	A: MatRef<'_, T>,
	p: T,
	q: T,
	x: MatMut<'_, T>,
	eps_norm: T,
	mut par: Par,
	params: EvdFromSchurParams,
) -> T {
	let n = A.nrows();
	let one = one::<T>;
	let zero = zero::<T>;
	let ref p = p;
	let ref q = q;
	let mut x = x;
	let big = sqrt_max_positive::<T>();
	let mut scale = one();
	if par.degree() == 0 || n < params.recursion_threshold {
		let mut i = n;
		loop {
			if i == 0 {
				break;
			}
			i -= 1;
			if i == 0 || A[(i, i - 1)] == zero() {
				let (re, im) = dot2x1(
					x.rb().subrows(i + 1, n - i - 1).col(0).transpose(),
					x.rb().subrows(i + 1, n - i - 1).col(1).transpose(),
					A.row(i).subcols(i + 1, n - i - 1).transpose(),
				);
				x[(i, 0)] = &x[(i, 0)] - re;
				x[(i, 1)] = &x[(i, 1)] - im;
				let mut z_re = &A[(i, i)] - p;
				let mut z_im = -q;
				let mut z = z_re.hypot(&z_im);
				if z < eps_norm {
					z_re = eps_norm.copy();
					z_im = zero();
					z = z_re.copy();
				}
				let ref z = z;
				let ref z_re = (z_re / z) / z;
				let ref z_im = (-z_im / z) / z;
				let ref x_re = x[(i, 0)].copy();
				let ref x_im = x[(i, 1)].copy();
				x[(i, 0)] = x_re * z_re - x_im * z_im;
				x[(i, 1)] = x_re * z_im + x_im * z_re;
				let a0 = x[(i, 0)].abs();
				let a1 = x[(i, 1)].abs();
				let m = if a0 > a1 { a0 } else { a1 };
				if m > big {
					let r = m.recip();
					for j in 0..x.nrows() {
						x[(j, 0)] *= r.copy();
						x[(j, 1)] *= r.copy();
					}
					scale *= r;
				}
			} else {
				let (re0, re1, im0, im1) = dot2x2(
					A.row(i - 1).subcols(i + 1, n - i - 1),
					A.row(i).subcols(i + 1, n - i - 1),
					x.rb().col(0).subrows(i + 1, n - i - 1),
					x.rb().col(1).subrows(i + 1, n - i - 1),
				);
				x[(i - 1, 0)] = &x[(i - 1, 0)] - re0;
				x[(i - 1, 1)] = &x[(i - 1, 1)] - im0;
				x[(i, 0)] = &x[(i, 0)] - re1;
				x[(i, 1)] = &x[(i, 1)] - im1;
				let ref a_re = &A[(i, i)] - p;
				let ref a_im = -q;
				let ref b = A[(i - 1, i)];
				let ref c = A[(i, i - 1)];
				let ref r0_re = x[(i - 1, 0)];
				let ref r0_im = x[(i - 1, 1)];
				let ref r1_re = x[(i, 0)];
				let ref r1_im = x[(i, 1)];
				let mut z_re = a_re.abs2() - a_im.abs2() - b * c;
				let mut z_im = (a_re * a_im).mul_pow2(from_f64::<T>(2.0));
				let mut z = z_re.hypot(&z_im);
				if z < eps_norm {
					z_re = eps_norm.copy();
					z_im = zero();
					z = z_re.copy();
				}
				let ref z = z;
				let ref z_re = (z_re / z) / z;
				let ref z_im = (-z_im / z) / z;
				let ref x0_re = (a_re * r0_re - a_im * r0_im) - b * r1_re;
				let ref x0_im = (a_re * r0_im + a_im * r0_re) - b * r1_im;
				let ref x1_re = (a_re * r1_re - a_im * r1_im) - c * r0_re;
				let ref x1_im = (a_re * r1_im + a_im * r1_re) - c * r0_im;
				x[(i - 1, 0)] = x0_re * z_re - x0_im * z_im;
				x[(i - 1, 1)] = x0_re * z_im + x0_im * z_re;
				x[(i, 0)] = x1_re * z_re - x1_im * z_im;
				x[(i, 1)] = x1_re * z_im + x1_im * z_re;
				let mut m = x[(i - 1, 0)].abs();
				let v = x[(i - 1, 1)].abs();
				if v > m {
					m = v;
				}
				let v = x[(i, 0)].abs();
				if v > m {
					m = v;
				}
				let v = x[(i, 1)].abs();
				if v > m {
					m = v;
				}
				if m > big {
					let r = m.recip();
					for j in 0..x.nrows() {
						x[(j, 0)] *= r.copy();
						x[(j, 1)] *= r.copy();
					}
					scale *= r;
				}
				i -= 1;
			}
		}
		scale
	} else {
		let mut mid = n / 2;
		if A[(mid, mid - 1)] != zero() {
			mid -= 1;
		}
		let (A00, A01, _, A11) = A.split_at(mid, mid);
		let (mut x0, mut x1) = x.rb_mut().split_at_row_mut(mid);
		let f1 = solve_cplx_shifted_upper_quasi_triangular_system(
			A11,
			p.copy(),
			q.copy(),
			x1.rb_mut(),
			eps_norm.copy(),
			par.reborrow(),
			params,
		);
		for j in 0..x0.nrows() {
			x0[(j, 0)] *= f1.copy();
			x0[(j, 1)] *= f1.copy();
		}
		linalg::matmul::matmul(
			x0.rb_mut(),
			Accum::Add,
			A01,
			x1.rb(),
			-one(),
			par.reborrow(),
		);
		let f0 = solve_cplx_shifted_upper_quasi_triangular_system(
			A00,
			p.copy(),
			q.copy(),
			x0.rb_mut(),
			eps_norm,
			par,
			params,
		);
		for j in 0..x1.nrows() {
			x1[(j, 0)] *= f0.copy();
			x1[(j, 1)] *= f0.copy();
		}
		f0 * f1
	}
}
fn solve_shifted_upper_triangular_system<T: ComplexField>(
	A: MatRef<'_, T>,
	conj_A: Conj,
	p: T,
	x: ColMut<'_, T>,
	eps_norm: T::Real,
	mut par: Par,
	params: EvdFromSchurParams,
) -> T::Real {
	let n = A.nrows();
	let mut scale = one::<T::Real>();
	let one = one::<T>;
	let zero = zero::<T>;
	let ref p = p;
	let mut x = x;
	let big = sqrt_max_positive::<T::Real>();
	if par.degree() == 0 || n < params.recursion_threshold {
		let mut i = n;
		loop {
			if i == 0 {
				break;
			}
			i -= 1;
			let dot = linalg::matmul::dot::inner_prod(
				A.row(i).subcols(i + 1, n - i - 1),
				conj_A,
				x.rb().subrows(i + 1, n - i - 1),
				Conj::No,
			);
			x[i] -= dot;
			let mut z = if conj_A == Conj::Yes {
				A[(i, i)].conj()
			} else {
				A[(i, i)].copy()
			} - p;
			if z.abs() < eps_norm {
				z = eps_norm.to_cplx();
			}
			if x[i] != zero() {
				x[i] *= z.recip();
			}
			let m = x[i].abs();
			if m > big {
				let r = m.recip();
				let rc: T = r.to_cplx();
				for j in 0..x.nrows() {
					x[j] *= rc.copy();
				}
				scale *= r;
			}
		}
		scale
	} else {
		let mid = n / 2;
		let (A00, A01, _, A11) = A.split_at(mid, mid);
		let (mut x0, mut x1) = x.rb_mut().split_at_row_mut(mid);
		let f1 = solve_shifted_upper_triangular_system(
			A11,
			conj_A,
			p.copy(),
			x1.rb_mut(),
			eps_norm.copy(),
			par.reborrow(),
			params,
		);
		let f1c: T = f1.to_cplx();
		for j in 0..x0.nrows() {
			x0[j] *= f1c.copy();
		}
		linalg::matmul::matmul_with_conj(
			x0.rb_mut().as_mat_mut(),
			Accum::Add,
			A01,
			conj_A,
			x1.rb().as_mat(),
			Conj::No,
			-one(),
			par.reborrow(),
		);
		let f0 = solve_shifted_upper_triangular_system(
			A00,
			conj_A,
			p.copy(),
			x0.rb_mut(),
			eps_norm,
			par,
			params,
		);
		let f0c: T = f0.to_cplx();
		for j in 0..x1.nrows() {
			x1[j] *= f0c.copy();
		}
		f0 * f1
	}
}
/// computes the layout of the workspace required to compute a matrix's
/// eigendecomposition
pub fn evd_scratch<T: ComplexField>(
	dim: usize,
	eigen_left: ComputeEigenvectors,
	eigen_right: ComputeEigenvectors,
	par: ParShape,
	params: Spec<EvdParams, T>,
) -> StackReq {
	let n = dim;
	if n == 0 {
		return StackReq::EMPTY;
	}
	let compute_eigen = eigen_left == ComputeEigenvectors::Yes
		|| eigen_right == ComputeEigenvectors::Yes;
	let bs = linalg::qr::no_pivoting::factor::recommended_block_size::<T>(
		n - 1,
		n - 1,
	);
	let H = temp_mat_scratch::<T>(n, n);
	let X = H;
	let Z = temp_mat_scratch::<T>(n, if compute_eigen { n } else { 0 });
	let householder = temp_mat_scratch::<T>(bs, n);
	let apply = linalg::householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<T>(n - 1, bs, n - 1);
	StackReq::all_of(&[
		H,
		Z,
		StackReq::any_of(&[
			householder.and(
				hessenberg::hessenberg_in_place_scratch::<T>(
					n,
					bs,
					par,
					params.hessenberg.into(),
				)
				.or(apply),
			),
			schur::multishift_qr_scratch::<T>(
				n,
				n,
				compute_eigen,
				compute_eigen,
				par,
				params.schur,
			),
			X,
		]),
	])
}
/// Workspace for [`hessenberg_schur_real`] and
/// [`hessenberg_schur_cplx`].
///
/// # Examples
/// ```
/// use faer::Par;
/// use faer::linalg::evd::hessenberg_schur_scratch;
/// let req =
/// 	hessenberg_schur_scratch::<f64>(3, true, faer::ParShape::Seq, Default::default());
/// let _buffer = dyn_stack::MemBuffer::new(req);
/// ```
pub fn hessenberg_schur_scratch<T: ComplexField>(
	dim: usize,
	compute_z: bool,
	par: ParShape,
	params: Spec<SchurParams, T>,
) -> StackReq {
	schur::multishift_qr_scratch::<T>(dim, dim, true, compute_z, par, *params)
}
/// Overwrites an upper-Hessenberg `h` with its real Schur form `t`.
/// Updates optional `z` to `z * q`, where `h_original = q * t * q.transpose()`.
/// Real conjugate pairs have equal diagonal entries, opposite-sign
/// off-diagonals, and eigenvalues `wr ± i wi`, with the positive imaginary part
/// first. `wr` and `wi` each have length `n`; `z`, when present, is `n` by `n`.
///
/// # Panics
/// Panics on incompatible dimensions or insufficient scratch. The input must
/// be upper Hessenberg (entries below its first subdiagonal must be zero).
///
/// # Errors
/// Returns [`EvdError::NoConvergence`] for nonfinite input or QR
/// nonconvergence. After nonconvergence, `h` and `z` contain the accumulated
/// similarity updates.
///
/// # Examples
/// ```
/// use dyn_stack::{MemBuffer, MemStack};
/// use faer::linalg::evd::*;
/// use faer::{Col, Mat, Par};
/// let mut h = faer::mat![[1.0f64, 2.0], [-2.0, 1.0]];
/// let mut z = Mat::identity(2, 2);
/// let (mut wr, mut wi) = (Col::zeros(2), Col::zeros(2));
/// let mut mem = MemBuffer::new(hessenberg_schur_scratch::<f64>(
/// 	2,
/// 	true,
/// 	faer::ParShape::Seq,
/// 	Default::default(),
/// ));
/// hessenberg_schur_real(
/// 	h.as_mut(),
/// 	Some(z.as_mut()),
/// 	wr.as_mut(),
/// 	wi.as_mut(),
/// 	Par::Seq,
/// 	MemStack::new(&mut mem),
/// 	Default::default(),
/// )
/// .unwrap();
/// assert!(wi[0] > 0.0 && wi[1] < 0.0);
/// ```
pub fn hessenberg_schur_real<T: RealField>(
	mut h: MatMut<'_, T>,
	z: Option<MatMut<'_, T>>,
	mut wr: ColMut<'_, T>,
	mut wi: ColMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<SchurParams, T>,
) -> Result<(), EvdError> {
	let n = h.nrows();
	assert!(h.ncols() == n && wr.nrows() == n && wi.nrows() == n);
	if let Some(z) = z.rb() {
		assert!(z.nrows() == n && z.ncols() == n);
	}
	if !h.is_all_finite() {
		return Err(EvdError::NoConvergence);
	}
	if n <= 1 {
		if n == 1 {
			wr[0] = h[(0, 0)].copy();
			wi[0] = zero();
		}
		return Ok(());
	}
	let (info, _, _) = schur::real_schur::multishift_qr(
		true,
		h.rb_mut(),
		z,
		wr,
		wi,
		0,
		n,
		par,
		stack,
		*params,
	);
	if info != 0 {
		return Err(EvdError::NoConvergence);
	}
	// The QR kernel uses entries below the Hessenberg band as workspace.
	for j in 0..n {
		for i in j + 2..n {
			h[(i, j)] = zero();
		}
	}
	Ok(())
}
/// Complex counterpart of [`hessenberg_schur_real`]: `h` becomes upper
/// triangular, `w` receives its diagonal, and optional `z` becomes `z * q`.
/// The similarity is `h_original = q * t * q.adjoint()`.
///
/// # Panics
/// Panics on incompatible dimensions, a real scalar type, or insufficient
/// scratch; `h` must be upper Hessenberg.
///
/// # Errors
/// Returns [`EvdError::NoConvergence`] for nonfinite input or QR
/// nonconvergence; accumulated similarity updates remain in `h` and `z`.
///
/// # Examples
/// ```
/// use dyn_stack::{MemBuffer, MemStack};
/// use faer::linalg::evd::*;
/// use faer::{Col, Mat, Par, c64};
/// let mut h = Mat::<c64>::identity(2, 2);
/// let mut w = Col::zeros(2);
/// let mut mem = MemBuffer::new(hessenberg_schur_scratch::<c64>(
/// 	2,
/// 	false,
/// 	faer::ParShape::Seq,
/// 	Default::default(),
/// ));
/// hessenberg_schur_cplx(
/// 	h.as_mut(),
/// 	None,
/// 	w.as_mut(),
/// 	Par::Seq,
/// 	MemStack::new(&mut mem),
/// 	Default::default(),
/// )
/// .unwrap();
/// assert_eq!(w[0], c64::new(1.0, 0.0));
/// ```
pub fn hessenberg_schur_cplx<T: ComplexField>(
	mut h: MatMut<'_, T>,
	z: Option<MatMut<'_, T>>,
	mut w: ColMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<SchurParams, T>,
) -> Result<(), EvdError> {
	assert!(!T::IS_REAL);
	let n = h.nrows();
	assert!(h.ncols() == n && w.nrows() == n);
	if let Some(z) = z.rb() {
		assert!(z.nrows() == n && z.ncols() == n);
	}
	if !h.is_all_finite() {
		return Err(EvdError::NoConvergence);
	}
	if n <= 1 {
		if n == 1 {
			w[0] = h[(0, 0)].copy();
		}
		return Ok(());
	}
	let (info, _, _) = schur::complex_schur::multishift_qr(
		true,
		h.rb_mut(),
		z,
		w,
		0,
		n,
		par,
		stack,
		*params,
	);
	if info != 0 {
		return Err(EvdError::NoConvergence);
	}
	// Return a full triangular matrix, without the QR kernel's workspace.
	for j in 0..n {
		for i in j + 1..n {
			h[(i, j)] = zero();
		}
	}
	Ok(())
}
/// An adjacent real Schur block swap was rejected as ill-conditioned.
/// `position` is the current first row of the moving block.
///
/// # Examples
/// ```
/// use faer::linalg::evd::SchurReorderError;
/// let err = SchurReorderError { position: 2 };
/// assert_eq!(err.position, 2);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SchurReorderError {
	/// Current first row of the moving block, after the last successful swap.
	pub position: usize,
}
/// Workspace for [`reorder_schur_real`] and [`reorder_schur_cplx`]: a copy of
/// the rows and columns of `t` and the columns of `q` that one adjacent swap
/// updates, so that a rejected swap can be undone.
///
/// # Examples
/// ```
/// let req = faer::linalg::evd::reorder_schur_scratch::<f64>(3);
/// let _buffer = dyn_stack::MemBuffer::new(req);
/// ```
pub fn reorder_schur_scratch<T: ComplexField>(dim: usize) -> StackReq {
	StackReq::all_of(&[
		temp_mat_scratch::<T>(4, dim),
		temp_mat_scratch::<T>(dim, 4),
		temp_mat_scratch::<T>(dim, 4),
	])
}
/// Moves a real Schur block from `ifst` to `ilst`, updating optional `q`.
/// Indices are zero-based rows; the second row of a pair denotes its first row.
/// Like `xTREXC`, when moving down across a differently sized destination
/// block, the final first row is adjusted so the moving block follows that
/// block. Returns the final first row; conjugate pairs are never split.
///
/// # Panics
/// Panics on invalid dimensions/indices. `t` must be in standardized real Schur
/// form.
///
/// # Errors
/// Returns [`SchurReorderError`] when an adjacent swap is rejected. As in
/// LAPACK `xTREXC`/`xLAEXC`, a swap involving a 2 by 2 block is performed
/// provisionally on a copy of its diagonal window and rejected when the
/// provisional entries that must vanish (and, for a moved 1 by 1 block, its
/// diagonal value) are off by more than `max(10 eps |window|_max,
/// safmin / eps)`. A swap whose update of `t` or `q` overflows is rejected
/// too. `t` and `q` are left exactly as after the last successful swap (in
/// valid Schur form, with the block standardization); the error gives the
/// moving block's current position. On success, `t` and `q` are finite.
///
/// # Examples
/// ```
/// use dyn_stack::{MemBuffer, MemStack};
/// use faer::Par;
/// use faer::linalg::evd::*;
/// let mut t = faer::mat![[1.0f64, 0.0], [0.0, 2.0]];
/// let mut mem = MemBuffer::new(reorder_schur_scratch::<f64>(2));
/// assert_eq!(
/// 	reorder_schur_real(
/// 		t.as_mut(),
/// 		None,
/// 		0,
/// 		1,
/// 		Par::Seq,
/// 		MemStack::new(&mut mem)
/// 	)
/// 	.unwrap(),
/// 	1
/// );
/// assert_eq!(t[(1, 1)], 1.0);
/// ```
pub fn reorder_schur_real<T: RealField>(
	mut t: MatMut<'_, T>,
	mut q: Option<MatMut<'_, T>>,
	mut ifst: usize,
	mut ilst: usize,
	_par: Par,
	stack: &mut MemStack,
) -> Result<usize, SchurReorderError> {
	let n = t.nrows();
	assert!(t.ncols() == n && ifst < n && ilst < n);
	if let Some(q) = q.rb() {
		assert!(q.nrows() == n && q.ncols() == n);
	}
	let (mut saved_rows, stack) = temp_mat_zeroed::<T, _, _>(4, n, stack);
	let (mut saved_cols, stack) = temp_mat_zeroed::<T, _, _>(n, 4, stack);
	let (mut saved_q, _) = temp_mat_zeroed::<T, _, _>(n, 4, stack);
	let mut saved_rows = saved_rows.as_mat_mut();
	let mut saved_cols = saved_cols.as_mat_mut();
	let mut saved_q = saved_q.as_mat_mut();
	if ifst > 0 && t[(ifst, ifst - 1)] != zero() {
		ifst -= 1;
	}
	if ilst > 0 && t[(ilst, ilst - 1)] != zero() {
		ilst -= 1;
	}
	let block_size = |t: MatRef<'_, T>, i: usize| {
		if i + 1 < n && t[(i + 1, i)] != zero() {
			2
		} else {
			1
		}
	};
	let moving = block_size(t.rb(), ifst);
	if ifst < ilst {
		let dest = block_size(t.rb(), ilst);
		ilst = ilst + dest - moving;
	}
	let mut here = ifst;
	while here != ilst {
		let down = here < ilst;
		let next = if down {
			block_size(t.rb(), here + moving)
		} else if here > 1 && t[(here - 1, here - 2)] != zero() {
			2
		} else {
			1
		};
		let start = if down { here } else { here - next };
		let (n1, n2) = if down { (moving, next) } else { (next, moving) };
		// A swap updates rows `start..start + k` (from column `start`) and
		// columns `start..start + k` (up to row `start + k`) of `t`, and
		// columns `start..start + k` of `q`. Keep a copy so that a rejected
		// or overflowing swap leaves the state of the last successful one.
		let k = n1 + n2;
		let w = n - start;
		let h = start + k;
		saved_rows
			.rb_mut()
			.submatrix_mut(0, 0, k, w)
			.copy_from(t.rb().submatrix(start, start, k, w));
		saved_cols
			.rb_mut()
			.submatrix_mut(0, 0, h, k)
			.copy_from(t.rb().submatrix(0, start, h, k));
		if let Some(q) = q.rb() {
			saved_q
				.rb_mut()
				.submatrix_mut(0, 0, n, k)
				.copy_from(q.submatrix(0, start, n, k));
		}
		let status = schur::real_schur::schur_swap_checked(
			t.rb_mut(),
			q.rb_mut(),
			start,
			n1,
			n2,
			true,
		);
		let finite = t.rb().submatrix(start, start, k, w).is_all_finite()
			&& t.rb().submatrix(0, start, h, k).is_all_finite()
			&& q.rb()
				.is_none_or(|q| q.submatrix(0, start, n, k).is_all_finite());
		if status != 0 || !finite {
			t.rb_mut()
				.submatrix_mut(0, start, h, k)
				.copy_from(saved_cols.rb().submatrix(0, 0, h, k));
			t.rb_mut()
				.submatrix_mut(start, start, k, w)
				.copy_from(saved_rows.rb().submatrix(0, 0, k, w));
			if let Some(q) = q.rb_mut() {
				q.submatrix_mut(0, start, n, k)
					.copy_from(saved_q.rb().submatrix(0, 0, n, k));
			}
			return Err(SchurReorderError { position: here });
		}
		here = if down { here + next } else { here - next };
	}
	Ok(here)
}
/// Moves a complex triangular Schur diagonal entry from `ifst` to `ilst`,
/// updating optional `q`. All blocks are 1 by 1; the complex swap cannot
/// reject.
///
/// # Panics
/// Panics on invalid dimensions/indices. `t` must be upper triangular.
///
/// # Examples
/// ```
/// use dyn_stack::{MemBuffer, MemStack};
/// use faer::linalg::evd::*;
/// use faer::{Mat, Par, c64};
/// let mut t = Mat::<c64>::identity(2, 2);
/// let mut mem = MemBuffer::new(reorder_schur_scratch::<c64>(2));
/// assert_eq!(
/// 	reorder_schur_cplx(
/// 		t.as_mut(),
/// 		None,
/// 		0,
/// 		1,
/// 		Par::Seq,
/// 		MemStack::new(&mut mem)
/// 	),
/// 	1
/// );
/// ```
pub fn reorder_schur_cplx<T: ComplexField>(
	mut t: MatMut<'_, T>,
	mut q: Option<MatMut<'_, T>>,
	ifst: usize,
	ilst: usize,
	_par: Par,
	_stack: &mut MemStack,
) -> usize {
	let n = t.nrows();
	assert!(t.ncols() == n && ifst < n && ilst < n);
	if let Some(q) = q.rb() {
		assert!(q.nrows() == n && q.ncols() == n);
	}
	let mut here = ifst;
	while here != ilst {
		let start = if here < ilst { here } else { here - 1 };
		schur::complex_schur::schur_swap(t.rb_mut(), q.rb_mut(), start);
		here = if here < ilst { here + 1 } else { here - 1 };
	}
	here
}
/// Workspace for [`eigenvectors_from_schur_real`] and
/// [`eigenvectors_from_schur_cplx`].
///
/// # Examples
/// ```
/// let req = faer::linalg::evd::eigenvectors_from_schur_scratch::<f64>(3);
/// let _buffer = dyn_stack::MemBuffer::new(req);
/// ```
pub fn eigenvectors_from_schur_scratch<T: ComplexField>(
	dim: usize,
) -> StackReq {
	temp_mat_scratch::<T>(dim, dim)
}
/// Computes all right eigenvectors of standardized real Schur `t` in diagonal
/// order. If `q` is supplied, returns eigenvectors of `q * t * q.transpose()`.
/// Each real column has unit 2-norm. A pair of columns encodes `re + i im` for
/// the positive-imaginary eigenvalue, normalized as one complex vector.
/// `x` is output only: its previous contents are never read.
///
/// # Panics
/// Panics on incompatible square dimensions or insufficient scratch.
///
/// # Errors
/// Returns [`EvdError::NoConvergence`] for nonfinite input or
/// nonfinite/degenerate output.
///
/// # Examples
/// ```
/// use dyn_stack::{MemBuffer, MemStack};
/// use faer::linalg::evd::*;
/// use faer::{Mat, Par};
/// let t = faer::mat![[1.0f64, 2.0], [0.0, 3.0]];
/// let mut x = Mat::zeros(2, 2);
/// let mut mem = MemBuffer::new(eigenvectors_from_schur_scratch::<f64>(2));
/// eigenvectors_from_schur_real(
/// 	t.as_ref(),
/// 	None,
/// 	x.as_mut(),
/// 	Par::Seq,
/// 	MemStack::new(&mut mem),
/// 	Default::default(),
/// )
/// .unwrap();
/// assert!((x.col(1).norm_l2() - 1.0).abs() < 1e-14);
/// ```
pub fn eigenvectors_from_schur_real<T: RealField>(
	t: MatRef<'_, T>,
	q: Option<MatRef<'_, T>>,
	mut x: MatMut<'_, T>,
	mut par: Par,
	stack: &mut MemStack,
	params: Spec<EvdFromSchurParams, T>,
) -> Result<(), EvdError> {
	let n = t.nrows();
	assert!(t.ncols() == n && x.nrows() == n && x.ncols() == n);
	if let Some(q) = q {
		assert!(q.nrows() == n && q.ncols() == n);
	}
	if !t.is_all_finite() {
		return Err(EvdError::NoConvergence);
	}
	let (mut v, _) = temp_mat_zeroed::<T, _, _>(n, n, stack);
	let mut v = v.as_mat_mut();
	evd_from_real_schur_imp(t, v.rb_mut(), par.reborrow(), *params);
	if let Some(q) = q {
		linalg::matmul::matmul(
			x.rb_mut(),
			Accum::Replace,
			q,
			v.rb(),
			one(),
			par,
		);
	} else {
		x.copy_from(v.rb());
	}
	normalize_eigenvectors(x.rb_mut(), t);
	if eigenvectors_ok(x.rb()) {
		Ok(())
	} else {
		Err(EvdError::NoConvergence)
	}
}
/// Complex counterpart of [`eigenvectors_from_schur_real`]. `t` is upper
/// triangular; every column is one unit-norm right eigenvector, in diagonal
/// order. Optional `q` back-transforms by `q * x` (for `q * t * q.adjoint()`).
/// `x` is output only: its previous contents are never read.
///
/// # Panics
/// Panics on incompatible square dimensions or insufficient scratch.
///
/// # Errors
/// Returns [`EvdError::NoConvergence`] for nonfinite input or
/// nonfinite/degenerate output.
///
/// # Examples
/// ```
/// use dyn_stack::{MemBuffer, MemStack};
/// use faer::linalg::evd::*;
/// use faer::{Mat, Par, c64};
/// let t = Mat::<c64>::identity(2, 2);
/// let mut x = Mat::zeros(2, 2);
/// let mut mem = MemBuffer::new(eigenvectors_from_schur_scratch::<c64>(2));
/// eigenvectors_from_schur_cplx(
/// 	t.as_ref(),
/// 	None,
/// 	x.as_mut(),
/// 	Par::Seq,
/// 	MemStack::new(&mut mem),
/// 	Default::default(),
/// )
/// .unwrap();
/// assert_eq!(x, t);
/// ```
pub fn eigenvectors_from_schur_cplx<T: ComplexField>(
	t: MatRef<'_, T>,
	q: Option<MatRef<'_, T>>,
	mut x: MatMut<'_, T>,
	mut par: Par,
	stack: &mut MemStack,
	params: Spec<EvdFromSchurParams, T>,
) -> Result<(), EvdError> {
	let n = t.nrows();
	assert!(t.ncols() == n && x.nrows() == n && x.ncols() == n);
	if let Some(q) = q {
		assert!(q.nrows() == n && q.ncols() == n);
	}
	if !t.is_all_finite() {
		return Err(EvdError::NoConvergence);
	}
	let (mut v, _) = temp_mat_zeroed::<T, _, _>(n, n, stack);
	let mut v = v.as_mat_mut();
	evd_from_cplx_schur_imp(t, Conj::No, v.rb_mut(), par.reborrow(), *params);
	if let Some(q) = q {
		linalg::matmul::matmul(
			x.rb_mut(),
			Accum::Replace,
			q,
			v.rb(),
			one(),
			par,
		);
	} else {
		x.copy_from(v.rb());
	}
	normalize_eigenvectors(x.rb_mut(), t);
	if eigenvectors_ok(x.rb()) {
		Ok(())
	} else {
		Err(EvdError::NoConvergence)
	}
}
fn evd_imp<T: ComplexField>(
	A: MatRef<'_, T>,
	s: ColMut<'_, T>,
	s_im: Option<ColMut<'_, T>>,
	u_left: Option<MatMut<'_, T>>,
	u_right: Option<MatMut<'_, T>>,
	mut par: Par,
	stack: &mut MemStack,
	params: EvdParams,
) -> Result<(), EvdError> {
	let n = A.nrows();
	if n == 0 {
		return Ok(());
	}
	for j in 0..n {
		for i in 0..n {
			if !A[(i, j)].is_finite() {
				return Err(EvdError::NoConvergence);
			}
		}
	}
	let bs = linalg::qr::no_pivoting::factor::recommended_block_size::<T>(
		n - 1,
		n - 1,
	);
	let mut s = s;
	let mut s_im = s_im;
	let (mut H, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
	let (mut Z, stack) = unsafe {
		temp_mat_uninit::<T, _, _>(
			n,
			if u_left.is_some() || u_right.is_some() {
				n
			} else {
				0
			},
			stack,
		)
	};
	let mut H = H.as_mat_mut();
	let mut Z = if u_left.is_some() || u_right.is_some() {
		Some(Z.as_mat_mut())
	} else {
		None
	};
	H.copy_from(A);
	{
		let (mut householder, stack) =
			unsafe { temp_mat_uninit::<T, _, _>(bs, n - 1, stack) };
		let mut householder = householder.as_mat_mut();
		hessenberg::hessenberg_in_place(
			H.rb_mut(),
			householder.rb_mut(),
			par.reborrow(),
			stack,
			params.hessenberg.into(),
		);
		if let Some(mut Z) = Z.rb_mut() {
			Z.fill(zero());
			Z.rb_mut().diagonal_mut().fill(one());
			linalg::householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
				H.rb().submatrix(1, 0, n - 1, n - 1),
				householder.rb(),
				Conj::No,
				Z.rb_mut().submatrix_mut(1, 1, n - 1, n - 1),
				par.reborrow(),
				stack,
			);
		}
		for j in 0..n {
			for i in j + 2..n {
				H[(i, j)] = zero();
			}
		}
	}
	if const { T::IS_REAL } {
		let (info, _, _) = schur::real_schur::multishift_qr::<T::Real>(
			unsafe { core::mem::transmute(Z.is_some()) },
			unsafe { core::mem::transmute(H.rb_mut()) },
			unsafe { core::mem::transmute(Z.rb_mut()) },
			unsafe { core::mem::transmute(s.rb_mut()) },
			unsafe { core::mem::transmute(s_im.rb_mut().unwrap()) },
			0,
			n,
			par.reborrow(),
			stack,
			params.schur,
		);
		if info != 0 {
			return Err(EvdError::NoConvergence);
		}
	} else {
		let (info, _, _) = schur::complex_schur::multishift_qr::<T>(
			Z.is_some(),
			H.rb_mut(),
			Z.rb_mut(),
			s.rb_mut(),
			0,
			n,
			par.reborrow(),
			stack,
			params.schur,
		);
		if info != 0 {
			return Err(EvdError::NoConvergence);
		}
	}
	let H = H.rb();
	if let (Some(mut u), Some(Z)) = (u_right, Z.rb()) {
		let (mut X, _) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
		let mut X = X.as_mat_mut();
		if const { T::IS_REAL } {
			evd_from_real_schur_imp::<T::Real>(
				unsafe { core::mem::transmute(H) },
				unsafe { core::mem::transmute(X.rb_mut()) },
				par.reborrow(),
				params.evd_from_schur,
			);
		} else {
			evd_from_cplx_schur_imp::<T>(
				H,
				Conj::No,
				X.rb_mut(),
				par.reborrow(),
				params.evd_from_schur,
			);
		}
		linalg::matmul::triangular::matmul(
			u.rb_mut(),
			BlockStructure::Rectangular,
			Accum::Replace,
			Z,
			BlockStructure::Rectangular,
			X.rb(),
			BlockStructure::TriangularUpper,
			one(),
			par.reborrow(),
		);
		normalize_eigenvectors(u.rb_mut(), H);
		if !eigenvectors_ok(u.rb()) {
			return Err(EvdError::NoConvergence);
		}
	}
	if let (Some(mut u), Some(Z)) = (u_left, Z.rb()) {
		let (mut X, _) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
		let mut X = X.as_mat_mut().reverse_rows_mut();
		if const { T::IS_REAL } {
			evd_from_real_schur_imp::<T::Real>(
				unsafe {
					core::mem::transmute(H.transpose().reverse_rows_and_cols())
				},
				unsafe { core::mem::transmute(X.rb_mut()) },
				par.reborrow(),
				params.evd_from_schur,
			);
		} else {
			evd_from_cplx_schur_imp::<T>(
				H.transpose().reverse_rows_and_cols(),
				Conj::Yes,
				X.rb_mut(),
				par.reborrow(),
				params.evd_from_schur,
			);
		}
		linalg::matmul::triangular::matmul(
			u.rb_mut(),
			BlockStructure::Rectangular,
			Accum::Replace,
			Z,
			BlockStructure::Rectangular,
			X.rb().reverse_rows_and_cols(),
			BlockStructure::TriangularLower,
			one(),
			par,
		);
		normalize_eigenvectors(u.rb_mut(), H);
		if !eigenvectors_ok(u.rb()) {
			return Err(EvdError::NoConvergence);
		}
	}
	Ok(())
}
/// computes the matrix $A$'s eigendecomposition
///
/// the eigenvalues are stored in $S$, the left eigenvectors in $U_L$, and the
/// right eigenvectors in $U_R$.
///
/// every eigenvector is scaled to unit 2-norm. if the Schur iteration does not
/// converge, or the back substitution would overflow for a finite input,
/// [`EvdError::NoConvergence`] is returned instead of nonfinite entries.
#[track_caller]
pub fn evd_cplx<T: RealField>(
	A: MatRef<'_, Complex<T>>,
	s: DiagMut<'_, Complex<T>>,
	u_left: Option<MatMut<'_, Complex<T>>>,
	u_right: Option<MatMut<'_, Complex<T>>>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<EvdParams, Complex<T>>,
) -> Result<(), EvdError> {
	let n = A.nrows();
	assert!(all(A.nrows() == n, A.ncols() == n, s.dim() == n));
	if let Some(u) = u_left.rb() {
		assert!(all(u.nrows() == n, u.ncols() == n));
	}
	if let Some(u) = u_right.rb() {
		assert!(all(u.nrows() == n, u.ncols() == n));
	}
	evd_imp(
		A,
		s.column_vector_mut(),
		None,
		u_left,
		u_right,
		par,
		stack,
		params.config,
	)
}
/// computes the matrix $A$'s eigendecomposition
///
/// the eigenvalues are stored in $S$, the left eigenvectors in $U_L$, and the
/// right eigenvectors in $U_R$.
///
/// every eigenvector is scaled to unit 2-norm: a real eigenvalue gets a unit
/// norm column, and the two real columns `re` and `im` of a conjugate pair are
/// scaled by a common factor so that `re + i * im` has unit 2-norm (the same
/// contract `xGEEV` provides). if the Schur iteration does not converge, or the
/// back substitution would overflow for a finite input,
/// [`EvdError::NoConvergence`] is returned instead of nonfinite entries.
#[track_caller]
pub fn evd_real<T: RealField>(
	A: MatRef<'_, T>,
	s_re: DiagMut<'_, T>,
	s_im: DiagMut<'_, T>,
	u_left: Option<MatMut<'_, T>>,
	u_right: Option<MatMut<'_, T>>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<EvdParams, T>,
) -> Result<(), EvdError> {
	let n = A.nrows();
	assert!(all(
		A.nrows() == n,
		A.ncols() == n,
		s_re.dim() == n,
		s_im.dim() == n
	));
	if let Some(u) = u_left.rb() {
		assert!(all(u.nrows() == n, u.ncols() == n));
	}
	if let Some(u) = u_right.rb() {
		assert!(all(u.nrows() == n, u.ncols() == n));
	}
	evd_imp(
		A,
		s_re.column_vector_mut(),
		Some(s_im.column_vector_mut()),
		u_left,
		u_right,
		par,
		stack,
		params.config,
	)
}
#[cfg(test)]
mod general_tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;
	fn test_cplx_evd(mat: MatRef<'_, c64>) {
		let n = mat.nrows();
		let params = Spec::new(EvdParams {
			hessenberg: auto!(c64),
			schur: auto!(c64),
			evd_from_schur: EvdFromSchurParams {
				recursion_threshold: 8,
				..auto!(c64)
			},
			..auto!(c64)
		});
		let approx_eq = CwiseMat(ApproxEq::eps() * (8.0 * n as f64).sqrt());
		let mut s = Mat::zeros(n, n);
		{
			let mut ul = Mat::zeros(n, n);
			let mut ur = Mat::zeros(n, n);
			evd_cplx(
				mat.as_ref(),
				s.as_mut().diagonal_mut(),
				Some(ul.as_mut()),
				Some(ur.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(evd_scratch::<c64>(
					n,
					ComputeEigenvectors::Yes,
					ComputeEigenvectors::Yes,
					ParShape::Seq,
					params,
				))),
				params,
			)
			.unwrap();
			assert!(& ur * & s ~ mat * & ur);
			assert!(& s * ul.adjoint() ~ ul.adjoint() * mat);
		}
	}
	fn test_real_evd(mat: MatRef<'_, f64>) {
		let n = mat.nrows();
		let params = Spec::new(EvdParams {
			hessenberg: auto!(f64),
			schur: auto!(f64),
			evd_from_schur: EvdFromSchurParams {
				recursion_threshold: 8,
				..auto!(f64)
			},
			..auto!(f64)
		});
		let approx_eq =
			CwiseMat(ApproxEq::<f64>::eps() * (8.0 * n as f64).sqrt());
		let mut s_re = Diag::zeros(n);
		let mut s_im = Diag::zeros(n);
		{
			let mut ul = Mat::zeros(n, n);
			let mut ur = Mat::zeros(n, n);
			evd_real(
				mat.as_ref(),
				s_re.as_mut(),
				s_im.as_mut(),
				Some(ul.as_mut()),
				Some(ur.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(evd_scratch::<f64>(
					n,
					ComputeEigenvectors::Yes,
					ComputeEigenvectors::Yes,
					ParShape::Seq,
					params,
				))),
				params,
			)
			.unwrap();
			let mut i = 0;
			while i < n {
				if s_im[i] == 0.0 {
					let ur = ur.col(i);
					let ul = ul.col(i);
					let s = Scale(s_re[i]);
					assert!((& ur * s).as_mat() ~ (mat * & ur).as_mat());
					assert!(
						(& ul.adjoint() * s).as_mat() ~ (& ul.adjoint() * mat).as_mat()
					);
					i += 1;
				} else {
					let re = ur.col(i);
					let im = ur.col(i + 1);
					let ur = &Col::from_fn(n, |i| c64::new(re[i], im[i]));
					let re = ul.col(i);
					let im = ul.col(i + 1);
					let ul = &Col::from_fn(n, |i| c64::new(re[i], im[i]));
					let mat =
						&Mat::from_fn(n, n, |i, j| c64::from(mat[(i, j)]));
					let s = Scale(c64::new(s_re[i], s_im[i]));
					let approx_eq =
						CwiseMat(ApproxEq::eps() * (8.0 * n as f64).sqrt());
					assert!((ur * s).as_mat() ~ (mat * ur).as_mat());
					assert!((ul.adjoint() * s).as_mat() ~ (ul.adjoint() * mat).as_mat());
					i += 2;
				}
			}
		}
	}
	#[test]
	fn test_cplx() {
		let rng = &mut StdRng::seed_from_u64(1);
		for n in [2, 4, 10, 15, 20, 50, 100, 150] {
			let mat = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);
			test_cplx_evd(mat.as_ref());
		}
	}
	#[test]
	fn test_real() {
		let rng = &mut StdRng::seed_from_u64(0);
		for n in [2, 4, 10, 15, 20, 50, 100, 150] {
			let mat = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			test_real_evd(mat.as_ref());
		}
	}
}
#[cfg(test)]
mod eig_2131_tests {
	//! Regression tests for `tensor4all/tenferro-rs#2131`: the faer
	//! eigendecomposition used to return eigenvectors that were neither
	//! normalized nor overflow-safe. On the Jordan block `2 I + N` the back
	//! substitution grew by `1 / (eps * norm)` per row, overflowing to `NaN`
	//! for finite input (`f32` at `n >= 8`, `f64` at `n = 26`).
	use super::*;
	use crate::assert;
	use dyn_stack::MemBuffer;

	/// `||A v - lambda v|| / (||A||_1 * ||v||)`, where `||A||_1` is the maximum
	/// absolute column sum.
	fn relative_residual<T: RealField>(
		mat: MatRef<'_, T>,
		v: ColRef<'_, T>,
		lambda: T,
	) -> T {
		let n = mat.nrows();
		let mut anorm = zero::<T>();
		for j in 0..n {
			let mut col = zero::<T>();
			for i in 0..n {
				col += mat[(i, j)].abs1();
			}
			if col > anorm {
				anorm = col;
			}
		}
		let mut vnorm2 = zero::<T>();
		let mut rnorm2 = zero::<T>();
		for i in 0..n {
			let mut acc = zero::<T>();
			for j in 0..n {
				acc += &mat[(i, j)] * &v[j];
			}
			acc -= &v[i] * &lambda;
			rnorm2 += acc.abs2();
			vnorm2 += v[i].abs2();
		}
		rnorm2.sqrt() / (anorm * vnorm2.sqrt())
	}

	fn evd_real_vectors<T: RealField>(
		mat: MatRef<'_, T>,
		params: Spec<EvdParams, T>,
	) -> (Diag<T>, Diag<T>, Mat<T>, Mat<T>) {
		let n = mat.nrows();
		let mut s_re = Diag::zeros(n);
		let mut s_im = Diag::zeros(n);
		let mut u = Mat::zeros(n, n);
		let mut ul = Mat::zeros(n, n);
		let mut mem = MemBuffer::new(evd_scratch::<T>(
			n,
			ComputeEigenvectors::Yes,
			ComputeEigenvectors::Yes,
			ParShape::Seq,
			params,
		));
		evd_real(
			mat,
			s_re.as_mut(),
			s_im.as_mut(),
			Some(ul.as_mut()),
			Some(u.as_mut()),
			Par::Seq,
			MemStack::new(&mut mem),
			params,
		)
		.unwrap();
		(s_re, s_im, u, ul)
	}

	/// The Jordan block `2 I + N` with ones on the superdiagonal.
	fn jordan<T: RealField>(n: usize) -> Mat<T> {
		let two = from_f64::<T>(2.0);
		let one = one::<T>();
		Mat::<T>::from_fn(n, n, |i, j| {
			if i == j {
				two.copy()
			} else if j == i + 1 {
				one.copy()
			} else {
				zero::<T>()
			}
		})
	}

	fn check_jordan<T: RealField>(n: usize, threshold: usize, tol: f64) {
		let params = Spec::new(EvdParams {
			evd_from_schur: EvdFromSchurParams {
				recursion_threshold: threshold,
				..auto!(T)
			},
			..auto!(T)
		});
		let mat = jordan::<T>(n);
		let (s_re, _, u, _) = evd_real_vectors(mat.as_ref(), params);
		let tol = from_f64::<T>(tol);
		for j in 0..n {
			let v = u.col(j);
			assert!(v.iter().all(|x| x.is_finite()));
			assert!((v.norm_l2() - one::<T>()).abs() < from_f64::<T>(1e-5));
			assert!(relative_residual(mat.as_ref(), v, s_re[j].copy()) < tol);
		}
	}

	#[test]
	fn jordan_eigenvectors_are_finite_and_unit_norm() {
		// `threshold = 3` forces the recursive block solver; the default `64`
		// keeps small sizes sequential and exercises the recursion for the
		// larger ones. Both must produce finite, normalized, backward-stable
		// eigenvectors.
		for threshold in [3usize, 64] {
			for n in [2, 4, 8, 20, 26, 30, 64, 65, 100] {
				check_jordan::<f64>(n, threshold, 1e-10);
			}
			for n in [2, 4, 8, 16, 30] {
				check_jordan::<f32>(n, threshold, 1e-3);
			}
		}
	}

	#[test]
	fn conjugate_pair_eigenvector_has_unit_norm() {
		// A real Schur form with four real eigenvalues in front of a 2x2
		// conjugate pair, so the complex-shift quasi-triangular solver gets a
		// prefix large enough for its recursive branch at `threshold = 4`.
		fn run<T: RealField>(threshold: usize) {
			let n = 6;
			let mat = Mat::<T>::from_fn(n, n, |i, j| {
				if i == j {
					if i >= 4 {
						zero::<T>()
					} else {
						from_f64::<T>((i + 1) as f64)
					}
				} else if i == 4 && j == 5 {
					one::<T>()
				} else if i == 5 && j == 4 {
					-one::<T>()
				} else if i + 2 <= j {
					from_f64::<T>(0.25)
				} else {
					zero::<T>()
				}
			});
			let params = Spec::new(EvdParams {
				evd_from_schur: EvdFromSchurParams {
					recursion_threshold: threshold,
					..auto!(T)
				},
				..auto!(T)
			});
			let (s_re, s_im, u, ul) = evd_real_vectors(mat.as_ref(), params);
			let tol = from_f64::<T>(1e-5);
			// Right and left eigenvectors share the same conjugate-pair scaling.
			let mut j = 0;
			while j < n {
				let v = u.col(j);
				let vl = ul.col(j);
				if s_im[j] == zero::<T>() {
					assert!((v.norm_l2() - one::<T>()).abs() < tol);
					assert!((vl.norm_l2() - one::<T>()).abs() < tol);
					assert!(
						relative_residual(mat.as_ref(), v, s_re[j].copy())
							< from_f64::<T>(1e-4)
					);
					j += 1;
				} else {
					let nrm = v.norm_l2().hypot(&u.col(j + 1).norm_l2());
					assert!((nrm - one::<T>()).abs() < tol);
					let nrm = vl.norm_l2().hypot(&ul.col(j + 1).norm_l2());
					assert!((nrm - one::<T>()).abs() < tol);
					j += 2;
				}
			}
		}
		for threshold in [4usize, 64] {
			run::<f64>(threshold);
			run::<f32>(threshold);
		}
	}

	#[test]
	fn complex_eigenvectors_have_unit_norm() {
		// `threshold = 4` drives the complex solver's recursive branch.
		fn run<T: RealField>(threshold: usize) {
			let n = 6;
			let mat = Mat::<Complex<T>>::from_fn(n, n, |i, j| {
				Complex::new(
					from_f64::<T>(i as f64)
						+ from_f64::<T>(j as f64) * from_f64::<T>(0.1),
					from_f64::<T>(i as f64 - j as f64) * from_f64::<T>(0.3),
				)
			});
			let params = Spec::new(EvdParams {
				evd_from_schur: EvdFromSchurParams {
					recursion_threshold: threshold,
					..auto!(Complex<T>)
				},
				..auto!(Complex<T>)
			});
			let mut s = Diag::zeros(n);
			let mut u = Mat::zeros(n, n);
			let mut mem = MemBuffer::new(evd_scratch::<Complex<T>>(
				n,
				ComputeEigenvectors::No,
				ComputeEigenvectors::Yes,
				ParShape::Seq,
				params,
			));
			evd_cplx(
				mat.as_ref(),
				s.as_mut(),
				None,
				Some(u.as_mut()),
				Par::Seq,
				MemStack::new(&mut mem),
				params,
			)
			.unwrap();
			let tol = from_f64::<T>(1e-5);
			for j in 0..n {
				assert!((u.col(j).norm_l2() - one::<T>()).abs() < tol);
			}
		}
		for threshold in [4usize, 64] {
			run::<f64>(threshold);
			run::<f32>(threshold);
		}
	}
}
#[cfg(test)]
mod range_limit_vector_tests {
	//! Regression tests for `tensor4all/faer#77`: the Schur eigenvector
	//! extraction used the unscaled 1-norm of the Schur form as its backward
	//! error scale, so a finite matrix near the top of the range made the shift
	//! clamp `eps * norm` infinite and returned a wrong direction with `Ok`.
	//! The 2x2 block solve of a conjugate pair still forms products of two
	//! block entries, which overflow at the top of the range; there the call
	//! must fail rather than return a wrong vector.
	//! Residuals are measured on the dimensionless matrix `t / scale`, which is
	//! exact for a power of two.
	use super::*;
	use crate::assert;
	use dyn_stack::MemBuffer;

	/// the reported Schur form `[[1, 1], [0, 1/2]]`
	const SHIFT_2X2: [[f64; 2]; 2] = [[1.0, 1.0], [0.0, 0.5]];
	/// real Schur form with the conjugate pair `[[0, -1], [1, 0]]` in front of
	/// the real eigenvalue `1/2`
	const PAIR_3X3: [[f64; 3]; 3] =
		[[0.0, -1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.5]];
	/// two conjugate pairs, `+-i` and `+-i/2`
	const TWO_PAIRS_4X4: [[f64; 4]; 4] = [
		[0.0, -1.0, 1.0, 0.0],
		[1.0, 0.0, 0.0, 0.0],
		[0.0, 0.0, 0.0, -0.5],
		[0.0, 0.0, 0.5, 0.0],
	];
	/// whose repeated eigenvalue makes the shift clamp matter
	const JORDAN_2X2: [[f64; 2]; 2] = [[1.0, 1.0], [0.0, 1.0]];

	fn unit<T: RealField, const N: usize>(
		mat: &[[f64; N]; N],
	) -> Mat<Complex<T>> {
		Mat::from_fn(N, N, |i, j| {
			Complex::new(from_f64::<T>(mat[i][j]), zero::<T>())
		})
	}

	/// the dimensionless matrix scaled by the power of two `scale`
	fn scaled_real<T: RealField>(
		unit: MatRef<'_, Complex<T>>,
		scale: &T,
	) -> Mat<T> {
		Mat::from_fn(unit.nrows(), unit.ncols(), |i, j| {
			&unit[(i, j)].re * scale
		})
	}

	/// complex counterpart of `scaled_real`
	fn scaled_cplx<T: RealField>(
		unit: MatRef<'_, Complex<T>>,
		scale: &T,
	) -> Mat<Complex<T>> {
		Mat::from_fn(unit.nrows(), unit.ncols(), |i, j| {
			Complex::new(&unit[(i, j)].re * scale, &unit[(i, j)].im * scale)
		})
	}

	/// the complex analogue of `PAIR_3X3`: the imaginary eigenvalue of the pair
	/// sits on the diagonal, and the off-diagonal entries have a large real and
	/// imaginary part, so `abs1 = |re| + |im|` overflows on its own
	fn complex_unit<T: RealField>() -> Mat<Complex<T>> {
		let z = |re: f64, im: f64| {
			Complex::new(from_f64::<T>(re), from_f64::<T>(im))
		};
		Mat::from_fn(3, 3, |i, j| match (i, j) {
			(0, 1) => z(-1.0, 0.5),
			(0, 2) => z(1.0, -1.0),
			(1, 1) => z(0.5, 0.0),
			(2, 2) => z(0.25, 0.0),
			_ => z(0.0, 0.0),
		})
	}

	/// `(||a v - lambda v||_2 / (||a||_1 * ||v||_2), ||v||_2)` for the
	/// dimensionless matrix `a`, so that no reference product overflows
	fn residual<T: RealField>(
		a: MatRef<'_, Complex<T>>,
		v: &[Complex<T>],
		lambda: &Complex<T>,
	) -> (T, T) {
		let n = a.nrows();
		let mut anorm = zero::<T>();
		for j in 0..n {
			let mut col = zero::<T>();
			for i in 0..n {
				col += &a[(i, j)].re.abs() + &a[(i, j)].im.abs();
			}
			anorm = anorm.fmax(col);
		}
		let mut rnorm2 = zero::<T>();
		let mut vnorm2 = zero::<T>();
		for i in 0..n {
			let mut acc = Complex::new(zero::<T>(), zero::<T>());
			for j in 0..n {
				acc += &a[(i, j)] * &v[j];
			}
			acc -= lambda * &v[i];
			rnorm2 += acc.re.abs2() + acc.im.abs2();
			vnorm2 += v[i].re.abs2() + v[i].im.abs2();
		}
		(rnorm2.sqrt() / (anorm * vnorm2.sqrt()), vnorm2.sqrt())
	}

	fn try_real_vectors<T: RealField>(t: MatRef<'_, T>) -> Option<Mat<T>> {
		let n = t.nrows();
		let mut x = Mat::zeros(n, n);
		let mut mem = MemBuffer::new(eigenvectors_from_schur_scratch::<T>(n));
		eigenvectors_from_schur_real(
			t,
			None,
			x.as_mut(),
			Par::Seq,
			MemStack::new(&mut mem),
			Default::default(),
		)
		.ok()?;
		Some(x)
	}

	fn real_vectors<T: RealField>(t: MatRef<'_, T>) -> Mat<T> {
		try_real_vectors(t).unwrap()
	}

	fn cplx_vectors<T: ComplexField>(t: MatRef<'_, T>) -> Mat<T> {
		let n = t.nrows();
		let mut x = Mat::zeros(n, n);
		let mut mem = MemBuffer::new(eigenvectors_from_schur_scratch::<T>(n));
		eigenvectors_from_schur_cplx(
			t,
			None,
			x.as_mut(),
			Par::Seq,
			MemStack::new(&mut mem),
			Default::default(),
		)
		.unwrap();
		x
	}

	/// the real column `j` as a complex vector, for the diagonal eigenvalue
	fn real_column<T: RealField>(
		x: MatRef<'_, T>,
		j: usize,
	) -> Vec<Complex<T>> {
		(0..x.nrows())
			.map(|i| Complex::new(x[(i, j)].copy(), zero::<T>()))
			.collect()
	}

	/// the conjugate pair `j`, `j + 1` decoded as `re + i im`
	fn real_pair<T: RealField>(x: MatRef<'_, T>, j: usize) -> Vec<Complex<T>> {
		(0..x.nrows())
			.map(|i| Complex::new(x[(i, j)].copy(), x[(i, j + 1)].copy()))
			.collect()
	}

	fn cplx_column<T: RealField>(
		x: MatRef<'_, Complex<T>>,
		j: usize,
	) -> Vec<Complex<T>> {
		(0..x.nrows()).map(|i| x[(i, j)].copy()).collect()
	}

	fn assert_eigenvector<T: RealField>(
		unit: MatRef<'_, Complex<T>>,
		v: &[Complex<T>],
		lambda: &Complex<T>,
		tol: &T,
	) {
		let (res, nrm) = residual(unit, v, lambda);
		let nrm_err = (&nrm - &one::<T>()).abs();
		assert!(nrm_err <= *tol, "unit 2-norm {nrm:?}, expected 1");
		assert!(res <= *tol, "relative residual {res:?}");
	}

	fn check_shift_2x2<T: RealField>(scale: T, tol: T) {
		let unit = unit::<T, 2>(&SHIFT_2X2);
		let lambda = Complex::new(from_f64::<T>(0.5), zero::<T>());

		let x = real_vectors::<T>(scaled_real(unit.as_ref(), &scale).as_ref());
		let v = real_column(x.as_ref(), 1);
		assert_eigenvector(unit.as_ref(), &v, &lambda, &tol);

		let x = cplx_vectors::<Complex<T>>(
			scaled_cplx(unit.as_ref(), &scale).as_ref(),
		);
		let v = cplx_column(x.as_ref(), 1);
		assert_eigenvector(unit.as_ref(), &v, &lambda, &tol);
	}

	/// `Ok` must carry eigenvectors; at the top of the range the conjugate
	/// pair's 2x2 solve may instead fail with an error.
	fn check_pair_3x3<T: RealField>(scale: T, tol: T, may_fail: bool) {
		let unit = unit::<T, 3>(&PAIR_3X3);
		let Some(x) =
			try_real_vectors::<T>(scaled_real(unit.as_ref(), &scale).as_ref())
		else {
			assert!(may_fail, "extraction failed at scale {scale:?}");
			return;
		};
		// the positive imaginary part comes first
		let lambda = Complex::new(zero::<T>(), one::<T>());
		let v = real_pair(x.as_ref(), 0);
		assert_eigenvector(unit.as_ref(), &v, &lambda, &tol);
		let lambda = Complex::new(from_f64::<T>(0.5), zero::<T>());
		let v = real_column(x.as_ref(), 2);
		assert_eigenvector(unit.as_ref(), &v, &lambda, &tol);
	}

	fn check_cplx_triangular<T: RealField>(scale: T, tol: T) {
		let unit = complex_unit::<T>();
		let t = scaled_cplx(unit.as_ref(), &scale);
		let x = cplx_vectors(t.as_ref());
		for j in 0..3 {
			let lambda = unit[(j, j)].copy();
			let v = cplx_column(x.as_ref(), j);
			assert_eigenvector(unit.as_ref(), &v, &lambda, &tol);
		}
	}

	#[test]
	fn two_by_two_at_the_range_limit() {
		// `f32` and `c32`, then `f64` and `c64`
		for (scale, tol) in [(1.0f32, 1e-4f32), (2.0f32.powi(127), 1e-4)] {
			check_shift_2x2::<f32>(scale, tol);
		}
		for (scale, tol) in [(1.0f64, 1e-12f64), (2.0f64.powi(1023), 1e-12)] {
			check_shift_2x2::<f64>(scale, tol);
		}
	}

	#[test]
	fn complex_triangular_at_the_range_limit() {
		for (scale, tol) in [(1.0f32, 1e-4f32), (2.0f32.powi(127), 1e-4)] {
			check_cplx_triangular::<f32>(scale, tol);
		}
		for (scale, tol) in [(1.0f64, 1e-12f64), (2.0f64.powi(1023), 1e-12)] {
			check_cplx_triangular::<f64>(scale, tol);
		}
	}

	#[test]
	fn conjugate_pair_block_is_never_a_wrong_success() {
		check_pair_3x3::<f32>(1.0, 1e-4, false);
		check_pair_3x3::<f32>(2.0f32.powi(127), 1e-4, true);
		check_pair_3x3::<f64>(1.0, 1e-12, false);
		check_pair_3x3::<f64>(2.0f64.powi(1023), 1e-12, true);
		// the second pair's vector goes through the 2x2 solve of the first
		// pair with a complex shift
		for (scale, tol, may_fail) in
			[(1.0, 1e-12, false), (2.0f64.powi(1023), 1e-12, true)]
		{
			let unit = unit::<f64, 4>(&TWO_PAIRS_4X4);
			let t = scaled_real(unit.as_ref(), &scale);
			match try_real_vectors::<f64>(t.as_ref()) {
				Some(x) => {
					for (j, im) in [(0, 1.0), (2, 0.5)] {
						let v = real_pair(x.as_ref(), j);
						assert_eigenvector(
							unit.as_ref(),
							&v,
							&Complex::new(0.0, im),
							&tol,
						);
					}
				},
				None => assert!(may_fail, "two pairs failed at scale 1"),
			}
		}
		// the shifted diagonal `A[i, i] - p` of the pair overflows here
		let s = 2.0f64.powi(1023);
		let t = crate::mat![[s, -s, 1.0], [s, s, 0.0], [0.0, 0.0, -s]];
		if let Some(x) = try_real_vectors::<f64>(t.as_ref()) {
			let unit = Mat::from_fn(3, 3, |i, j| {
				Complex::new(t[(i, j)] / s, 0.0)
			});
			let v = real_pair(x.as_ref(), 0);
			assert_eigenvector(
				unit.as_ref(),
				&v,
				&Complex::new(1.0, 1.0),
				&1e-12,
			);
		}
	}

	/// A pair whose off-diagonal entries span the whole range is solved by the
	/// ordinary block arithmetic, and must stay solved.
	#[test]
	fn conjugate_pair_with_a_wide_dynamic_range() {
		let t = crate::mat![
			[0.0, -2.0f64.powi(1023), 1.0],
			[2.0f64.powi(-1022), 0.0, 0.0],
			[0.0, 0.0, 0.5],
		];
		let x = real_vectors::<f64>(t.as_ref());
		assert!(x.is_all_finite());
		let unit = Mat::from_fn(3, 3, |i, j| Complex::new(t[(i, j)], 0.0));
		let lambda = Complex::new(0.5, 0.0);
		let v = real_column(x.as_ref(), 2);
		assert_eigenvector(unit.as_ref(), &v, &lambda, &1e-12);
	}

	/// A subnormal Schur form must keep the clamp the plain norm gives: summing
	/// the terms as `eps * |a_ij|` instead rounds the floor to zero and loses
	/// the only eigenvector of a repeated eigenvalue.
	#[test]
	fn subnormal_shift_clamp_is_unchanged() {
		let scale = 2.0f64.powi(-1023);
		let unit = unit::<f64, 2>(&JORDAN_2X2);
		let x =
			real_vectors::<f64>(scaled_real(unit.as_ref(), &scale).as_ref());
		let lambda = Complex::new(1.0, 0.0);
		let v = real_column(x.as_ref(), 1);
		assert_eigenvector(unit.as_ref(), &v, &lambda, &1e-12);
	}
}
#[cfg(test)]
mod self_adjoint_tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;
	fn test_self_adjoint_evd<T: ComplexField>(mat: MatRef<'_, T>) {
		let n = mat.nrows();
		let params = Spec::new(SelfAdjointEvdParams {
			recursion_threshold: 8,
			..auto!(T)
		});
		let approx_eq = CwiseMat(
			ApproxEq::<T::Real>::eps()
				* from_f64::<T::Real>(8.0 * n as f64).sqrt(),
		);
		let mut s = Mat::zeros(n, n);
		{
			let mut u = Mat::zeros(n, n);
			self_adjoint_evd(
				mat.as_ref(),
				s.as_mut().diagonal_mut(),
				Some(u.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					self_adjoint_evd_scratch::<T>(
						n,
						ComputeEigenvectors::Yes,
						ParShape::Seq,
						params,
					),
				)),
				params,
			)
			.unwrap();
			let reconstructed = &u * &s * u.adjoint();
			assert!(reconstructed ~ mat);
		}
		{
			let mut s2 = Mat::zeros(n, n);
			self_adjoint_evd(
				mat.as_ref(),
				s2.as_mut().diagonal_mut(),
				None,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					self_adjoint_evd_scratch::<T>(
						n,
						ComputeEigenvectors::No,
						ParShape::Seq,
						params,
					),
				)),
				params,
			)
			.unwrap();
			assert!(s2 ~ s);
		}
	}
	#[test]
	fn test_real() {
		let rng = &mut StdRng::seed_from_u64(1);
		for n in [1, 2, 4, 10, 15, 20, 50, 100, 150] {
			let mat = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			let mat = &mat + mat.adjoint();
			test_self_adjoint_evd(mat.as_ref());
		}
	}
	#[test]
	fn test_cplx() {
		let rng = &mut StdRng::seed_from_u64(1);
		for n in [2, 4, 10, 15, 20, 50, 100, 150] {
			let mat = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);
			let mat = &mat + mat.adjoint();
			test_self_adjoint_evd(mat.as_ref());
		}
	}
	#[test]
	fn test_special() {
		for n in [1, 2, 4, 10, 15, 20, 50, 100, 150] {
			test_self_adjoint_evd(Mat::full(n, n, 0.0).as_ref());
			test_self_adjoint_evd(Mat::full(n, n, c64::ZERO).as_ref());
			test_self_adjoint_evd(Mat::full(n, n, 1.0).as_ref());
			test_self_adjoint_evd(Mat::full(n, n, c64::ONE).as_ref());
			test_self_adjoint_evd(Mat::<f64>::identity(n, n).as_ref());
			test_self_adjoint_evd(Mat::<c64>::identity(n, n).as_ref());
		}
	}
	#[test]
	fn test_pinv() {
		let rng = &mut StdRng::seed_from_u64(0);
		let n = 36;
		let mat = CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: StandardNormal,
		}
		.rand::<Mat<f64>>(rng);
		let mat = &mat + mat.adjoint();
		let pinv = mat.self_adjoint_eigen(Side::Lower, Par::Seq).unwrap().pseudoinverse(Par::Seq);
		let err = &mat * &pinv - Mat::<f64>::identity(n, n);
		assert!(err.norm_max() < 1e-10);
	}
}
#[cfg(test)]
mod scale_and_empty_tests {
	//! Regression tests for `tensor4all/faer#42` and `#44`: the 2x2 shortcut
	//! averaged the diagonal as `(a + d) * 0.5`, which overflows for a finite
	//! spectrum near the floating-point limit, and the scratch query computed
	//! `n - 1` on an empty matrix.
	use super::*;
	#[test]
	fn two_by_two_finite_spectrum_at_the_range_limit() {
		let a = crate::mat![[1e308f64, 0.0], [0.0, 1e308]];
		let eig = a.self_adjoint_eigen(Side::Lower, Par::Seq).unwrap();
		assert_eq!(eig.S()[0], 1e308);
		assert_eq!(eig.S()[1], 1e308);
		let a = crate::mat![[1e38f32, 0.0], [0.0, 1e38]];
		let eig = a.self_adjoint_eigen(Side::Lower, Par::Seq).unwrap();
		assert_eq!(eig.S()[0], 1e38);
		assert_eq!(eig.S()[1], 1e38);
	}
	/// halving a subnormal is not exact, so the direct trace form stays in
	/// charge wherever it does not overflow
	#[test]
	fn two_by_two_subnormal_diagonal() {
		let q = f64::from_bits(1);
		let a = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { q } else { 0.0 });
		let eig = a.self_adjoint_eigen(Side::Lower, Par::Seq).unwrap();
		assert_eq!(eig.S()[0], q);
		assert_eq!(eig.S()[1], q);
	}
	#[test]
	fn empty_matrix() {
		let a = Mat::<f64>::zeros(0, 0);
		let eig = a.self_adjoint_eigen(Side::Lower, Par::Seq).unwrap();
		assert_eq!(eig.U().shape(), (0, 0));
		assert_eq!(eig.S().dim(), 0);
	}
	/// the tridiagonal entry point with eigenvectors, for an empty complex
	/// matrix (`tensor4all/faer#31`)
	#[test]
	fn empty_tridiagonal_with_vectors() {
		fn check<T: ComplexField>() {
			let empty = Diag::<T>::zeros(0);
			let mut s = Diag::<T>::zeros(0);
			let mut u = Mat::<T>::zeros(0, 0);
			let params = default();
			let mut mem = dyn_stack::MemBuffer::new(
				tridiagonal_self_adjoint_evd_scratch::<T>(
					0,
					ComputeEigenvectors::Yes,
					ParShape::Seq,
					params,
				),
			);
			tridiagonal_self_adjoint_evd(
				empty.as_ref(),
				empty.as_ref(),
				s.as_mut(),
				Some(u.as_mut()),
				Par::Seq,
				MemStack::new(&mut mem),
				params,
			)
			.unwrap();
		}
		check::<f64>();
		check::<c64>();
		check::<c32>();
	}
}
#[cfg(test)]
mod tridiagonal_range_tests {
	//! Regression tests for `tensor4all/faer#57`: the tridiagonal backend
	//! normalized by the reciprocal of the largest entry, which has no finite
	//! value for a subnormal spectrum, and the divide-and-conquer backend did
	//! not normalize at all, so a spectrum near the top of the range overflowed
	//! instead. Both spectra below are finite and exactly representable.
	use super::*;
	use crate::assert;
	use crate::diag::{DiagMut, DiagRef};
	use crate::dyn_stack::MemBuffer;

	fn tridiagonal_evd(
		diag: &[f64],
		offdiag: &[f64],
		threshold: usize,
	) -> Result<Vec<f64>, EvdError> {
		let n = diag.len();
		let mut s = vec![0.0; n];
		let mut u = Mat::<f64>::zeros(n, n);
		let mut params = Spec::<SelfAdjointEvdParams, f64>::default();
		params.recursion_threshold = threshold;
		let mut mem =
			MemBuffer::new(tridiagonal_self_adjoint_evd_scratch::<f64>(
				n,
				ComputeEigenvectors::Yes,
				crate::ParShape::Seq,
				params,
			));
		tridiagonal_self_adjoint_evd(
			DiagRef::from_slice(diag),
			DiagRef::from_slice(offdiag),
			DiagMut::from_slice_mut(&mut s),
			Some(u.as_mut()),
			Par::Seq,
			MemStack::new(&mut mem),
			params,
		)?;
		Ok(s)
	}

	/// the reported matrix, and the same 2x2 with a coupling that is not
	/// negligible
	#[test]
	fn two_by_two_at_the_range_limit() {
		let s = tridiagonal_evd(&[-1e308, 1e308], &[0.0], usize::MAX).unwrap();
		assert_eq!(s, [-1e308, 1e308]);
		let s =
			tridiagonal_evd(&[-1e308, 1e308], &[1e308], usize::MAX).unwrap();
		let sqrt2 = 2.0f64.sqrt();
		assert_eq!(s[0], -sqrt2 * 1e308);
		assert_eq!(s[1], sqrt2 * 1e308);
	}

	/// the reported subnormal matrix, whose spectrum is `s * (2 - sqrt2, 2,
	/// 2 + sqrt2)`
	#[test]
	fn subnormal_three_by_three() {
		let s = 1e-310f64;
		let sqrt2 = 2.0f64.sqrt();
		let got =
			tridiagonal_evd(&[2.0 * s; 3], &[-s, -s], usize::MAX).unwrap();
		for (got, exact) in
			got.iter()
				.zip([s * (2.0 - sqrt2), 2.0 * s, s * (2.0 + sqrt2)])
		{
			assert!(
				(got - exact).abs() <= 1e-13 * s,
				"got {got:e}, exact {exact:e}"
			);
		}
	}

	/// both backends, at both ends of the range, against the closed form of a
	/// constant tridiagonal matrix
	#[test]
	fn spectrum_at_both_range_limits() {
		let n = 5;
		for scale in [1e308f64, 1e-310] {
			let diag = vec![scale; n];
			let offdiag = vec![0.3 * scale; n - 1];
			// `usize::MAX` keeps the QR fallback in charge, `4` recurses
			for threshold in [usize::MAX, 4] {
				let s = tridiagonal_evd(&diag, &offdiag, threshold).unwrap();
				for (k, x) in s.iter().enumerate() {
					let exact = scale
						* (1.0
							+ 0.6
								* (core::f64::consts::PI * (n - k) as f64
									/ (n + 1) as f64)
									.cos());
					assert!(x.is_finite());
					assert!(
						(x - exact).abs() <= 1e-13 * scale,
						"threshold {threshold}: got {x:e}, exact {exact:e}"
					);
				}
			}
		}
	}
}

#[cfg(test)]
#[path = "tests/mod.rs"]
mod lapack;
