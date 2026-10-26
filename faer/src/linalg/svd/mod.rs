//! low level implementation of the svd of a matrix
//!
//! the svd of a matrix $A$ of shape $(m, n)$ is a decomposition into three
//! components $U$, $S$, and $V$, such that:
//!
//! - $U$ has shape $(m, m)$ and is a unitary matrix
//! - $V$ has shape $(n, n)$ and is a unitary matrix
//! - $S$ has shape $(m, n)$ and is zero everywhere except the main diagonal
//! - and finally:
//!
//! $$A = U S V^H$$
use crate::assert;
use crate::internal_prelude::*;
use bidiag::BidiagParams;
use linalg::qr::no_pivoting::factor::QrParams;
/// bidiagonalization
pub mod bidiag;
pub(crate) mod bidiag_svd;
/// whether the singular vectors should be computed
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComputeSvdVectors {
	/// do not compute singular vectors
	No,
	/// compute the first $\min(\text{nrows}, \text{ncols})$ singular vectors
	Thin,
	/// compute singular vectors
	Full,
}
/// svd error
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SvdError {
	/// reached max iterations
	NoConvergence,
}
/// svd tuning parameters
#[derive(Debug, Copy, Clone)]
pub struct SvdParams {
	/// bidiagonalization parameters
	pub bidiag: BidiagParams,
	/// $QR$ parameters
	pub qr: QrParams,
	/// threshold at which the implementation should stop recursing
	pub recursion_threshold: usize,
	/// threshold at which parallelism should be disabled
	pub qr_ratio_threshold: f64,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}
impl<T: ComplexField> Auto<T> for SvdParams {
	fn auto() -> Self {
		Self {
			recursion_threshold: 128,
			qr_ratio_threshold: 11.0 / 6.0,
			bidiag: auto!(T),
			qr: auto!(T),
			non_exhaustive: NonExhaustive(()),
		}
	}
}
fn svd_imp_scratch<T: ComplexField>(
	m: usize,
	n: usize,
	compute_u: ComputeSvdVectors,
	compute_v: ComputeSvdVectors,
	bidiag_svd_scratch: fn(
		n: usize,
		compute_u: bool,
		compute_v: bool,
		par: Par,
		params: SvdParams,
	) -> StackReq,
	params: SvdParams,
	par: Par,
) -> StackReq {
	assert!(m >= n);
	let householder_block_size =
		linalg::qr::no_pivoting::factor::recommended_block_size::<T>(m, n);
	let bid = temp_mat_scratch::<T>(m, n);
	let householder_left = temp_mat_scratch::<T>(householder_block_size, n);
	let householder_right = temp_mat_scratch::<T>(householder_block_size, n);
	let compute_bidiag =
		bidiag::bidiag_in_place_scratch::<T>(m, n, par, params.bidiag.into());
	let diag = temp_mat_scratch::<T>(n, 1);
	let subdiag = diag;
	let compute_ub = compute_v != ComputeSvdVectors::No;
	let compute_vb = compute_u != ComputeSvdVectors::No;
	let u_b = temp_mat_scratch::<T>(if compute_ub { n + 1 } else { 2 }, n + 1);
	let v_b = temp_mat_scratch::<T>(n, if compute_vb { n } else { 0 });
	let compute_bidiag_svd =
		bidiag_svd_scratch(n, compute_ub, compute_vb, par, params);
	let apply_householder_u = linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
		m,
		householder_block_size,
		match compute_u {
			ComputeSvdVectors::No => 0,
			ComputeSvdVectors::Thin => n,
			ComputeSvdVectors::Full => m,
		},
	);
	let apply_householder_v = linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
		n - 1,
		householder_block_size,
		match compute_v {
			ComputeSvdVectors::No => 0,
			_ => n,
		},
	);
	StackReq::all_of(&[
		bid,
		householder_left,
		householder_right,
		StackReq::any_of(&[
			compute_bidiag,
			StackReq::all_of(&[
				diag,
				subdiag,
				u_b,
				v_b,
				StackReq::any_of(&[
					compute_bidiag_svd,
					StackReq::all_of(&[
						apply_householder_u,
						apply_householder_v,
					]),
				]),
			]),
		]),
	])
}
fn bidiag_cplx_svd_scratch<T: ComplexField>(
	n: usize,
	compute_u: bool,
	compute_v: bool,
	par: Par,
	params: SvdParams,
) -> StackReq {
	StackReq::all_of(&[
		temp_mat_scratch::<T>(n, 1).array(4),
		temp_mat_scratch::<T::Real>(n + 1, if compute_u { n + 1 } else { 0 }),
		temp_mat_scratch::<T::Real>(n, if compute_v { n } else { 0 }),
		bidiag_real_svd_scratch::<T::Real>(
			n, compute_u, compute_v, par, params,
		),
	])
}
fn bidiag_real_svd_scratch<T: RealField>(
	n: usize,
	compute_u: bool,
	compute_v: bool,
	par: Par,
	params: SvdParams,
) -> StackReq {
	if n < params.recursion_threshold {
		StackReq::EMPTY
	} else {
		StackReq::all_of(&[
			temp_mat_scratch::<T>(2, if compute_u { 0 } else { n + 1 }),
			bidiag_svd::divide_and_conquer_scratch::<T>(
				n,
				params.recursion_threshold,
				compute_u,
				compute_v,
				par,
			),
		])
	}
}
fn compute_bidiag_cplx_svd<T: ComplexField>(
	mut diag: ColMut<'_, T, usize, ContiguousFwd>,
	subdiag: ColMut<'_, T, usize, ContiguousFwd>,
	mut u: Option<MatMut<'_, T>>,
	mut v: Option<MatMut<'_, T>>,
	params: SvdParams,
	par: Par,
	stack: &mut MemStack,
) -> Result<(), SvdError> {
	let n = diag.nrows();
	let (mut diag_real, stack) =
		unsafe { temp_mat_uninit::<T::Real, _, _>(n, 1, stack) };
	let (mut subdiag_real, stack) =
		unsafe { temp_mat_uninit::<T::Real, _, _>(n, 1, stack) };
	let (mut u_real, stack) = unsafe {
		temp_mat_uninit::<T::Real, _, _>(
			n + 1,
			if u.is_some() { n + 1 } else { 0 },
			stack,
		)
	};
	let (mut v_real, stack) = unsafe {
		temp_mat_uninit::<T::Real, _, _>(
			n,
			if v.is_some() { n } else { 0 },
			stack,
		)
	};
	let (mut col_mul, stack) =
		unsafe { temp_mat_uninit::<T, _, _>(n, 1, stack) };
	let (mut row_mul, stack) =
		unsafe { temp_mat_uninit::<T, _, _>(n - 1, 1, stack) };
	let mut u_real = u.rb().map(|_| u_real.as_mat_mut());
	let mut v_real = v.rb().map(|_| v_real.as_mat_mut());
	let mut diag_real = diag_real
		.as_mat_mut()
		.col_mut(0)
		.try_as_col_major_mut()
		.unwrap();
	let mut subdiag_real = subdiag_real
		.as_mat_mut()
		.col_mut(0)
		.try_as_col_major_mut()
		.unwrap();
	let mut col_mul = col_mul.as_mat_mut().col_mut(0);
	let mut row_mul = row_mul.as_mat_mut().col_mut(0);
	let normalized = |x: T| {
		if x == zero() {
			one()
		} else {
			let norm1 = x.real().abs().fmax(x.imag().abs());
			// a subnormal component scale has an infinite reciprocal; rescale it the
			// way reference `dlarfg` does instead of dividing by a zero product
			let ref shift = if norm1 >= min_positive::<T::Real>() {
				from_f64::<T::Real>(1.0)
			} else {
				from_f64::<T::Real>(4503599627370496.0)
			};
			let y = x.mul_real(shift).mul_real((&norm1 * shift).recip());
			y.mul_real(y.abs().recip())
		}
	};
	let magnitude = |x: &T| {
		// `abs` squares components scaled by `sqrt(min_positive)` or
		// `sqrt(max_positive)`, which flushes a magnitude below `min_positive`
		// to zero and overflows above about `max_positive / 2`. Outside
		// `[min_positive, 1 / min_positive]`, rescale by that exact power of
		// two first; other entries keep the historical `abs` bit for bit.
		let sml = min_positive::<T::Real>();
		let big = sml.recip();
		let m = x.real().abs().fmax(x.imag().abs());
		if m < sml && m != zero() {
			x.mul_real(&big).abs() * sml
		} else if m > big && m.is_finite() {
			x.mul_real(&sml).abs() * big
		} else {
			x.abs()
		}
	};
	let mut col_normalized = normalized(diag[0].conj());
	col_mul[0] = col_normalized.copy();
	diag_real[0] = magnitude(&diag[0]);
	subdiag_real[n - 1] = zero();
	for i in 1..n {
		let row_normalized =
			normalized((&subdiag[i - 1] * col_normalized).conj());
		subdiag_real[i - 1] = magnitude(&subdiag[i - 1]);
		row_mul[i - 1] = row_normalized.conj();
		col_normalized = normalized((&diag[i] * row_normalized).conj());
		diag_real[i] = magnitude(&diag[i]);
		col_mul[i] = col_normalized.copy();
	}
	compute_bidiag_real_svd(
		diag_real.rb_mut(),
		subdiag_real.rb_mut(),
		u_real.rb_mut(),
		v_real.rb_mut(),
		params,
		par,
		stack,
	)?;
	for i in 0..n {
		diag[i] = diag_real[i].to_cplx();
	}
	let u_real = u_real.rb();
	let v_real = v_real.rb();
	if let (Some(mut u), Some(u_real)) = (u.rb_mut(), u_real) {
		z!(u.rb_mut().row_mut(0), u_real.row(0))
			.for_each(|uz!(u, r)| *u = r.to_cplx());
		z!(u.rb_mut().row_mut(n), u_real.row(n))
			.for_each(|uz!(u, r)| *u = r.to_cplx());
		for j in 0..u.ncols() {
			let mut u = u.rb_mut().col_mut(j).subrows_mut(1, n - 1);
			let u_real = u_real.rb().col(j).subrows(1, n - 1);
			z!(u.rb_mut(), u_real, row_mul.rb())
				.for_each(|uz!(u, re, f)| *u = f.mul_real(re));
		}
	}
	if let (Some(mut v), Some(v_real)) = (v.rb_mut(), v_real) {
		for j in 0..v.ncols() {
			let mut v = v.rb_mut().col_mut(j);
			let v_real = v_real.rb().col(j);
			z!(v.rb_mut(), v_real, col_mul.rb())
				.for_each(|uz!(v, re, f)| *v = f.mul_real(re));
		}
	}
	Ok(())
}
fn compute_bidiag_real_svd<T: RealField>(
	mut diag: ColMut<'_, T, usize, ContiguousFwd>,
	mut subdiag: ColMut<'_, T, usize, ContiguousFwd>,
	mut u: Option<MatMut<'_, T, usize, usize>>,
	mut v: Option<MatMut<'_, T, usize, usize>>,
	params: SvdParams,
	par: Par,
	stack: &mut MemStack,
) -> Result<(), SvdError> {
	let n = diag.nrows();
	for i in 0..n {
		if !(diag[i].is_finite() && subdiag[i].is_finite()) {
			return Err(SvdError::NoConvergence);
		}
	}
	if n < params.recursion_threshold {
		if let Some(mut u) = u.rb_mut() {
			u.fill(zero());
			u.diagonal_mut().fill(one());
		}
		if let Some(mut v) = v.rb_mut() {
			v.fill(zero());
			v.diagonal_mut().fill(one());
		}
		bidiag_svd::qr_algorithm(
			diag.rb_mut(),
			subdiag.rb_mut(),
			u.rb_mut().map(|u| u.submatrix_mut(0, 0, n, n)),
			v.rb_mut(),
		)?;
		return Ok(());
	} else {
		let (mut u2, stack) = unsafe {
			temp_mat_uninit::<T::Real, _, _>(
				2,
				if u.is_some() { 0 } else { n + 1 },
				stack,
			)
		};
		bidiag_svd::divide_and_conquer(
			diag.as_row_shape_mut(n),
			subdiag.as_row_shape_mut(n),
			match u {
				Some(u) => bidiag_svd::MatU::Full(u),
				None => bidiag_svd::MatU::TwoRowsStorage(u2.as_mat_mut()),
			},
			v.map(|m| m.as_shape_mut(n, n)),
			par,
			stack,
			params.recursion_threshold,
		)
	}
}
/// Workspace for [`bidiagonal_svd`], including the padded
/// lower-bidiagonal storage used by the existing divide-and-conquer
/// implementation.
///
/// # Examples
/// ```
/// use faer::Par;
/// use faer::linalg::svd::bidiagonal_svd_scratch;
/// let _buffer = dyn_stack::MemBuffer::new(bidiagonal_svd_scratch::<f64>(
/// 	3,
/// 	Par::Seq,
/// 	Default::default(),
/// ));
/// ```
pub fn bidiagonal_svd_scratch<T: ComplexField>(
	dim: usize,
	par: Par,
	params: Spec<SvdParams, T>,
) -> StackReq {
	let n = dim;
	if n == 0 {
		return StackReq::EMPTY;
	}
	StackReq::all_of(&[
		temp_mat_scratch::<T>(n, 1).array(2),
		temp_mat_scratch::<T>(n + 1, n + 1),
		temp_mat_scratch::<T>(n, n),
		bidiag_cplx_svd_scratch::<T>(n, true, true, par, *params),
	])
}
/// Computes a square bidiagonal SVD `B = U * diag(s) * V.adjoint()`.
/// `diag` is overwritten with nonnegative, descending singular values (with
/// zero imaginary part). `offdiag` has length `n.saturating_sub(1)` and is
/// unchanged. `upper` selects the superdiagonal rather than the subdiagonal.
/// Both `u` and `v` are `n` by `n`; the output is **V**, not its adjoint.
/// Real and complex entries are supported via the existing phase reduction.
///
/// # Panics
/// Panics on incompatible dimensions or insufficient scratch.
///
/// # Errors
/// Returns [`SvdError::NoConvergence`] for nonfinite input, iteration
/// failure, or a result that would not be finite (for example when a complex
/// entry's modulus overflows). Finite input never yields `Ok` with nonfinite
/// output. Outputs are unspecified on error; `offdiag` is always unchanged.
///
/// # Examples
/// ```
/// use dyn_stack::{MemBuffer, MemStack};
/// use faer::linalg::svd::*;
/// use faer::{Mat, Par};
/// let mut d = [3.0f64, -2.0];
/// let (mut u, mut v) = (Mat::zeros(2, 2), Mat::zeros(2, 2));
/// let mut mem = MemBuffer::new(bidiagonal_svd_scratch::<f64>(
/// 	2,
/// 	Par::Seq,
/// 	Default::default(),
/// ));
/// bidiagonal_svd(
/// 	&mut d,
/// 	&[0.0],
/// 	true,
/// 	u.as_mut(),
/// 	v.as_mut(),
/// 	Par::Seq,
/// 	MemStack::new(&mut mem),
/// 	Default::default(),
/// )
/// .unwrap();
/// assert_eq!(d, [3.0, 2.0]);
/// ```
pub fn bidiagonal_svd<T: ComplexField>(
	diag: &mut [T],
	offdiag: &[T],
	upper: bool,
	mut u: MatMut<'_, T>,
	mut v: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<SvdParams, T>,
) -> Result<(), SvdError> {
	let n = diag.len();
	assert!(offdiag.len() == n.saturating_sub(1));
	assert!(
		u.nrows() == n && u.ncols() == n && v.nrows() == n && v.ncols() == n
	);
	if !diag.iter().chain(offdiag).all(|x| x.is_finite()) {
		return Err(SvdError::NoConvergence);
	}
	if n == 0 {
		return Ok(());
	}
	// SAFETY: Every entry is written below before the kernel reads the
	// diagonal.
	let (mut d, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, 1, stack) };
	let (mut e, stack) = temp_mat_zeroed::<T, _, _>(n, 1, stack);
	// SAFETY: The phase-reduction kernel fully writes each factor on success;
	// its error path returns before either factor is read here.
	let (mut left, stack) =
		unsafe { temp_mat_uninit::<T, _, _>(n + 1, n + 1, stack) };
	// SAFETY: Same full-overwrite contract as the left factor above.
	let (mut right, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
	let mut d = d.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap();
	let mut e = e.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap();
	for i in 0..n {
		d[i] = if upper {
			diag[i].conj()
		} else {
			diag[i].copy()
		};
		if i + 1 < n {
			e[i] = if upper {
				offdiag[i].conj()
			} else {
				offdiag[i].copy()
			};
		}
	}
	compute_bidiag_cplx_svd(
		d.rb_mut(),
		e.rb_mut(),
		Some(left.as_mat_mut()),
		Some(right.as_mat_mut()),
		*params,
		par,
		stack,
	)?;
	// Backstop: finite input never reports success with nonfinite output.
	if !(d.rb().is_all_finite()
		&& left.as_mat_ref().submatrix(0, 0, n, n).is_all_finite()
		&& right.as_mat_ref().is_all_finite())
	{
		return Err(SvdError::NoConvergence);
	}
	for i in 0..n {
		diag[i] = d[i].copy();
	}
	if upper {
		u.copy_from(right.as_mat_ref());
		v.copy_from(left.as_mat_ref().submatrix(0, 0, n, n));
	} else {
		u.copy_from(left.as_mat_ref().submatrix(0, 0, n, n));
		v.copy_from(right.as_mat_ref());
	}
	Ok(())
}
/// bidiag -> divide conquer svd / qr algo
fn svd_imp<T: ComplexField>(
	matrix: MatRef<'_, T>,
	s: ColMut<'_, T>,
	u: Option<MatMut<'_, T>>,
	v: Option<MatMut<'_, T>>,
	bidiag_svd: fn(
		diag: ColMut<'_, T, usize, ContiguousFwd>,
		subdiag: ColMut<'_, T, usize, ContiguousFwd>,
		u: Option<MatMut<'_, T, usize, usize>>,
		v: Option<MatMut<'_, T, usize, usize>>,
		params: SvdParams,
		par: Par,
		stack: &mut MemStack,
	) -> Result<(), SvdError>,
	par: Par,
	stack: &mut MemStack,
	params: SvdParams,
) -> Result<(), SvdError> {
	assert!(matrix.nrows() >= matrix.ncols());
	let m = matrix.nrows();
	let n = matrix.ncols();
	let bs = linalg::qr::no_pivoting::factor::recommended_block_size::<T>(m, n);
	let (mut bid, stack) = unsafe { temp_mat_uninit::<T, _, _>(m, n, stack) };
	let mut bid = bid.as_mat_mut();
	let (mut Hl, stack) = unsafe { temp_mat_uninit::<T, _, _>(bs, n, stack) };
	let (mut Hr, stack) =
		unsafe { temp_mat_uninit::<T, _, _>(bs, n - 1, stack) };
	let mut Hl = Hl.as_mat_mut();
	let mut Hr = Hr.as_mat_mut();
	bid.copy_from(matrix);
	bidiag::bidiag_in_place(
		bid.rb_mut(),
		Hl.rb_mut(),
		Hr.rb_mut(),
		par,
		stack,
		params.bidiag.into(),
	);
	let (mut diag, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, 1, stack) };
	let (mut subdiag, stack) =
		unsafe { temp_mat_uninit::<T, _, _>(n, 1, stack) };
	let mut diag = diag.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap();
	let mut subdiag = subdiag
		.as_mat_mut()
		.col_mut(0)
		.try_as_col_major_mut()
		.unwrap();
	let (mut ub, stack) = unsafe {
		temp_mat_uninit::<T, _, _>(
			n + 1,
			if v.is_some() { n + 1 } else { 0 },
			stack,
		)
	};
	let (mut vb, stack) = unsafe {
		temp_mat_uninit::<T, _, _>(n, if u.is_some() { n } else { 0 }, stack)
	};
	let mut ub = ub.as_mat_mut();
	let mut vb = vb.as_mat_mut();
	for i in 0..n {
		diag[i] = bid[(i, i)].conj();
		if i + 1 < n {
			subdiag[i] = bid[(i, i + 1)].conj();
		} else {
			subdiag[i] = zero();
		}
	}
	bidiag_svd(
		diag.rb_mut(),
		subdiag.rb_mut(),
		v.rb().map(|_| ub.rb_mut()),
		u.rb().map(|_| vb.rb_mut()),
		params,
		par,
		stack,
	)?;
	{ s }.copy_from(diag);
	if let Some(mut u) = u {
		let ncols = u.ncols();
		u.rb_mut().submatrix_mut(0, 0, n, n).copy_from(vb.rb());
		u.rb_mut().submatrix_mut(n, 0, m - n, ncols).fill(zero());
		u.rb_mut().submatrix_mut(0, n, n, ncols - n).fill(zero());
		u.rb_mut()
			.submatrix_mut(n, n, ncols - n, ncols - n)
			.diagonal_mut()
			.fill(one());
		linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(bid.rb(), Hl.rb(), Conj::No, u, par, stack);
	}
	if let Some(mut v) = v {
		v.copy_from(ub.rb().submatrix(0, 0, n, n));
		for j in 1..n {
			for i in 0..j {
				bid[(j, i)] = bid[(i, j)].copy();
			}
		}
		linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
			bid.rb().submatrix(1, 0, n - 1, n - 1),
			Hr.rb(),
			Conj::Yes,
			v.subrows_mut(1, n - 1),
			par,
			stack,
		);
	}
	Ok(())
}
fn compute_squareish_svd<T: ComplexField>(
	matrix: MatRef<'_, T>,
	s: ColMut<'_, T>,
	u: Option<MatMut<'_, T>>,
	v: Option<MatMut<'_, T>>,
	par: Par,
	stack: &mut MemStack,
	params: SvdParams,
) -> Result<(), SvdError> {
	if const { T::IS_REAL } {
		svd_imp::<T::Real>(
			unsafe { core::mem::transmute(matrix) },
			unsafe { core::mem::transmute(s) },
			unsafe { core::mem::transmute(u) },
			unsafe { core::mem::transmute(v) },
			compute_bidiag_real_svd::<T::Real>,
			par,
			stack,
			params,
		)
	} else {
		svd_imp::<T>(
			matrix,
			s,
			u,
			v,
			compute_bidiag_cplx_svd::<T>,
			par,
			stack,
			params,
		)
	}
}
/// computes the layout of the workspace required to compute a matrix's svd
pub fn svd_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	compute_u: ComputeSvdVectors,
	compute_v: ComputeSvdVectors,
	par: Par,
	params: Spec<SvdParams, T>,
) -> StackReq {
	let params = params.config;
	let mut m = nrows;
	let mut n = ncols;
	let mut compute_u = compute_u;
	let mut compute_v = compute_v;
	if n > m {
		core::mem::swap(&mut m, &mut n);
		core::mem::swap(&mut compute_u, &mut compute_v);
	}
	if n == 0 {
		return StackReq::EMPTY;
	}
	let bidiag_svd_scratch = if const { T::IS_REAL } {
		bidiag_real_svd_scratch::<T::Real>
	} else {
		bidiag_cplx_svd_scratch::<T>
	};
	if m as f64 / n as f64 <= params.qr_ratio_threshold {
		svd_imp_scratch::<T>(
			m,
			n,
			compute_u,
			compute_v,
			bidiag_svd_scratch,
			params,
			par,
		)
	} else {
		let bs =
			linalg::qr::no_pivoting::factor::recommended_block_size::<T>(m, n);
		StackReq::all_of(&[
			temp_mat_scratch::<T>(m, n),
			temp_mat_scratch::<T>(bs, n),
			StackReq::any_of(&[
				StackReq::all_of(&[
					temp_mat_scratch::<T>(n, n),
					svd_imp_scratch::<T>(n, n, compute_u, compute_v, bidiag_svd_scratch, params, par),
				]),
				linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
					m,
					bs,
					match compute_u {
						ComputeSvdVectors::No => 0,
						ComputeSvdVectors::Thin => n,
						ComputeSvdVectors::Full => m,
					},
				),
			]),
		])
	}
}
/// computes the svd of $A$, with the singular vectors being omitted, thin or
/// full
///
/// the singular are stored in $S$, and the singular vectors in $U$ and $V$ such
/// that the singular values are sorted in nonincreasing order
pub fn svd<T: ComplexField>(
	A: MatRef<'_, T>,
	s: DiagMut<'_, T>,
	u: Option<MatMut<'_, T>>,
	v: Option<MatMut<'_, T>>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<SvdParams, T>,
) -> Result<(), SvdError> {
	let params = params.config;
	let (m, n) = A.shape();
	let size = Ord::min(m, n);
	assert!(s.dim() == size);
	let s = s.column_vector_mut();
	if let Some(u) = u.rb() {
		assert!(all(
			u.nrows() == A.nrows(),
			any(u.ncols() == A.nrows(), u.ncols() == size),
		));
	}
	if let Some(v) = v.rb() {
		assert!(all(
			v.nrows() == A.ncols(),
			any(v.ncols() == A.ncols(), v.ncols() == size),
		));
	}
	#[cfg(feature = "perf-warn")]
	match (u.rb(), v.rb()) {
		(Some(matrix), _) | (_, Some(matrix)) => {
			if matrix.row_stride().unsigned_abs() != 1
				&& crate::__perf_warn!(QR_WARN)
			{
				if matrix.col_stride().unsigned_abs() == 1 {
					log::warn!(
						target : "faer_perf",
						"SVD prefers column-major singular vector matrices. Found row-major matrix."
					);
				} else {
					log::warn!(
						target : "faer_perf",
						"SVD prefers column-major singular vector matrices. Found matrix with generic strides."
					);
				}
			}
		},
		_ => {},
	}
	let mut u = u;
	let mut v = v;
	let mut matrix = A;
	let do_transpose = n > m;
	if do_transpose {
		matrix = matrix.transpose();
		core::mem::swap(&mut u, &mut v)
	}
	let (m, n) = matrix.shape();
	if n == 0 {
		if let Some(mut u) = u {
			u.fill(zero());
			u.rb_mut().diagonal_mut().fill(one());
		}
		return Ok(());
	}
	if m as f64 / n as f64 <= params.qr_ratio_threshold {
		compute_squareish_svd(
			matrix,
			s,
			u.rb_mut(),
			v.rb_mut(),
			par,
			stack,
			params,
		)?;
	} else {
		let bs =
			linalg::qr::no_pivoting::factor::recommended_block_size::<T>(m, n);
		let (mut qr, stack) =
			unsafe { temp_mat_uninit::<T, _, _>(m, n, stack) };
		let mut qr = qr.as_mat_mut();
		let (mut householder, stack) =
			unsafe { temp_mat_uninit::<T, _, _>(bs, n, stack) };
		let mut householder = householder.as_mat_mut();
		{
			qr.copy_from(matrix.rb());
			linalg::qr::no_pivoting::factor::qr_in_place(
				qr.rb_mut(),
				householder.rb_mut(),
				par,
				stack,
				params.qr.into(),
			);
		}
		{
			let (mut r, stack) =
				unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
			let mut r = r.as_mat_mut();
			z!(r.rb_mut()).for_each_triangular_lower(
				linalg::zip::Diag::Skip,
				|uz!(dst)| *dst = zero(),
			);
			z!(r.rb_mut(), qr.rb().submatrix(0, 0, n, n))
				.for_each_triangular_upper(
					linalg::zip::Diag::Include,
					|uz!(dst, src)| *dst = src.copy(),
				);
			compute_squareish_svd(
				r.rb(),
				s,
				u.rb_mut().map(|u| u.submatrix_mut(0, 0, n, n)),
				v.rb_mut(),
				par,
				stack,
				params,
			)?;
		}
		if let Some(mut u) = u.rb_mut() {
			u.rb_mut().subrows_mut(n, m - n).fill(zero());
			if u.ncols() == m {
				// the extra columns are `Q * [0; I]`, so their top block has
				// to start at zero instead of at the caller's old entries
				u.rb_mut().submatrix_mut(0, n, n, m - n).fill(zero());
				u.rb_mut()
					.submatrix_mut(n, n, m - n, m - n)
					.diagonal_mut()
					.fill(one());
			}
			linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
				qr.rb(),
				householder.rb(),
				Conj::No,
				u.rb_mut(),
				par,
				stack,
			);
		}
	}
	if do_transpose {
		if let Some(u) = u.rb_mut() {
			z!(u).for_each(|uz!(u)| *u = u.conj())
		}
		if let Some(v) = v.rb_mut() {
			z!(v).for_each(|uz!(v)| *v = v.conj())
		}
	}
	Ok(())
}
/// computes the layout of the workspace required to compute a matrix's
/// pseudoinverse, given the svd
pub fn pseudoinverse_from_svd_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	par: Par,
) -> StackReq {
	_ = par;
	let size = Ord::min(nrows, ncols);
	StackReq::all_of(&[
		temp_mat_scratch::<T>(nrows, size),
		temp_mat_scratch::<T>(ncols, size),
	])
}
/// computes a self-adjoint matrix's pseudoinverse, given the svd factors $S$,
/// $U$ and $V$
pub fn pseudoinverse_from_svd<T: ComplexField>(
	pinv: MatMut<'_, T>,
	s: DiagRef<'_, T>,
	u: MatRef<'_, T>,
	v: MatRef<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	pseudoinverse_from_svd_with_tolerance(
		pinv,
		s,
		u,
		v,
		zero(),
		eps::<T::Real>()
			* from_f64::<T::Real>(Ord::max(u.nrows(), v.nrows()) as f64),
		par,
		stack,
	);
}
/// computes a self-adjoint matrix's pseudoinverse, given the svd factors $S$,
/// $U$ and $V$, and tolerance parameters for determining zero singular values
pub fn pseudoinverse_from_svd_with_tolerance<T: ComplexField>(
	pinv: MatMut<'_, T>,
	s: DiagRef<'_, T>,
	u: MatRef<'_, T>,
	v: MatRef<'_, T>,
	abs_tol: T::Real,
	rel_tol: T::Real,
	par: Par,
	stack: &mut MemStack,
) {
	let mut pinv = pinv;
	let m = u.nrows();
	let n = v.nrows();
	let size = Ord::min(m, n);
	assert!(all(
		u.nrows() == m,
		v.nrows() == n,
		u.ncols() >= size,
		v.ncols() >= size,
		s.dim() >= size
	));
	let s = s.column_vector();
	let u = u.get(.., ..size);
	let v = v.get(.., ..size);
	let smax = s.norm_max();
	let tol = abs_tol.fmax(rel_tol * smax);
	let (mut u_trunc, stack) =
		unsafe { temp_mat_uninit::<T, _, _>(m, size, stack) };
	let (mut vp_trunc, _) =
		unsafe { temp_mat_uninit::<T, _, _>(n, size, stack) };
	let mut u_trunc = u_trunc.as_mat_mut();
	let mut vp_trunc = vp_trunc.as_mat_mut();
	let mut len = 0;
	for j in 0..size {
		let x = s[j].absmax();
		if x > tol {
			let ref p = s[j].real().recip();
			u_trunc.rb_mut().col_mut(len).copy_from(u.col(j));
			z!(vp_trunc.rb_mut().col_mut(len), v.col(j))
				.for_each(|uz!(dst, src)| *dst = src.mul_real(p));
			len += 1;
		}
	}
	let u_trunc = u_trunc.get(.., ..len);
	let vp_trunc = vp_trunc.get(.., ..len);
	linalg::matmul::matmul(
		pinv.rb_mut(),
		Accum::Replace,
		vp_trunc,
		u_trunc.adjoint(),
		one(),
		par,
	);
}
#[cfg(test)]
mod tests {
	mod lapack;
	mod svd_views;

	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;
	#[track_caller]
	fn test_svd<T: ComplexField>(mat: MatRef<'_, T>) {
		let (m, n) = mat.shape();
		let params = Spec::new(SvdParams {
			recursion_threshold: 8,
			qr_ratio_threshold: 1.0,
			..auto!(T)
		});
		let approx_eq = CwiseMat(
			ApproxEq::<T::Real>::eps()
				* from_f64::<T::Real>(8.0 * Ord::max(m, n) as f64).sqrt(),
		);
		{
			let mut s = Mat::zeros(m, n);
			let mut u = Mat::zeros(m, m);
			let mut v = Mat::zeros(n, n);
			svd(
				mat.as_ref(),
				s.as_mut().diagonal_mut(),
				Some(u.as_mut()),
				Some(v.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(svd_scratch::<T>(
					m,
					n,
					ComputeSvdVectors::Full,
					ComputeSvdVectors::Full,
					Par::Seq,
					params,
				))),
				params,
			)
			.unwrap();
			let reconstructed = &u * &s * v.adjoint();
			assert!(reconstructed ~ mat);
		}
		let size = Ord::min(m, n);
		let mut s = Mat::zeros(size, size);
		let mut u = Mat::zeros(m, size);
		let mut v = Mat::zeros(n, size);
		{
			svd(
				mat.as_ref(),
				s.as_mut().diagonal_mut(),
				Some(u.as_mut()),
				Some(v.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(svd_scratch::<T>(
					m,
					n,
					ComputeSvdVectors::Thin,
					ComputeSvdVectors::Thin,
					Par::Seq,
					params,
				))),
				params,
			)
			.unwrap();
			let reconstructed = &u * &s * v.adjoint();
			assert!(reconstructed ~ mat);
		}
		{
			let mut s2 = Mat::zeros(size, size);
			let mut u2 = Mat::zeros(m, size);
			svd(
				mat.as_ref(),
				s2.as_mut().diagonal_mut(),
				Some(u2.as_mut()),
				None,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(svd_scratch::<T>(
					m,
					n,
					ComputeSvdVectors::Thin,
					ComputeSvdVectors::No,
					Par::Seq,
					params,
				))),
				params,
			)
			.unwrap();
			assert!(s2 ~ s);
			assert!(u2 ~ u);
		}
		{
			let mut s2 = Mat::zeros(size, size);
			let mut v2 = Mat::zeros(n, size);
			svd(
				mat.as_ref(),
				s2.as_mut().diagonal_mut(),
				None,
				Some(v2.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(svd_scratch::<T>(
					m,
					n,
					ComputeSvdVectors::No,
					ComputeSvdVectors::Thin,
					Par::Seq,
					params,
				))),
				params,
			)
			.unwrap();
			assert!(s2 ~ s);
			assert!(v2 ~ v);
		}
		{
			let mut s2 = Mat::zeros(size, size);
			svd(
				mat.as_ref(),
				s2.as_mut().diagonal_mut(),
				None,
				None,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(svd_scratch::<T>(
					m,
					n,
					ComputeSvdVectors::No,
					ComputeSvdVectors::No,
					Par::Seq,
					params,
				))),
				params,
			)
			.unwrap();
			assert!(s2 ~ s);
		}
	}
	fn check_tiny_split<T: ComplexField>(a: MatRef<'_, T>) {
		let (m, n) = a.shape();
		let size = Ord::min(m, n);
		let compute = |threshold, want_u, want_v| {
			let params = Spec::new(SvdParams {
				recursion_threshold: threshold,
				..auto!(T)
			});
			let mut s = Col::zeros(size);
			let mut u = Mat::zeros(m, if want_u { size } else { 0 });
			let mut v = Mat::zeros(n, if want_v { size } else { 0 });
			let vectors = |want| {
				if want {
					ComputeSvdVectors::Thin
				} else {
					ComputeSvdVectors::No
				}
			};
			svd(
				a,
				s.as_mut().as_diagonal_mut(),
				if want_u { Some(u.as_mut()) } else { None },
				if want_v { Some(v.as_mut()) } else { None },
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(svd_scratch::<T>(
					m,
					n,
					vectors(want_u),
					vectors(want_v),
					Par::Seq,
					params,
				))),
				params,
			)
			.unwrap();
			(s, u, v)
		};
		let (reference, _, _) = compute(usize::MAX, true, true);
		for (want_u, want_v) in
			[(true, true), (false, false), (true, false), (false, true)]
		{
			let (s, u, v) = compute(128, want_u, want_v);
			assert!((&s - &reference).norm_max() < from_f64::<T::Real>(1e-13));
			if want_u {
				assert!(
					(u.adjoint() * &u - Mat::<T>::identity(size, size))
						.norm_l2() < from_f64::<T::Real>(1e-13)
				);
			}
			if want_v {
				assert!(
					(v.adjoint() * &v - Mat::<T>::identity(size, size))
						.norm_l2() < from_f64::<T::Real>(1e-13)
				);
			}
			if want_u && want_v {
				let back = &u * s.as_diagonal() * v.adjoint();
				assert!(
					(&back - a).norm_l2() / a.norm_l2()
						< from_f64::<T::Real>(1e-13)
				);
			}
		}
	}
	#[test]
	fn test_tiny_diagonal_at_split() {
		for n in [128, 256] {
			for tiny in [0.0, 1e-15, 1e-13, 1e-10, 1e-6] {
				let a = Mat::<f64>::from_fn(n, n, |i, j| {
					if (i == j && i < 64) || (j == i + 1 && i < 64) {
						1.0
					} else if i == 64 && j == 64 {
						tiny
					} else {
						0.0
					}
				});
				check_tiny_split(a.as_ref());
				let complex =
					Mat::<c64>::from_fn(n, n, |i, j| c64::new(0.0, a[(i, j)]));
				check_tiny_split(complex.as_ref());
			}
		}
	}
	/// A zero block at a divide-and-conquer split must still hand its parent merge the
	/// identity boundary entry that carries the nonzero part of the spectrum, whatever
	/// singular vectors the caller asked for (#33).
	#[test]
	fn test_zero_block_at_split() {
		fn check<T: ComplexField>(
			n: usize,
			threshold: usize,
			want: (bool, bool),
			tol: f64,
		) {
			let (want_u, want_v) = want;
			let a = Mat::<T>::from_fn(n, n, |i, j| {
				if i == n / 2 && j == n / 2 {
					one()
				} else {
					zero()
				}
			});
			let params = Spec::new(SvdParams {
				recursion_threshold: threshold,
				..auto!(T)
			});
			let vectors = |want| {
				if want {
					ComputeSvdVectors::Full
				} else {
					ComputeSvdVectors::No
				}
			};
			let mut s = Col::<T>::zeros(n);
			let mut u = Mat::<T>::zeros(n, if want_u { n } else { 0 });
			let mut v = Mat::<T>::zeros(n, if want_v { n } else { 0 });
			svd(
				a.as_ref(),
				s.as_mut().as_diagonal_mut(),
				if want_u { Some(u.as_mut()) } else { None },
				if want_v { Some(v.as_mut()) } else { None },
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(svd_scratch::<T>(
					n,
					n,
					vectors(want_u),
					vectors(want_v),
					Par::Seq,
					params,
				))),
				params,
			)
			.unwrap();
			// `norm_max` skips NaN entries, so finiteness is stated separately
			assert!(s.is_all_finite());
			if want_u {
				assert!(u.is_all_finite());
			}
			if want_v {
				assert!(v.is_all_finite());
			}
			let expected =
				Col::<T>::from_fn(n, |i| if i == 0 { one() } else { zero() });
			let tol = from_f64::<T::Real>(tol);
			assert!((&s - &expected).norm_max() <= tol.copy());
			let identity = Mat::<T>::identity(n, n);
			if want_u && want_v {
				let back = &u * s.as_diagonal() * v.adjoint();
				assert!((&back - &a).norm_max() <= tol.copy());
			}
			if want_u {
				assert!(
					(u.adjoint() * &u - &identity).norm_max() <= tol.copy()
				);
			}
			if want_v {
				assert!(
					(v.adjoint() * &v - &identity).norm_max() <= tol.copy()
				);
			}
		}
		for n in [8usize, 16, 32] {
			for threshold in [4, 8] {
				for want in
					[(true, true), (false, false), (true, false), (false, true)]
				{
					check::<f32>(n, threshold, want, 1e-5);
					check::<c32>(n, threshold, want, 1e-5);
					check::<f64>(n, threshold, want, 1e-13);
					check::<c64>(n, threshold, want, 1e-13);
				}
			}
		}
		// the reported `singular_values()` call runs at the default threshold
		let threshold = <SvdParams as Auto<f64>>::auto().recursion_threshold;
		check::<f64>(256, threshold, (false, false), 1e-13);
	}
	/// An extreme but finite scale must not overflow or underflow the reduction that
	/// bidiagonalisation and the SVD share (#36): the spectrum and the reconstruction
	/// stay finite and accurate.
	#[test]
	fn test_extreme_scale() {
		fn check<T: ComplexField>(s: f64, tol: f64) {
			let scale = from_f64::<T>(s);
			let a = Mat::<T>::from_fn(2, 2, |i, j| match (i, j) {
				(0, 0) => scale.copy(),
				(0, 1) => scale.mul_real(from_f64::<T::Real>(0.5)),
				(1, 1) => scale.copy(),
				_ => zero(),
			});
			let params = Spec::new(SvdParams {
				recursion_threshold: 4,
				..auto!(T)
			});
			let mut sigma = Mat::<T>::zeros(2, 2);
			let mut u = Mat::<T>::zeros(2, 2);
			let mut v = Mat::<T>::zeros(2, 2);
			svd(
				a.as_ref(),
				sigma.as_mut().diagonal_mut(),
				Some(u.as_mut()),
				Some(v.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(svd_scratch::<T>(
					2,
					2,
					ComputeSvdVectors::Thin,
					ComputeSvdVectors::Thin,
					Par::Seq,
					params,
				))),
				params,
			)
			.unwrap();
			let rec = &u * &sigma * v.adjoint();
			let err = (&rec - &a).norm_max();
			assert!(rec.is_all_finite());
			assert!(err <= a.norm_max() * from_f64::<T::Real>(tol));
		}
		for s in [1.0f64, 1e160, 1e-300] {
			check::<f64>(s, 1e-14);
			check::<c64>(s, 1e-14);
		}
		for s in [1.0f64, 1e30, 1e-35] {
			check::<f32>(s, 1e-6);
			check::<c32>(s, 1e-6);
		}
	}
	/// A rank-one matrix's bidiagonalisation cascades down to subnormal entries, where the
	/// row norm's reciprocal (and, for complex input, the phase normalisation's) overflows.
	#[test]
	fn test_rank_one_subnormal_cascade() {
		for n in [16usize, 32, 150] {
			test_svd(Mat::<f32>::full(n, n, 1.0).as_ref());
			test_svd(Mat::<c32>::full(n, n, c32::ONE).as_ref());
		}
		for n in [32usize, 150] {
			test_svd(Mat::<f64>::full(n, n, 1.0).as_ref());
			test_svd(Mat::<c64>::full(n, n, c64::ONE).as_ref());
		}
	}
	#[test]
	fn test_zero_child_at_split() {
		for n in [128, 256] {
			for pos in [0, n / 2, n - 1] {
				let a = Mat::<f64>::from_fn(n, n, |i, j| {
					if i == pos && j == pos { 1.0 } else { 0.0 }
				});
				check_tiny_split(a.as_ref());
				let complex =
					Mat::<c64>::from_fn(n, n, |i, j| c64::new(0.0, a[(i, j)]));
				check_tiny_split(complex.as_ref());
			}
		}
	}
	#[test]
	fn test_real() {
		let rng = &mut StdRng::seed_from_u64(1);
		for (m, n) in [
			(3, 2),
			(2, 2),
			(4, 4),
			(15, 10),
			(10, 10),
			(15, 15),
			(50, 50),
			(100, 100),
			(150, 150),
			(150, 20),
			(20, 150),
		] {
			let mat = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			test_svd(mat.as_ref());
		}
	}
	#[test]
	fn test_cplx() {
		let rng = &mut StdRng::seed_from_u64(1);
		for (m, n) in [
			(1, 1),
			(2, 2),
			(3, 2),
			(2, 2),
			(3, 3),
			(4, 4),
			(15, 10),
			(10, 10),
			(15, 15),
			(16, 16),
			(17, 17),
			(18, 18),
			(19, 19),
			(20, 20),
			(30, 30),
			(50, 50),
			(100, 100),
			(150, 150),
			(150, 20),
			(20, 150),
		] {
			let mat = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);
			test_svd(mat.as_ref());
		}
	}
	#[test]
	fn test_special() {
		for (m, n) in [
			(3, 2),
			(2, 2),
			(4, 4),
			(15, 10),
			(10, 10),
			(15, 15),
			(50, 50),
			(100, 100),
			(150, 150),
			(150, 20),
			(20, 150),
		] {
			test_svd(Mat::<f64>::zeros(m, n).as_ref());
			test_svd(Mat::<c64>::zeros(m, n).as_ref());
			test_svd(Mat::<f64>::full(m, n, 1.0).as_ref());
			test_svd(Mat::<c64>::full(m, n, c64::ONE).as_ref());
			test_svd(Mat::<f64>::identity(m, n).as_ref());
			test_svd(Mat::<c64>::identity(m, n).as_ref());
		}
	}
	#[test]
	fn test_zink() {
		let diag = [
			-9.931833701529301,
			-10.920807536026027,
			-52.33647796311243,
			2.3685025127736967,
			2.421701994236093,
			-0.5051763005624579,
			-0.04808263896606017,
			-0.003875251886338955,
			-0.0006413264967716465,
			-0.003381944152463707,
			2.981152313236375e-5,
			5.4290648208388795e-6,
			-6.329275972084404e-7,
			-6.879142344209158e-7,
			-5.265228263479126e-9,
			-2.941999902335516e-9,
			-1.3060984997930294e-10,
			7.07516117218088e-12,
			1.8657003929029376e-12,
			-6.216080089659131e-14,
		];
		let subdiag = [
			-57.8029649868477,
			17.67263066467847,
			8.884153814270894,
			-9.01998231080713,
			-1.028638150814966,
			0.22247719217200435,
			0.016389886745811315,
			-0.004090989452162578,
			0.00036818904090536926,
			-0.0031394146217732367,
			-7.571300829706796e-6,
			3.0045718718618155e-6,
			2.1329796886727743e-6,
			9.259701025627789e-8,
			2.2291214755992877e-9,
			-2.3017207713252894e-9,
			6.807967994979358e-11,
			2.1677299575405587e-12,
			-3.07282771050034e-13,
			0.0,
		];
		let n = diag.len();
		let params = SvdParams {
			recursion_threshold: 8,
			qr_ratio_threshold: 1.0,
			..auto!(f64)
		};
		let mut d = ColRef::from_slice(&diag).to_owned();
		let mut s = ColRef::from_slice(&subdiag).to_owned();
		compute_bidiag_real_svd(
			d.as_mut().try_as_col_major_mut().unwrap(),
			s.as_mut().try_as_col_major_mut().unwrap(),
			None,
			None,
			params,
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(bidiag_real_svd_scratch::<f64>(
				n,
				false,
				false,
				Par::Seq,
				params,
			))),
		)
		.unwrap();
		assert!(d[n - 1] != 0.0);
	}
	#[test]
	fn test_pinv() {
		let rng = &mut StdRng::seed_from_u64(0);
		let m = 6;
		let n = 36;
		let mat = CwiseMatDistribution {
			nrows: m,
			ncols: n,
			dist: StandardNormal,
		}
		.rand::<Mat<f64>>(rng);
		let pinv = mat.svd().unwrap().pseudoinverse();
		let err = &mat * &pinv - Mat::<f64>::identity(m, m);
		assert!(err.norm_max() < 1e-10);
	}
	#[test]
	fn test_pinv2() {
		let rng = &mut StdRng::seed_from_u64(0);
		let m = 36;
		let n = 6;
		let mat = CwiseMatDistribution {
			nrows: m,
			ncols: n,
			dist: StandardNormal,
		}
		.rand::<Mat<f64>>(rng);
		let pinv = mat.svd().unwrap().pseudoinverse();
		let err = &pinv * &mat - Mat::<f64>::identity(n, n);
		assert!(err.norm_max() < 1e-10);
	}
	/// The phase reduction's magnitude and phase helpers outside the normal
	/// range: for an exact power of two `s`, the singular values of `s B` are
	/// `s` times those of `B`. Entries below `min_positive` were flushed to
	/// zero or made `inf`/NaN, entries above `1 / min_positive` overflowed.
	fn cplx_phase_reduction_is_scale_exact<T: ComplexField>(
		make: fn(f64, f64) -> T,
	) {
		let n = 3;
		let base_d = [make(1.8, -0.4), make(-0.3, 0.7), make(0.5, 0.5)];
		let base_e = [make(0.2, 0.6), make(-0.8, 0.1)];
		let params: SvdParams = auto!(T);
		let run = |s: &T::Real| {
			let mut d = Col::<T>::from_fn(n, |i| base_d[i].mul_real(s));
			let mut e = Col::<T>::from_fn(n, |i| {
				if i + 1 < n {
					base_e[i].mul_real(s)
				} else {
					zero()
				}
			});
			let mut u = Mat::<T>::zeros(n + 1, n + 1);
			let mut v = Mat::<T>::zeros(n, n);
			compute_bidiag_cplx_svd(
				d.as_mut().try_as_col_major_mut().unwrap(),
				e.as_mut().try_as_col_major_mut().unwrap(),
				Some(u.as_mut()),
				Some(v.as_mut()),
				params,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					bidiag_cplx_svd_scratch::<T>(
						n,
						true,
						true,
						Par::Seq,
						params,
					),
				)),
			)
			.unwrap();
			assert!(u.is_all_finite() && v.is_all_finite());
			d
		};
		let reference = run(&one());
		let sml = min_positive::<T::Real>();
		for s in [sml.recip(), &sml * from_f64::<T::Real>(0.0625)] {
			let d = run(&s);
			for i in 0..n {
				let scaled_back = d[i].real() / &s;
				assert!(
					(scaled_back - reference[i].real()).abs()
						<= from_f64::<T::Real>(1e3)
							* eps::<T::Real>() * reference[0].real()
				);
			}
		}
	}
	#[test]
	fn cplx_phase_reduction_outside_normal_range() {
		cplx_phase_reduction_is_scale_exact::<c64>(c64::new);
		cplx_phase_reduction_is_scale_exact::<c32>(|r, i| {
			c32::new(r as f32, i as f32)
		});
		cplx_phase_reduction_is_scale_exact::<f64>(|r, _| r);
		cplx_phase_reduction_is_scale_exact::<f32>(|r, _| r as f32);
	}
	/// full factors must be orthogonal even when the caller's output buffers
	/// hold old entries; the tall/wide QR path used to leave the top-right block
	/// of `U` (and, after transposition, of `V`) as it found it
	#[test]
	fn full_factors_ignore_old_output_entries() {
		for (m, n) in [(4, 2), (2, 4)] {
			let a = Mat::<f64>::identity(m, n);
			let size = Ord::min(m, n);
			let mut s = Diag::<f64>::zeros(size);
			let mut u = Mat::<f64>::full(m, m, 1.0);
			let mut v = Mat::<f64>::full(n, n, 1.0);
			svd(
				a.as_ref(),
				s.as_mut(),
				Some(u.as_mut()),
				Some(v.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(svd_scratch::<f64>(
					m,
					n,
					ComputeSvdVectors::Full,
					ComputeSvdVectors::Full,
					Par::Seq,
					default(),
				))),
				default(),
			)
			.unwrap();
			let orth_u =
				(u.transpose() * &u - Mat::<f64>::identity(m, m)).norm_max();
			let orth_v =
				(v.transpose() * &v - Mat::<f64>::identity(n, n)).norm_max();
			assert!(orth_u < 1e-12, "U orthogonality error {orth_u}");
			assert!(orth_v < 1e-12, "V orthogonality error {orth_v}");
			let rec = u.as_ref().get(.., ..size) * s.as_ref()
				* v.as_ref().get(.., ..size).transpose();
			assert!((&rec - &a).norm_max() < 1e-12);
		}
	}
}
