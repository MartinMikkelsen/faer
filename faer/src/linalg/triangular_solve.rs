//! Triangular solve module.
use crate::internal_prelude::*;
use crate::utils::thread::join_raw;
use crate::{assert, debug_assert};
use faer_traits::{Conjugate, SimdArch};
use reborrow::*;
#[inline(always)]
fn identity<T: ComplexField>(x: T) -> T {
	x.copy()
}
#[inline(always)]
fn conjugate<T: ComplexField>(x: T) -> T {
	x.conj()
}
#[inline(always)]
fn solve_unit_lower_triangular_in_place_base_case_generic_imp<
	'N,
	'K,
	T: ComplexField,
>(
	tril: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	rhs: MatMut<'_, T, Dim<'N>, Dim<'K>>,
	maybe_conj_lhs: impl Fn(T) -> T,
) {
	let N = tril.nrows();
	let n = N.unbound();
	match n {
		0 | 1 => {},
		2 => {
			let i0 = N.check(0);
			let i1 = N.check(1);
			let nl10_div_l11 = &maybe_conj_lhs(-&tril[(i1, i0)]);
			let (x0, rhs) = rhs.split_first_row_mut().unwrap();
			let (x1, rhs) = rhs.split_first_row_mut().unwrap();
			_ = rhs;
			zip!(x0, x1)
				.for_each(|unzip!(x0, x1)| *x1 = &*x1 + nl10_div_l11 * &*x0);
		},
		3 => {
			let i0 = N.check(0);
			let i1 = N.check(1);
			let i2 = N.check(2);
			let nl10_div_l11 = &maybe_conj_lhs(-&tril[(i1, i0)]);
			let nl20_div_l22 = &maybe_conj_lhs(-&tril[(i2, i0)]);
			let nl21_div_l22 = &maybe_conj_lhs(-&tril[(i2, i1)]);
			let (x0, rhs) = rhs.split_first_row_mut().unwrap();
			let (x1, rhs) = rhs.split_first_row_mut().unwrap();
			let (x2, rhs) = rhs.split_first_row_mut().unwrap();
			_ = rhs;
			zip!(x0, x1, x2).for_each(|unzip!(x0, x1, x2)| {
				let y0 = x0.copy();
				let mut y1 = x1.copy();
				let mut y2 = x2.copy();
				y1 = y1 + nl10_div_l11 * &y0;
				y2 = y2 + nl20_div_l22 * &y0 + nl21_div_l22 * &y1;
				*x0 = y0;
				*x1 = y1;
				*x2 = y2;
			});
		},
		4 => {
			let i0 = N.check(0);
			let i1 = N.check(1);
			let i2 = N.check(2);
			let i3 = N.check(3);
			let nl10_div_l11 = &maybe_conj_lhs(-&tril[(i1, i0)]);
			let nl20_div_l22 = &maybe_conj_lhs(-&tril[(i2, i0)]);
			let nl21_div_l22 = &maybe_conj_lhs(-&tril[(i2, i1)]);
			let nl30_div_l33 = &maybe_conj_lhs(-&tril[(i3, i0)]);
			let nl31_div_l33 = &maybe_conj_lhs(-&tril[(i3, i1)]);
			let nl32_div_l33 = &maybe_conj_lhs(-&tril[(i3, i2)]);
			let (x0, rhs) = rhs.split_first_row_mut().unwrap();
			let (x1, rhs) = rhs.split_first_row_mut().unwrap();
			let (x2, rhs) = rhs.split_first_row_mut().unwrap();
			let (x3, rhs) = rhs.split_first_row_mut().unwrap();
			_ = rhs;
			zip!(x0, x1, x2, x3).for_each(|unzip!(x0, x1, x2, x3)| {
				let y0 = x0.copy();
				let mut y1 = x1.copy();
				let mut y2 = x2.copy();
				let mut y3 = x3.copy();
				y1 = y1 + nl10_div_l11 * &y0;
				y2 = y2 + nl20_div_l22 * &y0 + nl21_div_l22 * &y1;
				y3 = y3
					+ nl30_div_l33 * &y0
					+ nl31_div_l33 * &y1
					+ nl32_div_l33 * &y2;
				*x0 = y0;
				*x1 = y1;
				*x2 = y2;
				*x3 = y3;
			});
		},
		_ => unreachable!(),
	}
}
#[inline(always)]
fn solve_lower_triangular_in_place_base_case_generic_imp<
	'N,
	'K,
	T: ComplexField,
>(
	tril: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	rhs: MatMut<'_, T, Dim<'N>, Dim<'K>>,
	maybe_conj_lhs: impl Fn(T) -> T,
) {
	let N = tril.nrows();
	let n = N.unbound();
	// Form residuals before division when a tiny pivot's reciprocal can overflow.
	if N.indices().any(|i| {
		let pivot = &tril[(i, i)];
		let max = pivot.real().abs().fmax(pivot.imag().abs());
		max != zero() && max < min_positive::<T::Real>()
	}) {
		let mut rhs = rhs;
		for i in N.indices() {
			let ref pivot = maybe_conj_lhs(tril[(i, i)].copy());
			let ref inv = pivot.recip();
			let ref scale = if T::IS_REAL || inv.is_finite() {
				one::<T::Real>()
			} else {
				// Leave headroom for both complex products when scaling a tiny divisor.
				from_f64::<T::Real>(0.25) * min_positive::<T::Real>().recip()
			};
			let ref scaled_pivot = pivot.mul_real(scale);
			for k in rhs.ncols().indices() {
				let mut value = rhs[(i, k)].copy();
				for j in 0..i.unbound() {
					let j = N.check(j);
					value -= maybe_conj_lhs(tril[(i, j)].copy()) * &rhs[(j, k)];
				}
				rhs[(i, k)] = if T::IS_REAL {
					value / pivot
				} else if inv.is_finite() {
					value * inv
				} else {
					value.mul_real(scale) / scaled_pivot
				};
			}
		}
		return;
	}
	match n {
		0 => {},
		1 => {
			let i0 = N.check(0);
			let inv = &maybe_conj_lhs(tril[(i0, i0)].recip());
			let (x0, rhs) = rhs.split_first_row_mut().unwrap();
			_ = rhs;
			zip!(x0).for_each(|unzip!(x0)| *x0 = &*x0 * inv);
		},
		2 => {
			let i0 = N.check(0);
			let i1 = N.check(1);
			let l00_inv = &maybe_conj_lhs(tril[(i0, i0)].recip());
			let l11_inv = &maybe_conj_lhs(tril[(i1, i1)].recip());
			let nl10_div_l11 = &(maybe_conj_lhs(-&tril[(i1, i0)]) * l11_inv);
			let (x0, rhs) = rhs.split_first_row_mut().unwrap();
			let (x1, rhs) = rhs.split_first_row_mut().unwrap();
			_ = rhs;
			zip!(x0, x1).for_each(|unzip!(x0, x1)| {
				*x0 = &*x0 * l00_inv;
				*x1 = &*x1 * l11_inv + nl10_div_l11 * &*x0;
			});
		},
		3 => {
			let i0 = N.check(0);
			let i1 = N.check(1);
			let i2 = N.check(2);
			let l00_inv = &maybe_conj_lhs(tril[(i0, i0)].recip());
			let l11_inv = &maybe_conj_lhs(tril[(i1, i1)].recip());
			let l22_inv = &maybe_conj_lhs(tril[(i2, i2)].recip());
			let nl10_div_l11 = &(maybe_conj_lhs(-&tril[(i1, i0)]) * l11_inv);
			let nl20_div_l22 = &(maybe_conj_lhs(-&tril[(i2, i0)]) * l22_inv);
			let nl21_div_l22 = &(maybe_conj_lhs(-&tril[(i2, i1)]) * l22_inv);
			let (x0, rhs) = rhs.split_first_row_mut().unwrap();
			let (x1, rhs) = rhs.split_first_row_mut().unwrap();
			let (x2, rhs) = rhs.split_first_row_mut().unwrap();
			_ = rhs;
			zip!(x0, x1, x2).for_each(|unzip!(x0, x1, x2)| {
				let mut y0 = x0.copy();
				let mut y1 = x1.copy();
				let mut y2 = x2.copy();
				y0 = y0 * l00_inv;
				y1 = y1 * l11_inv + nl10_div_l11 * &y0;
				y2 = y2 * l22_inv + nl20_div_l22 * &y0 + nl21_div_l22 * &y1;
				*x0 = y0;
				*x1 = y1;
				*x2 = y2;
			});
		},
		4 => {
			let i0 = N.check(0);
			let i1 = N.check(1);
			let i2 = N.check(2);
			let i3 = N.check(3);
			let l00_inv = &maybe_conj_lhs(tril[(i0, i0)].recip());
			let l11_inv = &maybe_conj_lhs(tril[(i1, i1)].recip());
			let l22_inv = &maybe_conj_lhs(tril[(i2, i2)].recip());
			let l33_inv = &maybe_conj_lhs(tril[(i3, i3)].recip());
			let nl10_div_l11 = &(maybe_conj_lhs(-&tril[(i1, i0)]) * l11_inv);
			let nl20_div_l22 = &(maybe_conj_lhs(-&tril[(i2, i0)]) * l22_inv);
			let nl21_div_l22 = &(maybe_conj_lhs(-&tril[(i2, i1)]) * l22_inv);
			let nl30_div_l33 = &(maybe_conj_lhs(-&tril[(i3, i0)]) * l33_inv);
			let nl31_div_l33 = &(maybe_conj_lhs(-&tril[(i3, i1)]) * l33_inv);
			let nl32_div_l33 = &(maybe_conj_lhs(-&tril[(i3, i2)]) * l33_inv);
			let (x0, rhs) = rhs.split_first_row_mut().unwrap();
			let (x1, rhs) = rhs.split_first_row_mut().unwrap();
			let (x2, rhs) = rhs.split_first_row_mut().unwrap();
			let (x3, rhs) = rhs.split_first_row_mut().unwrap();
			_ = rhs;
			zip!(x0, x1, x2, x3).for_each(|unzip!(x0, x1, x2, x3)| {
				let mut y0 = x0.copy();
				let mut y1 = x1.copy();
				let mut y2 = x2.copy();
				let mut y3 = x3.copy();
				y0 = y0 * l00_inv;
				y1 = y1 * l11_inv + nl10_div_l11 * &y0;
				y2 = y2 * l22_inv + nl20_div_l22 * &y0 + nl21_div_l22 * &y1;
				y3 = y3 * l33_inv
					+ nl30_div_l33 * &y0
					+ nl31_div_l33 * &y1
					+ nl32_div_l33 * &y2;
				*x0 = y0;
				*x1 = y1;
				*x2 = y2;
				*x3 = y3;
			});
		},
		_ => unreachable!(),
	}
}
#[inline]
fn block_size(n: usize) -> usize {
	let base_rem = n / 2;
	n - if n >= 32 {
		(base_rem + 15) / 16 * 16
	} else if n >= 16 {
		(base_rem + 7) / 8 * 8
	} else if n >= 8 {
		(base_rem + 3) / 4 * 4
	} else {
		base_rem
	}
}
#[inline]
fn recursion_threshold() -> usize {
	4
}
/// solves $L x = b$, implicitly conjugating $L$ if needed, and stores the
/// result in `rhs`
#[track_caller]
#[inline]
pub fn solve_lower_triangular_in_place_with_conj<
	T: ComplexField,
	N: Shape,
	K: Shape,
>(
	triangular_lower: MatRef<'_, T, N, N, impl Stride, impl Stride>,
	conj_lhs: Conj,
	rhs: MatMut<'_, T, N, K, impl Stride, impl Stride>,
	par: Par,
) {
	assert!(all(
		triangular_lower.nrows() == triangular_lower.ncols(),
		rhs.nrows() == triangular_lower.ncols(),
	));
	make_guard!(N);
	make_guard!(K);
	let N = rhs.nrows().bind(N);
	let K = rhs.ncols().bind(K);
	solve_lower_triangular_in_place_imp(
		triangular_lower.as_dyn_stride().as_shape(N, N),
		conj_lhs,
		rhs.as_dyn_stride_mut().as_shape_mut(N, K),
		par,
	);
}
/// solves $L x = b$, implicitly conjugating $L$ if needed, and stores the
/// result in `rhs`
#[inline]
#[track_caller]
pub fn solve_lower_triangular_in_place<
	T: ComplexField,
	LhsT: Conjugate<Canonical = T>,
	N: Shape,
	K: Shape,
>(
	triangular_lower: MatRef<'_, LhsT, N, N, impl Stride, impl Stride>,
	rhs: MatMut<'_, T, N, K, impl Stride, impl Stride>,
	par: Par,
) {
	let tri = triangular_lower.canonical();
	solve_lower_triangular_in_place_with_conj(
		tri,
		Conj::get::<LhsT>(),
		rhs,
		par,
	)
}
/// solves $L x = b$, replacing the diagonal of $L$ with ones, and implicitly
/// conjugating $L$ if needed, and stores the result in `rhs`
#[track_caller]
#[inline]
pub fn solve_unit_lower_triangular_in_place_with_conj<
	T: ComplexField,
	N: Shape,
	K: Shape,
>(
	triangular_unit_lower: MatRef<'_, T, N, N, impl Stride, impl Stride>,
	conj_lhs: Conj,
	rhs: MatMut<'_, T, N, K, impl Stride, impl Stride>,
	par: Par,
) {
	assert!(all(
		triangular_unit_lower.nrows() == triangular_unit_lower.ncols(),
		rhs.nrows() == triangular_unit_lower.ncols(),
	));
	make_guard!(N);
	make_guard!(K);
	let N = rhs.nrows().bind(N);
	let K = rhs.ncols().bind(K);
	solve_unit_lower_triangular_in_place_imp(
		triangular_unit_lower.as_dyn_stride().as_shape(N, N),
		conj_lhs,
		rhs.as_dyn_stride_mut().as_shape_mut(N, K),
		par,
	);
}
/// solves $L x = b$, replacing the diagonal of $L$ with ones, and implicitly
/// conjugating $L$ if needed, and stores the result in `rhs`
#[inline]
#[track_caller]
pub fn solve_unit_lower_triangular_in_place<
	T: ComplexField,
	LhsT: Conjugate<Canonical = T>,
	N: Shape,
	K: Shape,
>(
	triangular_unit_lower: MatRef<'_, LhsT, N, N, impl Stride, impl Stride>,
	rhs: MatMut<'_, T, N, K, impl Stride, impl Stride>,
	par: Par,
) {
	let tri = triangular_unit_lower.canonical();
	solve_unit_lower_triangular_in_place_with_conj(
		tri,
		Conj::get::<LhsT>(),
		rhs,
		par,
	)
}
/// solves $U x = b$, implicitly conjugating $U$ if needed, and stores the
/// result in `rhs`
#[track_caller]
#[inline]
pub fn solve_upper_triangular_in_place_with_conj<
	T: ComplexField,
	N: Shape,
	K: Shape,
>(
	triangular_upper: MatRef<'_, T, N, N, impl Stride, impl Stride>,
	conj_lhs: Conj,
	rhs: MatMut<'_, T, N, K, impl Stride, impl Stride>,
	par: Par,
) {
	assert!(all(
		triangular_upper.nrows() == triangular_upper.ncols(),
		rhs.nrows() == triangular_upper.ncols(),
	));
	make_guard!(N);
	make_guard!(K);
	let N = rhs.nrows().bind(N);
	let K = rhs.ncols().bind(K);
	solve_upper_triangular_in_place_imp(
		triangular_upper.as_dyn_stride().as_shape(N, N),
		conj_lhs,
		rhs.as_dyn_stride_mut().as_shape_mut(N, K),
		par,
	);
}
/// solves $U x = b$, implicitly conjugating $U$ if needed, and stores the
/// result in `rhs`
#[inline]
#[track_caller]
pub fn solve_upper_triangular_in_place<
	T: ComplexField,
	LhsT: Conjugate<Canonical = T>,
	N: Shape,
	K: Shape,
>(
	triangular_upper: MatRef<'_, LhsT, N, N, impl Stride, impl Stride>,
	rhs: MatMut<'_, T, N, K, impl Stride, impl Stride>,
	par: Par,
) {
	let tri = triangular_upper.canonical();
	solve_upper_triangular_in_place_with_conj(
		tri,
		Conj::get::<LhsT>(),
		rhs,
		par,
	)
}
/// solves $U x = b$, replacing the diagonal of $U$ with ones, and implicitly
/// conjugating $U$ if needed, and stores the result in `rhs`
#[track_caller]
#[inline]
pub fn solve_unit_upper_triangular_in_place_with_conj<
	T: ComplexField,
	N: Shape,
	K: Shape,
>(
	triangular_unit_upper: MatRef<'_, T, N, N, impl Stride, impl Stride>,
	conj_lhs: Conj,
	rhs: MatMut<'_, T, N, K, impl Stride, impl Stride>,
	par: Par,
) {
	assert!(all(
		triangular_unit_upper.nrows() == triangular_unit_upper.ncols(),
		rhs.nrows() == triangular_unit_upper.ncols(),
	));
	make_guard!(N);
	make_guard!(K);
	let N = rhs.nrows().bind(N);
	let K = rhs.ncols().bind(K);
	solve_unit_upper_triangular_in_place_imp(
		triangular_unit_upper.as_dyn_stride().as_shape(N, N),
		conj_lhs,
		rhs.as_dyn_stride_mut().as_shape_mut(N, K),
		par,
	);
}
/// solves $U x = b$, replacing the diagonal of $U$ with ones, and implicitly
/// conjugating $U$ if needed, and stores the result in `rhs`
#[inline]
#[track_caller]
pub fn solve_unit_upper_triangular_in_place<
	T: ComplexField,
	LhsT: Conjugate<Canonical = T>,
	N: Shape,
	K: Shape,
>(
	triangular_unit_upper: MatRef<'_, LhsT, N, N, impl Stride, impl Stride>,
	rhs: MatMut<'_, T, N, K, impl Stride, impl Stride>,
	par: Par,
) {
	let tri = triangular_unit_upper.canonical();
	solve_unit_upper_triangular_in_place_with_conj(
		tri,
		Conj::get::<LhsT>(),
		rhs,
		par,
	)
}
fn solve_unit_lower_triangular_in_place_imp<'N, 'K, T: ComplexField>(
	tril: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	conj_lhs: Conj,
	rhs: MatMut<'_, T, Dim<'N>, Dim<'K>>,
	mut par: Par,
) {
	let N = tril.nrows();
	let K = rhs.ncols();
	let n = N.unbound();
	let k = K.unbound();
	if k > 64 && n <= 128 {
		// One flat split of the right hand sides over the budget when the whole solve pays for one
		// section; the halving below (a join per level, each nested in the one above) is the shape
		// rule for a sequential run.
		#[cfg(feature = "rayon")]
		{
			let jobs = Ord::min(par.degree(), k.div_ceil(64));
			if jobs >= 2
				&& crate::utils::cost::par_pays::<T>(
					&par,
					n.saturating_mul(n).saturating_mul(k) / 2,
					crate::utils::cost::rate::gemm_mac(),
					1,
				) {
				crate::utils::thread::for_each_with_par(
					par,
					jobs,
					rhs.as_dyn_cols_mut().par_col_partition_mut(jobs),
					|chunk, par| {
						make_guard!(CHUNK);
						let width = chunk.ncols().bind(CHUNK);
						solve_unit_lower_triangular_in_place_imp(
							tril,
							conj_lhs,
							chunk.as_col_shape_mut(width),
							par,
						)
					},
				);
				return;
			}
		}
		// A split of the right hand sides, taken for every `par` (it is a shape rule, short
		// triangles against many columns); only the join is a cost: one section, the unit a
		// multiply-add of the solve, `n*n*k/2`, at the GEMM rate.
		let par = if crate::utils::cost::join_pays::<T>(
			&par,
			n.saturating_mul(n).saturating_mul(k) / 2,
			crate::utils::cost::rate::gemm_mac(),
			1,
		) {
			par
		} else {
			Par::Seq
		};
		make_guard!(LEFT);
		make_guard!(RIGHT);
		let mid = K.partition(IdxInc::new_checked(k / 2, K), LEFT, RIGHT);
		let (rhs_left, rhs_right) = rhs.split_cols_with_mut(mid);
		join_raw(
			|par| {
				solve_unit_lower_triangular_in_place_imp(
					tril, conj_lhs, rhs_left, par,
				)
			},
			|par| {
				solve_unit_lower_triangular_in_place_imp(
					tril, conj_lhs, rhs_right, par,
				)
			},
			par,
		);
		return;
	}
	debug_assert!(all(
		tril.nrows() == tril.ncols(),
		rhs.nrows() == tril.ncols(),
	));
	if n <= recursion_threshold() {
		T::Arch::default().dispatch(
			#[inline(always)]
			|| match conj_lhs {
				Conj::Yes => {
					solve_unit_lower_triangular_in_place_base_case_generic_imp(
						tril, rhs, conjugate,
					)
				},
				Conj::No => {
					solve_unit_lower_triangular_in_place_base_case_generic_imp(
						tril, rhs, identity,
					)
				},
			},
		);
		return;
	}
	make_guard!(HEAD);
	make_guard!(TAIL);
	let bs = N.partition(IdxInc::new_checked(block_size(n), N), HEAD, TAIL);
	let (tril_top_left, _, tril_bot_left, tril_bot_right) =
		tril.split_with(bs, bs);
	let (mut rhs_top, mut rhs_bot) = rhs.split_rows_with_mut(bs);
	solve_unit_lower_triangular_in_place_imp(
		tril_top_left,
		conj_lhs,
		rhs_top.rb_mut(),
		par.reborrow(),
	);
	crate::linalg::matmul::matmul_with_conj(
		rhs_bot.rb_mut(),
		Accum::Add,
		tril_bot_left,
		conj_lhs,
		rhs_top.into_const(),
		Conj::No,
		-one::<T>(),
		par.reborrow(),
	);
	solve_unit_lower_triangular_in_place_imp(
		tril_bot_right,
		conj_lhs,
		rhs_bot,
		par,
	);
}
fn solve_lower_triangular_in_place_imp<'N, 'K, T: ComplexField>(
	tril: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	conj_lhs: Conj,
	rhs: MatMut<'_, T, Dim<'N>, Dim<'K>>,
	mut par: Par,
) {
	let N = tril.nrows();
	let K = rhs.ncols();
	let n = N.unbound();
	let k = K.unbound();
	if k > 64 && n <= 128 {
		#[cfg(feature = "rayon")]
		{
			let jobs = Ord::min(par.degree(), k.div_ceil(64));
			if jobs >= 2
				&& crate::utils::cost::par_pays::<T>(
					&par,
					n.saturating_mul(n).saturating_mul(k) / 2,
					crate::utils::cost::rate::gemm_mac(),
					1,
				) {
				crate::utils::thread::for_each_with_par(
					par,
					jobs,
					rhs.as_dyn_cols_mut().par_col_partition_mut(jobs),
					|chunk, par| {
						make_guard!(CHUNK);
						let width = chunk.ncols().bind(CHUNK);
						solve_lower_triangular_in_place_imp(
							tril,
							conj_lhs,
							chunk.as_col_shape_mut(width),
							par,
						)
					},
				);
				return;
			}
		}
		// A split of the right hand sides, taken for every `par` (it is a shape rule, short
		// triangles against many columns); only the join is a cost: one section, the unit a
		// multiply-add of the solve, `n*n*k/2`, at the GEMM rate.
		let par = if crate::utils::cost::join_pays::<T>(
			&par,
			n.saturating_mul(n).saturating_mul(k) / 2,
			crate::utils::cost::rate::gemm_mac(),
			1,
		) {
			par
		} else {
			Par::Seq
		};
		make_guard!(LEFT);
		make_guard!(RIGHT);
		let mid = K.partition(IdxInc::new_checked(k / 2, K), LEFT, RIGHT);
		let (rhs_left, rhs_right) = rhs.split_cols_with_mut(mid);
		join_raw(
			|par| {
				solve_lower_triangular_in_place_imp(
					tril, conj_lhs, rhs_left, par,
				)
			},
			|par| {
				solve_lower_triangular_in_place_imp(
					tril, conj_lhs, rhs_right, par,
				)
			},
			par,
		);
		return;
	}
	debug_assert!(all(
		tril.nrows() == tril.ncols(),
		rhs.nrows() == tril.ncols(),
	));
	if n <= recursion_threshold() {
		T::Arch::default().dispatch(
			#[inline(always)]
			|| match conj_lhs {
				Conj::Yes => {
					solve_lower_triangular_in_place_base_case_generic_imp(
						tril, rhs, conjugate,
					)
				},
				Conj::No => {
					solve_lower_triangular_in_place_base_case_generic_imp(
						tril, rhs, identity,
					)
				},
			},
		);
		return;
	}
	make_guard!(HEAD);
	make_guard!(TAIL);
	let bs = N.partition(IdxInc::new_checked(block_size(n), N), HEAD, TAIL);
	let (tril_top_left, _, tril_bot_left, tril_bot_right) =
		tril.split_with(bs, bs);
	let (mut rhs_top, mut rhs_bot) = rhs.split_rows_with_mut(bs);
	solve_lower_triangular_in_place_imp(
		tril_top_left,
		conj_lhs,
		rhs_top.rb_mut(),
		par.reborrow(),
	);
	crate::linalg::matmul::matmul_with_conj(
		rhs_bot.rb_mut(),
		Accum::Add,
		tril_bot_left,
		conj_lhs,
		rhs_top.into_const(),
		Conj::No,
		-one::<T>(),
		par.reborrow(),
	);
	solve_lower_triangular_in_place_imp(tril_bot_right, conj_lhs, rhs_bot, par);
}
#[inline]
fn solve_unit_upper_triangular_in_place_imp<'N, 'K, T: ComplexField>(
	triu: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	conj_lhs: Conj,
	rhs: MatMut<'_, T, Dim<'N>, Dim<'K>>,
	par: Par,
) {
	solve_unit_lower_triangular_in_place_imp(
		triu.reverse_rows_and_cols(),
		conj_lhs,
		rhs.reverse_rows_mut(),
		par,
	);
}
#[inline]
fn solve_upper_triangular_in_place_imp<'N, 'K, T: ComplexField>(
	triu: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	conj_lhs: Conj,
	rhs: MatMut<'_, T, Dim<'N>, Dim<'K>>,
	par: Par,
) {
	solve_lower_triangular_in_place_imp(
		triu.reverse_rows_and_cols(),
		conj_lhs,
		rhs.reverse_rows_mut(),
		par,
	);
}
#[cfg(test)]
mod tests {
	use super::*;
	fn check_subnormal_solve<T: ComplexField>(pivot: T, coupled: bool) {
		let large =
			from_f64::<T>(0.0625).mul_real(min_positive::<T::Real>().recip());
		for n in [1, 2, 3, 4, 5, 8, 16, 17, 32, 64, 65, 128, 129] {
			for upper in [false, true] {
				let diagonal = |i| {
					if coupled && i % 2 == 1 {
						large.copy()
					} else {
						pivot.copy()
					}
				};
				let a = Mat::<T>::from_fn(n, n, |i, j| {
					if i == j {
						diagonal(i)
					} else if coupled
						&& (if upper { j == i + 1 } else { i == j + 1 })
					{
						-diagonal(i)
					} else {
						zero()
					}
				});
				for conj in [Conj::No, Conj::Yes] {
					for row_major in [false, true] {
						let b = |i| {
							let d = if !coupled
								|| i == (if upper { n - 1 } else { 0 })
							{
								diagonal(i)
							} else {
								zero()
							};
							if conj == Conj::Yes { d.conj() } else { d }
						};
						let mut rhs = if row_major {
							Mat::<T>::from_fn(2, n, |_, i| b(i))
						} else {
							Mat::<T>::from_fn(n, 2, |i, _| b(i))
						};
						let view = if row_major {
							rhs.as_mut().transpose_mut()
						} else {
							rhs.as_mut()
						};
						if upper {
							solve_upper_triangular_in_place_with_conj(
								a.as_ref(),
								conj,
								view,
								Par::Seq,
							);
						} else {
							solve_lower_triangular_in_place_with_conj(
								a.as_ref(),
								conj,
								view,
								Par::Seq,
							);
						}
						for j in 0..rhs.ncols() {
							for i in 0..rhs.nrows() {
								assert_eq!(
									rhs[(i, j)],
									one::<T>(),
									"n={n}, upper={upper}, conj={conj:?}, row_major={row_major}, coupled={coupled}"
								);
							}
						}
					}
				}
			}
		}
	}
	#[test]
	fn subnormal_real_diagonal_solve() {
		for tiny in [f32::from_bits(1), f32::MIN_POSITIVE / 2.0, 1.0] {
			for pivot in [tiny, -tiny] {
				check_subnormal_solve(pivot, false);
			}
		}
		for tiny in [f64::from_bits(1), f64::MIN_POSITIVE / 2.0, 1.0] {
			for pivot in [tiny, -tiny] {
				check_subnormal_solve(pivot, false);
			}
		}
	}
	#[test]
	fn subnormal_complex_diagonal_solve() {
		for tiny in [f32::from_bits(1), f32::MIN_POSITIVE / 2.0, 1.0] {
			for pivot in [
				c32::new(0.0, tiny),
				c32::new(tiny, 0.0),
				c32::new(tiny, tiny),
			] {
				check_subnormal_solve(pivot, false);
			}
		}
		for tiny in [f64::from_bits(1), f64::MIN_POSITIVE / 2.0, 1.0] {
			for pivot in [
				c64::new(0.0, tiny),
				c64::new(tiny, 0.0),
				c64::new(tiny, tiny),
			] {
				check_subnormal_solve(pivot, false);
			}
		}
	}
	#[test]
	fn subnormal_coupled_solve() {
		check_subnormal_solve(f32::from_bits(1), true);
		check_subnormal_solve(f64::from_bits(1), true);
		check_subnormal_solve(c32::new(0.0, f32::from_bits(1)), true);
		check_subnormal_solve(c64::new(0.0, f64::from_bits(1)), true);
		check_subnormal_solve(
			c32::new(f32::from_bits(1), f32::from_bits(1)),
			true,
		);
		check_subnormal_solve(
			c64::new(f64::from_bits(1), f64::from_bits(1)),
			true,
		);
	}
	#[test]
	fn tiny_pivot_scalar_solve() {
		fn check<T: ComplexField + Clone>(pivot: T) {
			let a = Mat::<T>::from_fn(1, 1, |_, _| pivot.clone());
			for conj in [Conj::No, Conj::Yes] {
				let mut rhs = Mat::<T>::from_fn(1, 1, |_, _| pivot.clone());
				solve_lower_triangular_in_place_with_conj(
					a.as_ref(),
					conj,
					rhs.as_mut(),
					Par::Seq,
				);
				assert_eq!(rhs[(0, 0)].clone(), one::<T>());
			}
		}
		check(1e-310f64);
		check(1e-44f32);
	}
	/// a purely imaginary subnormal pivot (`tensor4all/faer#31`)
	#[test]
	fn tiny_imaginary_pivot_scalar_solve() {
		fn check<T: ComplexField + Clone>(pivot: T) {
			let a = Mat::<T>::from_fn(1, 1, |_, _| pivot.clone());
			for conj in [Conj::No, Conj::Yes] {
				let mut rhs = Mat::<T>::from_fn(1, 1, |_, _| match conj {
					Conj::No => pivot.clone(),
					Conj::Yes => pivot.conj(),
				});
				solve_lower_triangular_in_place_with_conj(
					a.as_ref(),
					conj,
					rhs.as_mut(),
					Par::Seq,
				);
				assert_eq!(rhs[(0, 0)].clone(), one::<T>());
				let mut rhs = Mat::<T>::from_fn(1, 1, |_, _| match conj {
					Conj::No => pivot.clone(),
					Conj::Yes => pivot.conj(),
				});
				solve_upper_triangular_in_place_with_conj(
					a.as_ref(),
					conj,
					rhs.as_mut(),
					Par::Seq,
				);
				assert_eq!(rhs[(0, 0)].clone(), one::<T>());
			}
		}
		check(c64::new(0.0, f64::MIN_POSITIVE / 2.0));
		check(c32::new(0.0, f32::MIN_POSITIVE / 2.0));
	}
}

#[cfg(test)]
#[path = "tests/lapack_triangular.rs"]
mod lapack;
