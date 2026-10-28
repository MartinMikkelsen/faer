use super::SvdError;
use crate::internal_prelude::*;
use crate::perm::swap_cols_idx;
use core::mem::swap;
use linalg::jacobi::JacobiRotation;
/// secular eq must be increasing
///
/// Convergence safeguards compared with OpenBLAS v0.3.26
/// `lapack-netlib/SRC/dlasd4.f`; no LAPACK code is ported here.
pub(crate) fn secular_eq_root_finder<T: RealField>(
	secular_eq: &dyn Fn(T, T) -> T,
	batch_secular_eq: &dyn Fn(&[T; 4], &[T; 4]) -> [T; 4],
	ref left: T,
	ref right: T,
	last: bool,
) -> (T, T) {
	let ref two = from_f64::<T>(2.0);
	let ref eight = from_f64::<T>(8.0);
	let ref one_half = from_f64::<T>(0.5);
	let ref epsilon = eps::<T>();
	let mid = left + (right - left).mul_pow2(one_half);
	let [mut f_mid, f_max, f_mid_left_shift, f_mid_right_shift] =
		batch_secular_eq(
			&[zero(), left.copy(), left.copy(), right.copy()],
			&[
				mid,
				if last {
					right - left
				} else {
					(right - left).mul_pow2(one_half)
				},
				(right - left).mul_pow2(one_half),
				(left - right).mul_pow2(one_half),
			],
		);
	let (mut shift, mu) = if last || (f_mid > zero()) {
		(left.copy(), (right - left).mul_pow2(one_half))
	} else {
		(right.copy(), (left - right).mul_pow2(one_half))
	};
	if (f_mid_left_shift <= zero()) && (f_mid_right_shift > zero()) {
		return (shift, mu);
	}
	if !last {
		if shift == *left {
			if f_mid_left_shift < zero() {
				shift = right.copy();
				f_mid = f_mid_right_shift;
			}
		} else if f_mid_right_shift > zero() {
			shift = left.copy();
			f_mid = f_mid_left_shift;
		}
	}
	enum SecantError {
		OutOfBounds,
		PrecisionLimitReached,
	}
	let secant = {
		|mut mu_cur: T, mut mu_prev: T, mut f_cur: T, mut f_prev: T| {
			if f_prev.abs() < f_cur.abs() {
				swap(&mut f_prev, &mut f_cur);
				swap(&mut mu_prev, &mut mu_cur);
			}
			let mut left_candidate = None;
			let mut right_candidate = None;
			let mut use_bisection = false;
			let sme_sign = (f_prev > zero()) == (f_cur > zero());
			if !sme_sign {
				let (min, max) = if mu_cur < mu_prev {
					(mu_cur.copy(), mu_prev.copy())
				} else {
					(mu_prev.copy(), mu_cur.copy())
				};
				left_candidate = Some(min);
				right_candidate = Some(max);
			}
			let mut err = SecantError::PrecisionLimitReached;
			while f_cur != zero()
				&& (&mu_cur - &mu_prev).abs()
					> eight * epsilon * mu_cur.abs().fmax(mu_prev.abs())
				&& !use_bisection
			{
				// Flat function values do not imply convergence of the root.
				if (&f_cur - &f_prev).abs() <= *epsilon {
					use_bisection = true;
					break;
				}
				let ref a = (&f_cur - &f_prev) * (&mu_prev * &mu_cur)
					/ (&mu_prev - &mu_cur);
				let ref b = &f_cur - a / &mu_cur;
				let mu_zero = -a / b;
				let ref f_zero = secular_eq(shift.copy(), mu_zero.copy());
				if *f_zero < zero() {
					left_candidate = Some(mu_zero.copy());
				} else {
					right_candidate = Some(mu_zero.copy());
				}
				mu_prev = mu_cur.copy();
				f_prev = f_cur.copy();
				mu_cur = mu_zero.copy();
				f_cur = f_zero.copy();
				if shift == *left && (mu_cur < zero() || mu_cur > right - left)
				{
					err = SecantError::OutOfBounds;
					use_bisection = true;
				}
				if shift == *right && (mu_cur > zero() || mu_cur < left - right)
				{
					err = SecantError::OutOfBounds;
					use_bisection = true;
				}
				if f_cur.abs() > f_prev.abs() {
					let mut k = one::<T>();
					for _ in 0..4 {
						let mu_opposite = -a / (&k * f_zero + b);
						let f_opposite =
							secular_eq(shift.copy(), mu_opposite.copy());
						if *f_zero < zero() && f_opposite >= zero() {
							right_candidate = Some(mu_opposite);
							break;
						}
						if *f_zero > zero() && f_opposite <= zero() {
							left_candidate = Some(mu_opposite);
							break;
						}
						k = k.mul_pow2(two);
					}
					use_bisection = true;
				}
			}
			(use_bisection, mu_cur, left_candidate, right_candidate, err)
		}
	};
	let (mut left_shifted, mut f_left, mut right_shifted, mut f_right) =
		if shift == *left {
			(
				zero(),
				-infinity::<T>(),
				if last {
					right - left
				} else {
					(right - left) * one_half
				},
				if last { &f_max } else { &f_mid }.copy(),
			)
		} else {
			((left - right) * one_half, f_mid.copy(), zero(), infinity())
		};
	let mut iteration_count = 0;
	let mut f_prev = f_mid.copy();
	let ref half0 = one_half.copy();
	let ref half1 = half0.mul_pow2(half0);
	let ref half2 = half1.mul_pow2(half1);
	let ref half3 = half2.mul_pow2(half2);
	let mu_values = if shift == *left {
		[
			right_shifted.mul_pow2(half3),
			right_shifted.mul_pow2(half2),
			right_shifted.mul_pow2(half1),
			right_shifted.mul_pow2(half0),
		]
	} else {
		[
			left_shifted.mul_pow2(half3),
			left_shifted.mul_pow2(half2),
			left_shifted.mul_pow2(half1),
			left_shifted.mul_pow2(half0),
		]
	};
	let f_values = batch_secular_eq(&[(); 4].map(|_| shift.copy()), &mu_values);
	if shift == *left {
		let mut i = 0;
		for (idx, (mu, f)) in core::iter::zip(&mu_values, &f_values).enumerate()
		{
			if *f < zero() {
				left_shifted = mu.copy();
				f_left = f.copy();
				i = idx + 1;
			}
		}
		if i < f_values.len() {
			right_shifted = mu_values[i].copy();
			f_right = f_values[i].copy();
		}
	} else {
		let mut i = 0;
		for (idx, (mu, f)) in core::iter::zip(&mu_values, &f_values).enumerate()
		{
			if *f > zero() {
				right_shifted = mu.copy();
				f_right = f.copy();
				i = idx + 1;
			}
		}
		if i < f_values.len() {
			left_shifted = mu_values[i].copy();
			f_left = f_values[i].copy();
		}
	}
	while &right_shifted - &left_shifted
		> two * epsilon * left_shifted.abs().fmax(right_shifted.abs())
	{
		let mid_shifted_arithmetic =
			(&left_shifted + &right_shifted) * one_half;
		let mut mid_shifted_geometric =
			left_shifted.abs().sqrt() * right_shifted.abs().sqrt();
		if left_shifted < zero() {
			mid_shifted_geometric = -mid_shifted_geometric;
		}
		let mid_shifted = if mid_shifted_geometric == zero() {
			mid_shifted_arithmetic
		} else {
			mid_shifted_geometric
		};
		let f_mid = secular_eq(shift.copy(), mid_shifted.copy());
		if f_mid == zero() {
			return (shift, mid_shifted);
		} else if f_mid > zero() {
			right_shifted = mid_shifted;
			f_prev = f_right;
			f_right = f_mid;
		} else {
			left_shifted = mid_shifted;
			f_prev = f_left;
			f_left = f_mid;
		}
		if iteration_count == 4 {
			break;
		}
		iteration_count += 1;
	}
	let args = if left_shifted == zero() {
		(
			right_shifted.mul_pow2(two),
			right_shifted.copy(),
			f_prev,
			f_right,
		)
	} else if right_shifted == zero() {
		(
			left_shifted.mul_pow2(two),
			left_shifted.copy(),
			f_prev,
			f_left,
		)
	} else {
		(left_shifted.copy(), right_shifted.copy(), f_left, f_right)
	};
	let (use_bisection, mut mu_cur, left_candidate, right_candidate, _err) =
		secant(args.0, args.1, args.2, args.3);
	match (left_candidate, right_candidate) {
		(Some(left), Some(right)) if left < right => {
			if left > left_shifted {
				left_shifted = left;
			}
			if right < right_shifted {
				right_shifted = right;
			}
		},
		_ => {},
	}
	if use_bisection {
		while &right_shifted - &left_shifted
			> two * epsilon * left_shifted.abs().fmax(right_shifted.abs())
		{
			let mid_shifted = (&left_shifted + &right_shifted) * one_half;
			let f_mid = secular_eq(shift.copy(), mid_shifted.copy());
			if f_mid == zero() {
				break;
			} else if f_mid > zero() {
				right_shifted = mid_shifted;
			} else {
				left_shifted = mid_shifted;
			}
		}
		mu_cur = (&left_shifted + &right_shifted) * one_half;
	}
	(shift, mu_cur)
}
pub(super) fn qr_algorithm<T: RealField>(
	diag: ColMut<'_, T, usize, ContiguousFwd>,
	subdiag: ColMut<'_, T, usize, ContiguousFwd>,
	u: Option<MatMut<'_, T>>,
	v: Option<MatMut<'_, T>>,
) -> Result<(), SvdError> {
	let n = diag.nrows();
	let ref eps = eps::<T>();
	let ref sml = min_positive::<T>();
	if n == 0 {
		return Ok(());
	}
	let max_iters = Ord::max(30, nbits::<T>() / 2)
		.saturating_mul(n)
		.saturating_mul(n);
	let last = n - 1;
	let ref max = diag.norm_max().fmax(subdiag.norm_max());
	if *max == zero() {
		return Ok(());
	}
	let ref max_inv = max.recip();
	// Outside the normal range `max` has an infinite or subnormal (inexact)
	// reciprocal: divide instead. Other inputs keep the multiply, so their
	// results are bit-for-bit unchanged.
	let divide = !(max_inv.is_finite() && *max_inv >= min_positive::<T>());
	let mut diag = diag;
	let mut subdiag = subdiag;
	let mut u = u;
	let mut v = v;
	for x in diag.rb_mut().iter_mut() {
		if divide {
			*x = x.copy() / max;
		} else {
			*x *= max_inv;
		}
	}
	for x in subdiag.rb_mut().iter_mut() {
		if divide {
			*x = x.copy() / max;
		} else {
			*x *= max_inv;
		}
	}
	{
		let ref eps2 = eps * eps;
		for iter in 0..max_iters {
			for i0 in 0..last {
				let i1 = i0 + 1;
				if subdiag[i0].abs2()
					<= eps2 * (&diag[i0] * &diag[i1]).abs() + sml
				{
					subdiag[i0] = zero();
				}
			}
			let mut end = n;
			while end >= 2 && subdiag[end - 2].abs2() <= *sml {
				end -= 1;
			}
			if end == 1 {
				break;
			}
			let mut start = end - 1;
			while start >= 1 && !(subdiag[start - 1] == zero()) {
				start -= 1;
			}
			let mut deflated = false;
			for i in start..end - 1 {
				if diag[i].abs() <= *eps {
					deflated = true;
					let mut val = subdiag[i].copy();
					subdiag[i] = zero();
					for j in i + 1..end {
						let rot = JacobiRotation::make_givens(
							diag[j].copy(),
							val.copy(),
						);
						diag[j] = &rot.c * &diag[j] - &rot.s * &val;
						if j + 1 < end {
							val = &rot.s * &subdiag[j];
							subdiag[j] = &rot.c * &subdiag[j];
						}
						if let Some(v) = v.rb_mut() {
							rot.adjoint().apply_on_the_right_in_place(
								v.two_cols_mut(j, i),
							);
						}
					}
				}
			}
			if !deflated && end - start == 2 {
				let i = start;
				let j = i + 1;
				let ref f = diag[i].copy();
				let ref g = subdiag[i].copy();
				let ref h = diag[j].copy();
				// Symmetrize the block, then diagonalize it with a Jacobi rotation.
				let polar = JacobiRotation::make_givens(f + h, g.copy());
				let (s00, s01, _, s11) = polar.apply_on_the_left_2x2(
					f.copy(),
					zero(),
					g.copy(),
					h.copy(),
				);
				let ref gap = (&s11 - &s00).mul_pow2(from_f64::<T>(0.5));
				let ref t = if s01 == zero() {
					zero()
				} else {
					let denom = gap.abs() + gap.hypot(s01.copy());
					if *gap < zero() {
						&s01 / denom
					} else {
						-&s01 / denom
					}
				};
				let ref c = t.hypot(one::<T>()).recip();
				let right = JacobiRotation {
					c: c.copy(),
					s: t * c,
				};
				let left = JacobiRotation {
					c: &polar.c * &right.c + &polar.s * &right.s,
					s: &polar.c * &right.s - &polar.s * &right.c,
				};
				let mut d0 = &s00 + t * &s01;
				let mut d1 = &s11 - t * &s01;
				// Recover the smaller value from det(B), avoiding cancellation.
				if d0.abs() >= d1.abs() {
					d1 = (f / &d0) * h;
				} else {
					d0 = (f / &d1) * h;
				}
				diag[i] = d0;
				diag[j] = d1;
				subdiag[i] = zero();
				if let Some(u) = u.rb_mut() {
					left.apply_on_the_right_in_place(u.two_cols_mut(i, j));
				}
				if let Some(v) = v.rb_mut() {
					right.apply_on_the_right_in_place(v.two_cols_mut(i, j));
				}
				deflated = true;
			}
			if deflated {
				if iter + 1 == max_iters {
					for x in diag.rb_mut().iter_mut() {
						*x *= max;
					}
					for x in subdiag.rb_mut().iter_mut() {
						*x *= max;
					}
					return Err(SvdError::NoConvergence);
				}
				continue;
			}
			let end2 = end - 2;
			let end1 = end - 1;
			let t00 = diag[end2].abs2() + subdiag[end - 3].abs2();
			let t11 = diag[end1].abs2() + subdiag[end2].abs2();
			let t01 = &diag[end2] * &subdiag[end2];
			let ref t01_2 = t01.abs2();
			let mu;
			if *t01_2 > *sml {
				let ref d = (t00 - &t11).mul_pow2(from_f64::<T>(0.5));
				let mut delta = (d.abs2() + t01_2).sqrt();
				if *d < zero() {
					delta = -delta;
				}
				mu = t11 - t01_2 / (d + delta);
			} else {
				mu = t11;
			}
			let mut y = diag[start].abs2() - mu;
			let mut z = &diag[start] * &subdiag[start];
			for k in start..end1 {
				let rot = JacobiRotation::make_givens(y.copy(), z.copy());
				if k > start {
					subdiag[k - 1] = (&rot.c * &y - &rot.s * &z).abs();
				}
				let mut diag_k = diag[k].copy();
				let tmp = (
					&rot.c * &diag_k - &rot.s * &subdiag[k],
					&rot.s * &diag_k + &rot.c * &subdiag[k],
				);
				diag_k = tmp.0;
				subdiag[k] = tmp.1;
				let k1 = k + 1;
				y = diag_k.copy();
				z = -&rot.s * &diag[k1];
				diag[k1] = &rot.c * &diag[k1];
				if let Some(u) = u.rb_mut() {
					let (k, k1) = u.two_cols_mut(k, k1);
					rot.apply_on_the_right_in_place((k1, k));
				}
				let rot = JacobiRotation::make_givens(y.copy(), z.copy());
				diag_k = &rot.c * &y - &rot.s * &z;
				diag[k] = diag_k;
				let tmp = (
					&rot.c * &subdiag[k] - &rot.s * &diag[k1],
					&rot.s * &subdiag[k] + &rot.c * &diag[k1],
				);
				subdiag[k] = tmp.0;
				diag[k1] = tmp.1;
				if k < end - 2 {
					y = subdiag[k].copy();
					z = -&rot.s * &subdiag[k1];
					subdiag[k1] = &rot.c * &subdiag[k1];
				}
				if let Some(v) = v.rb_mut() {
					let (k, k1) = v.two_cols_mut(k, k1);
					rot.apply_on_the_right_in_place((k1, k));
				}
			}
			if iter + 1 == max_iters {
				for x in diag.rb_mut().iter_mut() {
					*x *= max;
				}
				for x in subdiag.rb_mut().iter_mut() {
					*x *= max;
				}
				return Err(SvdError::NoConvergence);
			}
		}
	}
	for j in 0..n {
		let d = diag.rb_mut().at_mut(j);
		if *d < zero() {
			*d = -&*d;
			if let Some(mut v) = v.rb_mut() {
				for i in 0..n {
					v[(i, j)] = -&v[(i, j)];
				}
			}
		}
	}
	for k in 0..n {
		let mut max = zero();
		let mut idx = k;
		for kk in k..n {
			if diag[kk] > max {
				max = diag[kk].copy();
				idx = kk;
			}
		}
		if k != idx {
			let dk = diag[k].copy();
			let di = diag[idx].copy();
			diag[idx] = dk;
			diag[k] = di;
			if let Some(u) = u.rb_mut() {
				swap_cols_idx(u, k, idx);
			}
			if let Some(v) = v.rb_mut() {
				swap_cols_idx(v, k, idx);
			}
		}
	}
	for x in diag.rb_mut().iter_mut() {
		*x *= max;
	}
	for x in subdiag.rb_mut().iter_mut() {
		*x *= max;
	}
	Ok(())
}
fn compute_svd_of_m<T: RealField>(
	um: Option<MatMut<'_, T, usize, usize, ContiguousFwd>>,
	vm: Option<MatMut<'_, T, usize, usize, ContiguousFwd>>,
	diag: ColMut<'_, T, usize, ContiguousFwd>,
	col0: ColRef<'_, T, usize, ContiguousFwd>,
	outer_perm: &[usize],
	stack: &mut MemStack,
) {
	let mut diag = diag;
	let mut um = um;
	let mut vm = vm;
	let n = diag.nrows();
	diag[0] = zero();
	let mut actual_n = n;
	while actual_n > 1 && (diag[actual_n - 1] == zero()) {
		actual_n -= 1;
	}
	let (perm, stack) = stack.collect(
		col0.iter()
			.take(actual_n)
			.map(|x| x.abs())
			.enumerate()
			.filter(|(_, x)| !(*x == zero()))
			.map(|(i, _)| i),
	);
	let perm = &*perm;
	with_dim!(o, perm.len());
	let (mut col0_perm, stack) = unsafe { temp_mat_uninit(o, 1, stack) };
	let (mut diag_perm, stack) = unsafe { temp_mat_uninit(o, 1, stack) };
	let mut col0_perm = col0_perm
		.as_mat_mut()
		.col_mut(0)
		.try_as_col_major_mut()
		.unwrap();
	let mut diag_perm = diag_perm
		.as_mat_mut()
		.col_mut(0)
		.try_as_col_major_mut()
		.unwrap();
	for (k, &p) in perm.iter().enumerate() {
		let k = o.idx(k);
		col0_perm[k] = col0[p].copy();
		diag_perm[k] = diag[p].copy();
	}
	let (mut shifts, stack) = unsafe { temp_mat_uninit(n, 1, stack) };
	let shifts = shifts.as_mat_mut();
	let (mut mus, stack) = unsafe { temp_mat_uninit(n, 1, stack) };
	let mus = mus.as_mat_mut();
	let (mut singular_vals, stack) = unsafe { temp_mat_uninit(n, 1, stack) };
	let singular_vals = singular_vals.as_mat_mut();
	let (mut zhat, stack) = unsafe { temp_mat_uninit(n, 1, stack) };
	let zhat = zhat.as_mat_mut();
	let mut shifts = shifts.col_mut(0).try_as_col_major_mut().unwrap();
	let mut mus = mus.col_mut(0).try_as_col_major_mut().unwrap();
	let mut s = singular_vals.col_mut(0).try_as_col_major_mut().unwrap();
	let mut zhat = zhat.col_mut(0).try_as_col_major_mut().unwrap();
	with_dim!(N, diag.nrows());
	compute_singular_values(
		shifts.rb_mut().as_row_shape_mut(N),
		mus.rb_mut().as_row_shape_mut(N),
		s.rb_mut().as_row_shape_mut(N),
		diag.rb().as_row_shape(N),
		diag_perm.rb(),
		col0.as_row_shape(N),
		col0_perm.rb(),
	);
	let perm = Idx::from_slice_ref_checked(perm, N);
	let outer_perm =
		Array::from_ref(Idx::from_slice_ref_checked(outer_perm, N), N);
	perturb_col0(
		zhat.rb_mut().as_row_shape_mut(N),
		col0.as_row_shape(N),
		diag.rb().as_row_shape(N),
		perm,
		s.rb().as_row_shape(N),
		shifts.rb().as_row_shape(N),
		mus.rb().as_row_shape(N),
	);
	let (mut col_perm, stack) = stack.make_with(actual_n, |i| i);
	let (mut col_perm_inv, _) = stack.make_with(actual_n, |i| i);
	for i0 in 0..actual_n - 1 {
		let i1 = i0 + 1;
		if s[i0] > s[i1] {
			let si = s[i0].copy();
			let sj = s[i1].copy();
			s[i0] = sj;
			s[i1] = si;
			col_perm.swap(i0, i1);
		}
	}
	for (i, p) in col_perm.iter().copied().enumerate() {
		col_perm_inv[p] = i;
	}
	compute_singular_vectors(
		um.rb_mut(),
		vm.rb_mut().map(|v| v.as_shape_mut(N, N)),
		zhat.rb().as_row_shape(N),
		diag.rb().as_row_shape(N),
		perm,
		outer_perm,
		&col_perm_inv,
		actual_n,
		shifts.rb().as_row_shape(N),
		mus.rb().as_row_shape(N),
	);
	for (idx, diag) in diag.rb_mut().get_mut(..actual_n).iter_mut().enumerate()
	{
		*diag = s[actual_n - idx - 1].copy();
	}
	for (idx, diag) in diag.rb_mut().get_mut(actual_n..).iter_mut().enumerate()
	{
		*diag = s[actual_n + idx].copy();
	}
}
#[inline(never)]
fn compute_singular_vectors<'N, T: RealField>(
	mut um: Option<MatMut<T, usize, usize, ContiguousFwd>>,
	mut vm: Option<MatMut<T, Dim<'N>, Dim<'N>, ContiguousFwd>>,
	zhat: ColRef<T, Dim<'N>, ContiguousFwd>,
	diag: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
	perm: &[Idx<'N>],
	outer_perm: &Array<'N, Idx<'N>>,
	col_perm_inv: &[usize],
	actual_n: usize,
	shifts: ColRef<T, Dim<'N>, ContiguousFwd>,
	mus: ColRef<T, Dim<'N>, ContiguousFwd>,
) {
	let n = diag.nrows();
	for k in n.indices() {
		let actual_k = if *k >= actual_n {
			k
		} else {
			n.idx(actual_n - col_perm_inv[*k] - 1)
		};
		if let Some(mut u) = um.rb_mut() {
			u[(*n, *actual_k)] = zero();
		}
		let mut u = um
			.rb_mut()
			.map(|u| u.submatrix_mut(0, 0, n, n).col_mut(actual_k));
		let mut v = vm.rb_mut().map(|v| {
			v.submatrix_mut(IdxInc::ZERO, IdxInc::ZERO, n, n)
				.col_mut(actual_k)
		});
		if zhat[k] == zero() {
			if let Some(mut u) = u.rb_mut() {
				u[outer_perm[k]] = one();
			}
			if let Some(mut v) = v.rb_mut() {
				v[outer_perm[k]] = one();
			}
			continue;
		}
		let ref mu = mus[k].copy();
		let ref shift = shifts[k].copy();
		if let Some(mut u) = u.rb_mut() {
			for &i in perm {
				u[outer_perm[i]] = (&zhat[i] / ((&diag[i] - shift) - mu))
					/ (&diag[i] + (shift + mu));
			}
			let ref norm_inv = u.norm_l2().recip();
			z!(u.rb_mut()).for_each(|uz!(x)| *x *= norm_inv);
		}
		if let Some(mut v) = v {
			v[outer_perm[n.idx(0)]] = -one::<T>();
			for &i in &perm[1..] {
				v[outer_perm[i]] = ((&diag[i] * &zhat[i])
					/ ((&diag[i] - shift) - mu))
					/ (&diag[i] + (shift + mu));
			}
			let ref norm_inv = v.norm_l2().recip();
			z!(v.rb_mut()).for_each(|uz!(x)| *x *= norm_inv);
		}
	}
	if let Some(mut u) = um.rb_mut() {
		u[(*n, *n)] = one();
	}
}
fn perturb_col0<'N, T: RealField>(
	zhat: ColMut<'_, T, Dim<'N>, ContiguousFwd>,
	col0: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
	diag: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
	perm: &[Idx<'N>],
	s: ColRef<T, Dim<'N>, ContiguousFwd>,
	shifts: ColRef<T, Dim<'N>, ContiguousFwd>,
	mus: ColRef<T, Dim<'N>, ContiguousFwd>,
) {
	let mut zhat = zhat;
	let n = diag.nrows();
	let m = perm.len();
	if m == 0 {
		z!(zhat).for_each(|uz!(x): Zip![&mut _]| *x = zero());
		return;
	}
	let last_idx = perm[m - 1];
	for k in n.indices() {
		if col0[k] == zero() {
			zhat[k] = zero();
			continue;
		}
		let ref dk = diag[k];
		let mut prod =
			(&s[last_idx] + dk) * (&mus[last_idx] + (&shifts[last_idx] - dk));
		for (l, &i) in perm.iter().enumerate() {
			if i == k {
				continue;
			}
			if i >= k && l == 0 {
				prod = zero();
				break;
			}
			let j = if i < k {
				i
			} else if l > 0 {
				perm[l - 1]
			} else {
				i
			};
			let term = ((&s[j] + dk) / (&diag[i] + dk))
				* ((&mus[j] + (&shifts[j] - dk)) / (&diag[i] - dk));
			prod = prod * term;
		}
		let tmp = prod.sqrt();
		if col0[k] > zero() {
			zhat[k] = tmp;
		} else {
			zhat[k] = -tmp;
		}
	}
}
fn compute_singular_values<'N, 'O, T: RealField>(
	mut shifts: ColMut<T, Dim<'N>, ContiguousFwd>,
	mut mus: ColMut<T, Dim<'N>, ContiguousFwd>,
	mut s: ColMut<T, Dim<'N>, ContiguousFwd>,
	diag: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
	diag_perm: ColRef<'_, T, Dim<'O>, ContiguousFwd>,
	col0: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
	col0_perm: ColRef<'_, T, Dim<'O>, ContiguousFwd>,
) {
	let n = diag.nrows();
	let mut actual_n = *n;
	while actual_n > 1 && (col0[n.idx(actual_n - 1)] == zero()) {
		actual_n -= 1;
	}
	let actual_n = actual_n;
	let first = n.idx(0);
	'kth_iter: for k in n.indices() {
		s[k] = zero();
		shifts[k] = zero();
		mus[k] = zero();
		if col0[k] == zero() || actual_n == 1 {
			s[k] = if *k == 0 {
				col0[first].copy()
			} else {
				diag[k].copy()
			};
			shifts[k] = s[k].copy();
			mus[k] = zero();
			continue 'kth_iter;
		}
		let last_k = *k == actual_n - 1;
		let left = diag[k].copy();
		let right = if last_k {
			&diag[n.idx(actual_n - 1)] + col0.norm_l2()
		} else {
			let mut l = n.idx(*k + 1);
			while col0[l] == zero() {
				l = n.idx(*l + 1);
			}
			diag[l].copy()
		};
		let (shift, mu) = secular_eq_root_finder(
			&|shift, mu| secular_eq(shift, mu, col0_perm, diag_perm),
			&|shift, mu| batch_secular_eq(shift, mu, col0_perm, diag_perm),
			left,
			right,
			last_k,
		);
		s[k] = &shift + &mu;
		shifts[k] = shift;
		mus[k] = mu;
	}
}
fn secular_eq<'N, T: RealField>(
	ref shift: T,
	ref mu: T,
	col0_perm: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
	diag_perm: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
) -> T {
	let mut res = one();
	let n = diag_perm.nrows();
	for i in n.indices() {
		let ref c = col0_perm[i].copy();
		let ref d = diag_perm[i].copy();
		res += (c / ((d - shift) - mu)) * (c / ((d + shift) + mu));
	}
	res
}
fn batch_secular_eq<'N, const N: usize, T: RealField>(
	ref shift: &[T; N],
	ref mu: &[T; N],
	col0_perm: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
	diag_perm: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
) -> [T; N] {
	let n = col0_perm.nrows();
	let mut res = [(); N].map(|_| one());
	for i in n.indices() {
		let ref c = col0_perm[i].copy();
		let ref d = diag_perm[i].copy();
		for ((res, mu), shift) in
			res.iter_mut().zip(mu.iter()).zip(shift.iter())
		{
			*res += c.abs2() / (((d - shift) - mu) * ((d + shift) + mu));
		}
	}
	res
}
fn deflate<T: RealField>(
	diag: ColMut<'_, T, usize, ContiguousFwd>,
	col0: ColMut<'_, T, usize, ContiguousFwd>,
	jacobi_coeff: &mut [JacobiRotation<T>],
	jacobi_idx: &mut [usize],
	transpositions: &mut [usize],
	perm: &mut [usize],
	k: usize,
	stack: &mut MemStack,
) -> (usize, usize) {
	let mut diag = diag;
	let mut col0 = col0;
	let n = diag.nrows();
	let first = 0;
	let mut jacobi_0i = 0;
	let mut jacobi_ij = 0;
	let (max_diag, max_col0) =
		(diag.rb().get(first + 1..).norm_max(), col0.norm_max());
	let mx = max_diag.fmax(max_col0);
	let ref eps = eps::<T>();
	let ref sml = min_positive::<T>();
	let ref eps_strict = (eps * max_diag).fmax(sml);
	let ref eps_coarse = from_f64::<T>(8.0) * eps * mx;
	if diag[first] < *eps_coarse {
		diag[first] = eps_coarse.copy();
		col0[first] = eps_coarse.copy();
	}
	for i in first + 1..n {
		if col0[i].abs() < *eps_strict {
			col0[i] = zero();
		}
	}
	for i in first + 1..n {
		if diag[i] < *eps_coarse {
			if let Some(rot) = deflation_43(diag.rb_mut(), col0.rb_mut(), i) {
				jacobi_coeff[jacobi_0i] = rot;
				jacobi_idx[jacobi_0i] = i;
				jacobi_0i += 1;
			}
		}
	}
	let mut total_deflation = true;
	for i in first + 1..n {
		if !(col0[i].abs() < *sml) {
			total_deflation = false;
			break;
		}
	}
	// The entries of `diag` that are exactly zero are collected first so that
	// the reversal below leaves them at the end, where `compute_svd_of_m`
	// counts them as trailing zeros. The group's own order is by index, so it
	// is only interchangeable when every member is zero: a nonzero subnormal
	// value must keep its place in the magnitude-ordered merge, which is why
	// this is an exact-zero test and not the `sml` tolerance used for the
	// deflation tests.
	perm[first] = first;
	let mut p = 1;
	for i in first + 1..n {
		if diag[i] == zero() {
			perm[p] = i;
			p += 1;
		}
	}
	let mut i = 1;
	let mut j = k + 1;
	for p in &mut perm.as_mut()[p..] {
		if i >= k + 1 {
			*p = j;
			j += 1;
		} else if j >= n {
			*p = i;
			i += 1;
		} else if diag[i] < diag[j] {
			*p = j;
			j += 1;
		} else {
			*p = i;
			i += 1;
		}
	}
	if total_deflation {
		for i in first + 1..n {
			let i1 = i - 1;
			let pi = perm[i];
			// the same exact-zero classification as the group built above
			if diag[pi] == zero() || diag[pi] > diag[first] {
				perm[i1] = perm[i];
			} else {
				perm[i1] = first;
				break;
			}
		}
	}
	let (mut real_ind, stack) = stack.collect(0..n);
	let (mut real_col, _) = stack.collect(0..n);
	for i in (if total_deflation { first } else { first + 1 })..n {
		let pi = perm[n - if total_deflation { i + 1 } else { i }];
		let j = real_col[pi];
		let (a, b) = (diag[i].copy(), diag[j].copy());
		diag[i] = b;
		diag[j] = a;
		if i != 0 && j != 0 {
			let (a, b) = (col0[i].copy(), col0[j].copy());
			col0[i] = b;
			col0[j] = a;
		}
		transpositions[i] = j;
		let real_i = real_ind[i];
		real_col[real_i] = j;
		real_col[pi] = i;
		real_ind[j] = real_i;
		real_ind[i] = pi;
	}
	col0[first] = diag[first].copy();
	for i in 0..n {
		perm[i] = i;
	}
	for (i, &j) in transpositions.iter().enumerate() {
		perm.as_mut().swap(i, j);
	}
	let mut i = n - 1;
	while i > 0 && (diag[i].abs() < *sml || col0[i].abs() < *sml) {
		i -= 1;
	}
	while i > 1 {
		let i1 = i - 1;
		if &diag[i] - &diag[i1] < *eps_strict {
			if let Some(rot) = deflation_44(diag.rb_mut(), col0.rb_mut(), i1, i)
			{
				jacobi_coeff[jacobi_0i + jacobi_ij] = rot;
				jacobi_idx[jacobi_0i + jacobi_ij] = i;
				jacobi_ij += 1;
			}
		}
		i = i1;
	}
	(jacobi_0i, jacobi_ij)
}
fn deflation_43<T: RealField>(
	diag: ColMut<'_, T, usize, ContiguousFwd>,
	col0: ColMut<'_, T, usize, ContiguousFwd>,
	i: usize,
) -> Option<JacobiRotation<T>> {
	let mut diag = diag;
	let mut col0 = col0;
	let first = 0;
	let ref p = col0[first].copy();
	let ref q = col0[i].copy();
	if *p == zero() && *q == zero() {
		return None;
	}
	let rot = JacobiRotation::make_givens(p.copy(), q.copy());
	let r = &rot.c * p - &rot.s * q;
	col0[first] = r.copy();
	diag[first] = r.copy();
	col0[i] = zero();
	Some(rot)
}
fn deflation_44<T: RealField>(
	diag: ColMut<'_, T, usize, ContiguousFwd>,
	col0: ColMut<'_, T, usize, ContiguousFwd>,
	i: usize,
	j: usize,
) -> Option<JacobiRotation<T>> {
	let mut diag = diag;
	let mut col0 = col0;
	let ref p = col0[i].copy();
	let ref q = col0[j].copy();
	if *p == zero() && *q == zero() {
		diag[i] = diag[j].copy();
		return None;
	}
	let rot = JacobiRotation::make_givens(p.copy(), q.copy());
	let r = &rot.c * p - &rot.s * q;
	col0[i] = r.copy();
	col0[j] = zero();
	diag[i] = diag[j].copy();
	Some(rot)
}
#[derive(Debug)]
pub(super) enum MatU<'a, T: RealField> {
	Full(MatMut<'a, T>),
	TwoRows(MatMut<'a, T>),
	TwoRowsStorage(MatMut<'a, T>),
}
impl<'short, T: RealField> ReborrowMut<'short> for MatU<'_, T> {
	type Target = MatU<'short, T>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		match self {
			Self::Full(u) => MatU::Full(u.rb_mut()),
			Self::TwoRows(u) => MatU::TwoRows(u.rb_mut()),
			Self::TwoRowsStorage(u) => MatU::TwoRowsStorage(u.rb_mut()),
		}
	}
}
pub(super) fn divide_and_conquer_scratch<T: ComplexField>(
	n: usize,
	qr_fallback_threshold: usize,
	compute_u: bool,
	compute_v: bool,
	par: ParShape,
) -> StackReq {
	let qr_fallback_threshold = Ord::max(qr_fallback_threshold, 4);
	if n < qr_fallback_threshold {
		temp_mat_scratch::<T>(n + 1, n + 1)
	} else {
		let perm = StackReq::new::<usize>(n);
		let jacobi_coeffs = StackReq::new::<JacobiRotation<T>>(n);
		let um = temp_mat_scratch::<T>(n + 1, n + 1);
		let vm = temp_mat_scratch::<T>(n, if compute_v { n } else { 0 });
		let combined_u =
			temp_mat_scratch::<T>(if compute_u { n + 1 } else { 2 }, n + 1);
		let combined_v = vm;
		let compute_m = StackReq::all_of(&[
			perm,
			temp_mat_scratch::<T>(n, 1).array(6),
			perm.array(2),
		]);
		let merge = StackReq::all_of(&[
			perm,
			jacobi_coeffs,
			perm,
			StackReq::any_of(&[
				perm.array(3),
				StackReq::all_of(&[
					um,
					vm,
					StackReq::any_of(&[
						compute_m,
						StackReq::all_of(&[combined_u, combined_v]),
					]),
				]),
			]),
		]);
		let k = n / 2;
		let rem = n - k - 1;
		let child = divide_and_conquer_scratch::<T>(
			Ord::max(k, rem),
			qr_fallback_threshold,
			compute_u,
			compute_v,
			par,
		);
		// Recursion halves the bytes without aligning the second child.
		let recurse =
			StackReq::new::<u8>(child.unaligned_bytes_required()).array(2);
		StackReq::any_of(&[merge, recurse])
	}
}
/// `sqrt(x^2 + y^2)` scaled by the larger argument: `w * sqrt(1 + (z / w)^2)`
/// with `w = max(|x|, |y|)` and `z = min(|x|, |y|)`.
///
/// Squaring first is not usable for the merge's boundary rotation below.
/// `alpha * lambda` and `beta * phi` are products of entries of an already
/// rescaled subproblem, so a deeply split bidiagonal makes them as small as
/// `1e-41`, where `x.abs2() + y.abs2()` is subnormal or zero even after the
/// `sqrt(max_positive)` rescaling that `hypot` applies.
fn hypot_scaled<T: RealField>(x: &T, y: &T) -> T {
	let (x, y) = (x.abs(), y.abs());
	let (w, z) = if x > y { (x, y) } else { (y, x) };
	if w == zero() || !w.is_finite() {
		w
	} else {
		let ref r = z / &w;
		&w * (one::<T>() + r * r).sqrt()
	}
}
pub(super) fn divide_and_conquer<T: RealField>(
	diag: ColMut<'_, T, usize, ContiguousFwd>,
	subdiag: ColMut<'_, T, usize, ContiguousFwd>,
	u: MatU<'_, T>,
	v: Option<MatMut<'_, T>>,
	mut par: Par,
	stack: &mut MemStack,
	qr_fallback_threshold: usize,
) -> Result<(), SvdError> {
	let qr_fallback_threshold = Ord::max(qr_fallback_threshold, 4);
	let n = diag.nrows();
	let mut diag = diag;
	let mut subdiag = subdiag;
	let mut u = u;
	let mut v = v;
	if n < qr_fallback_threshold {
		let (mut u_alloc, _) = unsafe { temp_mat_uninit(n + 1, n + 1, stack) };
		let mut u_alloc = u_alloc.as_mat_mut();
		u_alloc.fill(zero());
		u_alloc.rb_mut().diagonal_mut().fill(one());
		if let Some(mut v) = v.rb_mut() {
			v.fill(zero());
			v.rb_mut().diagonal_mut().fill(one());
		}
		let mut val = subdiag[n - 1].copy();
		subdiag[n - 1] = zero();
		let j = n;
		let mut i = n;
		while i > 0 {
			i -= 1;
			let rot = JacobiRotation::make_givens(diag[i].copy(), val.copy());
			diag[i] = &rot.c * &diag[i] - &rot.s * &val;
			if i > 0 {
				val = &rot.s * &subdiag[i - 1];
				subdiag[i - 1] = &rot.c * &subdiag[i - 1];
			}
			rot.adjoint().apply_on_the_right_in_place(
				u_alloc.rb_mut().two_cols_mut(i, j),
			);
		}
		qr_algorithm(
			diag.rb_mut(),
			subdiag.rb_mut(),
			Some(u_alloc.rb_mut()),
			v.rb_mut(),
		)?;
		match u.rb_mut() {
			MatU::Full(u) => {
				u.submatrix_mut(0, 0, n + 1, n + 1).copy_from(u_alloc.rb())
			},
			MatU::TwoRows(u) => {
				let (mut top, mut bot) =
					u.subcols_mut(0, n + 1).two_rows_mut(0, 1);
				top.copy_from(u_alloc.rb().row(0));
				bot.copy_from(u_alloc.rb().row(n));
			},
			MatU::TwoRowsStorage(_) => {},
		}
		return Ok(());
	}
	let ref max = diag.norm_max().fmax(subdiag.norm_max());
	if *max == zero() {
		match u {
			MatU::Full(mut u) => {
				u.fill(zero());
				u.diagonal_mut().fill(one());
			},
			MatU::TwoRows(mut u) => {
				u.fill(zero());
				// rows 0 and n of the identity block, as reference `dlasd1` takes for a
				// subproblem's own singular vectors: the parent merge consumes the
				// boundary entry in the bottom row through `lambda` and `phi`.
				u[(0, 0)] = one();
				u[(1, n)] = one();
			},
			MatU::TwoRowsStorage(_) => {},
		}
		if let Some(mut u) = v.rb_mut() {
			u.fill(zero());
			u.diagonal_mut().fill(one());
		}
		return Ok(());
	}
	let ref max_inv = max.recip();
	// See `qr_algorithm`: divide when `max` has no finite normal reciprocal.
	let divide = !(max_inv.is_finite() && *max_inv >= min_positive::<T>());
	for i in 0..n {
		if divide {
			diag[i] = diag[i].copy() / max;
			subdiag[i] = subdiag[i].copy() / max;
		} else {
			diag[i] *= max_inv;
			subdiag[i] *= max_inv;
		}
	}
	let k = n / 2;
	let rem = n - k - 1;
	let (ref alpha, ref beta) = {
		let (d1, d2) = diag.rb_mut().split_at_row_mut(k);
		let (subd1, subd2) = subdiag.rb_mut().split_at_row_mut(k);
		let (alpha, d2) = d2.split_at_row_mut(1);
		let (beta, subd2) = subd2.split_at_row_mut(1);
		let mut d1 = d1;
		let mut d2 = d2;
		let mut subd1 = subd1;
		let mut subd2 = subd2;
		let alpha = alpha[0].copy();
		let beta = beta[0].copy();
		let (mut u1, mut u2) = match u.rb_mut() {
			MatU::Full(u) => {
				let (u1, u2) = u.split_at_row_mut(k + 1);
				(
					MatU::Full(u1.submatrix_mut(0, 1, k + 1, k + 1)),
					MatU::Full(u2.submatrix_mut(0, k + 1, rem + 1, rem + 1)),
				)
			},
			MatU::TwoRows(u) | MatU::TwoRowsStorage(u) => {
				let (u1, u2) = u.split_at_col_mut(k + 1);
				(MatU::TwoRows(u1), MatU::TwoRows(u2))
			},
		};
		let (mut v1, mut v2) = match v.rb_mut() {
			Some(v) => {
				let (v1, v2) = v.split_at_row_mut(k);
				let v2 = v2.split_at_row_mut(1).1;
				(Some(v1.subcols_mut(1, k)), Some(v2.subcols_mut(k + 1, rem)))
			},
			None => (None, None),
		};
		let stack_bytes = stack.len_bytes();
		let (mut stack1, stack2) = stack.make_uninit::<u8>(stack_bytes / 2);
		let stack1 = MemStack::new(&mut stack1);
		let mut r1 = Ok(());
		let mut r2 = Ok(());
		// one section for the join of the two halves, each a whole solve of size `n / 2`
		let join_par =
			if crate::utils::cost::halves_pays::<T>(&par, n, crate::utils::cost::rate::bidiag_svd_dc(), 2) {
				par.reborrow()
			} else {
				Par::Seq
			};
		crate::utils::thread::join_raw(
			|par| {
				r1 = divide_and_conquer(
					d1.rb_mut(),
					subd1.rb_mut(),
					u1.rb_mut(),
					v1.rb_mut(),
					par,
					stack1,
					qr_fallback_threshold,
				);
			},
			|par| {
				r2 = divide_and_conquer(
					d2.rb_mut(),
					subd2.rb_mut(),
					u2.rb_mut(),
					v2.rb_mut(),
					par,
					stack2,
					qr_fallback_threshold,
				);
			},
			join_par,
		);
		r1?;
		r2?;
		match u1.rb_mut() {
			MatU::TwoRows(mut u1) => {
				for i in (0..k).rev() {
					with_dim!(ncols, u1.ncols());
					swap_cols_idx(
						u1.rb_mut().as_col_shape_mut(ncols),
						ncols.idx(i),
						ncols.idx(i + 1),
					);
				}
			},
			_ => {},
		}
		(alpha, beta)
	};
	if let Some(mut v) = v.rb_mut() {
		v[(k, 0)] = one();
	}
	for i in (0..k).rev() {
		diag[i + 1] = diag[i].copy();
	}
	let lambda = match u.rb_mut() {
		MatU::Full(u) => u[(k, k + 1)].copy(),
		MatU::TwoRows(u) | MatU::TwoRowsStorage(u) => u[(1, 0)].copy(),
	};
	let phi = match u.rb_mut() {
		MatU::Full(u) => u[(k + 1, n)].copy(),
		MatU::TwoRows(u) | MatU::TwoRowsStorage(u) => u[(0, n)].copy(),
	};
	let ref al = alpha * lambda;
	let ref bp = beta * phi;
	let ref squares = al.abs2() + bp.abs2();
	// The boundary rotation has to stay orthogonal: it is applied to the first
	// and last column of the merged left factor. Squaring `al` and `bp` loses
	// that once the merge sits deep enough in the recursion for them to fall
	// below `sqrt(min_positive)`, because then `squares` is subnormal, `r0` is
	// accurate to a few bits only and `c0^2 + s0^2` drifts from one. Inside the
	// normal range the squares are exact to rounding, so the historical
	// arithmetic stays bit for bit.
	let scaled = !(squares.is_finite() && *squares >= min_positive::<T>());
	let ref r0 = if scaled {
		hypot_scaled(al, bp)
	} else {
		squares.sqrt()
	};
	let (ref c0, ref s0) = if *r0 == zero() {
		(one(), zero())
	} else if scaled {
		// A subnormal `r0` cannot produce an orthogonal pair from `al / r0`;
		// normalizing by a ratio keeps `c0^2 + s0^2` one to rounding.
		let JacobiRotation { c, s } =
			JacobiRotation::make_givens(al.copy(), bp.copy());
		(c, -s)
	} else {
		(al / r0, bp / r0)
	};
	let mut col0 = subdiag;
	diag[0] = r0.copy();
	col0[0] = r0.copy();
	match u.rb_mut() {
		MatU::Full(u) => {
			let (u1, u2) = u.split_at_row_mut(k + 1);
			let (mut u0_top, u1) = u1.split_at_col_mut(1);
			let (u1, mut un_top) = u1.split_at_col_mut(n - 1);
			let (mut u0_bot, u2) = u2.split_at_col_mut(1);
			let (u2, mut un_bot) = u2.split_at_col_mut(n - 1);
			for j in 1..k + 1 {
				col0[j] = alpha * &u1[(k, j - 1)];
			}
			for j in k + 1..n {
				col0[j] = beta * &u2[(0, j - 1)];
			}
			z!(
				u0_top.rb_mut().col_mut(0),
				un_top.rb_mut().col_mut(0),
				u1.col_mut(k)
			)
			.for_each(|uz!(x0, xn, a)| {
				*x0 = c0 * &*a;
				*xn = -s0 * &*a;
				if cfg!(debug_assertions) {
					*a = zero();
				}
			});
			z!(u0_bot.rb_mut().col_mut(0), un_bot.rb_mut().col_mut(0))
				.for_each(|uz!(x0, xn)| {
					*x0 = s0 * &*xn;
					*xn = c0 * &*xn;
				});
		},
		MatU::TwoRows(mut u) => {
			for j in 1..k + 1 {
				col0[j] = alpha * &u[(1, j)];
				u[(1, j)] = zero();
			}
			for j in k + 1..n {
				col0[j] = beta * &u[(0, j)];
				u[(0, j)] = zero();
			}
			let ref q10 = u[(0, 0)].copy();
			let ref q21 = u[(1, n)].copy();
			u[(0, 0)] = c0 * q10;
			u[(0, n)] = -s0 * q10;
			u[(1, 0)] = s0 * q21;
			u[(1, n)] = c0 * q21;
		},
		MatU::TwoRowsStorage(u) => {
			for j in 1..k + 1 {
				col0[j] = alpha * &u[(1, j)];
			}
			for j in k + 1..n {
				col0[j] = beta * &u[(0, j)];
			}
		},
	}
	let (mut perm, stack) = stack.collect(0..n);
	let perm = &mut *perm;
	let (mut jacobi_coeff, stack) =
		stack.make_with(n, |_| JacobiRotation::<T> {
			c: zero(),
			s: zero(),
		});
	let jacobi_coeff = &mut *jacobi_coeff;
	let (mut jacobi_idx, stack) = stack.collect(0..n);
	let jacobi_idx = &mut *jacobi_idx;
	let (jacobi_0i, jacobi_ij) = {
		let (mut transpositions, stack) = stack.collect(0..n);
		deflate(
			diag.rb_mut(),
			col0.rb_mut(),
			jacobi_coeff,
			jacobi_idx,
			&mut transpositions,
			perm,
			k,
			stack,
		)
	};
	let allocate_vm = v.is_some() as usize;
	let (mut um, stack) = temp_mat_zeroed(n + 1, n + 1, stack);
	let (mut vm, stack) = temp_mat_zeroed(n, allocate_vm * n, stack);
	let mut um = um.as_mat_mut().try_as_col_major_mut().unwrap();
	let mut vm = if v.is_some() {
		Some(
			vm.as_mat_mut()
				.subcols_mut(0, n)
				.try_as_col_major_mut()
				.unwrap(),
		)
	} else {
		None
	};
	compute_svd_of_m(
		Some(um.rb_mut()),
		vm.rb_mut(),
		diag.rb_mut(),
		col0.rb(),
		perm,
		stack,
	);
	for i in 0..n {
		col0[i] = zero();
	}
	for (rot, &i) in core::iter::zip(
		&jacobi_coeff[jacobi_0i..][..jacobi_ij],
		&jacobi_idx[jacobi_0i..][..jacobi_ij],
	)
	.rev()
	{
		let (i, j) = (i - 1, i);
		let actual_i = perm[i];
		let actual_j = perm[j];
		{
			let (i, j) = um
				.rb_mut()
				.subcols_mut(0, n)
				.two_rows_mut(actual_i, actual_j);
			rot.apply_on_the_left_in_place((j, i));
		}
		if let Some(mut vm) = vm.rb_mut() {
			let (i, j) = vm.rb_mut().two_rows_mut(actual_i, actual_j);
			rot.apply_on_the_left_in_place((j, i));
		}
	}
	for (rot, &i) in
		core::iter::zip(&jacobi_coeff[..jacobi_0i], &jacobi_idx[..jacobi_0i])
			.rev()
	{
		let (i, j) = um.rb_mut().subcols_mut(0, n).two_rows_mut(0, i);
		rot.apply_on_the_left_in_place((j, i));
	}
	let _v_is_none = v.is_none();
	let update_v =
		|mut v: Option<MatMut<'_, T>>, mut par: Par, stack: &mut MemStack| {
			let (mut combined_v, _) =
				unsafe { temp_mat_uninit(n, allocate_vm * n, stack) };
			let mut combined_v = if v.is_some() {
				Some(
					combined_v
						.as_mat_mut()
						.subcols_mut(0, n)
						.try_as_col_major_mut()
						.unwrap(),
				)
			} else {
				None
			};
			if let (Some(mut v), Some(v_rhs), Some(mut combined_v)) =
				(v.rb_mut(), vm.rb(), combined_v.rb_mut())
			{
				let mut combined_v = combined_v.rb_mut();
				let (mut combined_v1, combined_v2) =
					combined_v.rb_mut().split_at_row_mut(k);
				let mut combined_v2 = combined_v2.submatrix_mut(1, 0, rem, n);
				let v_lhs = v.rb();
				let v_lhs1 = v_lhs.submatrix(0, 1, k, k);
				let v_lhs2 = v_lhs.submatrix(k + 1, k + 1, rem, rem);
				let (v_rhs1, v_rhs2) = v_rhs.split_at_row(1).1.split_at_row(k);
				// one section; the two products `k x k` and `rem x rem` against `n` columns
				let join_par = if crate::utils::cost::join_pays::<T>(
					&par,
					(k * k + rem * rem).saturating_mul(n),
					crate::utils::cost::rate::gemm_mac(),
					1,
				) {
					par.reborrow()
				} else {
					Par::Seq
				};
				crate::utils::thread::join_raw(
					|par| {
						crate::linalg::matmul::matmul(
							combined_v1.rb_mut(),
							Accum::Replace,
							v_lhs1,
							v_rhs1,
							one(),
							par,
						)
					},
					|par| {
						crate::linalg::matmul::matmul(
							combined_v2.rb_mut(),
							Accum::Replace,
							v_lhs2,
							v_rhs2,
							one(),
							par,
						)
					},
					join_par,
				);
				crate::linalg::matmul::matmul(
					combined_v.rb_mut().submatrix_mut(k, 0, 1, n),
					Accum::Replace,
					v_lhs.submatrix(k, 0, 1, 1),
					v_rhs.submatrix(0, 0, 1, n),
					one(),
					par,
				);
				v.copy_from(combined_v.rb());
			}
		};
	let update_u = |mut u: MatMut<'_, T>, par: Par, stack: &mut MemStack| {
		let n = n;
		let k = k;
		let rem = rem;
		let (mut combined_u, _) =
			unsafe { temp_mat_uninit(n + 1, n + 1, stack) };
		let mut combined_u = combined_u.as_mat_mut();
		let (mut combined_u1, mut combined_u2) =
			combined_u.rb_mut().split_at_row_mut(k + 1);
		let u_lhs = u.rb();
		let u_rhs = um.rb();
		let (u_lhs1, u_lhs2) = (
			u_lhs.submatrix(0, 0, k + 1, k + 1),
			u_lhs.submatrix(k + 1, k + 1, rem + 1, rem + 1),
		);
		let (u_rhs1, u_rhs2) = u_rhs.split_at_row(k + 1);
		// one section; the two products `(k+1) x (k+1)` and `(rem+1) x (rem+1)` against `n + 1`
		// columns
		let join_par = if crate::utils::cost::join_pays::<T>(
			&par,
			((k + 1) * (k + 1) + (rem + 1) * (rem + 1)).saturating_mul(n + 1),
			crate::utils::cost::rate::gemm_mac(),
			1,
		) {
			par
		} else {
			Par::Seq
		};
		crate::utils::thread::join_raw(
			|mut par| {
				crate::linalg::matmul::matmul(
					combined_u1.rb_mut(),
					Accum::Replace,
					u_lhs1,
					u_rhs1,
					one(),
					par.reborrow(),
				);
				crate::linalg::matmul::matmul(
					combined_u1.rb_mut(),
					Accum::Add,
					u_lhs.col(n).subrows(0, k + 1).as_mat(),
					u_rhs2.row(rem).as_mat(),
					one(),
					par,
				);
			},
			|mut par| {
				crate::linalg::matmul::matmul(
					combined_u2.rb_mut(),
					Accum::Replace,
					u_lhs2,
					u_rhs2,
					one(),
					par.reborrow(),
				);
				crate::linalg::matmul::matmul(
					combined_u2.rb_mut(),
					Accum::Add,
					u_lhs.col(0).subrows(k + 1, rem + 1).as_mat(),
					u_rhs1.row(0).as_mat(),
					one(),
					par,
				);
			},
			join_par,
		);
		u.copy_from(combined_u.rb());
	};
	match u.rb_mut() {
		MatU::TwoRowsStorage(_) => {
			update_v(v.rb_mut(), par, stack);
		},
		MatU::TwoRows(mut u) => {
			update_v(v.rb_mut(), par.reborrow(), stack);
			let (mut combined_u, _) =
				unsafe { temp_mat_uninit(2, n + 1, stack) };
			let mut combined_u = combined_u.as_mat_mut();
			crate::linalg::matmul::matmul(
				combined_u.rb_mut(),
				Accum::Replace,
				u.rb(),
				um.rb(),
				one(),
				par,
			);
			u.copy_from(combined_u.rb());
		},
		MatU::Full(u) => match par {
			#[cfg(feature = "rayon")]
			Par::Rayon(_) if !_v_is_none => {
				let req_v = crate::linalg::temp_mat_scratch::<T>(n, n);
				let (mem_v, stack_u) = stack.make_aligned_uninit::<u8>(
					req_v.size_bytes(),
					req_v.align_bytes(),
				);
				let stack_v = MemStack::new(mem_v);
				// one section; the products of both updates
				let join_par = if crate::utils::cost::join_pays::<T>(
					&par,
					(k * k + rem * rem).saturating_mul(n).saturating_add(
						((k + 1) * (k + 1) + (rem + 1) * (rem + 1)).saturating_mul(n + 1),
					),
					crate::utils::cost::rate::gemm_mac(),
					1,
				) {
					par
				} else {
					Par::Seq
				};
				crate::utils::thread::join_raw(
					|par| update_v(v.rb_mut(), par, stack_v),
					|par| update_u(u, par, stack_u),
					join_par,
				);
			},
			_ => {
				update_v(v.rb_mut(), par.reborrow(), stack);
				update_u(u, par, stack);
			},
		},
	}
	for i in 0..n {
		diag[i] *= max;
	}
	Ok(())
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::utils::approx::*;
	use crate::{Col, Mat, MatMut, assert};
	use dyn_stack::MemBuffer;
	use std::mem::MaybeUninit;
	use std::path::{Path, PathBuf};
	use std::vec;
	fn bidiag_to_mat<T: RealField>(
		diag: ColRef<'_, T, usize, ContiguousFwd>,
		subdiag: ColRef<'_, T, usize, ContiguousFwd>,
	) -> Mat<T> {
		let n = diag.nrows();
		let mut m = Mat::zeros(n, n);
		for i in 0..n {
			m[(i, i)] = diag[i].copy();
			if i + 1 < n {
				m[(i + 1, i)] = subdiag[i].copy();
			}
		}
		m
	}
	fn bidiag_to_mat2<T: RealField>(
		diag: ColRef<'_, T, usize, ContiguousFwd>,
		subdiag: ColRef<'_, T, usize, ContiguousFwd>,
	) -> Mat<T> {
		let n = diag.nrows();
		let mut m = Mat::zeros(n + 1, n);
		for i in 0..n {
			m[(i, i)] = diag[i].copy();
			m[(i + 1, i)] = subdiag[i].copy();
		}
		m
	}
	fn parse_bidiag(path: &Path) -> (Col<f64>, Col<f64>) {
		let file = &*std::fs::read_to_string(path).unwrap();
		let mut diag = vec![];
		let mut subdiag = vec![];
		let mut iter = file.lines();
		for line in &mut iter {
			if line.starts_with("diag") {
				continue;
			}
			if line.starts_with("subdiag") {
				break;
			}
			let line = line.trim();
			let line = line.strip_suffix(",").unwrap_or(line);
			if line.is_empty() {
				continue;
			}
			diag.push(line.parse::<f64>().unwrap());
		}
		for line in iter {
			let line = line.trim();
			let line = line.strip_suffix(",").unwrap_or(line);
			if line.is_empty() {
				continue;
			}
			subdiag.push(line.parse::<f64>().unwrap());
		}
		assert!(diag.len() == subdiag.len());
		(
			Col::from_fn(diag.len(), |i| diag[i]),
			Col::from_fn(diag.len(), |i| subdiag[i]),
		)
	}
	#[test]
	fn test_secular_eq_flat_away_from_root() {
		for tiny in [1e-6_f64, 1e-13 / 65.0_f64.sqrt(), 1e-30] {
			for (near_right, last) in
				[(false, false), (false, true), (true, false)]
			{
				let right = 0.0483274904722646;
				let f = |shift: f64, mu: f64| {
					if near_right {
						-1.0 + (tiny / ((right - shift) - mu)).powi(2)
					} else {
						1.0 - (tiny / (shift + mu)).powi(2)
					}
				};
				let (shift, mu) = secular_eq_root_finder(
					&f,
					&|shifts, mus| {
						core::array::from_fn(|i| f(shifts[i], mus[i]))
					},
					0.0,
					right,
					last,
				);
				let distance = if near_right {
					(right - shift) - mu
				} else {
					shift + mu
				};
				assert!((distance / tiny - 1.0).abs() < 1e-14);
			}
		}
	}
	#[test]
	fn test_issue81_qr_near_repeated_pair() {
		fn check<T: RealField>(f: T, g: T, h: T, expected: [f64; 3]) {
			for scale in [1.0, 2.0, 1e-8, 1e8] {
				let ref scale = from_f64::<T>(scale);
				for (want_u, want_v) in
					[(true, true), (true, false), (false, true), (false, false)]
				{
					let mut d = Col::<T>::from_fn(3, |i| {
						(match i {
							0 => one(),
							1 => f.copy(),
							_ => h.copy(),
						}) * scale
					});
					let mut e = Col::<T>::from_fn(3, |i| {
						if i == 1 { &g * scale } else { zero() }
					});
					let original = bidiag_to_mat(
						d.as_ref().try_as_col_major().unwrap(),
						e.as_ref().try_as_col_major().unwrap(),
					);
					let mut u = Mat::<T>::identity(3, 3);
					let mut v = Mat::<T>::identity(3, 3);
					qr_algorithm(
						d.as_mut().try_as_col_major_mut().unwrap(),
						e.as_mut().try_as_col_major_mut().unwrap(),
						if want_u { Some(u.as_mut()) } else { None },
						if want_v { Some(v.as_mut()) } else { None },
					)
					.unwrap();
					let tol = from_f64::<T>(32.0) * eps::<T>();
					for i in 0..3 {
						let expected = from_f64::<T>(expected[i]);
						assert!(
							((&d[i] / scale) - &expected).abs()
								<= &tol * expected
						);
						assert_eq!(e[i], zero::<T>());
					}
					if want_u {
						assert!(
							(u.adjoint() * &u - Mat::<T>::identity(3, 3))
								.norm_max() <= tol
						);
					}
					if want_v {
						assert!(
							(v.adjoint() * &v - Mat::<T>::identity(3, 3))
								.norm_max() <= tol
						);
					}
					if want_u && want_v {
						let back = &u * d.as_diagonal() * v.adjoint();
						assert!(
							(&back - &original).norm_max()
								/ original.norm_max() <= tol
						);
					}
				}
			}
		}
		let f = 0.02874731110238686_f64;
		let h = 0.028747311102386864_f64;
		check(f, 2.0_f64.powi(-57), h, [1.0, h, f]);
		let f = f as f32;
		let h = f32::from_bits(f.to_bits() + 1);
		check(f, 2.0_f32.powi(-28), h, [1.0, h as f64, f as f64]);
		check(
			1.0_f64,
			1.0,
			1.0,
			[1.618033988749895, 1.0, 0.6180339887498948],
		);
		check(
			1.0_f32,
			1.0,
			-1.0,
			[1.618033988749895, 1.0, 0.6180339887498948],
		);
		check(
			1.0_f64,
			-1.0,
			1e-200,
			[
				core::f64::consts::SQRT_2,
				1.0,
				1e-200 / core::f64::consts::SQRT_2,
			],
		);
		check(
			-1.0_f32,
			1.0,
			1e-30,
			[
				core::f64::consts::SQRT_2,
				1.0,
				1e-30 / core::f64::consts::SQRT_2,
			],
		);
		// Multiplying f*h first would lose precision in this subnormal value.
		let mut d = Col::from_fn(3, |i| match i {
			0 => 1.0,
			1 => 1e-15,
			_ => f64::MIN_POSITIVE,
		});
		let mut e = Col::from_fn(3, |i| if i == 1 { 1e-15 } else { 0.0 });
		qr_algorithm(
			d.as_mut().try_as_col_major_mut().unwrap(),
			e.as_mut().try_as_col_major_mut().unwrap(),
			None,
			None,
		)
		.unwrap();
		assert!(
			(d[2] - f64::MIN_POSITIVE / core::f64::consts::SQRT_2).abs()
				<= 64.0 * f64::EPSILON * f64::MIN_POSITIVE
		);
	}
	#[test]
	fn test_qr_algorithm() {
		for file in std::fs::read_dir(
			PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/svd/"),
		)
		.unwrap()
		{
			let (diag, mut subdiag) = parse_bidiag(&file.unwrap().path());
			subdiag[diag.nrows() - 1] = 0.0;
			let n = diag.nrows();
			if n > 512 {
				continue;
			}
			let diag = diag.as_ref().try_as_col_major().unwrap();
			let subdiag = subdiag.as_ref().try_as_col_major().unwrap();
			let mut d = diag.to_owned();
			let mut subd = subdiag.to_owned();
			let mut u = Mat::identity(n, n);
			let mut v = Mat::identity(n, n);
			for i in 0..n {
				u[(i, i)] = 1.0;
				v[(i, i)] = 1.0;
			}
			qr_algorithm(
				d.as_mut().try_as_col_major_mut().unwrap(),
				subd.as_mut().try_as_col_major_mut().unwrap(),
				Some(u.as_mut()),
				Some(v.as_mut()),
			)
			.unwrap();
			for &x in subd.iter() {
				assert!(x == 0.0);
			}
			let mut approx_eq = CwiseMat(ApproxEq::<f64>::eps());
			approx_eq.0.abs_tol *=
				f64::max(diag.norm_max(), subdiag.norm_max())
					* (n as f64).sqrt();
			approx_eq.0.rel_tol *=
				f64::max(diag.norm_max(), subdiag.norm_max())
					* (n as f64).sqrt();
			let reconstructed = &u * d.as_diagonal() * v.adjoint();
			assert!(reconstructed ~ bidiag_to_mat(diag, subdiag));
		}
	}
	#[test]
	fn test_divide_and_conquer_exact_scratch() {
		for n in 1..=70 {
			for threshold in [4, 5, 8] {
				for u_mode in 0..3 {
					for want_v in [false, true] {
						check_divide_and_conquer_scratch::<f32>(
							n,
							threshold,
							u_mode,
							want_v,
							Par::Seq,
						);
						check_divide_and_conquer_scratch::<f64>(
							n,
							threshold,
							u_mode,
							want_v,
							Par::Seq,
						);
					}
				}
			}
		}
	}
	#[test]
	#[cfg(feature = "rayon")]
	fn test_divide_and_conquer_exact_scratch_rayon() {
		let pool = rayon::ThreadPoolBuilder::new()
			.num_threads(2)
			.build()
			.unwrap();
		pool.install(|| {
			for n in [4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65] {
				for threshold in [4, 5, 8] {
					check_divide_and_conquer_scratch::<f32>(
						n,
						threshold,
						0,
						true,
						Par::rayon(2),
					);
					check_divide_and_conquer_scratch::<f64>(
						n,
						threshold,
						0,
						true,
						Par::rayon(2),
					);
				}
			}
		});
	}
	fn check_divide_and_conquer_scratch<T: RealField>(
		n: usize,
		threshold: usize,
		u_mode: usize,
		want_v: bool,
		mut par: Par,
	) {
		for rank_one in [false, true] {
			for unaligned in [false, true] {
				let mut d = Col::<T>::from_fn(n, |i| {
					if rank_one {
						if i == 0 { one() } else { zero() }
					} else {
						from_f64(2.0 + (i % 11) as f64 / 11.0)
					}
				});
				let mut e = Col::<T>::from_fn(n, |i| {
					if rank_one {
						zero()
					} else {
						from_f64(0.25 + (i % 7) as f64 / 16.0)
					}
				});
				let norm = (d.squared_norm_l2() + e.squared_norm_l2()).sqrt();
				let u_rows = if u_mode == 0 { n + 1 } else { 2 };
				let mut u = Mat::<T>::zeros(u_rows, n + 1);
				let mut v = Mat::<T>::zeros(n, if want_v { n } else { 0 });
				let req = divide_and_conquer_scratch::<T>(
					n,
					threshold,
					u_mode == 0,
					want_v,
					par.shape(),
				);
				let mut mem = MemBuffer::new(StackReq::new_aligned::<u8>(
					req.unaligned_bytes_required() + 1,
					req.align_bytes(),
				));
				let (offset, len) = if unaligned {
					(1, req.unaligned_bytes_required())
				} else {
					(0, req.size_bytes())
				};
				assert_eq!(mem[offset..].as_ptr() as usize % 2, offset);
				let stack = MemStack::new(&mut mem[offset..][..len]);
				assert_eq!(stack.len_bytes(), len);
				let u_arg = match u_mode {
					0 => MatU::Full(u.as_mut()),
					1 => MatU::TwoRows(u.as_mut()),
					_ => MatU::TwoRowsStorage(u.as_mut()),
				};
				divide_and_conquer(
					d.as_mut().try_as_col_major_mut().unwrap(),
					e.as_mut().try_as_col_major_mut().unwrap(),
					u_arg,
					if want_v { Some(v.as_mut()) } else { None },
					par.reborrow(),
					stack,
					threshold,
				)
				.unwrap();
				assert!(
					d.is_all_finite() && u.is_all_finite() && v.is_all_finite()
				);
				let tol = from_f64::<T>(128.0 * n as f64) * eps::<T>();
				assert!((d.norm_l2() - &norm).abs() <= &tol * norm);
				if rank_one {
					let expected = Col::<T>::from_fn(n, |i| {
						if i == 0 { one() } else { zero() }
					});
					assert!((&d - &expected).norm_max() <= tol);
				}
				if u_mode != 2 {
					assert!(
						(&u * u.adjoint() - Mat::<T>::identity(u_rows, u_rows))
							.norm_max() <= tol
					);
				}
				if want_v {
					assert!(
						(v.adjoint() * &v - Mat::<T>::identity(n, n))
							.norm_max() <= tol
					);
				}
			}
		}
	}
	#[test]
	fn test_divide_and_conquer() {
		for file in std::fs::read_dir(
			PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/svd/"),
		)
		.unwrap()
		{
			let (diag, subdiag) = parse_bidiag(&file.unwrap().path());
			let n = diag.nrows();
			if n > 1024 {
				continue;
			}
			let diag = diag.as_ref().try_as_col_major().unwrap();
			let subdiag = subdiag.as_ref().try_as_col_major().unwrap();
			let mut d = diag.to_owned();
			let mut subd = subdiag.to_owned();
			let mut u = Mat::zeros(n + 1, n + 1);
			let mut v = Mat::zeros(n, n);
			for i in 0..n {
				u[(i, i)] = 1.0;
				v[(i, i)] = 1.0;
			}
			u[(n, n)] = 1.0;
			divide_and_conquer(
				d.as_mut().try_as_col_major_mut().unwrap(),
				subd.as_mut().try_as_col_major_mut().unwrap(),
				MatU::Full(u.as_mut()),
				Some(v.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					divide_and_conquer_scratch::<f64>(
						n,
						4,
						true,
						true,
						ParShape::Seq,
					),
				)),
				4,
			)
			.unwrap();
			for x in subd.iter_mut() {
				*x = 0.0;
			}
			let mut d2 = Mat::zeros(n + 1, n);
			for i in 0..n {
				d2[(i, i)] = d[i];
			}
			let mut approx_eq = CwiseMat(ApproxEq::<f64>::eps());
			approx_eq.0.abs_tol *=
				f64::max(diag.norm_max(), subdiag.norm_max())
					* (n as f64).sqrt()
					* 10.0;
			approx_eq.0.rel_tol *=
				f64::max(diag.norm_max(), subdiag.norm_max())
					* (n as f64).sqrt()
					* 10.0;
			let reconstructed = &u * &d2 * v.adjoint();
			assert!(reconstructed ~ bidiag_to_mat2(diag, subdiag));
		}
	}
	#[test]
	#[ignore]
	fn test_josef() {
		for file in std::fs::read_dir(
			PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/svd/"),
		)
		.unwrap()
		{
			let (diag, subdiag) = parse_bidiag(&file.unwrap().path());
			let diag = z!(&diag).map(|uz!(x)| *x as f32);
			let subdiag = z!(&subdiag).map(|uz!(x)| *x as f32);
			let n = diag.nrows();
			if n <= 1024 {
				continue;
			}
			let diag = diag.as_ref().try_as_col_major().unwrap();
			let subdiag = subdiag.as_ref().try_as_col_major().unwrap();
			let mut d = diag.to_owned();
			let mut subd = subdiag.to_owned();
			let mut u = Mat::zeros(n + 1, n + 1);
			let mut v = Mat::zeros(n, n);
			for i in 0..n {
				u[(i, i)] = 1.0;
				v[(i, i)] = 1.0;
			}
			u[(n, n)] = 1.0;
			divide_and_conquer(
				d.as_mut().try_as_col_major_mut().unwrap(),
				subd.as_mut().try_as_col_major_mut().unwrap(),
				MatU::Full(u.as_mut()),
				Some(v.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					divide_and_conquer_scratch::<f32>(
						n,
						4,
						true,
						true,
						ParShape::Seq,
					),
				)),
				4,
			)
			.unwrap();
			for x in subd.iter_mut() {
				*x = 0.0;
			}
			let mut d2 = Mat::zeros(n + 1, n);
			for i in 0..n {
				d2[(i, i)] = d[i];
			}
			let mut approx_eq = CwiseMat(ApproxEq::<f32>::eps());
			approx_eq.0.abs_tol *=
				f32::max(diag.norm_max(), subdiag.norm_max())
					* (n as f32).sqrt()
					* 10.0;
			approx_eq.0.rel_tol *=
				f32::max(diag.norm_max(), subdiag.norm_max())
					* (n as f32).sqrt()
					* 10.0;
			let reconstructed = &u * &d2 * v.adjoint();
			assert!(reconstructed ~ bidiag_to_mat2(diag, subdiag));
		}
	}
	#[test]
	fn test_deflation43() {
		let approx_eq = CwiseMat(ApproxEq::<f32>::eps());
		let n = 7;
		let k = 3;
		let diag_orig = &*vec![1.0, 5.0, 3.0, 1e-7, 4.0, 2.0, 2e-7_f32];
		let col_orig = &*vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0_f32];
		let n_jacobi = 2;
		let jacobi_coeffs =
			&mut *vec![JacobiRotation { c: 0.0, s: 0.0 }; n_jacobi];
		let jacobi_indices = &mut *vec![0; n_jacobi];
		let perm = &mut *vec![0; n];
		let diag = &mut *diag_orig.to_vec();
		let col = &mut *col_orig.to_vec();
		let mut diag = MatMut::from_column_major_slice_mut(diag, n, 1)
			.col_mut(0)
			.try_as_col_major_mut()
			.unwrap();
		let mut col = MatMut::from_column_major_slice_mut(col, n, 1)
			.col_mut(0)
			.try_as_col_major_mut()
			.unwrap();
		let (jacobi_0i, jacobi_ij) = deflate(
			diag.rb_mut(),
			col.rb_mut(),
			jacobi_coeffs,
			jacobi_indices,
			&mut *vec![0; n],
			perm,
			k,
			MemStack::new_any(&mut *vec![MaybeUninit::new(0usize); 2 * n]),
		);
		assert!(all(jacobi_0i == 2, jacobi_ij == 0));
		let perm_inv = &mut *vec![0; n];
		for (i, &p) in perm.iter().enumerate() {
			perm_inv[p] = i;
		}
		let P = crate::perm::PermRef::new_checked(perm, perm_inv, n);
		let mut M_orig = Mat::zeros(n, n);
		for i in 0..n {
			M_orig[(i, i)] = diag_orig[i];
			M_orig[(i, 0)] = col_orig[i];
		}
		let mut M = Mat::zeros(n, n);
		for i in 0..n {
			M[(i, i)] = diag[i];
			M[(i, 0)] = col[i];
		}
		M = P.inverse() * M * P;
		for (&rot, &i) in
			core::iter::zip(&*jacobi_coeffs, &*jacobi_indices).rev()
		{
			let (x, y) = M.two_rows_mut(0, i);
			rot.apply_on_the_left_in_place((y, x));
		}
		assert!(M ~ M_orig);
	}
	#[test]
	fn test_deflation44() {
		let approx_eq = CwiseMat(ApproxEq::<f32>::eps());
		let n = 7;
		let k = 3;
		let diag_orig = &*vec![1.0, 5.0, 3.0, 1.0, 4.0, 2.0, 1.0];
		let col_orig = &*vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0_f32];
		let n_jacobi = 1;
		let jacobi_coeffs =
			&mut *vec![JacobiRotation { c: 0.0, s: 0.0 }; n_jacobi];
		let jacobi_indices = &mut *vec![0; n_jacobi];
		let perm = &mut *vec![0; n];
		let diag = &mut *diag_orig.to_vec();
		let col = &mut *col_orig.to_vec();
		let mut diag = MatMut::from_column_major_slice_mut(diag, n, 1)
			.col_mut(0)
			.try_as_col_major_mut()
			.unwrap();
		let mut col = MatMut::from_column_major_slice_mut(col, n, 1)
			.col_mut(0)
			.try_as_col_major_mut()
			.unwrap();
		let (jacobi_0i, jacobi_ij) = deflate(
			diag.rb_mut(),
			col.rb_mut(),
			jacobi_coeffs,
			jacobi_indices,
			&mut *vec![0; n],
			perm,
			k,
			MemStack::new_any(&mut *vec![MaybeUninit::new(0usize); 2 * n]),
		);
		assert!(all(jacobi_0i == 0, jacobi_ij == 1));
		let perm_inv = &mut *vec![0; n];
		for (i, &p) in perm.iter().enumerate() {
			perm_inv[p] = i;
		}
		let P = crate::perm::PermRef::new_checked(perm, perm_inv, n);
		let mut M_orig = Mat::zeros(n, n);
		for i in 0..n {
			M_orig[(i, i)] = diag_orig[i];
			M_orig[(i, 0)] = col_orig[i];
		}
		let mut M = Mat::zeros(n, n);
		for i in 0..n {
			M[(i, i)] = diag[i];
			M[(i, 0)] = col[i];
		}
		M = P.inverse() * M * P;
		for (&rot, &i) in
			core::iter::zip(&*jacobi_coeffs, &*jacobi_indices).rev()
		{
			let (i, j) = (i - 1, i);
			let (pi, pj) = (perm[i], perm[j]);
			let (x, y) = M.two_rows_mut(pi, pj);
			rot.apply_on_the_left_in_place((y, x));
			let (x, y) = M.two_cols_mut(pi, pj);
			rot.transpose().apply_on_the_right_in_place((y, x));
		}
		assert!(M ~ M_orig);
	}
	#[test]
	fn test_both_deflation() {
		let approx_eq = CwiseMat(ApproxEq::<f32>::eps());
		let n = 7;
		let k = 3;
		let diag_orig = &*vec![1.0, 5.0, 3.0, 2.0, 4.0, 2.0, 0.0];
		let col_orig = &*vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0_f32];
		let n_jacobi = 2;
		let jacobi_coeffs =
			&mut *vec![JacobiRotation { c: 0.0, s: 0.0 }; n_jacobi];
		let jacobi_indices = &mut *vec![0; n_jacobi];
		let perm = &mut *vec![0; n];
		let diag = &mut *diag_orig.to_vec();
		let col = &mut *col_orig.to_vec();
		let mut diag = MatMut::from_column_major_slice_mut(diag, n, 1)
			.col_mut(0)
			.try_as_col_major_mut()
			.unwrap();
		let mut col = MatMut::from_column_major_slice_mut(col, n, 1)
			.col_mut(0)
			.try_as_col_major_mut()
			.unwrap();
		let (jacobi_0i, jacobi_ij) = deflate(
			diag.rb_mut(),
			col.rb_mut(),
			jacobi_coeffs,
			jacobi_indices,
			&mut *vec![0; n],
			perm,
			k,
			MemStack::new_any(&mut *vec![MaybeUninit::new(0usize); 2 * n]),
		);
		assert!(all(jacobi_0i == 1, jacobi_ij == 1));
		let perm_inv = &mut *vec![0; n];
		for (i, &p) in perm.iter().enumerate() {
			perm_inv[p] = i;
		}
		let P = crate::perm::PermRef::new_checked(perm, perm_inv, n);
		let mut M_orig = Mat::zeros(n, n);
		for i in 0..n {
			M_orig[(i, i)] = diag_orig[i];
			M_orig[(i, 0)] = col_orig[i];
		}
		let mut M = Mat::zeros(n, n);
		for i in 0..n {
			M[(i, i)] = diag[i];
			M[(i, 0)] = col[i];
		}
		M = P.inverse() * M * P;
		for (&rot, &i) in
			core::iter::zip(&jacobi_coeffs[1..], &jacobi_indices[1..]).rev()
		{
			let (i, j) = (i - 1, i);
			let (pi, pj) = (perm[i], perm[j]);
			let (x, y) = M.two_rows_mut(pi, pj);
			rot.apply_on_the_left_in_place((y, x));
			let (x, y) = M.two_cols_mut(pi, pj);
			rot.transpose().apply_on_the_right_in_place((y, x));
		}
		for (&rot, &i) in
			core::iter::zip(&jacobi_coeffs[..1], &jacobi_indices[..1]).rev()
		{
			let (x, y) = M.two_rows_mut(0, i);
			rot.apply_on_the_left_in_place((y, x));
		}
		assert!(M ~ M_orig);
	}
	/// Pre-scaling by `1 / max` in the QR and divide-and-conquer solvers: for
	/// an exact power of two `s` outside the normal range of `1 / max`, the
	/// singular values of `s B` must be `s` times those of `B`.
	fn prescale_is_scale_exact<T: RealField>() {
		let base_d = [1.8, -0.7, 0.4, 1.1, -0.5, 0.9];
		let base_e = [0.3, -1.2, 0.6, 0.2, -0.8];
		for n in [3usize, 6] {
			let run = |s: &T| {
				let mut d =
					Col::<T>::from_fn(n, |i| from_f64::<T>(base_d[i]) * s);
				let mut e = Col::<T>::from_fn(n, |i| {
					if i + 1 < n {
						from_f64::<T>(base_e[i]) * s
					} else {
						zero()
					}
				});
				let mut u = Mat::<T>::identity(n + 1, n + 1);
				let mut v = Mat::<T>::identity(n, n);
				if n < 4 {
					qr_algorithm(
						d.as_mut().try_as_col_major_mut().unwrap(),
						e.as_mut().try_as_col_major_mut().unwrap(),
						Some(u.as_mut().submatrix_mut(0, 0, n, n)),
						Some(v.as_mut()),
					)
					.unwrap();
				} else {
					divide_and_conquer(
						d.as_mut().try_as_col_major_mut().unwrap(),
						e.as_mut().try_as_col_major_mut().unwrap(),
						MatU::Full(u.as_mut()),
						Some(v.as_mut()),
						Par::Seq,
						MemStack::new(&mut MemBuffer::new(
							divide_and_conquer_scratch::<T>(
								n,
								4,
								true,
								true,
								ParShape::Seq,
							),
						)),
						4,
					)
					.unwrap();
				}
				assert!(u.is_all_finite() && v.is_all_finite());
				d
			};
			let reference = run(&one());
			let mut max = zero::<T>();
			for i in 0..n {
				max = max.fmax(reference[i].abs());
			}
			let sml = min_positive::<T>();
			for s in [sml.recip(), &sml * from_f64::<T>(0.0625)] {
				let d = run(&s);
				for i in 0..n {
					assert!(
						(&d[i] / &s - &reference[i]).abs()
							<= from_f64::<T>(1e3) * eps::<T>() * &max
					);
				}
			}
		}
	}
	/// A nonzero value below `min_positive` is a real singular value, not a
	/// deflated zero. The merge collects the exactly-zero entries into a group
	/// that the final reversal leaves at the end; that group's own order is by
	/// index, so a nonzero subnormal value placed in it comes out unordered.
	/// Group by an exact-zero test instead and the spectrum stays sorted
	/// (#82).
	fn subnormal_diag_at_split_is_ordered<T: RealField>(threshold: usize) {
		let n = 128;
		let tiny = min_positive::<T>().mul_pow2(from_f64::<T>(0.5));
		assert!(tiny > zero::<T>() && tiny < min_positive::<T>());
		for pos in [n / 2 + 1, n / 2 + 36, n - 1] {
			let mut d = Col::<T>::zeros(n);
			let mut e = Col::<T>::zeros(n);
			d[0] = one();
			d[1] = from_f64::<T>(0.5);
			d[pos] = tiny.copy();
			let d0 = Col::<T>::from_fn(n, |i| d[i].copy());
			let e0 = Col::<T>::from_fn(n, |i| e[i].copy());
			let (d0, e0) = (
				d0.as_ref().try_as_col_major().unwrap(),
				e0.as_ref().try_as_col_major().unwrap(),
			);
			let mut u = Mat::<T>::zeros(n + 1, n + 1);
			let mut v = Mat::<T>::zeros(n, n);
			divide_and_conquer(
				d.as_mut().try_as_col_major_mut().unwrap(),
				e.as_mut().try_as_col_major_mut().unwrap(),
				MatU::Full(u.as_mut()),
				Some(v.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					divide_and_conquer_scratch::<T>(
						n,
						threshold,
						true,
						true,
						ParShape::Seq,
					),
				)),
				threshold,
			)
			.unwrap();
			for i in 0..n {
				assert!(d[i].is_finite() && d[i] >= zero());
				if i + 1 < n {
					assert!(
						d[i] >= d[i + 1],
						"pos={pos} threshold={threshold}: d[{i}] = {:?} < \
						 d[{}] = {:?}",
						d[i],
						i + 1,
						d[i + 1],
					);
				}
			}
			assert!((&d[0] - one::<T>()).abs() <= eps::<T>());
			assert!((&d[1] - from_f64::<T>(0.5)).abs() <= eps::<T>());
			// singular values below the merge's deflation floor
			// `8 * eps * ||B||` (the subnormal one and the zeros) are only
			// determined to that normwise bound, which the whole spectrum
			// `[1, 1/2, tiny, 0, ...]` must meet
			for i in 0..n {
				let want = match i {
					0 => one::<T>(),
					1 => from_f64::<T>(0.5),
					2 => tiny.copy(),
					_ => zero::<T>(),
				};
				assert!(
					(&d[i] - want).abs() <= from_f64::<T>(8.0) * eps::<T>(),
					"pos={pos} threshold={threshold}: d[{i}] = {:?}",
					d[i],
				);
			}
			let mut sigma = Mat::<T>::zeros(n + 1, n);
			for i in 0..n {
				sigma[(i, i)] = d[i].copy();
			}
			assert!(u.is_all_finite() && v.is_all_finite());
			let rec = &u * &sigma * v.transpose();
			let residual = (&rec - bidiag_to_mat2(d0, e0)).norm_max();
			// the merge's deflated-value perturbation is `8 * eps`, which is
			// the accuracy the reconstruction carries
			assert!(residual <= from_f64::<T>(100.0) * eps::<T>());
		}
	}
	#[test]
	fn subnormal_diag_at_split() {
		for threshold in [128, 4] {
			subnormal_diag_at_split_is_ordered::<f64>(threshold);
			subnormal_diag_at_split_is_ordered::<f32>(threshold);
		}
	}
	#[test]
	fn prescale_outside_normal_range() {
		prescale_is_scale_exact::<f64>();
		prescale_is_scale_exact::<f32>();
	}
}
