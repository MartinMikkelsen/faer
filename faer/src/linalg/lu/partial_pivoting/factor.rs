use crate::internal_prelude::*;
use crate::perm::swap_rows_idx;
use crate::{assert, debug_assert};
#[inline]
pub(crate) fn swap_elems<T: ComplexField>(
	col: ColMut<'_, T>,
	i: usize,
	j: usize,
) {
	debug_assert!(all(i < col.nrows(), j < col.nrows()));
	let rs = col.row_stride();
	let col = col.as_ptr_mut();
	unsafe {
		let a = col.offset(i as isize * rs);
		let b = col.offset(j as isize * rs);
		core::ptr::swap(a, b);
	}
}
fn lu_in_place_unblocked<I: Index, T: ComplexField>(
	matrix: MatMut<'_, T>,
	start: usize,
	end: usize,
	trans: &mut [I],
) -> usize {
	let mut matrix = matrix;
	let m = matrix.nrows();
	if start == end {
		return 0;
	}
	let mut n_trans = 0;
	for j in start..end {
		let col = j;
		let row = j - start;
		let t = &mut trans[row];
		let mut imax = row;
		let mut max = zero();
		for i in imax..m {
			let abs = matrix[(i, col)].abs1();
			if abs > max {
				max = abs;
				imax = i;
			}
		}
		*t = I::truncate(imax - row);
		if imax != row {
			swap_rows_idx(matrix.rb_mut(), row, imax);
			n_trans += 1;
		}
		let mut matrix = matrix.rb_mut().get_mut(.., start..end);
		// An exactly zero pivot column is a singular factor, not a division: OpenBLAS's
		// `getf2_k` leaves the (all zero) column alone instead of scaling it by infinity.
		let pivot = matrix[(row, row)].copy();
		if pivot != zero() {
			if pivot.abs() >= min_positive() {
				let ref inv = pivot.recip();
				for i in row + 1..m {
					matrix[(i, row)] *= inv;
				}
			} else {
				// the reciprocal of a subnormal pivot overflows, and the complex quotient
				// forms its squared denominator, so rescale both operands instead: a power
				// of two keeps the scaling exact and the squares in range (`dgetf2`
				// divides elementwise below `SFMIN`)
				let ref shift = max_positive::<T::Real>();
				let ref pivot = pivot.mul_real(shift);
				for i in row + 1..m {
					matrix[(i, row)] =
						&matrix[(i, row)].mul_real(shift) / pivot;
				}
			}
		}
		let (_, A01, A10, A11) = matrix.rb_mut().split_at_mut(row + 1, row + 1);
		let A01 = A01.row(row);
		let A10 = A10.col(row);
		linalg::matmul::matmul(
			A11,
			Accum::Add,
			A10.as_mat(),
			A01.as_mat(),
			-one::<T>(),
			Par::Seq,
		);
	}
	n_trans
}
pub(crate) fn lu_in_place_recursion<I: Index, T: ComplexField>(
	A: MatMut<'_, T>,
	start: usize,
	end: usize,
	trans: &mut [I],
	par: Par,
	params: Spec<PartialPivLuParams, T>,
) -> usize {
	let params = params.config;
	let mut A = A;
	let m = A.nrows();
	let ncols = A.ncols();
	let n = end - start;
	if n <= params.recursion_threshold {
		return lu_in_place_unblocked(A, start, end, trans);
	}
	let half = n / 2;
	let pow = Ord::min(16, half.next_power_of_two());
	let block_size = half.next_multiple_of(pow);
	let mut n_trans = 0;
	assert!(n <= m);
	n_trans += lu_in_place_recursion(
		A.rb_mut().get_mut(.., start..end),
		0,
		block_size,
		&mut trans[..block_size],
		par,
		params.into(),
	);
	{
		let mut A = A.rb_mut().get_mut(.., start..end);
		let (A00, mut A01, A10, mut A11) =
			A.rb_mut().split_at_mut(block_size, block_size);
		let A00 = A00.rb();
		let A10 = A10.rb();
		// The panel solve and the trailing product below are this level's whole
		// parallel work, and the block size halves every level of the recursion, so
		// past a few levels they are a fork and a join around a handful of thousands
		// of flops. Below the work floor the swap stage already uses, take
		// `Par::Seq`; the recursion keeps the caller's `par`, so every level decides
		// for its own trailing update. The crossover comes from a sweep over 256x256,
		// 512x512 and 1024x1024 threaded `f64`, not from one shape, so it is a
		// statement about *flops*: `par_threshold` counts `f64` elements, and a
		// complex element carries four times the arithmetic. Comparing raw areas
		// made the floor four times too high for `c64`, which left its `128x128`
		// trailing update sequential at the size where the `f64` one is already
		// threaded. The comparison saturates, so `usize::MAX` still disables it.
		let work = A11
			.nrows()
			.saturating_mul(A11.ncols())
			.saturating_mul(if T::IS_REAL { 2 } else { 8 });
		let par_here = if work > params.par_threshold.saturating_mul(2) {
			par
		} else {
			Par::Seq
		};
		{
			linalg::triangular_solve::solve_unit_lower_triangular_in_place(
				A00.rb(),
				A01.rb_mut(),
				par_here,
			);
		}
		linalg::matmul::matmul(
			A11.rb_mut(),
			Accum::Add,
			A10.rb(),
			A01.rb(),
			-one::<T>(),
			par_here,
		);
		n_trans += lu_in_place_recursion(
			A.rb_mut().get_mut(block_size..m, ..),
			block_size,
			n,
			&mut trans[block_size..n],
			par,
			params.into(),
		);
	}
	let swap = |mat: MatMut<'_, T>| {
		let mut mat = mat;
		for j in 0..mat.ncols() {
			let mut col = mat.rb_mut().col_mut(j);
			if col.row_stride() == 1 {
				for (j, &t) in trans[..n].iter().enumerate() {
					swap_elems(col.rb_mut(), j, t.zx() + j);
				}
			} else {
				for (j, &t) in trans[..n].iter().enumerate() {
					swap_elems(col.rb_mut(), j, t.zx() + j);
				}
			}
		}
	};
	let (A_left, A_right) = A.rb_mut().split_at_col_mut(start);
	let A_right = A_right.get_mut(.., end - start..ncols - start);
	let par = if m * (ncols - n) > params.par_threshold {
		par
	} else {
		Par::Seq
	};
	match par {
		Par::Seq => {
			swap(A_left);
			swap(A_right);
		},
		#[cfg(feature = "rayon")]
		_ => {
		let nthreads = core::num::NonZeroUsize::new(par.degree()).unwrap();
			let nthreads = nthreads.get();
			let len = (A_left.ncols() + A_right.ncols()) as f64;
			let left_threads = Ord::min(
				(nthreads as f64 * (A_left.ncols() as f64 / len)) as usize,
				nthreads,
			);
			let right_threads = nthreads - left_threads;
			crate::utils::thread::join_raw(
				|_| {
					if A_left.ncols() > 0 {
						crate::utils::thread::for_each(par, 
							left_threads,
							A_left.par_col_partition_mut(left_threads),
							|A| swap(A),
						)
					}
				},
				|_| {
					if A_right.ncols() > 0 {
						crate::utils::thread::for_each(par, 
							right_threads,
							A_right.par_col_partition_mut(right_threads),
							|A| swap(A),
						)
					}
				},
				par,
			);
		},
	}
	n_trans
}
/// $LU$ factorization tuning parameters
#[derive(Copy, Clone, Debug)]
pub struct PartialPivLuParams {
	/// threshold at which the implementation should stop recursing
	pub recursion_threshold: usize,
	/// blocking variant step size
	pub block_size: usize,
	/// threshold at which size parallelism should be disabled, counted in `f64`
	/// elements of the submatrix that would be threaded: a complex element is
	/// four real MACs, so it reaches the same floor at a quarter of the area
	pub par_threshold: usize,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}
/// information about the resulting $LU$ factorization
#[derive(Copy, Clone, Debug)]
pub struct PartialPivLuInfo {
	/// number of transpositions that were performed, can be used to compute
	/// the determinant of $P$
	pub transposition_count: usize,
}
/// error in the $LU$ factorization
#[derive(Copy, Clone, Debug)]
pub enum LdltError {
	ZeroPivot { index: usize },
}
impl<T: ComplexField> Auto<T> for PartialPivLuParams {
	#[inline]
	fn auto() -> Self {
		Self {
			recursion_threshold: 16,
			block_size: 64,
			par_threshold: 128 * 128,
			non_exhaustive: NonExhaustive(()),
		}
	}
}
#[inline]
pub fn lu_in_place_scratch<I: Index, T: ComplexField>(
	nrows: usize,
	ncols: usize,
	par: Par,
	params: Spec<PartialPivLuParams, T>,
) -> StackReq {
	_ = par;
	_ = params;
	StackReq::new::<I>(Ord::min(nrows, ncols))
}
pub fn lu_in_place<'out, I: Index, T: ComplexField>(
	A: MatMut<'_, T>,
	perm: &'out mut [I],
	perm_inv: &'out mut [I],
	par: Par,
	stack: &mut MemStack,
	params: Spec<PartialPivLuParams, T>,
) -> (PartialPivLuInfo, PermRef<'out, I>) {
	let _ = &params;
	let truncate = I::truncate;
	#[cfg(feature = "perf-warn")]
	if (A.col_stride().unsigned_abs() == 1
		|| A.row_stride().unsigned_abs() != 1)
		&& crate::__perf_warn!(LU_WARN)
	{
		log::warn!(
			target : "faer_perf",
			"LU with partial pivoting prefers column-major or row-major matrix. Found matrix with generic strides."
		);
	}
	let mut matrix = A;
	let mut stack = stack;
	let m = matrix.nrows();
	let n = matrix.ncols();
	// Row-major input: the unblocked core walks each pivot column twice (the
	// search, then the scaling) and the swap stage swaps elements inside a
	// column, which is a cache line per element when the columns are not
	// contiguous -- `size * m` touches that way against `m * n` elements the copy
	// moves contiguously twice. Factor a column-major copy and write the factor
	// back, which is what LAPACK's row-major interface pays internally; the
	// arithmetic is the copy's, so the factor and the permutation are the
	// column-major ones. The guard is the panel's own cost model, the same one
	// `qr_in_place` states: a wide matrix has `size == m` over `n` columns, where
	// the copy would cost more than it saved.
	if matrix.row_stride().unsigned_abs() != 1
		&& matrix.nrows() >= matrix.ncols()
		&& matrix.ncols() > 1
		&& linalg::row_major_copy_pays(matrix.nrows(), matrix.ncols())
	{
		let mut work = alloc::vec![zero::<T>(); m * n];
		let mut work = MatMut::from_column_major_slice_mut(&mut work, m, n);
		linalg::copy_across_layouts(work.rb_mut(), matrix.rb(), par);
		let ret = lu_in_place(
			work.rb_mut(),
			perm,
			perm_inv,
			par,
			stack,
			params,
		);
		linalg::copy_across_layouts(matrix, work.rb(), par);
		return ret;
	}
	let size = Ord::min(n, m);
	for i in 0..m {
		let p = &mut perm[i];
		*p = truncate(i);
	}
	let (mut transpositions, _) =
		stack.rb_mut().make_with(size, |_| truncate(0));
	let transpositions = transpositions.as_mut();
	let n_transpositions = lu_in_place_recursion(
		matrix.rb_mut(),
		0,
		size,
		transpositions.as_mut(),
		par,
		params,
	);
	for idx in 0..size {
		let t = transpositions[idx];
		perm.as_mut().swap(idx, idx + t.zx());
	}
	if m < n {
		let (left, right) = matrix.split_at_col_mut(size);
		linalg::triangular_solve::solve_unit_lower_triangular_in_place(
			left.rb(),
			right,
			par,
		);
	}
	for i in 0..m {
		perm_inv[perm[i].zx()] = truncate(i);
	}
	(
		PartialPivLuInfo {
			transposition_count: n_transpositions,
		},
		unsafe { PermRef::new_unchecked(perm, perm_inv, m) },
	)
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, assert, c32, c64};
	use dyn_stack::MemBuffer;
	#[test]
	fn test_plu() {
		let rng = &mut StdRng::seed_from_u64(0);
		let approx_eq = CwiseMat(ApproxEq {
			abs_tol: 1e-13,
			rel_tol: 1e-13,
		});
		for n in [1, 2, 3, 128, 255, 256, 257] {
			let A = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			let A = A.as_ref();
			let mut LU = A.cloned();
			let perm = &mut *vec![0usize; n];
			let perm_inv = &mut *vec![0usize; n];
			let params = PartialPivLuParams {
				recursion_threshold: 2,
				block_size: 2,
				..auto!(f64)
			};
			let p = lu_in_place(
				LU.as_mut(),
				perm,
				perm_inv,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(lu_in_place_scratch::<
					usize,
					f64,
				>(
					n,
					n,
					Par::Seq,
					params.into(),
				))),
				params.into(),
			)
			.1;
			let mut L = LU.as_ref().cloned();
			let mut U = LU.as_ref().cloned();
			for j in 0..n {
				for i in 0..j {
					L[(i, j)] = 0.0;
				}
				L[(j, j)] = 1.0;
			}
			for j in 0..n {
				for i in j + 1..n {
					U[(i, j)] = 0.0;
				}
			}
			let L = L.as_ref();
			let U = U.as_ref();
			assert!(p.inverse() * L * U ~ A);
		}
		for m in [8, 128, 255, 256, 257] {
			let n = 8;
			let A = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			let A = A.as_ref();
			let mut LU = A.cloned();
			let perm = &mut *vec![0usize; m];
			let perm_inv = &mut *vec![0usize; m];
			let p = lu_in_place(
				LU.as_mut(),
				perm,
				perm_inv,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(lu_in_place_scratch::<
					usize,
					f64,
				>(
					n, n, Par::Seq, default()
				))),
				default(),
			)
			.1;
			let mut L = LU.as_ref().cloned();
			let mut U = LU.as_ref().cloned();
			for j in 0..n {
				for i in 0..j {
					L[(i, j)] = 0.0;
				}
				L[(j, j)] = 1.0;
			}
			for j in 0..n {
				for i in j + 1..m {
					U[(i, j)] = 0.0;
				}
			}
			let L = L.as_ref();
			let U = U.as_ref();
			let U = U.subrows(0, n);
			assert!(p.inverse() * L * U ~ A);
		}
	}
	fn factor_lu(A: &Mat<f64>, par: Par) -> (Mat<f64>, Vec<usize>, usize) {
		let n = A.nrows();
		let mut LU = A.cloned();
		let perm = &mut *vec![0usize; n];
		let perm_inv = &mut *vec![0usize; n];
		let params: Spec<PartialPivLuParams, f64> = default();
		let stack =
			&mut MemBuffer::new(lu_in_place_scratch::<usize, f64>(n, n, par, params));
		let stack = MemStack::new(stack);
		let (info, _) = lu_in_place(LU.as_mut(), perm, perm_inv, par, stack, params);
		(LU, perm.to_vec(), info.transposition_count)
	}
	fn factor_lu_c64(A: &Mat<c64>, par: Par) -> (Mat<c64>, Vec<usize>, usize) {
		let n = A.nrows();
		let mut LU = A.cloned();
		let perm = &mut *vec![0usize; n];
		let perm_inv = &mut *vec![0usize; n];
		let params: Spec<PartialPivLuParams, c64> = default();
		let stack =
			&mut MemBuffer::new(lu_in_place_scratch::<usize, c64>(n, n, par, params));
		let stack = MemStack::new(stack);
		let (info, _) = lu_in_place(LU.as_mut(), perm, perm_inv, par, stack, params);
		(LU, perm.to_vec(), info.transposition_count)
	}
	/// The two products at a recursion level take `Par::Seq` while the trailing update's work is
	/// below the floor, instead of the caller's hint. That is a choice of how to schedule the
	/// same arithmetic, so both sides of the floor have to return the same factorization: `96`,
	/// `128` and `192` are below `128 * 128` `f64` elements at the top level, `256` and `512`
	/// above it, and every level in between halves down through it. A `c64` element is four real
	/// MACs, so the same floor is reached at a quarter of the area -- `128` is below it and `256`
	/// above it for `c64` too, which is the case the element-only comparison got wrong.
	#[test]
	fn threaded_lu_matches_the_sequential_one() {
		let rng = &mut StdRng::seed_from_u64(0);
		let pool = rayon::ThreadPoolBuilder::new()
			.num_threads(4)
			.build()
			.unwrap();
		let approx_eq = CwiseMat(ApproxEq {
			abs_tol: 1e-12,
			rel_tol: 1e-12,
		});
		for n in [96, 128, 192, 256, 512] {
			let A = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			let (seq, perm_seq, trans_seq) = factor_lu(&A, Par::Seq);
			let (par, perm_par, trans_par) =
				pool.install(|| factor_lu(&A, Par::rayon(4)));
			assert!(seq.as_ref() ~ par.as_ref());
			assert!(perm_seq == perm_par);
			assert!(trans_seq == trans_par);
		}
		for n in [96, 128, 192, 256, 512] {
			let real = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			let imag = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			let A = Mat::<c64>::from_fn(n, n, |i, j| c64::new(real[(i, j)], imag[(i, j)]));
			let (seq, perm_seq, trans_seq) = factor_lu_c64(&A, Par::Seq);
			let (par, perm_par, trans_par) =
				pool.install(|| factor_lu_c64(&A, Par::rayon(4)));
			for j in 0..n {
				for i in 0..n {
					let diff = (seq[(i, j)] - par[(i, j)]).abs();
					assert!(diff <= 1e-10, "({i}, {j}) differs by {diff:e}");
				}
			}
			assert!(perm_seq == perm_par);
			assert!(trans_seq == trans_par);
		}
	}
	/// An exactly zero pivot column is a singular matrix, not a failed factorization: the
	/// factors stay finite and still reconstruct the input, with the singularity left in `U`.
	/// A subnormal pivot must not be scaled by its infinite reciprocal either. OpenBLAS's
	/// `getf2_k` skips the column when the pivot is exactly zero, and reference `dgetf2`
	/// divides elementwise below `SFMIN`.
	#[test]
	fn zero_pivot_column_reconstructs() {
		fn zero_pivot<T: ComplexField>(n: usize) -> Mat<T> {
			Mat::<T>::from_fn(n, n, |i, j| {
				if j > 0 {
					from_f64::<T>((i + 2 * j) as f64)
				} else {
					zero()
				}
			})
		}
		// the pivot is the largest entry of a column, so a subnormal pivot needs a
		// subnormal fixture: `tiny` is the smallest useful one for each dtype
		fn subnormal<T: ComplexField>(tiny: f64) -> Mat<T> {
			let tiny = from_f64::<T>(tiny);
			Mat::<T>::from_fn(2, 2, |i, j| match (i, j) {
				(0, 0) => tiny.copy(),
				(0, 1) => one(),
				(1, 0) => tiny.mul_real(from_f64::<T::Real>(0.5)),
				(1, 1) => from_f64::<T>(2.0),
				_ => zero(),
			})
		}
		for n in [1usize, 2, 3, 128] {
			check_factor::<f32>(zero_pivot::<f32>(n).as_ref(), None);
			check_factor::<c32>(zero_pivot::<c32>(n).as_ref(), None);
			check_factor::<f64>(zero_pivot::<f64>(n).as_ref(), None);
			check_factor::<c64>(zero_pivot::<c64>(n).as_ref(), None);
		}
		// a power of two so that the subnormal and its half are both exact and the
		// expected multiplier is exactly `0.5`
		check_factor::<f32>(
			subnormal::<f32>(2f64.powi(-147)).as_ref(),
			Some(0.5f32),
		);
		check_factor::<c32>(
			subnormal::<c32>(2f64.powi(-147)).as_ref(),
			Some(c32::new(0.5, 0.0)),
		);
		check_factor::<f64>(
			subnormal::<f64>(2f64.powi(-1072)).as_ref(),
			Some(0.5f64),
		);
		check_factor::<c64>(
			subnormal::<c64>(2f64.powi(-1072)).as_ref(),
			Some(c64::new(0.5, 0.0)),
		);
	}
	fn check_factor<T: ComplexField>(a: MatRef<'_, T>, l10: Option<T>) {
		let tol = eps::<T::Real>() * from_f64::<T::Real>(64.0);
		let approx_eq = CwiseMat(ApproxEq {
			abs_tol: tol.copy(),
			rel_tol: tol,
		});
		let n = a.nrows();
		for par in [Par::Seq, Par::rayon(8)] {
			let mut LU = a.cloned();
			let perm = &mut *vec![0usize; n];
			let perm_inv = &mut *vec![0usize; n];
			let params = PartialPivLuParams {
				recursion_threshold: 2,
				block_size: 2,
				..auto!(T)
			};
			let p = lu_in_place(
				LU.as_mut(),
				perm,
				perm_inv,
				par,
				MemStack::new(&mut MemBuffer::new(lu_in_place_scratch::<
					usize,
					T,
				>(
					n, n, par, params.into()
				))),
				params.into(),
			)
			.1;
			assert!(LU.as_ref().is_all_finite());
			let mut L = LU.as_ref().cloned();
			let mut U = LU.as_ref().cloned();
			for j in 0..n {
				for i in 0..j {
					L[(i, j)] = zero();
				}
				L[(j, j)] = one();
			}
			for j in 0..n {
				for i in j + 1..n {
					U[(i, j)] = zero();
				}
			}
			assert!(p.inverse() * L.as_ref() * U.as_ref() ~ a);
			if let Some(ref x) = l10 {
				assert!(
					(L[(1, 0)].copy() - x.copy()).abs()
						<= eps::<T::Real>() * from_f64::<T::Real>(4.0)
				);
			}
		}
	}
}
