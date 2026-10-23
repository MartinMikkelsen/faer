//! block householder transformations
//!
//! a householder reflection is linear transformation that describes a
//! reflection about a hyperplane that crosses the origin of the space
//!
//! let $v$ be a unit vector that is orthogonal to the hyperplane. then the
//! corresponding householder transformation in matrix form is $I - 2vv^H$,
//! where $I$ is the identity matrix
//!
//! in practice, a non unit vector $v$ is used, so the transformation is written
//! as $$H = I - \frac{vv^H}{\tau}$$
//!
//! a block householder transformation is a sequence of such transformations
//! $H_0, H_1, \dots, H_{b -1 }$ applied one after the other, with the
//! restriction that the first $i$ components of the vector $v_i$ of the $i$-th
//! transformation are zero, and the component at index $i$ is one
//!
//! the matrix $V = [v_0\ v_1\ \dots\ v_{b-1}]$ is thus a lower trapezoidal
//! matrix with unit diagonal. we call it the householder basis
//!
//! there exists a unique upper triangular matrix $T$, that we call the
//! householder factor, such that $$H_0 \times \dots \times H_{b-1} = I -
//! VT^{-1}V^H$$
//!
//! a block householder sequence is a sequence of such transformations, composed
//! of two matrices:
//! - a lower trapezoidal matrix with unit diagonal, which is the horizontal
//!   concatenation of the
//! bases of each block householder transformation,
//! - a horizontal concatenation of the householder factors.
//!
//! examples on how to create and manipulate block householder sequences are
//! provided in the documentation of the $QR$ module.
use crate::assert;
use crate::internal_prelude::*;
use crate::linalg::matmul::triangular::{self, BlockStructure};
use crate::linalg::matmul::{dot, matmul, matmul_with_conj};
use crate::linalg::triangular_solve;
use crate::utils::simd::SimdCtx;
use crate::utils::thread::join_raw;
/// Householder information
#[derive(Clone, Debug)]
pub struct HouseholderInfo<T: ComplexField> {
	/// The tau value of the householder transformation
	pub tau: T::Real,
	/// The reciprocal of head with beta added
	pub head_with_beta_inv: T,
	/// The norm
	pub norm: T::Real,
}
/// computes the householder reflection $I - \frac{v v^H}{\tau}$ such that when
/// multiplied by $x$ from the left, the result is $\beta e_0$. $\tau$ and
/// $(\text{head} - \beta)^{-1}$ are returned and $\tau$ is real-valued. $\beta$
/// is stored in `head`
///
/// $x$ is determined by $x_0$, contained in `head`, and $|x_{1\dots}|$,
/// contained in `tail_norm`. the vector $v$ is such that $v_0 = 1$ and
/// $v_{1\dots}$ is stored in `essential` (when provided)
fn make_householder_imp<T: ComplexField>(
	head: &mut T,
	out: ColMut<'_, T>,
	input: Option<ColRef<'_, T>>,
) -> HouseholderInfo<T> {
	let tail = input.unwrap_or(out.rb());
	let tail_norm = tail.norm_l2();
	let mut head_norm = (*head).abs();
	if head_norm < min_positive() {
		*head = zero();
		head_norm = zero();
	}
	if tail_norm < min_positive() {
		return HouseholderInfo {
			tau: infinity(),
			head_with_beta_inv: infinity(),
			norm: head_norm,
		};
	}
	let ref head_norm = head_norm;
	let ref one_half = from_f64::<T::Real>(0.5);
	let norm = head_norm.hypot(&tail_norm);
	let sign = if *head_norm != zero() {
		head.mul_real(head_norm.recip())
	} else {
		one()
	};
	let ref signed_norm = sign * norm.to_cplx::<T>();
	let ref head_with_beta = &*head + signed_norm;
	let head_with_beta_inv = head_with_beta.recip();
	match input {
		None => zip!(out).for_each(|unzip!(e)| {
			*e *= &head_with_beta_inv;
		}),
		Some(input) => {
			zip!(out, input).for_each(|unzip!(o, e): Zip!(&mut T, &T)| {
				*o = e * &head_with_beta_inv;
			})
		},
	}
	*head = -signed_norm;
	let tau = one_half
		* (one::<T::Real>() + (tail_norm * head_with_beta_inv.abs()).abs2());
	HouseholderInfo {
		tau,
		head_with_beta_inv,
		norm,
	}
}
/// computes the householder reflection $I - \frac{v v^H}{\tau}$ such that when
/// multiplied by $x$ from the left, the result is $\beta e_0$. $\tau$ and
/// $(\text{head} - \beta)^{-1}$ are returned and $\tau$ is real-valued. $\beta$
/// is stored in `head`
///
/// $x$ is determined by $x_0$, contained in `head`, and $|x_{1\dots}|$,
/// contained in `tail_norm`. the vector $v$ is such that $v_0 = 1$ and
/// $v_{1\dots}$ is stored in `essential` (when provided)
#[inline]
pub fn make_householder_in_place<T: ComplexField>(
	head: &mut T,
	tail: ColMut<'_, T>,
) -> HouseholderInfo<T> {
	make_householder_imp(head, tail, None)
}
#[inline]
pub(crate) fn make_householder_out_of_place<T: ComplexField>(
	head: &mut T,
	out: ColMut<'_, T>,
	tail: ColRef<'_, T>,
) -> HouseholderInfo<T> {
	make_householder_imp(head, out, Some(tail))
}
#[doc(hidden)]
/// bases up to this width have their `V^H V` formed as inner products on the calling thread
/// (see `upgrade_householder_factor`)
const NARROW_BASIS: usize = 8;
pub fn upgrade_householder_factor<T: ComplexField>(
	householder_factor: MatMut<'_, T>,
	essentials: MatRef<'_, T>,
	block_size: usize,
	prev_block_size: usize,
	par: Par,
) {
	assert!(all(
		householder_factor.nrows() == householder_factor.ncols(),
		essentials.ncols() == householder_factor.ncols(),
	));
	if block_size == prev_block_size
		|| householder_factor.nrows().unbound() <= prev_block_size
	{
		return;
	}
	let n = essentials.ncols();
	// Below, this function threads its `V^H V` products, and its own `deep` predicate states when
	// the reduction has the shape to pay for that: deep against its `n x n` output, over a work
	// floor, with at least one row per thread. Below that floor the same products still open a
	// region each -- and a blocked factorisation calls this function once per block, at a block
	// size the recursion halves every level, so a `128 x 128` QR pays the fork-join cost of
	// `n`-wide reductions ~50 times for work the floor itself calls too small to split. The
	// decision here is the floor, asked once: no region at all when the products below would not
	// split either.
	let threads = crate::utils::thread::parallelism_degree(par);
	let deep_rows = essentials.nrows().unbound().saturating_sub(n);
	let above_floor =
		deep_rows >= 2 * n && deep_rows >= 4 * threads && deep_rows * n * n >= 8 * 1024;
	let par = if above_floor { par } else { Par::Seq };
	let mut householder_factor = householder_factor;
	let essentials = essentials;
	assert!(householder_factor.nrows() == householder_factor.ncols());
	let block_count = householder_factor.nrows().msrv_div_ceil(block_size);
	if block_count > 1 {
		assert!(all(
			block_size > prev_block_size,
			block_size % prev_block_size == 0,
		));
		let mid = block_count / 2;
		let (tau_tl, _, _, tau_br) = householder_factor.split_at_mut(mid, mid);
		let (basis_left, basis_right) = essentials.split_at_col(mid);
		let basis_right = basis_right.split_at_row(mid).1;
		join_raw(
			|parallelism| {
				upgrade_householder_factor(
					tau_tl,
					basis_left,
					block_size,
					prev_block_size,
					parallelism,
				)
			},
			|parallelism| {
				upgrade_householder_factor(
					tau_br,
					basis_right,
					block_size,
					prev_block_size,
					parallelism,
				)
			},
			par,
		);
		return;
	}
	if prev_block_size < 8 {
		let (basis_top, basis_bot) = essentials.split_at_row(n);
		let acc_structure = BlockStructure::UnitTriangularUpper;
		triangular::matmul(
			householder_factor.rb_mut(),
			acc_structure,
			Accum::Replace,
			basis_top.adjoint(),
			BlockStructure::UnitTriangularUpper,
			basis_top,
			BlockStructure::UnitTriangularLower,
			one(),
			par,
		);
		// `T += V_bot^H V_bot` reduces over every row below the basis' diagonal, so on a tall panel
		// it is one serial kernel per task with an `n x n` output -- the same shape as the reduction
		// the block apply splits, and it does not scale for the same reason: 0.271 ms at eight
		// threads for `m = 4096, n = 16`, against 0.040 when its rows are split. The partials here
		// are `n x n` each, tiny against the basis, and only the strict upper triangle is stored
		// (the diagonal is the caller's `tau`), so the split costs almost nothing.
		let threads = crate::utils::thread::parallelism_degree(par);
		let rows = basis_bot.nrows().unbound();
		let deep = cfg!(feature = "rayon")
			&& threads > 1
			&& rows >= 2 * n
			&& rows >= 4 * threads
			&& rows * n * n >= 8 * 1024;
		if n <= NARROW_BASIS {
			// a narrow basis: its `n (n - 1) / 2` inner products on the calling thread. the gemm
			// kernel runs this `n x rows x n` shape at 7 GB/s, and the split above pulls the basis
			// the leaf just wrote out of this core's cache into the workers' for less work than
			// the transfer costs (measured in the recursive QR at 4096 rows, n = 3: 23 us per call
			// split, 13 us as one gemm, 3 us as inner products)
			for col in 1..n {
				for row in 0..col {
					let s = crate::linalg::matmul::dot::inner_prod(
						basis_bot.col(row).transpose(),
						Conj::Yes,
						basis_bot.col(col),
						Conj::No,
					);
					householder_factor[(row, col)] =
						householder_factor[(row, col)].clone() + s;
				}
			}
		} else if deep {
			#[cfg(feature = "rayon")]
			{
				use rayon::prelude::*;
				let chunk = rows.div_ceil(threads);
				let n_chunks = rows.div_ceil(chunk);
				let mut partials = vec![T::zero(); n_chunks * n * n];
				let source = basis_bot;
				// through spindle, like every other region in this crate: a raw rayon iterator
				// finds the pool's threads occupied when the caller holds the worker group
				// (`spindle::with_lock`), and the split then runs starved
				spindle::for_each(
					n_chunks,
					partials.par_chunks_mut(n * n).enumerate(),
					|(index, part)| {
						let lo = index * chunk;
						let hi = rows.min(lo + chunk);
						let mut part =
							MatMut::from_column_major_slice_mut(part, n, n);
						matmul_with_conj(
							part.rb_mut(),
							Accum::Replace,
							source.subrows(lo, hi - lo).transpose(),
							Conj::Yes,
							source.subrows(lo, hi - lo),
							Conj::No,
							one(),
							Par::Seq,
						);
					},
				);
				for index in 0..n_chunks {
					let part = MatRef::from_column_major_slice(
						&partials[index * n * n..(index + 1) * n * n],
						n,
						n,
					);
					for col in 0..n {
						for row in 0..col {
							householder_factor[(row, col)] =
								householder_factor[(row, col)].clone()
									+ part[(row, col)].clone();
						}
					}
				}
			}
		} else {
			triangular::matmul(
				householder_factor.rb_mut(),
				acc_structure,
				Accum::Add,
				basis_bot.adjoint(),
				BlockStructure::Rectangular,
				basis_bot,
				BlockStructure::Rectangular,
				one(),
				par,
			);
		}
	} else {
		let prev_block_count =
			householder_factor.nrows().msrv_div_ceil(prev_block_size);
		let mid = (prev_block_count / 2) * prev_block_size;
		let (tau_tl, mut tau_tr, _, tau_br) =
			householder_factor.split_at_mut(mid, mid);
		let (basis_left, basis_right) = essentials.split_at_col(mid);
		let basis_right = basis_right.split_at_row(mid).1;
		join_raw(
			|parallelism| {
				join_raw(
					|parallelism| {
						upgrade_householder_factor(
							tau_tl,
							basis_left,
							block_size,
							prev_block_size,
							parallelism,
						)
					},
					|parallelism| {
						upgrade_householder_factor(
							tau_br,
							basis_right,
							block_size,
							prev_block_size,
							parallelism,
						)
					},
					parallelism,
				);
			},
			|parallelism| {
				let basis_left = basis_left.split_at_row(mid).1;
				let row_mid = basis_right.ncols();
				let (basis_left_top, basis_left_bot) =
					basis_left.split_at_row(row_mid);
				let (basis_right_top, basis_right_bot) =
					basis_right.split_at_row(row_mid);
				triangular::matmul(
					tau_tr.rb_mut(),
					BlockStructure::Rectangular,
					Accum::Replace,
					basis_left_top.adjoint(),
					BlockStructure::Rectangular,
					basis_right_top,
					BlockStructure::UnitTriangularLower,
					one(),
					parallelism,
				);
				matmul(
					tau_tr.rb_mut(),
					Accum::Add,
					basis_left_bot.adjoint(),
					basis_right_bot,
					one(),
					parallelism,
				);
			},
			par,
		);
	}
}
/// computes the layout of required workspace for applying a block householder
/// transformation to a right-hand-side matrix in place
pub fn apply_block_householder_on_the_left_in_place_scratch<T: ComplexField>(
	householder_basis_nrows: usize,
	block_size: usize,
	rhs_ncols: usize,
) -> StackReq {
	let _ = householder_basis_nrows;
	temp_mat_scratch::<T>(block_size, rhs_ncols)
}
/// computes the layout of required workspace for applying the transpose of a
/// block householder transformation to a right-hand-side matrix in place
pub fn apply_block_householder_transpose_on_the_left_in_place_scratch<
	T: ComplexField,
>(
	householder_basis_nrows: usize,
	block_size: usize,
	rhs_ncols: usize,
) -> StackReq {
	let _ = householder_basis_nrows;
	temp_mat_scratch::<T>(block_size, rhs_ncols)
}
/// computes the layout of required workspace for applying a block householder
/// transformation to a left-hand-side matrix in place
pub fn apply_block_householder_on_the_right_in_place_scratch<
	T: ComplexField,
>(
	householder_basis_nrows: usize,
	block_size: usize,
	lhs_nrows: usize,
) -> StackReq {
	let _ = householder_basis_nrows;
	temp_mat_scratch::<T>(block_size, lhs_nrows)
}
/// computes the layout of required workspace for applying the transpose of a
/// block householder transformation to a left-hand-side matrix in place
pub fn apply_block_householder_transpose_on_the_right_in_place_scratch<
	T: ComplexField,
>(
	householder_basis_nrows: usize,
	block_size: usize,
	lhs_nrows: usize,
) -> StackReq {
	let _ = householder_basis_nrows;
	temp_mat_scratch::<T>(block_size, lhs_nrows)
}
/// computes the layout of required workspace for applying the transpose of a
/// sequence of block householder transformations to a right-hand-side matrix in
/// place
pub fn apply_block_householder_sequence_transpose_on_the_left_in_place_scratch<
	T: ComplexField,
>(
	householder_basis_nrows: usize,
	block_size: usize,
	rhs_ncols: usize,
) -> StackReq {
	let _ = householder_basis_nrows;
	temp_mat_scratch::<T>(block_size, rhs_ncols)
}
/// computes the layout of required workspace for applying a sequence of block
/// householder transformations to a right-hand-side matrix in place
pub fn apply_block_householder_sequence_on_the_left_in_place_scratch<
	T: ComplexField,
>(
	householder_basis_nrows: usize,
	block_size: usize,
	rhs_ncols: usize,
) -> StackReq {
	let _ = householder_basis_nrows;
	temp_mat_scratch::<T>(block_size, rhs_ncols)
}
/// computes the layout of required workspace for applying the transpose of a
/// sequence of block householder transformations to a left-hand-side matrix in
/// place
pub fn apply_block_householder_sequence_transpose_on_the_right_in_place_scratch<
	T: ComplexField,
>(
	householder_basis_nrows: usize,
	block_size: usize,
	lhs_nrows: usize,
) -> StackReq {
	let _ = householder_basis_nrows;
	temp_mat_scratch::<T>(block_size, lhs_nrows)
}
/// computes the layout of required workspace for applying a sequence of block
/// householder transformations to a left-hand-side matrix in place
pub fn apply_block_householder_sequence_on_the_right_in_place_scratch<
	T: ComplexField,
>(
	householder_basis_nrows: usize,
	block_size: usize,
	lhs_nrows: usize,
) -> StackReq {
	let _ = householder_basis_nrows;
	temp_mat_scratch::<T>(block_size, lhs_nrows)
}
#[track_caller]
fn apply_block_householder_on_the_left_in_place_generic<
	'M,
	'N,
	'K,
	T: ComplexField,
>(
	householder_basis: MatRef<'_, T, Dim<'M>, Dim<'N>>,
	householder_factor: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	conj_lhs: Conj,
	matrix: MatMut<'_, T, Dim<'M>, Dim<'K>>,
	forward: bool,
	par: Par,
	stack: &mut MemStack,
) {
	assert!(all(
		householder_factor.nrows() == householder_factor.ncols(),
		householder_basis.ncols() == householder_factor.nrows(),
		matrix.nrows() == householder_basis.nrows(),
	));
	let mut matrix = matrix;
	let M = householder_basis.nrows();
	let N = householder_basis.ncols();
	make_guard!(TAIL);
	let midpoint = M.head_partition(N, TAIL);
	// The N == 1 kernel below is one sequential rank-one update per target column, and a panel of a
	// tall blocked QR is made almost entirely of those. When the caller asks for parallelism and
	// there is enough work to pay for its temporaries, the generic path below is the better one: it
	// partitions the target's columns and threads its two matrix products, which this one cannot.
	let big_rank_one = householder_basis.nrows().unbound()
		* matrix.ncols().unbound()
		>= 8 * 1024;
	let use_simd = N.unbound() == 1
		&& T::SIMD_CAPABILITIES.is_simd()
		&& !(crate::utils::thread::parallelism_degree(par) > 1 && big_rank_one);
	if let (Some(householder_basis), Some(matrix), true) = (
		householder_basis.try_as_col_major(),
		matrix.rb_mut().try_as_col_major_mut(),
		use_simd,
	) {
		struct ApplyOnLeft<'a, 'TAIL, 'K, T: ComplexField, const CONJ: bool> {
			tau_inv: &'a T,
			essential: ColRef<'a, T, Dim<'TAIL>, ContiguousFwd>,
			rhs0: RowMut<'a, T, Dim<'K>>,
			rhs: MatMut<'a, T, Dim<'TAIL>, Dim<'K>, ContiguousFwd>,
		}
		impl<'TAIL, 'K, T: ComplexField, const CONJ: bool> pulp::WithSimd
			for ApplyOnLeft<'_, 'TAIL, 'K, T, CONJ>
		{
			type Output = ();

			#[inline(always)]
			fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
				let Self {
					tau_inv,
					essential,
					mut rhs,
					mut rhs0,
				} = self;
				if rhs.nrows().unbound() == 0 {
					return;
				}
				let N = rhs.nrows();
				let K = rhs.ncols();
				let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), N);
				let indices = simd.indices();
				for idx in K.indices() {
					let col0 = rhs0.rb_mut().at_mut(idx);
					let mut col = rhs.rb_mut().col_mut(idx);
					let essential = essential;
					let dot = if const { CONJ } {
						&*col0
							+ dot::inner_prod_no_conj_simd(
								simd,
								essential.rb(),
								col.rb(),
							)
					} else {
						&*col0
							+ dot::inner_prod_conj_lhs_simd(
								simd,
								essential.rb(),
								col.rb(),
							)
					};
					let ref k = -dot * tau_inv;
					*col0 += k;
					let k = simd.splat(k);

					simd_iter!(for i in [indices] {
						let mut a = simd.read(col.rb(), i);
						let b = simd.read(essential.rb(), i);
						if const { CONJ } {
							a = simd.conj_mul_add(b, k, a);
						} else {
							a = simd.mul_add(b, k, a);
						}
						simd.write(col.rb_mut(), i, a);
					});
				}
			}
		}
		let N0 = N.check(0);
		let essential = householder_basis.col(N0).split_rows_with(midpoint).1;
		let (rhs0, rhs) = matrix.split_rows_with_mut(midpoint);
		let rhs0 = rhs0.row_mut(N0);
		let tau_inv: T = householder_factor[(N0, N0)].real().recip().to_cplx();
		if const { T::IS_REAL } {
			type Apply<'a, 'TAIL, 'K, T> = ApplyOnLeft<'a, 'TAIL, 'K, T, false>;
			dispatch!(
				Apply {
					tau_inv: &tau_inv,
					essential,
					rhs,
					rhs0,
				},
				Apply,
				T
			);
		} else if matches!(conj_lhs, Conj::No) {
			type Apply<'a, 'TAIL, 'K, T> = ApplyOnLeft<'a, 'TAIL, 'K, T, false>;
			dispatch!(
				Apply {
					tau_inv: &tau_inv,
					essential,
					rhs,
					rhs0,
				},
				Apply,
				T
			);
		} else {
			type Apply<'a, 'TAIL, 'K, T> = ApplyOnLeft<'a, 'TAIL, 'K, T, true>;
			dispatch!(
				Apply {
					tau_inv: &tau_inv,
					essential,
					rhs,
					rhs0,
				},
				Apply,
				T
			);
		}
	} else {
		let (essentials_top, essentials_bot) =
			householder_basis.split_rows_with(midpoint);
		let M = matrix.nrows();
		let K = matrix.ncols();
		let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(N, K, stack) };
		let mut tmp = tmp.as_mat_mut();
		let threads = crate::utils::thread::parallelism_degree(par);
		// The split below slices the reduction's rows, but the top `n` rows carry the unit
		// diagonal and are applied through their own `n x n` factor -- a chunk whose rows
		// are a *subrange* of those `n` would need that factor column-sliced, and the
		// triangular kernel cannot express the trapezoid that leaves. So the split is taken
		// only while one chunk swallows the whole top block. Below that the products further
		// down stay exactly as they were.
		let row_split = cfg!(feature = "rayon")
			&& threads > 1
			&& M.unbound() >= 2 * K.unbound()
			&& M.unbound() >= 4 * threads
			&& M.unbound() * K.unbound() >= 8 * 1024
			&& M.unbound().div_ceil(threads) >= N.unbound();
		// The column split is the only parallelism when the rows are not split, so it takes
		// the whole budget and leaves `inner_parallelism` at one thread: the block apply a
		// blocked QR performs is short and wide (`N` columns of basis against the whole
		// trailing block), and capping the split at four tasks made the rest nest a second
		// region inside each -- `512 x 512` `qr_in_place` f64 at eight threads: 5.27 ms with
		// the cap, 4.74 ms with eight tasks, and `1024 x 1024`: 20.5 -> 16.9 ms. When the
		// rows *are* split, that split already owns the thread budget and the four-task cap
		// stays, because the nested column split is then only a second level over a much
		// smaller target (`4096 x 16` basis, 240-column target: 0.397 ms capped, 0.405 ms
		// uncapped at eight threads).
		let mut n_tasks = Ord::min(
			Ord::min(threads, K.unbound()),
			if row_split { 4 } else { threads },
		);
		if (M.unbound() * K.unbound()).saturating_mul(4 * M.unbound())
			< gemm::get_threading_threshold()
		{
			n_tasks = 1;
		}
		let inner_parallelism = match par {
			Par::Seq => Par::Seq,
			#[cfg(feature = "rayon")]
			Par::Rayon(par) => {
				let par = par.get();
				if par >= 2 * n_tasks {
					Par::rayon(par / n_tasks)
				} else {
					Par::Seq
				}
			},
		};
		// The reduction `tmp = V^H C` has an `N x K` output and reduces over every row of the
		// target. The products below are good at that while the output is wide; with `N` small and
		// the target long they are one serial reduction inside one task, so the column split cannot
		// help -- and that is exactly the shape a tall blocked QR's panel and its trailing update
		// take. Split the rows instead, one `N x K` partial per thread, then add them: the partials
		// are tiny against the target. `1T` and short targets keep the products below unchanged.
		// Splitting the rows is a split along the reduction's depth: it adds parallel work there, and
		// it adds `threads` partials of `N x K` to write, read and add. It pays only while the serial
		// kernel it replaces is depth-bound, which is a property of the shape, not of one size: a
		// sweep over `M` against `K` at eight threads puts the crossover at `M ~ 2 K` -- a reduction
		// four times deeper than it is wide gains 3-4x, and one as wide as it is deep (the 512x512
		// trailing update, `M = 512`, `K = 496`) loses for the partials it pays for. The thread and
		// work floors keep small problems on the products below.
		if row_split {
			#[cfg(feature = "rayon")]
			{
				use rayon::prelude::*;
				let rows = M.unbound();
				let (n, k) = (N.unbound(), K.unbound());
				// The first `n` rows carry the unit diagonal; below them the basis is rectangular.
				let split_at = n;
				let chunk = rows.div_ceil(threads);
				let n_chunks = rows.div_ceil(chunk);
				let mut partials = vec![T::zero(); n_chunks * n * k];
				let source = matrix.rb().as_dyn_rows().as_dyn_cols();
				let top_basis = essentials_top.as_dyn_rows().as_dyn_cols();
				let bot_basis = essentials_bot.as_dyn_rows().as_dyn_cols();
				let conj = Conj::Yes.compose(conj_lhs);
				spindle::for_each(
					n_chunks,
					partials.par_chunks_mut(n * k).enumerate(),
					|(index, part)| {
						let lo = index * chunk;
						let hi = rows.min(lo + chunk);
						let mut part =
							MatMut::from_column_major_slice_mut(part, n, k);
						let top_hi = split_at.min(hi);
						if lo < top_hi {
							triangular::matmul_with_conj(
								part.rb_mut(),
								BlockStructure::Rectangular,
								Accum::Replace,
								top_basis.transpose(),
								BlockStructure::UnitTriangularUpper,
								conj,
								source.subrows(lo, top_hi - lo),
								BlockStructure::Rectangular,
								Conj::No,
								one(),
								Par::Seq,
							);
						}
						let bot_lo = split_at.max(lo);
						if hi > bot_lo {
							matmul_with_conj(
								part.rb_mut(),
								if lo < top_hi {
									Accum::Add
								} else {
									Accum::Replace
								},
								bot_basis
									.transpose()
									.subcols(bot_lo - split_at, hi - bot_lo),
								conj,
								source.subrows(bot_lo, hi - bot_lo),
								Conj::No,
								one(),
								Par::Seq,
							);
						}
					},
				);
				let mut tmp_dynamic =
					tmp.rb_mut().as_dyn_rows_mut().as_dyn_cols_mut();
				for col in 0..k {
					for row in 0..n {
						let mut sum = T::zero();
						for index in 0..n_chunks {
							sum +=
								partials[index * n * k + row + col * n].clone();
						}
						tmp_dynamic[(row, col)] = sum;
					}
				}
			}
		}
		let func = |(mut tmp, mut matrix): (
			MatMut<'_, T, Dim<'N>>,
			MatMut<'_, T, Dim<'M>>,
		)| {
			let (mut top, mut bot) =
				matrix.rb_mut().split_rows_with_mut(midpoint);
			if !row_split {
				triangular::matmul_with_conj(
					tmp.rb_mut(),
					BlockStructure::Rectangular,
					Accum::Replace,
					essentials_top.transpose(),
					BlockStructure::UnitTriangularUpper,
					Conj::Yes.compose(conj_lhs),
					top.rb(),
					BlockStructure::Rectangular,
					Conj::No,
					one(),
					inner_parallelism,
				);
				matmul_with_conj(
					tmp.rb_mut(),
					Accum::Add,
					essentials_bot.transpose(),
					Conj::Yes.compose(conj_lhs),
					bot.rb(),
					Conj::No,
					one(),
					inner_parallelism,
				);
			}
			if forward {
				triangular_solve::solve_lower_triangular_in_place_with_conj(
					householder_factor.transpose(),
					Conj::Yes.compose(conj_lhs),
					tmp.rb_mut(),
					inner_parallelism,
				);
			} else {
				triangular_solve::solve_upper_triangular_in_place_with_conj(
					householder_factor,
					Conj::No.compose(conj_lhs),
					tmp.rb_mut(),
					inner_parallelism,
				);
			}
			triangular::matmul_with_conj(
				top.rb_mut(),
				BlockStructure::Rectangular,
				Accum::Add,
				essentials_top,
				BlockStructure::UnitTriangularLower,
				Conj::No.compose(conj_lhs),
				tmp.rb(),
				BlockStructure::Rectangular,
				Conj::No,
				-one::<T>(),
				inner_parallelism,
			);
			matmul_with_conj(
				bot.rb_mut(),
				Accum::Add,
				essentials_bot,
				Conj::No.compose(conj_lhs),
				tmp.rb(),
				Conj::No,
				-one::<T>(),
				inner_parallelism,
			);
		};
		if n_tasks <= 1 {
			func((tmp.as_dyn_cols_mut(), matrix.as_dyn_cols_mut()));
			return;
		} else {
			#[cfg(feature = "rayon")]
			{
				use rayon::prelude::*;
				spindle::for_each(
					n_tasks,
					tmp.rb_mut()
						.par_col_partition_mut(n_tasks)
						.zip_eq(matrix.rb_mut().par_col_partition_mut(n_tasks)),
					func,
				);
			}
		}
	}
}
/// computes the product of the matrix, multiplied by the given block
/// householder transformation, and stores the result in `matrix`
#[track_caller]
pub fn apply_block_householder_on_the_right_in_place_with_conj<
	T: ComplexField,
>(
	householder_basis: MatRef<'_, T>,
	householder_factor: MatRef<'_, T>,
	conj_rhs: Conj,
	matrix: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	apply_block_householder_transpose_on_the_left_in_place_with_conj(
		householder_basis,
		householder_factor,
		conj_rhs,
		matrix.transpose_mut(),
		par,
		stack,
	)
}
/// computes the product of the matrix, multiplied by the transpose of the given
/// block householder transformation, and stores the result in `matrix`
#[track_caller]
pub fn apply_block_householder_transpose_on_the_right_in_place_with_conj<
	T: ComplexField,
>(
	householder_basis: MatRef<'_, T>,
	householder_factor: MatRef<'_, T>,
	conj_rhs: Conj,
	matrix: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	apply_block_householder_on_the_left_in_place_with_conj(
		householder_basis,
		householder_factor,
		conj_rhs,
		matrix.transpose_mut(),
		par,
		stack,
	)
}
/// computes the product of the given block householder transformation,
/// multiplied by `matrix`, and stores the result in `matrix`
#[track_caller]
pub fn apply_block_householder_on_the_left_in_place_with_conj<
	T: ComplexField,
>(
	householder_basis: MatRef<'_, T>,
	householder_factor: MatRef<'_, T>,
	conj_lhs: Conj,
	matrix: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	make_guard!(M);
	make_guard!(N);
	make_guard!(K);
	let M = householder_basis.nrows().bind(M);
	let N = householder_basis.ncols().bind(N);
	let K = matrix.ncols().bind(K);
	apply_block_householder_on_the_left_in_place_generic(
		householder_basis.as_shape(M, N).as_dyn_stride(),
		householder_factor.as_shape(N, N).as_dyn_stride(),
		conj_lhs,
		matrix.as_shape_mut(M, K).as_dyn_stride_mut(),
		false,
		par,
		stack,
	)
}
/// computes the product of the transpose of the given block householder
/// transformation, multiplied by `matrix`, and stores the result in `matrix`
#[track_caller]
pub fn apply_block_householder_transpose_on_the_left_in_place_with_conj<
	T: ComplexField,
>(
	householder_basis: MatRef<'_, T>,
	householder_factor: MatRef<'_, T>,
	conj_lhs: Conj,
	matrix: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	with_dim!(M, householder_basis.nrows());
	with_dim!(N, householder_basis.ncols());
	with_dim!(K, matrix.ncols());
	apply_block_householder_on_the_left_in_place_generic(
		householder_basis.as_shape(M, N).as_dyn_stride(),
		householder_factor.as_shape(N, N).as_dyn_stride(),
		conj_lhs.compose(Conj::Yes),
		matrix.as_shape_mut(M, K).as_dyn_stride_mut(),
		true,
		par,
		stack,
	)
}
/// computes the product of a sequence of block householder transformations
/// given by `householder_basis` and `householder_factor`, multiplied by
/// `matrix`, and stores the result in `matrix`
#[track_caller]
pub fn apply_block_householder_sequence_on_the_left_in_place_with_conj<
	T: ComplexField,
>(
	householder_basis: MatRef<'_, T>,
	householder_factor: MatRef<'_, T>,
	conj_lhs: Conj,
	matrix: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	let mut matrix = matrix;
	let mut stack = stack;
	let m = householder_basis.nrows();
	let n = householder_basis.ncols();
	assert!(all(
		householder_factor.nrows() > 0,
		householder_factor.ncols() == Ord::min(m, n),
	));
	let size = householder_factor.ncols();
	let mut j = size;
	let mut block_size = size % householder_factor.nrows();
	if block_size == 0 {
		block_size = householder_factor.nrows();
	}
	while j > 0 {
		let j_prev = j - block_size;
		block_size = householder_factor.nrows();
		let essentials = householder_basis.get(j_prev.., j_prev..j);
		let householder =
			householder_factor.get(.., j_prev..j).subrows(0, j - j_prev);
		let matrix = matrix.rb_mut().get_mut(j_prev.., ..);
		apply_block_householder_on_the_left_in_place_with_conj(
			essentials,
			householder,
			conj_lhs,
			matrix,
			par,
			stack.rb_mut(),
		);
		j = j_prev;
	}
}
/// computes the product of the transpose of a sequence block householder
/// transformations given by `householder_basis` and `householder_factor`,
/// multiplied by `matrix`, and stores the result in `matrix`
#[track_caller]
pub fn apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj<
	T: ComplexField,
>(
	householder_basis: MatRef<'_, T>,
	householder_factor: MatRef<'_, T>,
	conj_lhs: Conj,
	matrix: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	let mut matrix = matrix;
	let mut stack = stack;
	let block_size = householder_factor.nrows();
	let m = householder_basis.nrows();
	let n = householder_basis.ncols();
	assert!(all(
		householder_factor.nrows() > 0,
		householder_factor.ncols() == Ord::min(m, n),
	));
	let size = householder_factor.ncols();
	let mut j = 0;
	while j < size {
		let block_size = Ord::min(block_size, size - j);
		let essentials = householder_basis.get(j.., j..j + block_size);
		let householder = householder_factor
			.get(.., j..j + block_size)
			.subrows(0, block_size);
		let matrix = matrix.rb_mut().get_mut(j.., ..);
		apply_block_householder_transpose_on_the_left_in_place_with_conj(
			essentials,
			householder,
			conj_lhs,
			matrix,
			par,
			stack.rb_mut(),
		);
		j += block_size;
	}
}
/// computes the product of `matrix`, multiplied by a sequence of block
/// householder transformations given by `householder_basis` and
/// `householder_factor`, and stores the result in `matrix`
#[track_caller]
pub fn apply_block_householder_sequence_on_the_right_in_place_with_conj<
	T: ComplexField,
>(
	householder_basis: MatRef<'_, T>,
	householder_factor: MatRef<'_, T>,
	conj_rhs: Conj,
	matrix: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
		householder_basis,
		householder_factor,
		conj_rhs,
		matrix.transpose_mut(),
		par,
		stack,
	)
}
/// computes the product of `matrix`, multiplied by the transpose of a sequence
/// of block householder transformations given by `householder_basis` and
/// `householder_factor`, and stores the result in `matrix`
#[track_caller]
pub fn apply_block_householder_sequence_transpose_on_the_right_in_place_with_conj<
	T: ComplexField,
>(
	householder_basis: MatRef<'_, T>,
	householder_factor: MatRef<'_, T>,
	conj_rhs: Conj,
	matrix: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	apply_block_householder_sequence_on_the_left_in_place_with_conj(
		householder_basis,
		householder_factor,
		conj_rhs,
		matrix.transpose_mut(),
		par,
		stack,
	)
}

#[cfg(all(test, feature = "rayon"))]
mod tests {
	use super::{
		apply_block_householder_transpose_on_the_left_in_place_scratch,
		apply_block_householder_transpose_on_the_left_in_place_with_conj,
		make_householder_in_place, upgrade_householder_factor,
	};
	use crate::dyn_stack::{MemBuffer, MemStack};
	use crate::{Conj, Mat, Par};
	use num_complex::ComplexFloat;
	use reborrow::ReborrowMut;

	/// The `N == 1` kernel is one sequential rank-one update per target column, and it is skipped in
	/// favour of the generic path -- which partitions the columns and threads its products -- when
	/// the caller asks for threads and the update is large enough to pay for that path's
	/// temporaries. The two must compute the same map, at the sizes that take that branch and at the
	/// sizes that do not.
	fn check_f64(m: usize, k: usize) {
		let pool = rayon::ThreadPoolBuilder::new()
			.num_threads(4)
			.build()
			.unwrap();
		// A unit-lower basis and its 1x1 compact-WY factor, as a QR panel leaves them.
		let mut a: Mat<f64> =
			Mat::from_fn(m, 1, |i, _| ((i % 7) as f64) * 0.25 - 0.75);
		let tau = {
			let mut col = a.as_mut().col_mut(0);
			let (mut head, tail) = col.rb_mut().split_at_row_mut(1);
			let tau = make_householder_in_place(&mut head[0], tail).tau;
			head[0] = 1.0;
			tau
		};
		let factor = Mat::<f64>::from_fn(1, 1, |_, _| tau);
		let target: Mat<f64> = Mat::from_fn(m, k, |i, j| {
			(((i * 5 + j * 11) % 9) as f64) * 0.125 - 0.5
		});
		let mut mem = MemBuffer::new(
			apply_block_householder_transpose_on_the_left_in_place_scratch::<f64>(
				m, 1, k,
			),
		);
		let mut sequential = target.clone();
		apply_block_householder_transpose_on_the_left_in_place_with_conj(
			a.as_ref(),
			factor.as_ref(),
			Conj::Yes,
			sequential.as_mut(),
			Par::Seq,
			MemStack::new(&mut mem),
		);
		let mut threaded = target.clone();
		pool.install(|| {
			apply_block_householder_transpose_on_the_left_in_place_with_conj(
				a.as_ref(),
				factor.as_ref(),
				Conj::Yes,
				threaded.as_mut(),
				Par::rayon(4),
				MemStack::new(&mut mem),
			)
		});
		for i in 0..m {
			for j in 0..k {
				let d = (threaded[(i, j)] - sequential[(i, j)]).abs();
				assert!(d < 1e-12, "({i}, {j}): {d}");
			}
		}
	}

	/// The complex scalars take the same branch, through the conjugated kernel.
	fn check_c64(m: usize, k: usize) {
		use num_complex::Complex64 as C64;
		let pool = rayon::ThreadPoolBuilder::new()
			.num_threads(4)
			.build()
			.unwrap();
		let mut a: Mat<C64> = Mat::from_fn(m, 1, |i, _| {
			C64::new(((i % 7) as f64) * 0.25 - 0.75, 0.125)
		});
		let tau = {
			let mut col = a.as_mut().col_mut(0);
			let (mut head, tail) = col.rb_mut().split_at_row_mut(1);
			let tau = make_householder_in_place(&mut head[0], tail).tau;
			head[0] = C64::new(1.0, 0.0);
			tau
		};
		let factor = Mat::<C64>::from_fn(1, 1, |_, _| C64::new(tau, 0.0));
		let target: Mat<C64> = Mat::from_fn(m, k, |i, j| {
			C64::new((((i * 5 + j * 11) % 9) as f64) * 0.125 - 0.5, 0.0)
		});
		let mut mem = MemBuffer::new(
			apply_block_householder_transpose_on_the_left_in_place_scratch::<C64>(
				m, 1, k,
			),
		);
		let mut sequential = target.clone();
		apply_block_householder_transpose_on_the_left_in_place_with_conj(
			a.as_ref(),
			factor.as_ref(),
			Conj::Yes,
			sequential.as_mut(),
			Par::Seq,
			MemStack::new(&mut mem),
		);
		let mut threaded = target.clone();
		pool.install(|| {
			apply_block_householder_transpose_on_the_left_in_place_with_conj(
				a.as_ref(),
				factor.as_ref(),
				Conj::Yes,
				threaded.as_mut(),
				Par::rayon(4),
				MemStack::new(&mut mem),
			)
		});
		for i in 0..m {
			for j in 0..k {
				let d = (threaded[(i, j)] - sequential[(i, j)]).abs();
				assert!(d < 1e-12, "({i}, {j}): {d}");
			}
		}
	}

	/// The same for a panel of reflectors: `b > 1` gives the row split both kinds of basis -- the
	/// unit-triangular top and the rectangular bottom -- and is the shape a trailing update takes.
	fn check_panel(m: usize, b: usize, k: usize) {
		check_panel_with(m, b, k, 4);
	}

	fn check_panel_with(m: usize, b: usize, k: usize, threads: usize) {
		let pool = rayon::ThreadPoolBuilder::new()
			.num_threads(threads)
			.build()
			.unwrap();
		let mut a: Mat<f64> = Mat::from_fn(m, b, |i, j| {
			if i == j {
				4.0 + i as f64
			} else {
				((i * 3 + j * 7) % 5) as f64 * 0.25 - 0.5
			}
		});
		let mut tile = Mat::<f64>::zeros(b, b);
		{
			let mut qr = a.as_mut();
			for j in 0..b {
				let mut col = qr.rb_mut().col_mut(j).subrows_mut(j, m - j);
				let (mut head, tail) = col.rb_mut().split_at_row_mut(1);
				let info = make_householder_in_place(&mut head[0], tail);
				tile[(j, j)] = info.tau;
				head[0] = 1.0;
			}
		}
		let mut factor = tile.clone();
		upgrade_householder_factor(factor.as_mut(), a.as_ref(), b, 1, Par::Seq);
		let target: Mat<f64> = Mat::from_fn(m, k, |i, j| {
			(((i * 5 + j * 11) % 9) as f64) * 0.125 - 0.5
		});
		let mut mem = MemBuffer::new(
			apply_block_householder_transpose_on_the_left_in_place_scratch::<f64>(
				m, b, k,
			),
		);
		let mut sequential = target.clone();
		apply_block_householder_transpose_on_the_left_in_place_with_conj(
			a.as_ref(),
			factor.as_ref(),
			Conj::Yes,
			sequential.as_mut(),
			Par::Seq,
			MemStack::new(&mut mem),
		);
		let mut threaded = target.clone();
		pool.install(|| {
			apply_block_householder_transpose_on_the_left_in_place_with_conj(
				a.as_ref(),
				factor.as_ref(),
				Conj::Yes,
				threaded.as_mut(),
				Par::rayon(threads),
				MemStack::new(&mut mem),
			)
		});
		for i in 0..m {
			for j in 0..k {
				let d = (threaded[(i, j)] - sequential[(i, j)]).abs();
				assert!(d < 1e-10, "({i}, {j}): {d}");
			}
		}
	}

	/// The flat branch of `upgrade_householder_factor` builds `T = diag(tau) + strict_upper(V^H V)`,
	/// and its deep half runs on one thread per task unless the reduction's rows are split across
	/// the caller's threads. Both must produce the same `T`, diagonal `tau`s included.
	fn check_upgrade(m: usize, n: usize) {
		let pool = rayon::ThreadPoolBuilder::new()
			.num_threads(4)
			.build()
			.unwrap();
		let basis: Mat<f64> = Mat::from_fn(m, n, |i, j| {
			((i * 3 + j * 7) % 11) as f64 * 0.25 - 1.0
		});
		let taus = |tile: &mut Mat<f64>| {
			for j in 0..n {
				tile[(j, j)] = 1.5 + j as f64;
			}
		};
		let mut sequential = Mat::<f64>::zeros(n, n);
		taus(&mut sequential);
		upgrade_householder_factor(
			sequential.as_mut(),
			basis.as_ref(),
			n,
			1,
			Par::Seq,
		);
		let mut threaded = Mat::<f64>::zeros(n, n);
		taus(&mut threaded);
		pool.install(|| {
			upgrade_householder_factor(
				threaded.as_mut(),
				basis.as_ref(),
				n,
				1,
				Par::rayon(4),
			)
		});
		for col in 0..n {
			for row in 0..n {
				let d = (threaded[(row, col)] - sequential[(row, col)]).abs();
				assert!(d < 1e-12, "({row}, {col}): {d}");
			}
		}
	}

	#[test]
	fn threaded_upgrade_matches_the_sequential_one() {
		check_upgrade(4096, 16);
		check_upgrade(2048, 8);
		check_upgrade(4096, 1);
		// Below the split's floors: the products must still be right.
		check_upgrade(64, 8);
	}

	#[test]
	fn threaded_panel_matches_the_sequential_one() {
		check_panel(2048, 4, 32);
		check_panel(4096, 16, 240);
		// Not tall enough for the row split: the column path must still be right.
		check_panel(128, 8, 16);
		// Short and wide, so the column split takes the whole eight-thread budget
		// (`n_tasks == 8`), which is the shape a blocked QR's trailing update takes.
		check_panel_with(256, 32, 224, 8);
		check_panel_with(512, 48, 464, 8);
	}

	/// The row split slices the reduction's rows, but the top `n` rows are applied through one
	/// `n x n` factor: the split is only valid while a single chunk carries that whole top block.
	/// A panel wider than the row chunk does not -- and it used to trip the triangular shape
	/// assertion (64 factor columns against a 32-row chunk) from `qr_in_place`, which is the abort
	/// [tensor4all/faer#12](https://github.com/tensor4all/faer/issues/12) reports. Both sides of
	/// the threshold must compute the same map.
	#[test]
	fn threaded_panel_narrower_than_the_row_chunk_matches_the_sequential_one() {
		check_panel(256, 128, 64);
	}

	#[test]
	fn threaded_rank_one_matches_the_sequential_one() {
		check_f64(2048, 8);
		check_f64(16384, 1);
		// Small enough to stay on the sequential kernel, which is the same map.
		check_f64(64, 8);
		check_c64(2048, 8);
		check_c64(64, 8);
	}

	/// On both paths the reflector maps its own vector to its negative -- `v^H v = 2 tau` is what
	/// makes `H v = -v` -- which is the one exact property the kernel's convention fixes.
	#[test]
	fn rank_one_negates_its_own_reflector_column() {
		let (m, pool) = (
			2048usize,
			rayon::ThreadPoolBuilder::new()
				.num_threads(4)
				.build()
				.unwrap(),
		);
		let mut a: Mat<f64> =
			Mat::from_fn(m, 1, |i, _| if i % 13 == 0 { 3.0 } else { 0.25 });
		let tau = {
			let mut col = a.as_mut().col_mut(0);
			let (mut head, tail) = col.rb_mut().split_at_row_mut(1);
			let tau = make_householder_in_place(&mut head[0], tail).tau;
			head[0] = 1.0;
			tau
		};
		let factor = Mat::<f64>::from_fn(1, 1, |_, _| tau);
		let mut mem = MemBuffer::new(
			apply_block_householder_transpose_on_the_left_in_place_scratch::<f64>(
				m, 1, 1,
			),
		);
		for par in [Par::Seq, Par::rayon(4)] {
			let mut out = a.clone();
			pool.install(|| {
				apply_block_householder_transpose_on_the_left_in_place_with_conj(
					a.as_ref(),
					factor.as_ref(),
					Conj::Yes,
					out.as_mut(),
					par,
					MemStack::new(&mut mem),
				)
			});
			let err = (0..m)
				.map(|i| (out[(i, 0)] + a[(i, 0)]).abs())
				.fold(0.0, f64::max);
			assert!(err < 1e-12, "{err}");
		}
	}
}
