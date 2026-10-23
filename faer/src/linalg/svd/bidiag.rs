use crate::assert;
use crate::internal_prelude::*;
use linalg::householder::*;
use linalg::matmul::{dot, matmul};

/// width of one blocked panel: `BLOCK_SIZE` columns and rows are reduced with level-2 work
/// against deferred corrections, then the trailing block is brought up to date once with two
/// level-3 products: the block reduction of Dongarra, Hammarling and Sorensen, *Block reduction of
/// matrices to condensed forms for eigenvalue computations* (J. Comput. Appl. Math. 27, 1989), the
/// algorithm LAPACK's `dlabrd`/`dgebrd` implement. the recurrences here are derived from this
/// crate's reflector conventions, not translated from any implementation.
///
/// measured (`docs/worklogs/2026-10-23-blocked-bidiag.md`): the per-step recurrence costs grow
/// with the width while the update's cost shrinks with it, and 8 to 16 is flat at every size
/// and budget tried, with 32 and 48 slower.
const BLOCK_SIZE: usize = 16;
/// the blocked loop runs while the remaining dimension `min(m - p, n - p)` exceeds this floor;
/// the tail below it, and every input with `min(m, n) <= BLOCK_FLOOR`, keeps the unblocked
/// per-step update exactly as it was.
const BLOCK_FLOOR: usize = 128;

/// computes the layout of the workspace required to compute a matrix's
/// bidiagonalization
pub fn bidiag_in_place_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	par: Par,
	params: Spec<BidiagParams, T>,
) -> StackReq {
	_ = par;
	_ = params;
	let tail = StackReq::all_of(&[
		temp_mat_scratch::<T>(nrows, 1),
		temp_mat_scratch::<T>(ncols, 1),
		temp_mat_scratch::<T>(ncols, 1),
		temp_mat_scratch::<T>(ncols, 1),
	]);
	if Ord::min(nrows, ncols) > BLOCK_FLOOR {
		// the correction matrices live through a panel and its trailing update, and are dead
		// before the tail allocates its own pair, so the two phases share the stack
		let blocked = StackReq::all_of(&[
			temp_mat_scratch::<T>(nrows, BLOCK_SIZE),
			temp_mat_scratch::<T>(ncols, BLOCK_SIZE),
			temp_mat_scratch::<T>(BLOCK_SIZE, 2),
			temp_mat_scratch::<T>(ncols, 1),
		]);
		StackReq::any_of(&[blocked, tail])
	} else {
		tail
	}
}
/// bidiagonalization tuning parameters.
#[derive(Debug, Copy, Clone)]
pub struct BidiagParams {
	/// threshold at which parallelism should be disabled
	pub par_threshold: usize,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}
impl<T: ComplexField> Auto<T> for BidiagParams {
	fn auto() -> Self {
		Self {
			par_threshold: 192 * 256,
			non_exhaustive: NonExhaustive(()),
		}
	}
}
/// computes a matrix $A$'s bidiagonalization such that $A = U B V^H$
///
/// $B$ is a bidiagonal matrix stored in $A$'s diagonal and superdiagonal
///
/// $U$ is a sequence of householder reflections stored in the unit lower
/// triangular half of $A$, with the householder coefficients being stored in
/// `H_left`
///
/// $V$ is a sequence of householder reflections stored in the unit upper
/// triangular half of $A$ (excluding the diagonal), with the householder
/// coefficients being stored in `H_right`
pub fn bidiag_in_place<T: ComplexField>(
	A: MatMut<'_, T>,
	H_left: MatMut<'_, T>,
	H_right: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<BidiagParams, T>,
) {
	let params = params.config;
	let m = A.nrows();
	let n = A.ncols();
	let size = Ord::min(m, n);
	let bl = H_left.nrows();
	let br = H_right.nrows();
	assert!(H_left.ncols() == size);
	assert!(H_right.ncols() == size.saturating_sub(1));
	let mut A = A;
	let mut Hl = H_left;
	let mut Hr = H_right;

	// blocked phase: whole panels while the remaining dimension clears the floor. its regions are
	// two vector products per step and one update per panel, and they run fastest when the caller
	// holds the worker group around the call (`faer::spindle::with_lock`, see the crate docs):
	// spinning workers that keep their column range of the trailing block. the library does not
	// take the lock itself; measured at n = 512, 8T: 19.2 ms held against 25.5 ms with plain regions.
	let mut p = 0;
	if size > BLOCK_FLOOR {
		p = bidiag_blocked_phase(
			A.rb_mut(),
			Hl.rb_mut().row_mut(0),
			Hr.rb_mut().row_mut(0),
			par,
			params.par_threshold,
			stack,
		);
	}

	// unblocked tail: the trailing block is already current, so it starts with fresh local
	// indexing and no pending deferred pair
	bidiag_unblocked(
		A.rb_mut().get_mut(p.., p..),
		Hl.rb_mut().row_mut(0).get_mut(p..),
		Hr.rb_mut().row_mut(0).get_mut(p..),
		par,
		stack,
		params.par_threshold,
		p,
	);

	let mut j = 0;
	while j < size {
		let bl = Ord::min(bl, size - j);
		let mut Hl = Hl.rb_mut().get_mut(..bl, j..j + bl);
		for k in 0..bl {
			Hl[(k, k)] = Hl[(0, k)].copy();
		}
		upgrade_householder_factor(
			Hl.rb_mut(),
			A.rb().get(j.., j..j + bl),
			bl,
			1,
			par,
		);
		j += bl;
	}
	if size > 0 {
		let size = size - 1;
		let A = A.rb().get(..size, 1..);
		let mut Hr = Hr.rb_mut().get_mut(.., ..size);
		let mut j = 0;
		while j < size {
			let br = Ord::min(br, size - j);
			let mut Hr = Hr.rb_mut().get_mut(..br, j..j + br);
			for k in 0..br {
				Hr[(k, k)] = Hr[(0, k)].copy();
			}
			upgrade_householder_factor(
				Hr.rb_mut(),
				A.transpose().get(j.., j..j + br),
				br,
				1,
				par,
			);
			j += br;
		}
	}
}

/// the blocked phase: whole panels of `BLOCK_SIZE` while the remaining dimension exceeds
/// `BLOCK_FLOOR`. returns the offset at which the unblocked tail starts.
fn bidiag_blocked_phase<T: ComplexField>(
	A: MatMut<'_, T>,
	Hl: RowMut<'_, T>,
	Hr: RowMut<'_, T>,
	par: Par,
	par_threshold: usize,
	stack: &mut MemStack,
) -> usize {
	let mut A = A;
	let mut Hl = Hl;
	let mut Hr = Hr;
	let m = A.nrows();
	let n = A.ncols();
	let size = Ord::min(m, n);
	let (mut X, stack) =
		unsafe { temp_mat_uninit::<T, _, _>(m, BLOCK_SIZE, stack) };
	let (mut Y, stack) =
		unsafe { temp_mat_uninit::<T, _, _>(n, BLOCK_SIZE, stack) };
	let (mut de, stack) =
		unsafe { temp_mat_uninit::<T, _, _>(BLOCK_SIZE, 2, stack) };
	let (mut w, _) = unsafe { temp_mat_uninit::<T, _, _>(n, 1, stack) };
	let mut X = X.as_mat_mut();
	let mut Y = Y.as_mat_mut();
	let mut de = de.as_mat_mut();
	let mut w = w.as_mat_mut();
	let mut p = 0;
	while size - p > BLOCK_FLOOR {
		// the decision is made on the full remaining area, never latched by a panel
		let par = if (m - p) * (n - p) < par_threshold {
			Par::Seq
		} else {
			par
		};
		bidiag_blocked_panel(
			A.rb_mut().get_mut(p.., p..),
			Hl.rb_mut().get_mut(p..p + BLOCK_SIZE),
			Hr.rb_mut().get_mut(p..p + BLOCK_SIZE),
			X.rb_mut().get_mut(p.., ..),
			Y.rb_mut().get_mut(p.., ..),
			de.rb_mut(),
			w.rb_mut().col_mut(0).get_mut(p..),
			par,
		);
		p += BLOCK_SIZE;
	}
	p
}

/// reduces the first `ib` columns and rows of `A` (`ib = X.ncols()`), then updates the trailing
/// block `A[ib.., ib..]` once with two level-3 products.
///
/// conventions, the ones the rest of this file uses: the left reflector of step `i` is the column
/// `u_i = A[i.., i]` with unit head and `H_i = I - u_i u_i^H / tl_i`; the right one is the row
/// `r_i = A[i, i + 1..]` with unit head and `G_i = I - r_i^H r_i / tr_i`. after `ib` steps
///
/// `A_new = A - V Y^H - X R`
///
/// where `V = [u_0 .. u_ib-1]`, `Y[.., i] = A_i^H u_i / tl_i` (`A_i` is the matrix after `i`
/// full steps), `R = [r_0; ..; r_ib-1]` and `X[.., i] = A_i' r_i^H / tr_i` (`A_i'` is `A_i` after
/// the left reflection of step `i`). `Y[.., i]` is meaningful at rows `> i`, `X[.., i]` at rows
/// `> i`; the rows at or above the diagonal hold the recurrence's short temporaries, as in
/// LAPACK's `dlabrd`. the reflector heads stay `1` in `A` during the panel and the update, and
/// the diagonal and superdiagonal they replace are kept in `de` and written back at the end.
///
/// the row of each step is gathered into the contiguous `w` (length `n`) for its update, its
/// reflector and the products it is the vector of, and scattered back once: a row of `A` has a
/// stride of `m` elements, and both the vector products and the reflector norm want a contiguous
/// operand.
fn bidiag_blocked_panel<T: ComplexField>(
	A: MatMut<'_, T>,
	Hl: RowMut<'_, T>,
	Hr: RowMut<'_, T>,
	X: MatMut<'_, T>,
	Y: MatMut<'_, T>,
	de: MatMut<'_, T>,
	w: ColMut<'_, T>,
	par: Par,
) {
	let mut A = A;
	let mut w = w;
	let mut Hl = Hl;
	let mut Hr = Hr;
	let mut X = X;
	let mut Y = Y;
	let mut de = de;
	let m = A.nrows();
	let n = A.ncols();
	let ib = X.ncols();
	assert!(all(
		Y.ncols() == ib,
		X.nrows() == m,
		Y.nrows() == n,
		Ord::min(m, n) > ib,
		Hl.ncols() == ib,
		Hr.ncols() == ib,
	));

	for i in 0..ib {
		// 1. column i, rows i.., against the corrections of the steps before it:
		//    A[i.., i] -= V[i.., ..i] Y[i, ..i]^H + X[i.., ..i] R[..i, i]
		let tl = {
			let (left, right) = A.rb_mut().split_at_col_mut(i);
			let (top, mut col) = right.col_mut(0).split_at_row_mut(i);
			if i > 0 {
				matmul(
					col.rb_mut(),
					Accum::Add,
					left.rb().get(i.., ..),
					Y.rb().row(i).get(..i).adjoint(),
					-one::<T>(),
					Par::Seq,
				);
				matmul(
					col.rb_mut(),
					Accum::Add,
					X.rb().get(i.., ..i),
					top.rb(),
					-one::<T>(),
					Par::Seq,
				);
			}
			// 2. left reflector
			let (mut head, tail) = col.split_at_row_mut(1);
			let head = &mut head[0];
			let HouseholderInfo { tau: tl, .. } =
				make_householder_in_place(head, tail);
			de[(i, 0)] = head.copy();
			*head = one();
			Hl[i] = tl.to_cplx();
			tl
		};
		let ref tl_inv = tl.recip();

		// 3. Y[i+1.., i] = A_i[i.., i+1..]^H u_i / tl_i, with A_i expressed through the stored
		//    A and the corrections:
		//    A[i.., i+1..]^H u - Y[i+1.., ..i] (V[i.., ..i]^H u) - R[..i, i+1..]^H (X[i.., ..i]^H u)
		{
			let u = A.rb().col(i).get(i..);
			let (y_prev, y_i) = Y.rb_mut().split_at_col_mut(i);
			let (mut t, mut y) = y_i.col_mut(0).split_at_row_mut(i + 1);
			matmul(
				y.rb_mut(),
				Accum::Replace,
				A.rb().get(i.., i + 1..).adjoint(),
				u,
				one(),
				par,
			);
			if i > 0 {
				let mut t = t.rb_mut().get_mut(..i);
				matmul(
					t.rb_mut(),
					Accum::Replace,
					A.rb().get(i.., ..i).adjoint(),
					u,
					one(),
					Par::Seq,
				);
				matmul(
					y.rb_mut(),
					Accum::Add,
					y_prev.rb().get(i + 1.., ..),
					t.rb(),
					-one::<T>(),
					Par::Seq,
				);
				matmul(
					t.rb_mut(),
					Accum::Replace,
					X.rb().get(i.., ..i).adjoint(),
					u,
					one(),
					Par::Seq,
				);
				matmul(
					y.rb_mut(),
					Accum::Add,
					A.rb().get(..i, i + 1..).adjoint(),
					t.rb(),
					-one::<T>(),
					Par::Seq,
				);
			}
			z!(y.rb_mut()).for_each(|uz!(y)| *y = y.mul_real(tl_inv));
		}

		// 4. row i, columns i+1.., after the left reflection of this step and the corrections
		//    of the steps before it, computed on the contiguous gather `w = A[i, i+1..]^T`:
		//    w -= conj(Y[i+1.., ..i+1]) V[i, ..i+1]^T + R[..i, i+1..]^T X[i, ..i]^T
		let tr = {
			let (top, bot) = A.rb_mut().split_at_row_mut(i);
			let (left, mut row) = bot.row_mut(0).split_at_col_mut(i + 1);
			let mut w = w.rb_mut().get_mut(i + 1..);
			w.copy_from(row.rb().transpose());
			matmul(
				w.rb_mut(),
				Accum::Add,
				Y.rb().get(i + 1.., ..i + 1).conjugate(),
				left.rb().transpose(),
				-one::<T>(),
				Par::Seq,
			);
			if i > 0 {
				matmul(
					w.rb_mut(),
					Accum::Add,
					top.rb().get(.., i + 1..).transpose(),
					X.rb().row(i).get(..i).transpose(),
					-one::<T>(),
					Par::Seq,
				);
			}
			// 5. right reflector, on the gather, then scattered back into the row
			let (mut head, tail) = w.rb_mut().split_at_row_mut(1);
			let head = &mut head[0];
			let HouseholderInfo { tau: tr, .. } =
				make_householder_in_place(head, tail);
			de[(i, 1)] = head.copy();
			*head = one();
			Hr[i] = tr.to_cplx();
			row.rb_mut().transpose_mut().copy_from(w.rb());
			tr
		};
		let ref tr_inv = tr.recip();

		// 6. X[i+1.., i] = A_i'[i+1.., i+1..] r_i^H / tr_i, with `r^H = conj(w)`:
		//    A[i+1.., i+1..] r^H - V[i+1.., ..i+1] (Y[i+1.., ..i+1]^H r^H) - X[i+1.., ..i] (R[..i, i+1..] r^H)
		{
			let r = w.rb().get(i + 1..).conjugate();
			let (x_prev, x_i) = X.rb_mut().split_at_col_mut(i);
			let (mut t, mut x) = x_i.col_mut(0).split_at_row_mut(i + 1);
			matmul(
				x.rb_mut(),
				Accum::Replace,
				A.rb().get(i + 1.., i + 1..),
				r,
				one(),
				par,
			);
			matmul(
				t.rb_mut(),
				Accum::Replace,
				Y.rb().get(i + 1.., ..i + 1).adjoint(),
				r,
				one(),
				Par::Seq,
			);
			matmul(
				x.rb_mut(),
				Accum::Add,
				A.rb().get(i + 1.., ..i + 1),
				t.rb(),
				-one::<T>(),
				Par::Seq,
			);
			if i > 0 {
				let mut t = t.rb_mut().get_mut(..i);
				matmul(
					t.rb_mut(),
					Accum::Replace,
					A.rb().get(..i, i + 1..),
					r,
					one(),
					Par::Seq,
				);
				matmul(
					x.rb_mut(),
					Accum::Add,
					x_prev.rb().get(i + 1.., ..),
					t.rb(),
					-one::<T>(),
					Par::Seq,
				);
			}
			z!(x.rb_mut()).for_each(|uz!(x)| *x = x.mul_real(tr_inv));
		}
	}

	// trailing update, once per panel: A[ib.., ib..] -= V[ib.., ..] Y[ib.., ..]^H + X[ib.., ..] R[.., ib..]
	{
		let (top, bot) = A.rb_mut().split_at_row_mut(ib);
		let (v, mut a22) = bot.split_at_col_mut(ib);
		matmul(
			a22.rb_mut(),
			Accum::Add,
			v.rb(),
			Y.rb().get(ib.., ..).adjoint(),
			-one::<T>(),
			par,
		);
		matmul(
			a22.rb_mut(),
			Accum::Add,
			X.rb().get(ib.., ..),
			top.rb().get(.., ib..),
			-one::<T>(),
			par,
		);
	}
	// the heads were 1 for the products; put the bidiagonal back
	for i in 0..ib {
		A[(i, i)] = de[(i, 0)].copy();
		A[(i, i + 1)] = de[(i, 1)].copy();
	}
}

/// the unblocked bidiagonalization of `A`, one reflector pair per step with the trailing update
/// deferred by one step and fused with the next step's product. `row_offset` is `A`'s row
/// position in the matrix it is a tail of, used only as a SIMD alignment hint.
///
/// the row of each step lives in a contiguous buffer while it is worked on: a row of `A` has a
/// stride of `m`, and the deferred update, the norm, the scaling, the right reflector and the
/// inner product each walk it once. `w` holds the current row (columns `k + 1..`), `wp` the
/// previous step's row reflector, which the deferred update and the fused product read; the row
/// is gathered once at the start of the step and scattered back once at its end.
fn bidiag_unblocked<T: ComplexField>(
	A: MatMut<'_, T>,
	Hl: RowMut<'_, T>,
	Hr: RowMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
	par_threshold: usize,
	row_offset: usize,
) {
	let m = A.nrows();
	let n = A.ncols();
	let size = Ord::min(m, n);
	alloca!('stack: {
		let mut y = unsafe { row![uninit::<T>, n] };
		let mut z = unsafe { col![uninit::<T>, m] };
		let mut w0 = unsafe { row![uninit::<T>, n] };
		let mut w1 = unsafe { row![uninit::<T>, n] };
	});
	let mut A = A;
	let mut Hl = Hl;
	let mut Hr = Hr;
	let mut par = par;
	let mut w = w0.rb_mut();
	let mut wp = w1.rb_mut();
	for k in 0..size {
		let mut A = A.rb_mut();
		let (_, _, A10, A11) = A.rb_mut().split_at_mut(k, k);
		let (A10, A20) = A10.split_first_row_mut().unwrap();
		let (mut A11, A12, A21, mut A22) = A11.split_at_mut(1, 1);
		let mut A12 = A12.row_mut(0);
		let mut A21 = A21.col_mut(0);
		let a11 = &mut A11[(0, 0)];
		let (y1, mut y2) = y.rb_mut().split_at_col_mut(k).1.split_at_col_mut(1);
		let (z1, mut z2) = z.rb_mut().split_at_row_mut(k).1.split_at_row_mut(1);
		// the row of this step, contiguous, and the previous step's row reflector
		let mut wk = w.rb_mut().split_at_col_mut(k + 1).1;
		wk.copy_from(A12.rb());
		let vp = wp.rb().split_at_col(k + 1).1;
		if k > 0 {
			let ref y1 = y1[0].copy();
			let ref z1 = z1[0].copy();
			let k1 = k - 1;
			let ref up0 = A10[k1];
			let up = A20.rb().col(k1);
			*a11 -= up0 * y1 + z1;
			z!(A21.rb_mut(), up.rb(), z2.rb()).for_each(
				|uz!(a, u, z): Zip!(&mut T, &T, &T)| *a -= u * y1 + z,
			);
			z!(wk.rb_mut(), y2.rb(), vp).for_each(
				|uz!(a, y, v): Zip!(&mut T, &T, &T)| {
					*a -= up0 * y + z1 * v;
				},
			);
		}
		let HouseholderInfo { tau: tl, .. } =
			make_householder_in_place(a11, A21.rb_mut());
		let ref tl_inv = tl.recip();
		Hl[k] = tl.to_cplx();
		if (m - k - 1) * (n - k - 1) < par_threshold {
			par = Par::Seq;
		}
		if k > 0 {
			let k1 = k - 1;
			let up = A20.rb().col(k1);
			match par {
				Par::Seq => bidiag_fused_op(
					A22.rb_mut(),
					A21.rb(),
					up.rb(),
					z2.rb(),
					y2.rb_mut(),
					vp,
					simd_align(row_offset + k + 1),
				),
				#[cfg(feature = "rayon")]
				Par::Rayon(nthreads) => {
					use rayon::prelude::*;
					let nthreads = nthreads.get();
					spindle::for_each(
						nthreads,
						A22.rb_mut()
							.par_col_partition_mut(nthreads)
							.zip_eq(y2.rb_mut().par_partition_mut(nthreads))
							.zip_eq(vp.par_partition(nthreads)),
						|((A22, y2), vp)| {
							bidiag_fused_op(
								A22,
								A21.rb(),
								up.rb(),
								z2.rb(),
								y2,
								vp.rb(),
								simd_align(row_offset + k + 1),
							);
						},
					);
				},
			}
		} else {
			matmul(
				y2.rb_mut(),
				Accum::Replace,
				A21.rb().adjoint(),
				A22.rb(),
				one(),
				par,
			);
		}
		z!(y2.rb_mut(), wk.rb_mut()).for_each(|uz!(y, a)| {
			*y = (&*y + &*a).mul_real(tl_inv);
			*a -= &*y;
		});
		// a subnormal row norm has an infinite reciprocal, so rescale the row first: reference
		// `dlarfg` does the same when its normaliser underflows (main's fix, carried onto the
		// contiguous row). the row's own norm is kept for the head's restoration below.
		let norm = wk.rb().norm_l2();
		if norm != zero() {
			let ref norm_inv = if norm >= min_positive() {
				(&norm).recip()
			} else {
				let ref shift = from_f64::<T::Real>(4503599627370496.0);
				z!(wk.rb_mut()).for_each(|uz!(a)| *a = a.mul_real(shift));
				(&norm * shift).recip()
			};
			z!(wk.rb_mut()).for_each(|uz!(a)| *a = a.mul_real(norm_inv));
		}
		matmul(
			z2.rb_mut(),
			Accum::Replace,
			A22.rb(),
			wk.rb().adjoint(),
			one(),
			par,
		);
		if k + 1 == size {
			A12.copy_from(wk.rb());
			break;
		}
		let (mut wk_a, mut wk_b) = wk.rb_mut().split_at_col_mut(1);
		let A22_a = A22.rb().col(0);
		let (y2_a, y2_b) = y2.rb().split_at_col(1);
		let y2_a = &y2_a[0];
		let a12_a = &mut wk_a[0];
		let HouseholderInfo {
			tau: tr,
			head_with_beta_inv: m,
			..
		} = make_householder_in_place(a12_a, wk_b.rb_mut().transpose_mut());
		let ref tr_inv = tr.recip();
		Hr[k] = tr.to_cplx();
		let beta = a12_a.copy();
		*a12_a = a12_a.mul_real(norm);
		let ref b = y2_a
			+ dot::inner_prod(y2_b, Conj::No, wk_b.rb().transpose(), Conj::Yes);
		if m != infinity() {
			z!(z2.rb_mut(), A21.rb(), A22_a.rb()).for_each(
				|uz!(z, u, a): Zip!(&mut T, &T, &T)| {
					let w = &*z - a * beta.conj();
					let w = w * m.conj();
					let w = w - u * b;
					*z = w.mul_real(tr_inv);
				},
			);
		} else {
			z!(z2.rb_mut(), A21.rb(), A22_a.rb()).for_each(
				|uz!(z, u, a): Zip!(&mut T, &T, &T)| {
					let w = a - u * b;
					*z = w.mul_real(tr_inv);
				},
			);
		}
		// the row goes back into `A`, where it is the stored right reflector; the next step reads
		// it from `wp`
		A12.copy_from(wk.rb());
		core::mem::swap(&mut w, &mut wp);
	}
}
fn bidiag_fused_op<T: ComplexField>(
	A22: MatMut<'_, T>,
	u: ColRef<'_, T>,
	up: ColRef<'_, T>,
	z: ColRef<'_, T>,
	y: RowMut<'_, T>,
	vp: RowRef<'_, T>,
	align: usize,
) {
	let mut A22 = A22;
	if const { T::SIMD_CAPABILITIES.is_simd() } {
		if let (Some(A22), Some(u), Some(up), Some(z)) = (
			A22.rb_mut().try_as_col_major_mut(),
			u.try_as_col_major(),
			up.try_as_col_major(),
			z.try_as_col_major(),
		) {
			bidiag_fused_op_simd(A22, u, up, z, y, vp, align);
		} else {
			bidiag_fused_op_fallback(A22, u, up, z, y, vp);
		}
	} else {
		bidiag_fused_op_fallback(A22, u, up, z, y, vp);
	}
}
fn bidiag_fused_op_fallback<T: ComplexField>(
	A22: MatMut<'_, T>,
	u: ColRef<'_, T>,
	up: ColRef<'_, T>,
	z: ColRef<'_, T>,
	y: RowMut<'_, T>,
	vp: RowRef<'_, T>,
) {
	let mut A22 = A22;
	let mut y = y;
	matmul(A22.rb_mut(), Accum::Add, up, y.rb(), -one::<T>(), Par::Seq);
	matmul(A22.rb_mut(), Accum::Add, z, vp, -one::<T>(), Par::Seq);
	matmul(
		y.rb_mut(),
		Accum::Replace,
		u.adjoint(),
		A22.rb(),
		one(),
		Par::Seq,
	);
}
fn bidiag_fused_op_simd<'M, 'N, T: ComplexField>(
	A22: MatMut<'_, T, usize, usize, ContiguousFwd>,
	u: ColRef<'_, T, usize, ContiguousFwd>,
	up: ColRef<'_, T, usize, ContiguousFwd>,
	z: ColRef<'_, T, usize, ContiguousFwd>,
	y: RowMut<'_, T, usize>,
	vp: RowRef<'_, T, usize>,
	align: usize,
) {
	struct Impl<'a, 'M, 'N, T: ComplexField> {
		A22: MatMut<'a, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
		u: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
		up: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
		z: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
		y: RowMut<'a, T, Dim<'N>>,
		vp: RowRef<'a, T, Dim<'N>>,
		align: usize,
	}
	impl<'a, 'M, 'N, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'N, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
			let Self {
				mut A22,
				u,
				up,
				z,
				mut y,
				vp,
				align,
			} = self;
			let m = A22.nrows();
			let n = A22.ncols();
			let simd = SimdCtx::<T, S>::new_align(T::simd_ctx(simd), m, align);
			let indices = simd.batch_indices::<4>();
			for j in n.indices() {
				let mut A = A22.rb_mut().col_mut(j);
				let mut acc = [simd.zero(); 4];
				let yj = simd.splat(-&y[j]);
				let vj = simd.splat(-&vp[j]);
				simd_iter!(for (IDX, i) in [indices; 4] {
					let mut a = simd.read(A.rb(), i);
					a = simd.mul_add(simd.read(up, i), yj, a);
					a = simd.mul_add(simd.read(z, i), vj, a);
					simd.write(A.rb_mut(), i, a);
					acc[IDX] = simd.conj_mul_add(simd.read(u, i), a, acc[IDX]);
				});
				let acc0 = simd.add(acc[0], acc[1]);
				let acc2 = simd.add(acc[2], acc[3]);
				let acc0 = simd.add(acc0, acc2);
				y[j] = simd.reduce_sum(acc0);
			}
		}
	}
	with_dim!(M, A22.nrows());
	with_dim!(N, A22.ncols());
	dispatch!(
		Impl {
			A22: A22.as_shape_mut(M, N),
			u: u.as_row_shape(M),
			up: up.as_row_shape(M),
			z: z.as_row_shape(M),
			y: y.as_col_shape_mut(N),
			vp: vp.as_col_shape(N),
			align,
		},
		Impl,
		T
	)
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, assert, c64};
	use dyn_stack::MemBuffer;

	/// bidiagonalizes `A` and rebuilds it from the factors
	fn reconstruct<T: ComplexField>(
		A: &Mat<T>,
		par: Par,
		params: BidiagParams,
	) -> (Mat<T>, Mat<T>) {
		let m = A.nrows();
		let n = A.ncols();
		let size = Ord::min(m, n);
		let bl = 4;
		let br = 3;
		let mut Hl = Mat::zeros(bl, size);
		let mut Hr = Mat::zeros(br, size - 1);
		let mut UV = A.clone();
		bidiag_in_place(
			UV.rb_mut(),
			Hl.rb_mut(),
			Hr.rb_mut(),
			par,
			MemStack::new(&mut MemBuffer::new(bidiag_in_place_scratch::<T>(
				m,
				n,
				par,
				params.into(),
			))),
			params.into(),
		);
		let mut B = A.clone();
		let mut B = B.as_mut();
		apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
			UV.rb().get(.., ..size),
			Hl.rb(),
			Conj::Yes,
			B.rb_mut(),
			par,
			MemStack::new(&mut MemBuffer::new(
				apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(
					n - 1, 1, m,
				),
			)),
		);
		let V = UV.rb().get(..size - 1, 1..size);
		let B1 = B.rb_mut().get_mut(.., 1..size);
		apply_block_householder_sequence_on_the_right_in_place_with_conj(
			V.transpose(),
			Hr.rb(),
			Conj::Yes,
			B1,
			par,
			MemStack::new(&mut MemBuffer::new(
				apply_block_householder_sequence_on_the_right_in_place_scratch::<
					T,
				>(n - 1, 1, m),
			)),
		);
		for j in 0..n {
			for i in 0..m {
				if i > j || j > i + 1 {
					UV[(i, j)] = zero();
				}
			}
		}
		(UV, B.to_owned())
	}

	/// the shapes the tests cover: the two historical small ones (unblocked), and three that
	/// cross at least two blocked panels before the unblocked tail
	const SHAPES: [(usize, usize); 5] =
		[(8, 4), (8, 8), (300, 300), (400, 300), (200, 200)];

	fn budgets() -> alloc::vec::Vec<(Par, BidiagParams)> {
		let seq = (Par::Seq, <BidiagParams as Auto<f64>>::auto());
		#[cfg(feature = "rayon")]
		{
			// the default threshold (192 * 256) leaves a 199^2 trailing area sequential, so the
			// threaded case lowers it to exercise the threaded panel update and tail
			let mut threaded = <BidiagParams as Auto<f64>>::auto();
			threaded.par_threshold = 0;
			alloc::vec![seq, (Par::rayon(4), threaded)]
		}
		#[cfg(not(feature = "rayon"))]
		{
			alloc::vec![seq]
		}
	}

	#[test]
	fn test_bidiag_real() {
		let rng = &mut StdRng::seed_from_u64(0);
		for (m, n) in SHAPES {
			let A = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			for (par, params) in budgets() {
				let (UV, B) = reconstruct(&A, par, params);
				let approx_eq =
					CwiseMat(ApproxEq::<f64>::eps() * ((m * n) as f64).sqrt());
				assert!(UV ~ B);
			}
		}
	}
	#[test]
	fn test_bidiag_cplx() {
		let rng = &mut StdRng::seed_from_u64(0);
		for (m, n) in SHAPES {
			let A = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);
			for (par, params) in budgets() {
				let (UV, B) = reconstruct(&A, par, params);
				let approx_eq =
					CwiseMat(ApproxEq::<f64>::eps() * ((m * n) as f64).sqrt());
				assert!(UV ~ B);
			}
		}
	}
}
