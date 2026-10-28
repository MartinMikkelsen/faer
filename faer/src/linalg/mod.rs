//! linear algebra module
//!
//! contains low level routines and the implementation of their corresponding
//! high level wrappers
//!
//! # memory allocation
//! since most `faer` crates aim to expose a low level api for optimal
//! performance, most algorithms try to defer memory allocation to the user
//!
//! however, since a lot of algorithms need some form of temporary space for
//! intermediate computations, they may ask for a slice of memory for that
//! purpose, by taking a [`stack: MemStack`](dyn_stack::MemStack) parameter. a
//! [`MemStack`] is a thin wrapper over a slice of memory bytes. this memory may
//! come from any valid source (heap allocation, fixed-size array on the stack,
//! etc.). the functions taking a [`MemStack`] parameter have a corresponding
//! function with a similar name ending in `_scratch` that returns the memory
//! requirements of the algorithm. for example:
//! [`householder::apply_block_householder_on_the_left_in_place_with_conj`] and
//! [`householder::apply_block_householder_on_the_left_in_place_scratch`]
//!
//! the memory stack may be reused in user-code to avoid repeated allocations,
//! and it is also possible to compute the sum ([`dyn_stack::StackReq::all_of`])
//! or union ([`dyn_stack::StackReq::any_of`]) of multiple scratch requirements,
//! in order to optimally combine them into a single allocation
//!
//! after computing a [`dyn_stack::StackReq`], one can query its layout to
//! allocate the required memory. the simplest way to do so is through
//! [`dyn_stack::MemBuffer::new`]
use crate::Shape;
use crate::internal_prelude::*;
use crate::mat::matown::align_for;
use crate::mat::{AsMatMut, MatMut};
use core::marker::PhantomData;
use dyn_stack::StackReq;
use faer_traits::ComplexField;
/// returns the stack requirements for creating a temporary matrix with the
/// given dimensions.
pub fn temp_mat_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
) -> StackReq {
	let align = align_for(
		core::mem::size_of::<T>(),
		core::mem::align_of::<T>(),
		core::mem::needs_drop::<T>(),
	);
	let mut col_stride = nrows;
	if align > core::mem::size_of::<T>() {
		col_stride =
			col_stride.msrv_next_multiple_of(align / core::mem::size_of::<T>());
	}
	let len = col_stride.checked_mul(ncols).unwrap();
	StackReq::new_aligned::<T>(len, align)
}
/// Copies `src` into `dst` elementwise, where the two have transposed strides:
/// one is column-major, the other row-major, over the same shape, in tiles that
/// are column-partitioned between threads when the caller asked for them.
/// Whether adapting a row-major input to a column-major copy pays for this shape.
///
/// The adaptation moves `nrows * ncols` elements contiguously twice and buys a
/// contiguous panel; on a small matrix the copy and the thread-region start-up
/// cost more than the strided panel they replace.
///
/// The floor is `64 * 64`, and it comes from the pair measured on 2026-10-19: at
/// `32 x 32` the copy cost 1.8x on `qr_factor` (row-major f64: 0.042 ms against
/// 0.024 ms at 8T), while at `64 x 64` and `96 x 96` removing it cost 1.10-1.31x
/// on the same family, and `lu`/`solve` were flat at both sizes. So the copy
/// pays from `4096` elements up and loses below it.
///
/// It is a floor on elements, not on a side, because the cost of the copy is the
/// area and the tall shapes the change was made for (`4096 x 256`, `1024 x 64`)
/// are asymmetric.
#[inline]
pub(crate) fn row_major_copy_pays(nrows: usize, ncols: usize) -> bool {
	nrows.saturating_mul(ncols) >= 64 * 64
}
pub(crate) fn copy_across_layouts<T: ComplexField>(
	mut dst: MatMut<'_, T>,
	src: MatRef<'_, T>,
	par: Par,
) {
	#[cfg(feature = "rayon")]
	{
		use rayon::prelude::*;
		let n_tasks = Ord::min(
			crate::utils::thread::parallelism_degree(&par),
			dst.ncols(),
		)
		.max(1);
		if n_tasks > 1 {
			crate::utils::thread::for_each(
				par,
				n_tasks,
				dst.rb_mut()
					.par_col_partition_mut(n_tasks)
					.zip_eq(src.par_col_partition(n_tasks)),
				|(mut d, s)| copy_across_layouts_block(d.rb_mut(), s),
			);
			return;
		}
	}
	let _ = par;
	copy_across_layouts_block(dst.rb_mut(), src);
}
/// One `16 x 16` tile of the copy, on one thread: read the tile along whichever
/// side is row-major and write it along whichever is column-major, so neither
/// side steps by the other's row stride. Each source value is cloned into an
/// `Option` tile and moved into the destination, without another clone or a
/// seed value. Every slot the second loop reads was written by the first.
fn copy_across_layouts_block<T: Clone>(
	mut dst: MatMut<'_, T>,
	src: MatRef<'_, T>,
) {
	const B: usize = 16;
	let (m, n) = dst.shape();
	// `src` row-major means `dst` is the column-major one, and the other way
	// round; exactly one of the two holds.
	let src_row_major = src.row_stride().unsigned_abs() != 1;
	let mut tile: [Option<T>; B * B] = core::array::from_fn(|_| None);
	let mut jb = 0;
	while jb < n {
		let j1 = Ord::min(jb + B, n);
		let mut ib = 0;
		while ib < m {
			let i1 = Ord::min(ib + B, m);
			if src_row_major {
				for i in ib..i1 {
					for j in jb..j1 {
						tile[(i - ib) * B + (j - jb)] =
							Some(src[(i, j)].clone());
					}
				}
				for j in jb..j1 {
					for i in ib..i1 {
						dst[(i, j)] =
							tile[(i - ib) * B + (j - jb)].take().unwrap();
					}
				}
			} else {
				for j in jb..j1 {
					for i in ib..i1 {
						tile[(i - ib) * B + (j - jb)] =
							Some(src[(i, j)].clone());
					}
				}
				for i in ib..i1 {
					for j in jb..j1 {
						dst[(i, j)] =
							tile[(i - ib) * B + (j - jb)].take().unwrap();
					}
				}
			}
			ib += B;
		}
		jb += B;
	}
}
struct DynMat<'a, T: ComplexField, Rows: Shape, Cols: Shape> {
	ptr: *mut T,
	nrows: Rows,
	ncols: Cols,
	col_stride: usize,
	__marker: PhantomData<(&'a T, T)>,
}
impl<'a, T: ComplexField, Rows: Shape, Cols: Shape> Drop
	for DynMat<'a, T, Rows, Cols>
{
	#[inline]
	fn drop(&mut self) {
		unsafe {
			core::ptr::drop_in_place(core::slice::from_raw_parts_mut(
				self.ptr,
				self.col_stride * self.ncols.unbound(),
			))
		};
	}
}
impl<'a, T: ComplexField, Rows: Shape, Cols: Shape> AsMatRef
	for DynMat<'a, T, Rows, Cols>
{
	type Cols = Cols;
	type Owned = Mat<T, Rows, Cols>;
	type Rows = Rows;
	type T = T;

	fn as_mat_ref(&self) -> crate::mat::MatRef<'_, T, Rows, Cols> {
		unsafe {
			MatRef::from_raw_parts(
				self.ptr as *const T,
				self.nrows,
				self.ncols,
				1,
				self.col_stride as isize,
			)
		}
	}
}
impl<'a, T: ComplexField, Rows: Shape, Cols: Shape> AsMatMut
	for DynMat<'a, T, Rows, Cols>
{
	fn as_mat_mut(&mut self) -> crate::mat::MatMut<'_, T, Rows, Cols> {
		unsafe {
			MatMut::from_raw_parts_mut(
				self.ptr,
				self.nrows,
				self.ncols,
				1,
				self.col_stride as isize,
			)
		}
	}
}
struct DropGuard<T> {
	ptr: *mut T,
	len: usize,
}
impl<T> Drop for DropGuard<T> {
	#[inline]
	fn drop(&mut self) {
		unsafe {
			core::ptr::drop_in_place(core::slice::from_raw_parts_mut(
				self.ptr, self.len,
			))
		};
	}
}
/// creates a temporary matrix of uninit values, from the given memory stack.
#[track_caller]
pub unsafe fn temp_mat_uninit<
	'a,
	T: ComplexField + 'a,
	Rows: Shape + 'a,
	Cols: Shape + 'a,
>(
	nrows: Rows,
	ncols: Cols,
	stack: &'a mut MemStack,
) -> (
	impl 'a + AsMatMut<T = T, Rows = Rows, Cols = Cols>,
	&'a mut MemStack,
) {
	let align = align_for(
		core::mem::size_of::<T>(),
		core::mem::align_of::<T>(),
		core::mem::needs_drop::<T>(),
	);
	let mut col_stride = nrows.unbound();
	if align > core::mem::size_of::<T>() {
		col_stride =
			col_stride.msrv_next_multiple_of(align / core::mem::size_of::<T>());
	}
	let len = col_stride.checked_mul(ncols.unbound()).unwrap();
	let (uninit, stack) = stack.make_aligned_uninit::<T>(len, align);
	let ptr = uninit.as_mut_ptr() as *mut T;
	if core::mem::needs_drop::<T>() {
		unsafe {
			let mut guard = DropGuard { ptr, len: 0 };
			for j in 0..len {
				let ptr = ptr.add(j);
				let val = T::nan_impl();
				ptr.write(val);
				guard.len += 1;
			}
			core::mem::forget(guard);
		}
	}
	(
		DynMat {
			ptr,
			nrows,
			ncols,
			col_stride,
			__marker: PhantomData,
		},
		stack,
	)
}
/// creates a temporary matrix of zero values, from the given memory stack.
#[track_caller]
pub fn temp_mat_zeroed<
	'a,
	T: ComplexField + 'a,
	Rows: Shape + 'a,
	Cols: Shape + 'a,
>(
	nrows: Rows,
	ncols: Cols,
	stack: &'a mut MemStack,
) -> (
	impl 'a + AsMatMut<T = T, Rows = Rows, Cols = Cols>,
	&'a mut MemStack,
) {
	let align = align_for(
		core::mem::size_of::<T>(),
		core::mem::align_of::<T>(),
		core::mem::needs_drop::<T>(),
	);
	let mut col_stride = nrows.unbound();
	if align > core::mem::size_of::<T>() {
		col_stride =
			col_stride.msrv_next_multiple_of(align / core::mem::size_of::<T>());
	}
	let len = col_stride.checked_mul(ncols.unbound()).unwrap();
	_ = stack.make_aligned_uninit::<T>(len, align);
	let (uninit, stack) = stack.make_aligned_uninit::<T>(len, align);
	let ptr = uninit.as_mut_ptr() as *mut T;
	unsafe {
		let mut guard = DropGuard { ptr, len: 0 };
		for j in 0..len {
			let ptr = ptr.add(j);
			let val = T::zero_impl();
			ptr.write(val);
			guard.len += 1;
		}
		core::mem::forget(guard);
	}
	(
		DynMat {
			ptr,
			nrows,
			ncols,
			col_stride,
			__marker: PhantomData,
		},
		stack,
	)
}
pub mod cholesky;
pub mod evd;
pub mod gevd;
pub mod householder;
/// jacobi rotation matrix
pub mod jacobi;
/// kronecker product
pub mod kron;
pub mod lu;
mod mat_ops;
pub mod matmul;
pub mod qr;
pub(crate) mod reductions;
/// high level solvers
pub mod solvers;
pub mod svd;
/// triangular matrix inverse
pub mod triangular_inverse;
/// triangular matrix solve
pub mod triangular_solve;
/// matrix zipping implementation
pub mod zip;
#[cfg(test)]
mod tests {
	use super::*;
	use dyn_stack::MemBuffer;

	/// A scalar the fixture can build and two results can be compared with
	/// exactly -- exact because every entry point below runs the same arithmetic
	/// on the same values once the copy has happened.
	trait Entry: ComplexField {
		fn at(i: usize, j: usize) -> Self;
		fn same(&self, other: &Self) -> bool;
		fn close(&self, other: &Self) -> bool;
	}
	impl Entry for f64 {
		fn at(i: usize, j: usize) -> Self {
			val(i, j)
		}
		fn same(&self, other: &Self) -> bool {
			self.to_bits() == other.to_bits()
		}
		fn close(&self, other: &Self) -> bool {
			// `inf` marks an unused diagonal entry and must compare equal to
			// itself; subtraction would make it `NaN`.
			if self.same(other) {
				return true;
			}
			(self - other).abs() <= 1e-10 * self.abs().max(other.abs()).max(1.0)
		}
	}
	impl Entry for crate::c64 {
		fn at(i: usize, j: usize) -> Self {
			crate::c64::new(val(i, j), val(i + 7, j + 3))
		}
		fn same(&self, other: &Self) -> bool {
			self.re.to_bits() == other.re.to_bits()
				&& self.im.to_bits() == other.im.to_bits()
		}
		fn close(&self, other: &Self) -> bool {
			if self.same(other) {
				return true;
			}
			(self.re - other.re).abs()
				<= 1e-10 * self.re.abs().max(other.re.abs()).max(1.0)
				&& (self.im - other.im).abs()
					<= 1e-10 * self.im.abs().max(other.im.abs()).max(1.0)
		}
	}
	/// A well-conditioned fixture with no structure for the two layouts to
	/// exploit: a `(i, j)`-hashed value in `[-5, 5)`.
	fn val(i: usize, j: usize) -> f64 {
		let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
			^ (j as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
		h ^= h >> 31;
		h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
		h ^= h >> 29;
		((h >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 10.0
	}
	/// Everything the factorization left behind, as stored: the factor at every
	/// `(i, j)`, then each array it wrote, entry by entry.
	fn compare<T: Entry>(what: &str, cm: &[T], rm: &[T]) {
		if cm.len() != rm.len() {
			panic!("{what}: {} entries against {}", cm.len(), rm.len());
		}
		for k in 0..cm.len() {
			if !cm[k].same(&rm[k]) {
				panic!("{what}: entry {k} differs between the two orders");
			}
		}
	}
	fn same_exact<T: Entry>(what: &str, cm: &[T], rm: &[T]) {
		compare(what, cm, rm);
	}
	/// The `tau` array follows the factor: identical above the copy floor, equal
	/// up to rounding below it, where the row-major side factors in place.
	fn same_tau<T: Entry>(m: usize, n: usize, what: &str, cm: &[T], rm: &[T]) {
		if crate::linalg::row_major_copy_pays(m, n) {
			compare(what, cm, rm);
			return;
		}
		if cm.len() != rm.len() {
			panic!("{what}: {} entries against {}", cm.len(), rm.len());
		}
		for k in 0..cm.len() {
			if !cm[k].close(&rm[k]) {
				panic!("{what}: entry {k} differs between the two orders");
			}
		}
	}
	fn same_index<T: PartialEq + core::fmt::Debug + Copy>(
		what: &str,
		cm: &[T],
		rm: &[T],
	) {
		if cm != rm {
			panic!("{what}: {cm:?} against {rm:?}");
		}
	}
	/// The same matrix, flat in each order: `cm` column-major, `rm` row-major.
	/// `flipped` builds both so that a view whose rows are walked backwards
	/// represents the same matrix in each -- the row-major view is read from its
	/// last row with a negative row stride.
	fn fixture<T: Entry>(m: usize, n: usize, flipped: bool) -> (Vec<T>, Vec<T>) {
		let flipped = flipped && m > 0 && n > 0;
		let row_of = |i: usize| if flipped { m - 1 - i } else { i };
		let mut cm = Vec::with_capacity(m * n);
		for j in 0..n {
			for i in 0..m {
				cm.push(T::at(row_of(i), j));
			}
		}
		let mut rm = Vec::with_capacity(m * n);
		for i in 0..m {
			for j in 0..n {
				rm.push(T::at(i, j));
			}
		}
		(cm, rm)
	}
	/// The row-major side's flat position for a factor entry read through a
	/// `flipped` view: the view's row `i` is storage row `m - 1 - i`.
	fn rm_strides(m: usize, n: usize, flipped: bool) -> (usize, isize, isize) {
		if flipped && m > 0 && n > 0 {
			((m - 1) * n, -(n as isize), 1)
		} else {
			(0, n as isize, 1)
		}
	}
	fn compare_row_major<T: Entry>(
		m: usize,
		n: usize,
		flipped: bool,
		what: &str,
		cm: &[T],
		rm: &[T],
	) {
		let flipped = flipped && m > 0 && n > 0;
		// Above the copy floor both orders take the same arithmetic and must
		// agree bit for bit; below it the row-major side factors in place
		// through a strided panel, which is the same matrix but a different
		// summation order.
		let exact = crate::linalg::row_major_copy_pays(m, n);
		for j in 0..n {
			for i in 0..m {
				let k = if flipped { m - 1 - i } else { i };
				let (a, b) = (cm[i + j * m].clone(), rm[k * n + j].clone());
				if !(if exact { a.same(&b) } else { a.close(&b) }) {
					panic!(
						"{what} {m}x{n}: entry ({i}, {j}) differs between the two orders"
					);
				}
			}
		}
	}

	/// Every entry point that adapts a row-major matrix to a column-major one has
	/// to leave behind the column-major factorization of the same matrix: same
	/// factor, same `tau` matrix, same permutations, same transposition count.
	/// This is what says so, for the column-pivoted QR and both LU flavours (the
	/// no-pivoting QR has its own test beside its `qr_in_place`, with the shapes
	/// its guard leaves alone). Each shape is run with a plain row-major view and
	/// with one whose rows are walked backwards, so the copy's own direction
	/// branch is covered on a negative stride too.
	#[test]
	fn test_row_major_adaptation_matches_the_column_major_one() {
		use crate::linalg::lu::full_pivoting::factor as flu;
		use crate::linalg::lu::partial_pivoting::factor as plu;
		use crate::linalg::qr::col_pivoting::factor as cpqr;

		fn check_cpqr<T: Entry>(m: usize, n: usize, flipped: bool, mut par: Par) {
			let size = Ord::min(m, n);
			let bs = cpqr::recommended_block_size::<T>(m, n);
			let scratch =
				cpqr::qr_in_place_scratch::<usize, T>(m, n, bs, par.shape(), default());
			let (mut cm, mut rm) = fixture::<T>(m, n, flipped);
			let mut h_cm = vec![T::zero(); bs * size];
			let mut h_rm = vec![T::zero(); bs * size];
			let mut p_cm = vec![0usize; n];
			let mut p_rm = vec![0usize; n];
			let mut pi_cm = vec![0usize; n];
			let mut pi_rm = vec![0usize; n];
			let mut run = |a: &mut [T],
			           h: &mut [T],
			           p: &mut [usize],
			           pi: &mut [usize],
			           (base, rs, cs): (usize, isize, isize)|
			 -> cpqr::ColPivQrInfo {
				unsafe {
					cpqr::qr_in_place(
						MatMut::from_raw_parts_mut(
							a.as_mut_ptr().add(base),
							m,
							n,
							rs,
							cs,
						),
						MatMut::from_raw_parts_mut(
							h.as_mut_ptr(),
							bs,
							size,
							1,
							bs as isize,
						),
						p,
						pi,
						par.reborrow(),
						MemStack::new(&mut MemBuffer::new(scratch)),
						default(),
					)
					.0
				}
			};
			let cm_info = run(&mut cm, &mut h_cm, &mut p_cm, &mut pi_cm, (0, 1, m as isize));
			let rm_info = run(
				&mut rm,
				&mut h_rm,
				&mut p_rm,
				&mut pi_rm,
				rm_strides(m, n, flipped),
			);
			compare_row_major(m, n, flipped, "col_pivoting qr", &cm, &rm);
			same_tau(m, n, "col_pivoting qr tau", &h_cm, &h_rm);
			same_index("col_pivoting qr perm", &p_cm, &p_rm);
			same_index("col_pivoting qr perm inv", &pi_cm, &pi_rm);
			same_index(
				"col_pivoting qr transpositions",
				&[cm_info.transposition_count],
				&[rm_info.transposition_count],
			);
		}

		fn check_plu<T: Entry>(m: usize, n: usize, flipped: bool, mut par: Par) {
			let scratch = plu::lu_in_place_scratch::<usize, T>(m, n, par.shape(), default());
			let (mut cm, mut rm) = fixture::<T>(m, n, flipped);
			let mut p_cm = vec![0usize; m];
			let mut p_rm = vec![0usize; m];
			let mut pi_cm = vec![0usize; m];
			let mut pi_rm = vec![0usize; m];
			let mut run = |a: &mut [T],
			           p: &mut [usize],
			           pi: &mut [usize],
			           (base, rs, cs): (usize, isize, isize)|
			 -> plu::PartialPivLuInfo {
				unsafe {
					plu::lu_in_place(
						MatMut::from_raw_parts_mut(
							a.as_mut_ptr().add(base),
							m,
							n,
							rs,
							cs,
						),
						p,
						pi,
						par.reborrow(),
						MemStack::new(&mut MemBuffer::new(scratch)),
						default(),
					)
					.0
				}
			};
			let cm_info = run(&mut cm, &mut p_cm, &mut pi_cm, (0, 1, m as isize));
			let rm_info =
				run(&mut rm, &mut p_rm, &mut pi_rm, rm_strides(m, n, flipped));
			compare_row_major(m, n, flipped, "partial pivoting lu", &cm, &rm);
			same_index("partial pivoting lu perm", &p_cm, &p_rm);
			same_index("partial pivoting lu perm inv", &pi_cm, &pi_rm);
			same_index(
				"partial pivoting lu transpositions",
				&[cm_info.transposition_count],
				&[rm_info.transposition_count],
			);
		}

		fn check_flu<T: Entry>(m: usize, n: usize, flipped: bool, mut par: Par) {
			let scratch = flu::lu_in_place_scratch::<usize, T>(m, n, par.shape(), default());
			let (mut cm, mut rm) = fixture::<T>(m, n, flipped);
			let mut p_cm = vec![0usize; m];
			let mut p_rm = vec![0usize; m];
			let mut pi_cm = vec![0usize; m];
			let mut pi_rm = vec![0usize; m];
			let mut c_cm = vec![0usize; n];
			let mut c_rm = vec![0usize; n];
			let mut ci_cm = vec![0usize; n];
			let mut ci_rm = vec![0usize; n];
			let mut run = |a: &mut [T],
			           p: &mut [usize],
			           pi: &mut [usize],
			           c: &mut [usize],
			           ci: &mut [usize],
			           (base, rs, cs): (usize, isize, isize)|
			 -> flu::FullPivLuInfo {
				unsafe {
					flu::lu_in_place(
						MatMut::from_raw_parts_mut(
							a.as_mut_ptr().add(base),
							m,
							n,
							rs,
							cs,
						),
						p,
						pi,
						c,
						ci,
						par.reborrow(),
						MemStack::new(&mut MemBuffer::new(scratch)),
						default(),
					)
					.0
				}
			};
			let cm_info =
				run(&mut cm, &mut p_cm, &mut pi_cm, &mut c_cm, &mut ci_cm, (0, 1, m as isize));
			let rm_info = run(
				&mut rm,
				&mut p_rm,
				&mut pi_rm,
				&mut c_rm,
				&mut ci_rm,
				rm_strides(m, n, flipped),
			);
			compare_row_major(m, n, flipped, "full pivoting lu", &cm, &rm);
			same_index("full pivoting lu row perm", &p_cm, &p_rm);
			same_index("full pivoting lu row perm inv", &pi_cm, &pi_rm);
			same_index("full pivoting lu col perm", &c_cm, &c_rm);
			same_index("full pivoting lu col perm inv", &ci_cm, &ci_rm);
			same_index(
				"full pivoting lu transpositions",
				&[cm_info.transposition_count],
				&[rm_info.transposition_count],
			);
		}

		for (m, n) in [
			(1, 1),
			(2, 2),
			(16, 16),
			(17, 15),
			(64, 64),
			(300, 17),
			(64, 256),
			(300, 8),
		] {
			for flipped in [false, true] {
				for mut par in [Par::Seq, Par::rayon(8)] {
					check_cpqr::<f64>(m, n, flipped, par.reborrow());
					check_cpqr::<crate::c64>(m, n, flipped, par);
				}
			}
		}
		for (m, n) in [
			(1, 1),
			(2, 2),
			(16, 16),
			(17, 15),
			(64, 64),
			(128, 128),
			(300, 17),
		] {
			for flipped in [false, true] {
				for mut par in [Par::Seq, Par::rayon(8)] {
					check_plu::<f64>(m, n, flipped, par.reborrow());
					check_plu::<crate::c64>(m, n, flipped, par.reborrow());
					check_flu::<f64>(m, n, flipped, par.reborrow());
					check_flu::<crate::c64>(m, n, flipped, par);
				}
			}
		}
	}
}
