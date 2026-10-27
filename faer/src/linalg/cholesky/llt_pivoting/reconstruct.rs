use crate::assert;
use crate::internal_prelude::*;
use linalg::matmul::triangular::BlockStructure;
pub fn reconstruct_scratch<I: Index, T: ComplexField>(
	dim: usize,
	par: ParShape,
) -> StackReq {
	_ = (dim, par);
	temp_mat_scratch::<T>(dim, dim)
}
#[track_caller]
pub fn reconstruct<I: Index, T: ComplexField>(
	out: MatMut<'_, T>,
	L: MatRef<'_, T>,
	perm: PermRef<'_, I>,
	par: Par,
	stack: &mut MemStack,
) {
	let mut out = out;
	let n = out.nrows();
	assert!(all(
		out.nrows() == n,
		out.ncols() == n,
		L.nrows() == n,
		L.ncols() == n,
	));
	let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
	let mut tmp = tmp.as_mat_mut();
	linalg::matmul::triangular::matmul(
		tmp.rb_mut(),
		BlockStructure::TriangularLower,
		Accum::Replace,
		L,
		BlockStructure::TriangularLower,
		L.adjoint(),
		BlockStructure::TriangularUpper,
		one(),
		par,
	);
	let p = perm.arrays().1;
	for j in 0..n {
		let jj = p[j].zx();
		for i in j..n {
			let ii = p[i].zx();
			if ii >= jj {
				out[(i, j)] = tmp[(ii, jj)].copy();
			} else {
				out[(i, j)] = tmp[(jj, ii)].conj();
			}
		}
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;
	use linalg::cholesky::llt_pivoting::*;
	#[test]
	fn test_reconstruct() {
		let rng = &mut StdRng::seed_from_u64(0);
		let n = 50;
		let A = CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand::<Mat<c64>>(rng);
		let A = &A * A.adjoint();
		let mut L = A.to_owned();
		let perm_fwd = &mut *vec![0usize; n];
		let perm_bwd = &mut *vec![0usize; n];
		let (_, perm) = factor::cholesky_in_place(
			L.as_mut(),
			perm_fwd,
			perm_bwd,
			Par::Seq,
			MemStack::new(&mut {
				MemBuffer::new(factor::cholesky_in_place_scratch::<usize, c64>(
					n,
					ParShape::Seq,
					default(),
				))
			}),
			default(),
		)
		.unwrap();
		let approx_eq = CwiseMat(ApproxEq::eps() * (n as f64));
		let mut A_rec = Mat::zeros(n, n);
		reconstruct::reconstruct(
			A_rec.as_mut(),
			L.as_ref(),
			perm,
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(
				reconstruct::reconstruct_scratch::<usize, c64>(n, ParShape::Seq),
			)),
		);
		for j in 0..n {
			for i in 0..j {
				A_rec[(i, j)] = A_rec[(j, i)].conj();
			}
		}
		assert!(A_rec ~ A);
	}
	/// a rank-deficient positive-definite matrix stops at its numerical rank;
	/// the inactive part of the factor must not keep the input's entries, or
	/// `reconstruct` multiplies them back in (`tensor4all/faer#45`)
	#[test]
	fn test_early_rank_reconstruct() {
		let n = 3;
		let A = Mat::<f64>::from_fn(n, n, |i, j| {
			if i == j { 1.0 + f64::EPSILON } else { 1.0 }
		});
		let mut L = A.clone();
		let perm_fwd = &mut *vec![0usize; n];
		let perm_bwd = &mut *vec![0usize; n];
		let (info, perm) = factor::cholesky_in_place(
			L.as_mut(),
			perm_fwd,
			perm_bwd,
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(
				factor::cholesky_in_place_scratch::<usize, f64>(
					n,
					ParShape::Seq,
					default(),
				),
			)),
			default(),
		)
		.unwrap();
		assert_eq!(info.rank, 1);
		let mut A_rec = Mat::zeros(n, n);
		reconstruct::reconstruct(
			A_rec.as_mut(),
			L.as_ref(),
			perm,
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(
				reconstruct::reconstruct_scratch::<usize, f64>(n, ParShape::Seq),
			)),
		);
		// the stored factor is the rank-1 one, so every stored (lower
		// triangular) entry is the truncated matrix's entry, not the input's
		for j in 0..n {
			for i in j..n {
				assert!((A_rec[(i, j)] - 1.0).abs() < 1e-12);
			}
		}
	}
	/// a zero first pivot means rank zero, as LAPACK `dpstrf` reports, not a
	/// factorization error (`tensor4all/faer#31`)
	#[test]
	fn test_zero_matrix_rank_zero() {
		for n in [1, 2, 5] {
			let mut L = Mat::<f64>::zeros(n, n);
			let perm_fwd = &mut *vec![0usize; n];
			let perm_bwd = &mut *vec![0usize; n];
			let (info, perm) = factor::cholesky_in_place(
				L.as_mut(),
				perm_fwd,
				perm_bwd,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					factor::cholesky_in_place_scratch::<usize, f64>(
						n,
						ParShape::Seq,
						default(),
					),
				)),
				default(),
			)
			.unwrap();
			assert_eq!(info.rank, 0);
			let mut A_rec = Mat::full(n, n, f64::NAN);
			reconstruct::reconstruct(
				A_rec.as_mut(),
				L.as_ref(),
				perm,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					reconstruct::reconstruct_scratch::<usize, f64>(n, ParShape::Seq),
				)),
			);
			for j in 0..n {
				for i in j..n {
					assert_eq!(A_rec[(i, j)], 0.0);
				}
			}
		}
	}
}
