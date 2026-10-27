use crate::assert;
use crate::internal_prelude::*;
use linalg::matmul::triangular::BlockStructure;
pub fn reconstruct_scratch<I: Index, T: ComplexField>(
	dim: usize,
	par: ParShape,
) -> StackReq {
	_ = par;
	temp_mat_scratch::<T>(dim, dim)
}
#[track_caller]
pub fn reconstruct<I: Index, T: ComplexField>(
	out: MatMut<'_, T>,
	L: MatRef<'_, T>,
	diagonal: DiagRef<'_, T>,
	subdiagonal: DiagRef<'_, T>,
	perm: PermRef<'_, I>,
	par: Par,
	stack: &mut MemStack,
) {
	let n = L.nrows();
	assert!(all(
		out.nrows() == n,
		out.ncols() == n,
		L.nrows() == n,
		L.ncols() == n,
		diagonal.dim() == n,
		subdiagonal.dim() == n,
		perm.len() == n,
	));
	let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
	let mut tmp = tmp.as_mat_mut();
	let mut out = out;
	let s = subdiagonal;
	out.fill(zero());
	out.rb_mut().diagonal_mut().fill(one());
	out.copy_from_strict_triangular_lower(L);
	let mut j = 0;
	while j < n {
		if s[j] == zero() {
			let d = &diagonal[j].real();
			for i in 0..n {
				out[(i, j)] = out[(i, j)].mul_real(d);
			}
			j += 1;
		} else {
			let akp1k = &s[j].copy();
			let ak = &diagonal[j].real();
			let akp1 = &diagonal[j + 1].real();
			for i in 0..n {
				let xk = &out[(i, j)];
				let xkp1 = &out[(i, j + 1)];
				(out[(i, j)], out[(i, j + 1)]) = (
					xk.mul_real(ak) + xkp1 * akp1k,
					xkp1.mul_real(akp1) + xk * akp1k.conj(),
				);
			}
			j += 2;
		}
	}
	linalg::matmul::triangular::matmul(
		tmp.rb_mut(),
		BlockStructure::TriangularLower,
		Accum::Replace,
		L,
		BlockStructure::UnitTriangularLower,
		out.rb().adjoint(),
		BlockStructure::Rectangular,
		one(),
		par,
	);
	let perm_inv = perm.arrays().1;
	for j in 0..n {
		let pj = perm_inv[j].zx();
		for i in j..n {
			let pi = perm_inv[i].zx();
			out[(i, j)] = if pi >= pj {
				tmp[(pi, pj)].copy()
			} else {
				tmp[(pj, pi)].conj()
			};
		}
	}
	for j in 0..n {
		out[(j, j)] = out[(j, j)].as_real();
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;
	use linalg::cholesky::lblt::*;
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
		let A = &A + A.adjoint();
		let mut LB = A.to_owned();
		let mut subdiag = Diag::zeros(n);
		let perm_fwd = &mut *vec![0usize; n];
		let perm_bwd = &mut *vec![0usize; n];
		let (_, perm) = factor::cholesky_in_place(
			LB.as_mut(),
			subdiag.as_mut(),
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
		);
		let approx_eq = CwiseMat(ApproxEq::eps() * (n as f64));
		let mut A_rec = Mat::zeros(n, n);
		reconstruct::reconstruct(
			A_rec.as_mut(),
			LB.as_ref(),
			LB.diagonal(),
			subdiag.as_ref(),
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
	/// a 1x1 pivot with no entries below it must still clear its subdiagonal
	/// entry, since `subdiag` is an output the caller may reuse
	/// (`tensor4all/faer#31`)
	#[test]
	fn test_factor_clears_stale_subdiag() {
		// n = 100 exceeds the default block size, so the blocked path runs too
		for n in [1usize, 3, 100] {
			let A = Mat::<f64>::from_fn(n, n, |i, j| {
				if i == j { 1.0 + i as f64 } else { 0.0 }
			});
			let mut LB = A.clone();
			let mut subdiag = Diag::full(n, 7.0);
			let perm_fwd = &mut *vec![0usize; n];
			let perm_bwd = &mut *vec![0usize; n];
			let (_, perm) = factor::cholesky_in_place(
				LB.as_mut(),
				subdiag.as_mut(),
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
			);
			for i in 0..n {
				assert!(subdiag[i] == 0.0);
			}
			let mut A_rec = Mat::zeros(n, n);
			reconstruct::reconstruct(
				A_rec.as_mut(),
				LB.as_ref(),
				LB.diagonal(),
				subdiag.as_ref(),
				perm,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					reconstruct::reconstruct_scratch::<usize, f64>(n, ParShape::Seq),
				)),
			);
			for j in 0..n {
				for i in j..n {
					assert!(A_rec[(i, j)] == A[(i, j)]);
				}
			}
		}
	}
	/// The high-level wrapper stores the block diagonal separately and keeps a unit
	/// diagonal in `L`, so reconstruction has to read `diagonal` for both block sizes.
	#[test]
	fn test_reconstruct_separate_diagonal() {
		fn check<T: ComplexField>() {
			let approx_eq =
				CwiseMat(ApproxEq::<T::Real>::eps() * from_f64::<T::Real>(8.0));
			for n in [1usize, 2, 3, 5] {
				let mut L = Mat::<T>::identity(n, n);
				for j in 0..n {
					for i in j + 1..n {
						L[(i, j)] = from_f64::<T>((i + j) as f64 * 0.5 - 1.0);
					}
				}
				let mut diagonal = Diag::<T>::zeros(n);
				let mut subdiagonal = Diag::<T>::zeros(n);
				for j in 0..n {
					diagonal[j] = from_f64::<T>(2.0 + j as f64);
					if j % 3 == 1 && j + 1 < n {
						subdiagonal[j] = from_f64::<T>(0.5 * j as f64);
					}
				}
				let mut B = Mat::<T>::zeros(n, n);
				for j in 0..n {
					B[(j, j)] = diagonal[j].copy();
					if subdiagonal[j] != zero() {
						B[(j + 1, j)] = subdiagonal[j].copy();
						B[(j, j + 1)] = subdiagonal[j].copy();
					}
				}
				let expected = &L * &B * L.adjoint();
				let identity = || (0..n).collect::<Vec<_>>().into_boxed_slice();
				let perm =
					Perm::<usize>::new_checked(identity(), identity(), n);
				let mut rec = Mat::<T>::zeros(n, n);
				reconstruct(
					rec.as_mut(),
					L.as_ref(),
					diagonal.as_ref(),
					subdiagonal.as_ref(),
					perm.as_ref(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(reconstruct_scratch::<
						usize,
						T,
					>(n, ParShape::Seq))),
				);
				// reconstruct fills the lower triangle only
				for j in 0..n {
					for i in 0..j {
						rec[(i, j)] = rec[(j, i)].copy();
					}
				}
				assert!(rec ~ expected);
			}
		}
		check::<f32>();
		check::<f64>();
	}
}
