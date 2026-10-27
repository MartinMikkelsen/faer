#![allow(dead_code, unused_imports)]
pub use dyn_stack::{MemBuffer, MemStack};
pub use faer::traits::ext::*;
pub use faer::traits::math_utils::*;
pub use faer::traits::{ComplexField, RealField};
pub use faer::{Col, Mat, MatRef, Par, ParShape, c32, c64};

pub const SIZES: [usize; 6] = [1, 2, 3, 8, 33, 100];
pub fn sample(i: usize, j: usize) -> f64 {
	// Deterministic pseudo-random entries, independent of the optional rand
	// feature.
	let mut x =
		(i as u64 + 7).wrapping_mul(0x9E3779B97F4A7C15) ^ (j as u64 + 19);
	x ^= x >> 30;
	x = x.wrapping_mul(0xBF58476D1CE4E5B9);
	x ^= x >> 27;
	(x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}
pub fn tolerance<T: ComplexField>(n: usize) -> T::Real {
	eps::<T::Real>() * from_f64::<T::Real>(200.0 * n.max(1) as f64)
}
pub fn close<T: ComplexField>(a: MatRef<'_, T>, b: MatRef<'_, T>) {
	let error = (a - b).norm_l2();
	let bound = tolerance::<T>(a.nrows()) * b.norm_l2().fmax(one::<T::Real>());
	assert!(error <= bound, "residual {error:?}, bound {bound:?}");
}
pub fn unitary<T: ComplexField>(q: MatRef<'_, T>) {
	close(
		(&q.adjoint() * q).as_ref(),
		Mat::identity(q.ncols(), q.ncols()).as_ref(),
	);
}
pub fn schur_form<T: ComplexField>(t: MatRef<'_, T>, real: bool) {
	let n = t.nrows();
	for j in 0..n {
		for i in j + if real { 2 } else { 1 }..n {
			assert_eq!(t[(i, j)], zero());
		}
		if real && j + 1 < n && t[(j + 1, j)] != zero() {
			assert_eq!(t[(j, j)], t[(j + 1, j + 1)]);
			assert!(t[(j, j + 1)].real() * t[(j + 1, j)].real() < zero());
			if j + 2 < n {
				assert_eq!(t[(j + 2, j + 1)], zero());
			}
		}
	}
}
