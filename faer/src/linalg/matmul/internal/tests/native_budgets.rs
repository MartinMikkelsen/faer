use super::*;
use crate::c64;
use crate::linalg::matmul::{internal, triangular};

fn check_triangular<T: ComplexField<Real = f64>>(
	scalar: impl Fn(f64, f64) -> T,
) {
	use triangular::BlockStructure::*;
	for n in [1, 32, 64, 128] {
		let a = Mat::from_fn(n, n, |i, k| {
			scalar(
				((i + 3 * k) % 11) as f64 * 0.25 - 1.25,
				((2 * i + k) % 7) as f64 * 0.125 - 0.375,
			)
		});
		let b = Mat::from_fn(n, n, |k, j| {
			scalar(
				((k + 2 * j) % 9) as f64 * 0.25 - 1.0,
				((3 * k + j) % 5) as f64 * 0.125 - 0.25,
			)
		});
		for row_major in [false, true] {
			let (a, b) = if row_major {
				(a.as_ref().transpose(), b.as_ref().transpose())
			} else {
				(a.as_ref(), b.as_ref())
			};
			for block in [
				TriangularLower,
				TriangularUpper,
				StrictTriangularLower,
				UnitTriangularUpper,
			] {
				for beta in [Accum::Add, Accum::Replace] {
					let initial =
						Mat::from_fn(n + 2, n + 2, |_, _| scalar(7.0, 2.0));
					let mut expected = initial.clone();
					let alpha = scalar(0.75, -0.25);
					{
						let mut reference = if row_major {
							expected.as_mut().transpose_mut()
						} else {
							expected.as_mut()
						};
						for j in 0..n {
							for i in 0..n {
								let stored = match block {
									TriangularLower => i >= j,
									TriangularUpper => i <= j,
									StrictTriangularLower => i > j,
									UnitTriangularUpper => i < j,
									_ => unreachable!(),
								};
								if !stored {
									continue;
								}
								let mut sum = T::zero();
								for k in 0..n {
									sum += a[(i, k)].conj() * b[(k, j)].clone();
								}
								let value = alpha.clone() * sum;
								if beta == Accum::Replace {
									reference[(i + 1, j + 1)] = value;
								} else {
									reference[(i + 1, j + 1)] += value;
								}
							}
						}
					}
					for par in [Par::Seq, Par::rayon(8)] {
						let mut actual = initial.clone();
						let view = if row_major {
							actual.as_mut().transpose_mut()
						} else {
							actual.as_mut()
						};
						triangular::matmul_with_conj(
							view.submatrix_mut(1, 1, n, n),
							block,
							beta,
							a,
							Rectangular,
							Conj::Yes,
							b,
							Rectangular,
							Conj::No,
							alpha.clone(),
							par,
						);
						std::assert!(actual.as_ref().is_all_finite());
						let error =
							(actual.as_ref() - expected.as_ref()).norm_max();
						std::assert!(
							error < 1e-10,
							"n={n} row_major={row_major} real={} error={error}",
							T::IS_REAL
						);
					}
				}
			}
		}
	}
}

fn check_indexed<T: ComplexField<Real = f64>>(scalar: impl Fn(f64, f64) -> T) {
	use triangular::BlockStructure::*;
	let (m, n, depth) = (128, 96, 64);
	let a = Mat::from_fn(m, depth, |i, k| {
		scalar(
			((i + 3 * k) % 11) as f64 * 0.25 - 1.25,
			((2 * i + k) % 7) as f64 * 0.125 - 0.375,
		)
	});
	let b = Mat::from_fn(depth, n, |k, j| {
		scalar(
			((k + 2 * j) % 9) as f64 * 0.25 - 1.0,
			((3 * k + j) % 5) as f64 * 0.125 - 0.25,
		)
	});
	let d =
		Mat::from_fn(depth, 1, |k, _| scalar(1.0 + (k % 3) as f64 * 0.25, 0.0));
	let rows: Vec<usize> = (0..m).map(|i| m - i).collect();
	let cols: Vec<usize> = (0..n).map(|j| n - j).collect();
	for block in [Rectangular, StrictTriangularLower] {
		for with_diagonal in [false, true] {
			let mut actual =
				Mat::from_fn(m + 2, n + 2, |_, _| scalar(7.0, 2.0));
			let mut expected = actual.clone();
			let alpha = scalar(0.75, -0.25);
			for j in 0..n {
				for i in 0..m {
					if block == StrictTriangularLower && i <= j {
						continue;
					}
					let mut sum = T::zero();
					for k in 0..depth {
						let weight =
							if with_diagonal { d[(k, 0)].real() } else { 1.0 };
						sum += a[(i, k)].conj().mul_real(&weight)
							* b[(k, j)].clone();
					}
					expected[(rows[i], cols[j])] += alpha.clone() * sum;
				}
			}
			let mut mem =
				dyn_stack::MemBuffer::new(internal::spicy_matmul_scratch::<T>(
					m,
					n,
					depth,
					true,
					with_diagonal,
				));
			internal::spicy_matmul::<usize, T>(
				actual.as_mut().submatrix_mut(0, 0, m + 1, n + 1),
				block,
				Some(&rows),
				Some(&cols),
				Accum::Add,
				a.as_ref(),
				Conj::Yes,
				b.as_ref(),
				Conj::No,
				with_diagonal.then_some(d.as_ref().col(0).as_diagonal()),
				alpha,
				Par::rayon(8),
				MemStack::new(&mut mem),
			);
			std::assert!(actual.as_ref().is_all_finite());
			let error = (actual.as_ref() - expected.as_ref()).norm_max();
			std::assert!(
				error < 1e-10,
				"indexed real={} diagonal={with_diagonal} error={error}",
				T::IS_REAL
			);
		}
	}
}

#[test]
fn structured_native_budgets_preserve_scalar_products_and_guards() {
	let pool = rayon::ThreadPoolBuilder::new()
		.num_threads(8)
		.build()
		.unwrap();
	pool.install(|| {
		std::assert_eq!(rayon::current_num_threads(), 8);
		check_triangular::<f64>(|re, _| re);
		check_triangular::<c64>(c64::new);
		check_indexed::<f64>(|re, _| re);
		check_indexed::<c64>(c64::new);
	});
}
