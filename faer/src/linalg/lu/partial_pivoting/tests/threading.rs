use super::*;
use crate::{Mat, c64};
use dyn_stack::MemBuffer;
use std::assert;

fn check<T: ComplexField<Real = f64>>(a: &Mat<T>, par: Par, row_major: bool) {
	let (m, n) = a.shape();
	let size = m.min(n);
	let mut buffer = vec![zero::<T>(); m * n];
	let mut lu = if row_major {
		MatMut::from_row_major_slice_mut(&mut buffer, m, n)
	} else {
		MatMut::from_column_major_slice_mut(&mut buffer, m, n)
	};
	lu.copy_from(a);
	let mut perm = vec![0usize; m];
	let mut perm_inv = vec![0usize; m];
	let mut mem =
		MemBuffer::new(lu_in_place_scratch::<usize, T>(m, n, par, default()));
	let (info, p) = lu_in_place(
		lu.rb_mut(),
		&mut perm,
		&mut perm_inv,
		par,
		MemStack::new(&mut mem),
		default(),
	);
	let l = Mat::from_fn(m, size, |i, j| {
		if i == j {
			one::<T>()
		} else if i > j {
			lu[(i, j)].copy()
		} else {
			zero::<T>()
		}
	});
	let u = Mat::from_fn(size, n, |i, j| {
		if i <= j {
			lu[(i, j)].copy()
		} else {
			zero::<T>()
		}
	});
	let reconstructed = p.inverse() * l * u;
	let residual = (&reconstructed - a).norm_l2() / a.norm_l2();
	assert!(
		residual < 1e-12,
		"{m}x{n}, row_major={row_major}, {par:?}: residual={residual:e}"
	);
	assert!(info.transposition_count <= size);
	for i in 0..m {
		assert!(perm[i] < m);
		assert_eq!(perm_inv[perm[i]], i);
	}
}

#[test]
fn recursive_products_preserve_real_and_complex_lu() {
	for threads in [1, 4, 8] {
		let pool = rayon::ThreadPoolBuilder::new()
			.num_threads(threads)
			.build()
			.unwrap();
		let par = if threads == 1 {
			Par::Seq
		} else {
			Par::rayon(threads)
		};
		pool.install(|| {
			for (m, n) in [(64, 64), (256, 256), (129, 65), (65, 129)] {
				let real = Mat::from_fn(m, n, |i, j| {
					((i * 17 + j * 11 + i * j) % 37) as f64 / 37.0
						+ if i == j { 2.0 } else { 0.0 }
				});
				let complex = Mat::from_fn(m, n, |i, j| {
					c64::new(
						real[(i, j)],
						((i * 7 + j * 13) % 23) as f64 / 23.0 - 0.5,
					)
				});
				for row_major in [false, true] {
					check(&real, par, row_major);
					check(&complex, par, row_major);
				}
			}
		});
	}
}
