mod common;
use common::*;
use faer::linalg::evd::*;

fn real<T: RealField>() {
	for n in SIZES {
		let original = Mat::from_fn(n, n, |i, j| {
			let pair_start = if i % 3 == 1 { i - 1 } else { i };
			if i == j {
				from_f64((pair_start + 1) as f64)
			} else if i + 1 == j && i % 3 == 0 {
				from_f64(0.5)
			} else if j + 1 == i && j % 3 == 0 {
				from_f64(-2.0)
			} else if i < j {
				from_f64(sample(i, j))
			} else {
				zero()
			}
		});
		for (ifst, ilst) in [(0, n - 1), (n - 1, 0)] {
			let mut t = original.clone();
			let mut q = Mat::identity(n, n);
			let start = if ifst > 0 && original[(ifst, ifst - 1)] != zero() {
				ifst - 1
			} else {
				ifst
			};
			let mut mem = MemBuffer::new(reorder_schur_scratch::<T>(n));
			let pos = reorder_schur_real(
				t.as_mut(),
				Some(q.as_mut()),
				ifst,
				ilst,
				Par::Seq,
				MemStack::new(&mut mem),
			)
			.unwrap();
			close((&q * &t * q.adjoint()).as_ref(), original.as_ref());
			unitary(q.as_ref());
			schur_form(t.as_ref(), true);
			assert!(
				(&t[(pos, pos)] - &original[(start, start)]).abs()
					<= tolerance::<T>(n)
			);
			let mut before: Vec<_> =
				(0..n).map(|i| original[(i, i)].copy()).collect();
			let mut after: Vec<_> = (0..n).map(|i| t[(i, i)].copy()).collect();
			before.sort_by(|a, b| a.partial_cmp(b).unwrap());
			after.sort_by(|a, b| a.partial_cmp(b).unwrap());
			for (a, b) in before.iter().zip(&after) {
				assert!((a - b).abs() <= tolerance::<T>(n));
			}
			if start + 1 < n && original[(start + 1, start)] != zero() {
				assert_ne!(t[(pos + 1, pos)], zero());
				let im = t[(pos + 1, pos)].abs().sqrt()
					* t[(pos, pos + 1)].abs().sqrt();
				assert!((im - one::<T>()).abs() <= tolerance::<T>(n));
			}
		}
	}
}
fn complex<T: ComplexField>(make: fn(f64, f64) -> T) {
	for n in SIZES {
		let original = Mat::from_fn(n, n, |i, j| {
			if i > j {
				zero()
			} else if i == j {
				make(i as f64 + 1.0, sample(i, j))
			} else {
				make(sample(i, j), sample(j, i))
			}
		});
		for (ifst, ilst) in [(0, n - 1), (n - 1, 0)] {
			let mut t = original.clone();
			let mut q = Mat::identity(n, n);
			let mut mem = MemBuffer::new(reorder_schur_scratch::<T>(n));
			assert_eq!(
				reorder_schur_cplx(
					t.as_mut(),
					Some(q.as_mut()),
					ifst,
					ilst,
					Par::Seq,
					MemStack::new(&mut mem)
				),
				ilst
			);
			close((&q * &t * q.adjoint()).as_ref(), original.as_ref());
			unitary(q.as_ref());
			schur_form(t.as_ref(), false);
			assert_eq!(t[(ilst, ilst)], original[(ifst, ifst)]);
			let mut order: Vec<_> = (0..n).collect();
			let moved = order.remove(ifst);
			order.insert(ilst, moved);
			for (i, old) in order.into_iter().enumerate() {
				assert_eq!(t[(i, i)], original[(old, old)]);
			}
		}
	}
}
#[test]
fn swaps_preserve_similarity_and_pairs() {
	real::<f64>();
	real::<f32>();
	complex::<c64>(c64::new);
	complex::<c32>(|r, i| c32::new(r as f32, i as f32));
}

#[test]
fn rejected_swap_preserves_last_successful_state() {
	// Coincident pairs with very different off-diagonal scales make the
	// Sylvester-based 2x2 swap fail its lower-left-block acceptance test.
	let small = 2.0f64.powi(-100);
	let middle = 2.0f64.powi(-50);
	let large = 2.0f64.powi(100);
	let original = faer::mat![
		[0.0, small, large, large * 0.3],
		[-1.0 / small, 0.0, -large * 0.7, large * 0.4],
		[0.0, 0.0, 0.0, middle],
		[0.0, 0.0, -1.0 / middle, 0.0],
	];
	let mut t = original.clone();
	let mut q = Mat::identity(4, 4);
	let mut mem = MemBuffer::new(reorder_schur_scratch::<f64>(4));
	assert_eq!(
		reorder_schur_real(
			t.as_mut(),
			Some(q.as_mut()),
			0,
			2,
			Par::Seq,
			MemStack::new(&mut mem)
		),
		Err(SchurReorderError { position: 0 })
	);
	assert_eq!(t, original);
	assert_eq!(q, Mat::identity(4, 4));
	schur_form(t.as_ref(), true);
}

#[test]
fn rejection_keeps_earlier_successful_swaps() {
	let small = 2.0f64.powi(-100);
	let middle = 2.0f64.powi(-50);
	let large = 2.0f64.powi(100);
	let pairs = faer::mat![
		[0.0, small, large, large * 0.3],
		[-1.0 / small, 0.0, -large * 0.7, large * 0.4],
		[0.0, 0.0, 0.0, middle],
		[0.0, 0.0, -1.0 / middle, 0.0]
	];
	let original = Mat::from_fn(5, 5, |i, j| {
		if i == 2 || j == 2 {
			if i == j { 5.0 } else { 0.0 }
		} else {
			pairs
				[(if i > 2 { i - 1 } else { i }, if j > 2 { j - 1 } else { j })]
		}
	});
	let mut expected = original.clone();
	let mut expected_q = Mat::identity(5, 5);
	let mut t = original.clone();
	let mut q = Mat::identity(5, 5);
	let mut mem = MemBuffer::new(reorder_schur_scratch::<f64>(5));
	assert_eq!(
		reorder_schur_real(
			expected.as_mut(),
			Some(expected_q.as_mut()),
			0,
			2,
			Par::Seq,
			MemStack::new(&mut mem)
		)
		.unwrap(),
		1
	);
	assert_eq!(
		reorder_schur_real(
			t.as_mut(),
			Some(q.as_mut()),
			0,
			3,
			Par::Seq,
			MemStack::new(&mut mem)
		),
		Err(SchurReorderError { position: 1 })
	);
	assert_eq!(t, expected);
	assert_eq!(q, expected_q);
	close((&q * &t * q.adjoint()).as_ref(), original.as_ref());
	schur_form(t.as_ref(), true);
}

/// Relative similarity residual and orthogonality error, computed after
/// scaling by the largest entry so that extreme inputs stay representable.
fn metrics<T: RealField>(original: &Mat<T>, t: &Mat<T>, q: &Mat<T>) -> (T, T) {
	let n = original.nrows();
	let mut scale = zero::<T>();
	for j in 0..n {
		for i in 0..n {
			scale = scale.fmax(original[(i, j)].abs()).fmax(t[(i, j)].abs());
		}
	}
	let a = Mat::from_fn(n, n, |i, j| &original[(i, j)] / &scale);
	let b = Mat::from_fn(n, n, |i, j| &t[(i, j)] / &scale);
	let residual = (q * b * q.transpose() - &a).norm_l2() / a.norm_l2();
	let orthogonality =
		(q.transpose() * q - Mat::<T>::identity(n, n)).norm_l2();
	(residual, orthogonality)
}

/// Runs a reordering with `q = I`. The public contract is "stable or reject":
/// success means finite `t`, `q` forming an orthogonal similarity in Schur
/// form; a rejection of the first swap leaves `t` and `q` untouched. Returns
/// whether the reordering was accepted.
fn stable_or_rejected<T: RealField>(
	original: &Mat<T>,
	ifst: usize,
	ilst: usize,
	start: usize,
) -> bool {
	let n = original.nrows();
	let mut t = original.clone();
	let mut q = Mat::<T>::identity(n, n);
	let mut mem = MemBuffer::new(reorder_schur_scratch::<T>(n));
	match reorder_schur_real(
		t.as_mut(),
		Some(q.as_mut()),
		ifst,
		ilst,
		Par::Seq,
		MemStack::new(&mut mem),
	) {
		Ok(_) => {
			assert!(t.is_all_finite() && q.is_all_finite());
			let (residual, orthogonality) = metrics(original, &t, &q);
			assert!(residual <= tolerance::<T>(n), "residual {residual:?}");
			assert!(orthogonality <= tolerance::<T>(n));
			schur_form(t.as_ref(), true);
			true
		},
		Err(err) => {
			assert_eq!(err, SchurReorderError { position: start });
			assert!(t == *original);
			assert!(q == Mat::<T>::identity(n, n));
			false
		},
	}
}

/// faer's historical 2x1/1x2 kernels returned `Ok` here with a broken `q`
/// (`||q t q^T - t|| / ||t|| = 1`, `||q^T q - I|| = sqrt 2`). LAPACK `dtrexc`
/// (macOS Accelerate) accepts both swaps with relative residuals 1.7e-16 and
/// 2.2e-16, so the `dlaexc` port must accept them too.
#[test]
fn reviewer_2x1_and_mirrored_1x2_swaps_are_accepted_and_stable() {
	let b = f64::from_bits(0x16687E92154EF7A9);
	let k = f64::from_bits(0x6974E718D7D7625D);
	let t = faer::mat![
		[0.0, b, k],
		[-k, 0.0, -3.0000000000000014e199],
		[0.0, 0.0, 0.0],
	];
	assert!(stable_or_rejected(&t, 0, 2, 0));
	// The mirrored 1x2 swap. Row 2 is the pair's second row, so the moving
	// block starts at row 1.
	let t = faer::mat![
		[0.0, -3.0000000000000014e199, k],
		[0.0, 0.0, b],
		[0.0, -k, 0.0],
	];
	assert!(stable_or_rejected(&t, 2, 0, 1));
}

fn ordinary<T: RealField>() {
	let f = |x: f64| from_f64::<T>(x);
	let pair_first = Mat::from_fn(3, 3, |i, j| {
		f([[1.0, 2.0, 0.3], [-0.5, 1.0, -0.7], [0.0, 0.0, 4.0]][i][j])
	});
	let single_first = Mat::from_fn(3, 3, |i, j| {
		f([[4.0, 0.3, -0.7], [0.0, 1.0, 2.0], [0.0, -0.5, 1.0]][i][j])
	});
	for (original, ifst, ilst, pos, single) in
		[(pair_first, 0, 2, 1, 0), (single_first, 0, 1, 2, 2)]
	{
		let mut t = original.clone();
		let mut q = Mat::<T>::identity(3, 3);
		let mut mem = MemBuffer::new(reorder_schur_scratch::<T>(3));
		assert_eq!(
			reorder_schur_real(
				t.as_mut(),
				Some(q.as_mut()),
				ifst,
				ilst,
				Par::Seq,
				MemStack::new(&mut mem),
			),
			Ok(pos)
		);
		let (residual, orthogonality) = metrics(&original, &t, &q);
		assert!(residual <= tolerance::<T>(3));
		assert!(orthogonality <= tolerance::<T>(3));
		schur_form(t.as_ref(), true);
		assert!((&t[(single, single)] - f(4.0)).abs() <= tolerance::<T>(3));
	}
}
#[test]
fn ordinary_2x1_and_1x2_swaps() {
	ordinary::<f64>();
	ordinary::<f32>();
}

/// A moderate diagonal window with entries near overflow outside it.
fn outside_window<T: RealField>(column: [f64; 3]) -> Mat<T> {
	let local = [[1.0, 2.0, 0.3], [-0.5, 1.0, -0.7], [0.0, 0.0, 4.0]];
	Mat::from_fn(4, 4, |i, j| {
		from_f64(if i < 3 && j < 3 {
			local[i][j]
		} else if j == 3 && i < 3 {
			column[i]
		} else if i == 3 && j == 3 {
			7.0
		} else {
			0.0
		})
	})
}
/// The direct `dlarfx` update of the last column overflows before
/// cancellation (faer's historical 2x1 kernel produced NaN, the first port
/// `inf`). LAPACK `dtrexc` / `strexc` (macOS Accelerate) accept every case
/// here with finite output.
#[test]
fn swap_keeps_large_outside_window_column_finite() {
	for column in [[1e308, 1e308, 1e308], [-1e308, 1e308, 1e308]] {
		assert!(stable_or_rejected(&outside_window::<f64>(column), 0, 2, 0));
	}
	for column in [[2.5e38, 2.5e38, 2.5e38], [-1e38, 1e38, 1e38]] {
		assert!(stable_or_rejected(&outside_window::<f32>(column), 0, 2, 0));
	}
}
/// When the reflected outside column is not representable, the swap is
/// rejected and `t`, `q` stay as they were. LAPACK `dtrexc` / `strexc` report
/// success with `inf`/NaN entries for these inputs.
#[test]
fn overflowing_swap_is_rejected_unchanged() {
	let column = [f64::MAX, -f64::MAX, f64::MAX];
	assert!(!stable_or_rejected(&outside_window::<f64>(column), 0, 2, 0));
	let column = [-2.5e38, 2.5e38, 2.5e38];
	assert!(!stable_or_rejected(&outside_window::<f32>(column), 0, 2, 0));
}
/// The standardization of the swapped pair needs `dlanv2`'s scaled norm:
/// `sqrt(c^2 + z^2)` and faer's generic `hypot` both overflow here, which made
/// `q` lose orthogonality. LAPACK `dtrexc` accepts with residual below 1e-15.
#[test]
fn standardization_of_extreme_pair_stays_orthogonal() {
	let t =
		faer::mat![[0.0, 1e308, 0.0], [-1e154, 0.0, -1e289], [0.0, 0.0, 0.0]];
	assert!(stable_or_rejected(&t, 0, 2, 0));
	let t =
		faer::mat![[0.0f32, 3e38, 0.0], [-1e19, 0.0, -1e34], [0.0, 0.0, 0.0]];
	assert!(stable_or_rejected(&t, 0, 2, 0));
}

/// The complex swap kernel formed the diagonal difference before scaling, so a
/// finite pair of diagonals near `max_positive` gave an infinite difference and
/// the rotation came out `NaN`: `reorder_schur_cplx` reported success with a
/// fully `NaN` `q`. The inputs are now halved (exact, and the ratio — hence the
/// rotation — is unchanged) when the difference overflows.
fn extreme_diagonal<T: ComplexField>(max: f64, make: fn(f64, f64) -> T) {
	let f = |x: f64| from_f64::<T::Real>(x);
	let original = Mat::<T>::from_fn(2, 2, |i, j| match (i, j) {
		(0, 0) => make(-max, 0.0),
		(0, 1) => make(1.0, 0.0),
		(1, 1) => make(max, 0.0),
		_ => make(0.0, 0.0),
	});
	let mut t = original.clone();
	let mut q = Mat::<T>::identity(2, 2);
	let mut mem = MemBuffer::new(reorder_schur_scratch::<T>(2));
	assert_eq!(
		reorder_schur_cplx(
			t.as_mut(),
			Some(q.as_mut()),
			0,
			1,
			Par::Seq,
			MemStack::new(&mut mem)
		),
		1
	);
	assert!(t.is_all_finite() && q.is_all_finite(), "t={t:?} q={q:?}");
	assert!(t[(0, 0)] == make(max, 0.0) && t[(1, 1)] == make(-max, 0.0));
	assert!(t[(1, 0)] == zero::<T>());
	unitary(q.as_ref());
	// Halve before the similarity: no intermediate may exceed `max_positive`.
	let ref half = f(0.5);
	let scaled = Mat::from_fn(2, 2, |i, j| t[(i, j)].mul_real(half));
	let expected = Mat::from_fn(2, 2, |i, j| original[(i, j)].mul_real(half));
	let back = &q * &scaled * q.adjoint();
	let tol = f(2048.0) * eps::<T::Real>() * expected[(0, 0)].abs();
	for j in 0..2 {
		for i in 0..2 {
			assert!(
				(&back[(i, j)] - &expected[(i, j)]).abs() <= tol,
				"{i},{j}: {back:?} vs {expected:?}"
			);
		}
	}
}
#[test]
fn extreme_diagonal_swap_stays_finite() {
	extreme_diagonal::<f64>(f64::MAX, |r, _| r);
	extreme_diagonal::<f32>(f32::MAX as f64, |r, _| r as f32);
	extreme_diagonal::<c64>(f64::MAX, c64::new);
	extreme_diagonal::<c32>(f32::MAX as f64, |r, i| {
		c32::new(r as f32, i as f32)
	});
}
