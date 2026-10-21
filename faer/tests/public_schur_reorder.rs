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
			let mut mem = MemBuffer::new(reorder_schur_scratch());
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
			let mut mem = MemBuffer::new(reorder_schur_scratch());
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
	let mut mem = MemBuffer::new(reorder_schur_scratch());
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
	let mut mem = MemBuffer::new(reorder_schur_scratch());
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
