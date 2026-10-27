// CHKBL/CHKBK properties, LAPACK 3.12.0, OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80; TESTING/{s,d,c,z}{bal,bak}.in.
// Selected first BAL/BAK records are diagonal/identity. Rust power-of-two and
// permutation fixtures extend them; packed SCALE/ILO/IHI are adapted deliberately.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD. Exact LAPACK balancing choices are not required.
fn metadata_error(
	a: MatRef<'_, Scalar>,
	b: MatRef<'_, Scalar>,
	perm: &[usize],
	scale: &[Real],
) -> Real {
	let n = a.nrows();
	if perm.len() != n || scale.len() != n {
		return Real::INFINITY;
	}
	let mut seen = vec![false; n];
	for &p in perm {
		if p >= n || seen[p] {
			return Real::INFINITY;
		}
		seen[p] = true;
	}
	if scale.iter().any(|v| !v.is_finite() || *v <= 0.0) {
		return Real::INFINITY;
	}
	let mut error = 0.0 as Real;
	for j in 0..n {
		for i in 0..n {
			let expected = (a[(perm[i], perm[j])] / scale[i]) * scale[j];
			let actual = b[(i, j)];
			if !is_finite(&expected) || !is_finite(&actual) {
				return Real::INFINITY;
			}
			error = error.max(
				magnitude(actual - expected)
					/ magnitude(actual)
						.max(magnitude(expected))
						.max(Real::MIN_POSITIVE),
			);
		}
	}
	error
}
fn back_error(
	actual: MatRef<'_, Scalar>,
	expected: MatRef<'_, Scalar>,
) -> Real {
	let mut error = 0.0 as Real;
	for j in 0..actual.ncols() {
		for i in 0..actual.nrows() {
			let a = actual[(i, j)];
			let e = expected[(i, j)];
			if !is_finite(&a) || !is_finite(&e) {
				return Real::INFINITY;
			}
			let delta = magnitude(a - e);
			error = error.max(if magnitude(a) > Real::MIN_POSITIVE {
				delta / magnitude(a) / Real::EPSILON
			} else {
				delta / Real::EPSILON
			});
		}
	}
	error
}
#[test]
fn lapack_balance_checkers_and_literal_backtransform() {
	let a = Mat::from_fn(3, 3, |i, j| z((1 + i + 3 * j) as Real));
	let p = [2, 0, 1];
	let d = [2.0, 0.5, 4.0];
	let b = Mat::from_fn(3, 3, |i, j| (a[(p[i], p[j])] / d[i]) * d[j]);
	assert_eq!(metadata_error(a.as_ref(), b.as_ref(), &p, &d), 0.0);
	crate::assert!(
		metadata_error(a.as_ref(), b.as_ref(), &[1, 2, 0], &d) > Real::EPSILON
	);
	let values = [
		(1.0, 1.0),
		(2.0, 2.0),
		(3.0, -1.0),
		(4.0, 0.5),
		(5.0, 2.0),
		(6.0, -3.0),
	];
	let mut x = Mat::from_fn(3, 2, |i, j| {
		let (r, im) = values[i * 2 + j];
		scalar(r, im)
	});
	let want = [
		(1.5, -0.5),
		(2.0, 0.25),
		(20.0, 8.0),
		(24.0, -12.0),
		(2.0, 2.0),
		(4.0, 4.0),
	];
	let expected = Mat::from_fn(3, 2, |i, j| {
		let (r, im) = want[i * 2 + j];
		scalar(r, im)
	});
	let mut mem = MemBuffer::new(balance_eigenvectors_scratch::<Scalar>(3));
	balance_eigenvectors_in_place(
		x.as_mut(),
		&p,
		&d,
		Par::Seq,
		MemStack::new(&mut mem),
	);
	assert_eq!(back_error(x.as_ref(), expected.as_ref()), 0.0);
}
fn balance_input(n: usize, kind: usize) -> Mat<Scalar> {
	if kind == 0 {
		return Mat::zeros(n, n);
	}
	if kind == 1 {
		return Mat::from_fn(n, n, |i, j| {
			if i == j {
				scalar(
					(i + 1) as Real,
					if <Scalar as ComplexField>::IS_REAL {
						0.0
					} else {
						[1.0, 1.0, 3.0, 1.0, 5.0][i % 5]
					},
				)
			} else {
				z(0.0)
			}
		});
	}
	if kind == 2 {
		return Mat::from_fn(
			n,
			n,
			|i, j| if i <= j { sample(i, j) } else { z(0.0) },
		);
	}
	let scale = match kind {
		5 => Real::MAX.sqrt(),
		6 => Real::MIN_POSITIVE.sqrt(),
		_ => 1.0,
	};
	Mat::from_fn(n, n, |i, j| {
		let base = if i == j {
			z((i + 1) as Real)
		} else {
			sample(i, j) * 0.25
		};
		let ei = if kind == 3 {
			0
		} else {
			(i % 3) as i32 * 12 - 12
		};
		let ej = if kind == 3 {
			0
		} else {
			(j % 3) as i32 * 12 - 12
		};
		(base * scale) * (2.0 as Real).powi(ei - ej)
	})
}
fn balance_case(
	run: &mut Run,
	a: MatRef<'_, Scalar>,
	layout: usize,
	kind: usize,
) {
	let n = a.nrows();
	let mut ap = guarded(n, n, layout);
	let mut xp = guarded(n, 3, layout);
	let mut perm = vec![usize::MAX; n];
	let mut scale = vec![Real::NAN; n];
	let mut mem = MemBuffer::new(balance_scratch());
	let mut xmem = MemBuffer::new(balance_eigenvectors_scratch::<Scalar>(n));
	for pass in 0..2 {
		let input = scaled(a, if pass == 0 { 1.0 } else { 0.5 });
		interior_mut(ap.as_mut(), layout).copy_from(&input);
		let ctx =
			format!("Balance kind={kind} n={n} layout={layout} pass={pass}");
		let mut active = None;
		if run.call(&ctx, || {
			active = Some(balance_in_place(
				interior_mut(ap.as_mut(), layout),
				&mut perm,
				&mut scale,
				Par::Seq,
				MemStack::new(&mut mem),
			)?);
			Ok::<_, BalanceError>(())
		}) {
			let active = active.unwrap();
			let b = interior(ap.as_ref(), layout);
			run.require(
				&ctx,
				"invalid active range",
				active.start <= active.end && active.end <= n,
			);
			let error = metadata_error(input.as_ref(), b, &perm, &scale);
			run.require(
				&ctx,
				&format!("metadata identity relative error={error:e}"),
				error.is_finite() && error <= 8.0 * Real::EPSILON,
			);
			run.require(
				&ctx,
				"scale is not a power of two",
				scale.iter().all(|s| {
					s.is_finite() && *s > 0.0 && s.log2() == s.log2().round()
				}),
			);
			run.require(
				&ctx,
				"isolation outside active range",
				(0..n).all(|j| {
					(j + 1..n).all(|i| {
						(j >= active.start && i < active.end)
							|| b[(i, j)] == z(0.0)
					})
				}),
			);
			let xb = Mat::from_fn(n, 3, |i, j| sample(i, j));
			let mut expected = Mat::<Scalar>::zeros(n, 3);
			for j in 0..3 {
				for i in 0..n {
					expected[(perm[i], j)] = xb[(i, j)] * scale[i];
				}
			}
			interior_mut(xp.as_mut(), layout).copy_from(&xb);
			if run.aux(&ctx, || {
				balance_eigenvectors_in_place(
					interior_mut(xp.as_mut(), layout),
					&perm,
					&scale,
					Par::Seq,
					MemStack::new(&mut xmem),
				);
				Ok::<_, ()>(())
			}) {
				run.ratio(
					&ctx,
					"CHKBK backtransform",
					back_error(
						interior(xp.as_ref(), layout),
						expected.as_ref(),
					),
					20.0,
				);
			}
		}
		run.require(
			&ctx,
			"guard overwritten",
			guards_ok(ap.as_ref()) && guards_ok(xp.as_ref()),
		);
	}
}
fn balance_vector_case(run: &mut Run, n: usize, layout: usize) {
	let q = unitary(n);
	let lambda: Vec<_> = (0..n).map(|i| (i + 1) as Real).collect();
	let d: Vec<_> = (0..n)
		.map(|i| (2.0 as Real).powi((i % 3) as i32 * 6 - 6))
		.collect();
	let diagonal =
		Mat::from_fn(n, n, |i, j| if i == j { z(lambda[i]) } else { z(0.0) });
	let base = product(
		product(q.as_ref(), diagonal.as_ref()).as_ref(),
		adjoint(q.as_ref()).as_ref(),
	);
	let a = Mat::from_fn(n, n, |i, j| base[(i, j)] * d[i] / d[j]);
	let original = Mat::from_fn(n, n, |i, j| q[(i, j)] * d[i]);
	let mut work = a.clone();
	let mut perm = vec![0; n];
	let mut scale = vec![0.0; n];
	let mut mem = MemBuffer::new(balance_scratch());
	let ctx = format!("Balance known eigenvectors n={n} layout={layout}");
	if run.call(&ctx, || {
		balance_in_place(
			work.as_mut(),
			&mut perm,
			&mut scale,
			Par::Seq,
			MemStack::new(&mut mem),
		)
		.map(|_| ())
	}) {
		let mut xp = guarded(n, n, layout);
		let mut x = interior_mut(xp.as_mut(), layout);
		for j in 0..n {
			for i in 0..n {
				x[(i, j)] = original[(perm[i], j)] / scale[i];
			}
		}
		let mut mem = MemBuffer::new(balance_eigenvectors_scratch::<Scalar>(n));
		if !run.aux(&ctx, || {
			balance_eigenvectors_in_place(
				x,
				&perm,
				&scale,
				Par::Seq,
				MemStack::new(&mut mem),
			);
			Ok::<_, ()>(())
		}) {
			return;
		}
		let x = interior(xp.as_ref(), layout);
		run.ratio(
			&ctx,
			"known vectors",
			back_error(x, original.as_ref()),
			20.0,
		);
		run.ratio(
			&ctx,
			"original right residual",
			eigenvector_ratios(a.as_ref(), x, &lambda, &vec![0.0; n], false).0,
			20.0,
		);
		run.require(&ctx, "guard overwritten", guards_ok(xp.as_ref()));
	}
}
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_balance_cases() {
	let mut run = Run::new("balance");
	for n in [0, 1, 2, 3, 5, 10, 16] {
		for kind in 0..7 {
			let a = balance_input(n, kind);
			for layout in 0..4 {
				balance_case(&mut run, a.as_ref(), layout, kind);
			}
		}
	}
	for n in [3, 5, 10] {
		for layout in 0..4 {
			balance_vector_case(&mut run, n, layout);
		}
	}
	// First BAK input record: identity with packed [1,2,3,4,5], ILO=IHI=1.
	let mut x = Mat::<Scalar>::identity(5, 5);
	let mut mem = MemBuffer::new(balance_eigenvectors_scratch::<Scalar>(5));
	let ctx = "CHKBK first identity record";
	run.call(ctx, || {
		balance_eigenvectors_in_place(
			x.as_mut(),
			&[0, 1, 2, 3, 4],
			&[1.0; 5],
			Par::Seq,
			MemStack::new(&mut mem),
		);
		Ok::<_, ()>(())
	});
	run.ratio(
		ctx,
		"identity fixture",
		back_error(x.as_ref(), Mat::<Scalar>::identity(5, 5).as_ref()),
		20.0,
	);
	let mut a = Mat::<Scalar>::identity(3, 3);
	a[(0, 1)] = if <Scalar as ComplexField>::IS_REAL {
		z(Real::NAN)
	} else {
		scalar(1.0, Real::NAN)
	};
	let before = a.clone();
	let mut perm = [usize::MAX; 3];
	let mut scale = [-1.0; 3];
	let mut mem = MemBuffer::new(balance_scratch());
	run.call("Balance nonfinite error", || {
		let result = balance_in_place(
			a.as_mut(),
			&mut perm,
			&mut scale,
			Par::Seq,
			MemStack::new(&mut mem),
		);
		assert_eq!(result, Err(BalanceError::NonFiniteInput));
		Ok::<_, ()>(())
	});
	run.require(
		"Balance nonfinite error",
		"input or metadata changed",
		same_bits(a.as_ref(), before.as_ref())
			&& perm == [usize::MAX; 3]
			&& scale == [-1.0; 3],
	);
	run.finish(7 * 7 * 4 * 2 + 3 * 4 + 2);
}
