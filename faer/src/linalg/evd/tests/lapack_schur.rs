// CHKHS 3-8 / HST01 properties adapted from LAPACK 3.12.0, OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Native fixed families replace LATMR.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD.
fn hessenberg_input(n: usize, kind: usize) -> Mat<Scalar> {
	let d = spectrum(n, kind);
	Mat::from_fn(n, n, |i, j| {
		if i > j + 1 {
			return z(0.0);
		}
		match kind {
			0 => z(0.0),
			1 => z(if i == j { 1.0 } else { 0.0 }),
			2 | 6 | 7 => {
				if i == j {
					z(d[i])
				} else {
					z(0.0)
				}
			},
			3 => z(if i == j || i == j + 1 { 1.0 } else { 0.0 }),
			4 => {
				if i == j {
					z(1.0)
				} else if i % 2 == 0 && j == i + 1 {
					z(1.0)
				} else if j % 2 == 0 && i == j + 1 {
					z(-1.0)
				} else {
					z(0.0)
				}
			},
			5 => {
				if i == j {
					z(d[i])
				} else if i < j {
					0.25 * phase(i + j)
				} else {
					z(0.0)
				}
			},
			_ => sample(i, j),
		}
	})
}
fn schur_form(t: MatRef<'_, Scalar>) -> bool {
	let n = t.nrows();
	if !max_entry(t).is_finite() {
		return false;
	}
	for j in 0..n {
		for i in j + 2..n {
			if t[(i, j)] != z(0.0) {
				return false;
			}
		}
	}
	let mut i = 0;
	while i + 1 < n {
		if t[(i + 1, i)] != z(0.0) {
			if !<Scalar as ComplexField>::IS_REAL {
				return false;
			}
			if i + 2 < n && t[(i + 2, i + 1)] != z(0.0) {
				return false;
			}
			if real(&t[(i, i)]) != real(&t[(i + 1, i + 1)]) {
				return false;
			}
			// Standardized blocks need b*c < 0; compare signs so that
			// an underflowing product cannot hide a valid block.
			let b = real(&t[(i, i + 1)]);
			let c = real(&t[(i + 1, i)]);
			if b == 0.0 || (b > 0.0) == (c > 0.0) {
				return false;
			}
			i += 2;
		} else {
			i += 1;
		}
	}
	true
}
fn eigenvalue_gap(a: &[(Real, Real)], b: &[(Real, Real)], eps: Real) -> Real {
	assert_eq!(a.len(), b.len());
	if a.iter()
		.chain(b)
		.any(|&(r, i)| !r.is_finite() || !i.is_finite())
	{
		return Real::INFINITY;
	}
	let mut a = a.to_vec();
	let mut b = b.to_vec();
	let order = |x: &(Real, Real), y: &(Real, Real)| {
		x.0.total_cmp(&y.0).then(x.1.total_cmp(&y.1))
	};
	a.sort_by(order);
	b.sort_by(order);
	let scale = b.iter().fold(0.0 as Real, |m, x| m.max(x.0.hypot(x.1)));
	if scale == 0.0 {
		return if a.iter().all(|x| x.0 == 0.0 && x.1 == 0.0) {
			0.0
		} else {
			Real::INFINITY
		};
	}
	a.iter().zip(b).fold(0.0 as Real, |m, (x, y)| {
		m.max((x.0 / scale - y.0 / scale).hypot(x.1 / scale - y.1 / scale))
	}) / eps
}
fn schur_case(
	run: &mut Run,
	h: MatRef<'_, Scalar>,
	kind: usize,
	layout: usize,
	mut par: Par,
) {
	let n = h.nrows();
	let mut hp = guarded(n, n, layout);
	let mut qp = guarded(n, n, layout);
	let mut wr = Col::full(n, Real::NAN);
	let mut wi = Col::full(n, Real::NAN);
	let params = Spec::new(SchurParams { ..auto!(Scalar) });
	let mut reference: [Option<Vec<(Real, Real)>>; 2] = [None, None];
	for want_q in [true, false] {
		let mut mem = MemBuffer::new(hessenberg_schur_scratch::<Scalar>(
			n, want_q, par.shape(), params,
		));
		for pass in 0..2 {
			let input = scaled(h, if pass == 0 { 1.0 } else { 0.5 });
			interior_mut(hp.as_mut(), layout).copy_from(&input);
			let initial = if pass == 0 {
				Mat::<Scalar>::identity(n, n)
			} else {
				unitary(n)
			};
			interior_mut(qp.as_mut(), layout).copy_from(&initial);
			let a = product(
				product(initial.as_ref(), input.as_ref()).as_ref(),
				adjoint(initial.as_ref()).as_ref(),
			);
			let ctx = format!(
				"Schur kind={kind} n={n} layout={layout} Q={want_q} pass={pass} par={par:?}"
			);
			wr.as_mut().fill(Real::NAN);
			wi.as_mut().fill(Real::NAN);
			if run.call(&ctx, || {
				schur_call(
					interior_mut(hp.as_mut(), layout),
					if want_q {
						Some(interior_mut(qp.as_mut(), layout))
					} else {
						None
					},
					wr.as_mut(),
					wi.as_mut(),
					par.reborrow(),
					MemStack::new(&mut mem),
					params,
				)
			}) {
				let t = interior(hp.as_ref(), layout);
				let values: Vec<_> = (0..n).map(|i| (wr[i], wi[i])).collect();
				run.require(&ctx, "invalid Schur form", schur_form(t));
				run.require(
					&ctx,
					"nonfinite spectrum",
					values.iter().all(|x| x.0.is_finite() && x.1.is_finite()),
				);
				if want_q {
					let q = interior(qp.as_ref(), layout);
					run.ratio(
						&ctx,
						"HST01 reconstruction",
						similarity_ratio(a.as_ref(), q, t, Real::EPSILON),
						20.0,
					);
					run.ratio(
						&ctx,
						"Q unitary",
						unitary_ratio(q, Real::EPSILON),
						20.0,
					);
				}
				// CHKHS's condition-weighted comparisons have no faer estimator;
				// compare modes only for these normal inputs, not Jordan/nonnormal cases.
				if matches!(kind, 0 | 1 | 2 | 4 | 6 | 7) {
					if let Some(s) = &reference[pass] {
						run.ratio(
							&ctx,
							"spectrum modes",
							eigenvalue_gap(&values, s, Real::EPSILON),
							20.0,
						);
					} else {
						reference[pass] = Some(values);
					}
				}
			}
			run.require(
				&ctx,
				"guard overwritten",
				guards_ok(hp.as_ref()) && guards_ok(qp.as_ref()),
			);
		}
	}
}
#[test]
fn lapack_schur_checkers() {
	let t = Mat::from_fn(2, 2, |i, j| {
		z(if i == j {
			1.0
		} else if i < j {
			1.0
		} else {
			-1.0
		})
	});
	assert_eq!(schur_form(t.as_ref()), <Scalar as ComplexField>::IS_REAL);
	let t = Mat::from_fn(2, 2, |i, j| z(if i < j { 0.0 } else { 1.0 }));
	crate::assert!(!schur_form(t.as_ref()));
	crate::assert!(
		eigenvalue_gap(&[(1.0, 0.0)], &[(2.0, 0.0)], Real::EPSILON) > 20.0
	);
	let q = unitary(2);
	let d = Mat::<Scalar>::identity(2, 2);
	crate::assert!(
		similarity_ratio(d.as_ref(), q.as_ref(), d.as_ref(), Real::EPSILON)
			< 20.0
	);
	let bad = Mat::<Scalar>::full(2, 2, z(2.0));
	crate::assert!(
		similarity_ratio(d.as_ref(), q.as_ref(), bad.as_ref(), Real::EPSILON)
			> 20.0
	);
}
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_schur_cases() {
	let mut run = Run::new("schur");
	for n in [0, 1, 2, 3, 5, 10, 16, 74, 75, 76] {
		for kind in 0..9 {
			let h = hessenberg_input(n, kind);
			for layout in 0..4 {
				schur_case(&mut run, h.as_ref(), kind, layout, Par::Seq);
			}
		}
	}
	run.finish(10 * 9 * 4 * 4);
}
#[cfg(feature = "rayon")]
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_schur_parallel() {
	let mut run = Run::new("schur_parallel");
	let h = hessenberg_input(257, 8);
	schur_case(&mut run, h.as_ref(), 8, 0, Par::Seq);
	for threads in [2, 4] {
		rayon::ThreadPoolBuilder::new()
			.num_threads(threads)
			.build()
			.unwrap()
			.install(|| {
				schur_case(&mut run, h.as_ref(), 8, 0, Par::rayon(threads))
			});
	}
	run.finish(12);
}
