// Adapted CHKST tests 9-12, 22-26 / SYT21-HET21, LAPACK 3.12.0, OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Deterministic Rust fixtures replace LATMS/LATMR.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD.

fn ordered_spectrum_ratio(a: &[Real], b: &[Real]) -> Real {
	if a.len() != b.len() || a.iter().chain(b).any(|x| !x.is_finite()) {
		return Real::INFINITY;
	}
	let scale = a.iter().chain(b).fold(0.0 as Real, |m, x| m.max(x.abs()));
	if scale == 0.0 {
		return 0.0;
	}
	let error = a.iter().zip(b).fold(0.0 as Real, |m, (x, y)| {
		m.max((x / scale - y / scale).abs())
	});
	error / (Real::MIN_POSITIVE / scale).max(Real::EPSILON * error.max(1.0))
}
fn self_adjoint_residual(
	a: MatRef<'_, Scalar>,
	q: MatRef<'_, Scalar>,
	s: MatRef<'_, Scalar>,
) -> Real {
	let scale = max_entry(a);
	if !scale.is_finite()
		|| !max_entry(s).is_finite()
		|| !max_entry(q).is_finite()
	{
		return Real::INFINITY;
	}
	let scale = if scale == 0.0 { 1.0 } else { scale };
	let an = Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)] / scale);
	let av = product(an.as_ref(), q);
	let r = Mat::from_fn(q.nrows(), q.ncols(), |i, j| {
		av[(i, j)] - q[(i, j)] * (s[(j, 0)] / scale)
	});
	let rn = norm_one(r.as_ref());
	let norm = norm_one(an.as_ref());
	if norm == 0.0 {
		return if rn == 0.0 { 0.0 } else { Real::INFINITY };
	}
	rn / norm / Real::EPSILON / a.nrows().max(1) as Real
}
fn valid_real_spectrum(s: MatRef<'_, Scalar>) -> bool {
	(0..s.nrows()).all(|i| {
		real(&s[(i, 0)]).is_finite()
			&& imag(&s[(i, 0)]) == 0.0
			&& (i == 0 || real(&s[(i - 1, 0)]) <= real(&s[(i, 0)]))
	})
}
#[test]
fn lapack_tridiagonal_evd_checkers() {
	crate::assert!(ordered_spectrum_ratio(&[1.0, 2.0], &[1.0, 3.0]) > 60.0);
	crate::assert!(!ordered_spectrum_ratio(&[Real::NAN], &[1.0]).is_finite());
	assert_eq!(ordered_spectrum_ratio(&[0.0], &[0.0]), 0.0);
	let t = Mat::from_fn(2, 2, |i, j| {
		if i == j {
			z(0.0)
		} else if i > j {
			phase(0)
		} else {
			conj(&phase(0))
		}
	});
	let r = (0.5 as Real).sqrt();
	let q = Mat::from_fn(2, 2, |i, j| {
		if i == 0 {
			z(r)
		} else {
			phase(0) * if j == 0 { -r } else { r }
		}
	});
	let d = Mat::from_fn(2, 2, |i, j| {
		if i == j {
			z(if i == 0 { -1.0 } else { 1.0 })
		} else {
			z(0.0)
		}
	});
	crate::assert!(
		similarity_ratio(t.as_ref(), q.as_ref(), d.as_ref(), Real::EPSILON)
			< 60.0
	);
	let spec = Mat::from_fn(2, 1, |i, _| z(if i == 0 { -1.0 } else { 1.0 }));
	crate::assert!(
		self_adjoint_residual(t.as_ref(), q.as_ref(), spec.as_ref()) < 60.0
	);
	let bad = scaled(spec.as_ref(), 2.0);
	crate::assert!(
		self_adjoint_residual(t.as_ref(), q.as_ref(), bad.as_ref()) > 60.0
	);
	let mut wrong = q.clone();
	wrong[(1, 0)] = -wrong[(1, 0)];
	crate::assert!(
		similarity_ratio(t.as_ref(), wrong.as_ref(), d.as_ref(), Real::EPSILON)
			> 60.0
	);
	let mut s = Mat::from_fn(2, 1, |i, _| z(i as Real));
	crate::assert!(valid_real_spectrum(s.as_ref()));
	s[(0, 0)] = scalar(0.0, Real::NAN);
	if !<Scalar as ComplexField>::IS_REAL {
		crate::assert!(!valid_real_spectrum(s.as_ref()));
	}
}
fn tridiagonal_fixture(n: usize, kind: usize) -> (Mat<Scalar>, Vec<Real>) {
	let mut a = Mat::zeros(n, n);
	let mut known = Vec::new();
	let scale = match kind {
		6 => Real::MAX.sqrt(),
		7 => Real::MIN_POSITIVE.sqrt(),
		_ => 1.0,
	};
	for i in 0..n {
		a[(i, i)] = z(scale
			* match kind {
				0 | 4 => 0.0,
				1 => 1.0,
				2 => i as Real - 0.5 * n as Real,
				8 => 1.0 + 8.0 * Real::EPSILON * i as Real,
				_ => 2.0,
			});
		if i + 1 < n {
			a[(i + 1, i)] = phase(i)
				* scale * match kind {
				0 | 1 | 2 | 8 => 0.0,
				4 => {
					if i % 2 == 0 {
						1.0
					} else {
						0.0
					}
				},
				5 => Real::EPSILON.powf(i as Real / n.max(1) as Real),
				_ => 1.0,
			};
			a[(i, i + 1)] = conj(&a[(i + 1, i)]);
		}
	}
	if kind == 3 || kind == 6 || kind == 7 {
		known = (1..=n)
			.map(|k| {
				scale
					* (2.0
						- 2.0
							* ((k as Real) * (core::f64::consts::PI as Real)
								/ (n + 1) as Real)
								.cos())
			})
			.collect();
	} else if kind == 4 {
		known = (0..n)
			.map(|i| {
				if i < n / 2 {
					-1.0
				} else if i < n - n / 2 {
					0.0
				} else {
					1.0
				}
			})
			.collect();
	} else if kind != 5 {
		known = (0..n).map(|i| real(&a[(i, i)])).collect();
	}
	known.sort_by(|a, b| a.partial_cmp(b).unwrap());
	(a, known)
}
fn tridiagonal_evd_case(
	run: &mut Run,
	n: usize,
	kind: usize,
	layout: usize,
	threshold: usize,
	mut par: Par,
) {
	let (a, known) = tridiagonal_fixture(n, kind);
	let mut dp = guarded(n, 1, layout);
	let mut ep = guarded(n.saturating_sub(1), 1, layout);
	let mut sp = guarded(n, 1, layout);
	let mut qp = guarded(n, n, layout);
	let params = Spec::new(SelfAdjointEvdParams {
		recursion_threshold: threshold,
		..auto!(Scalar)
	});
	let mut mem = MemBuffer::new(
		tridiagonal_self_adjoint_evd_scratch::<Scalar>(
			n,
			ComputeEigenvectors::Yes,
			par.shape(),
			params,
		)
		.or(tridiagonal_self_adjoint_evd_scratch::<Scalar>(
			n,
			ComputeEigenvectors::No,
			par.shape(),
			params,
		)),
	);
	for pass in 0..2 {
		let f = if pass == 0 { 1.0 } else { 0.5 };
		let input = scaled(a.as_ref(), f);
		for i in 0..n {
			interior_mut(dp.as_mut(), layout)[(i, 0)] = input[(i, i)];
			if i + 1 < n {
				interior_mut(ep.as_mut(), layout)[(i, 0)] = input[(i + 1, i)];
			}
		}
		let db = dp.clone();
		let eb = ep.clone();
		let mut mode: Option<Vec<Real>> = None;
		for vectors in [false, true] {
			let ctx = format!(
				"CHKST n={n} kind={kind} layout={layout} threshold={threshold} vectors={vectors} pass={pass} par={par:?}"
			);
			if run.call(&ctx, || {
				tridiagonal_self_adjoint_evd(
					interior(dp.as_ref(), layout).col(0).as_diagonal(),
					interior(ep.as_ref(), layout).col(0).as_diagonal(),
					interior_mut(sp.as_mut(), layout)
						.col_mut(0)
						.as_diagonal_mut(),
					if vectors {
						Some(interior_mut(qp.as_mut(), layout))
					} else {
						None
					},
					par.reborrow(),
					MemStack::new(&mut mem),
					params,
				)
			}) {
				let s = interior(sp.as_ref(), layout);
				run.require(
					&ctx,
					"spectrum not ordered real finite",
					valid_real_spectrum(s),
				);
				let values: Vec<_> = (0..n).map(|i| real(&s[(i, 0)])).collect();
				if let Some(ref v) = mode {
					run.ratio(
						&ctx,
						"values/vector modes",
						ordered_spectrum_ratio(&values, v),
						60.0,
					);
				} else {
					mode = Some(values.clone());
				}
				if known.len() == n {
					let expected: Vec<_> =
						known.iter().map(|x| x * f).collect();
					run.ratio(
						&ctx,
						"known spectrum (normwise)",
						ordered_spectrum_ratio(&values, &expected),
						60.0,
					);
				}
				if vectors {
					let q = interior(qp.as_ref(), layout);
					let d = Mat::from_fn(n, n, |i, j| {
						if i == j { s[(i, 0)] } else { z(0.0) }
					});
					run.ratio(
						&ctx,
						"TV-VD",
						self_adjoint_residual(input.as_ref(), q, s),
						60.0,
					);
					run.ratio(
						&ctx,
						"T-QDQH",
						similarity_ratio(
							input.as_ref(),
							q,
							d.as_ref(),
							Real::EPSILON,
						),
						60.0,
					);
					run.ratio(
						&ctx,
						"unitary",
						unitary_ratio(q, Real::EPSILON),
						60.0,
					);
				}
			}
			run.require(
				&ctx,
				"input changed",
				same_bits(dp.as_ref(), db.as_ref())
					&& same_bits(ep.as_ref(), eb.as_ref()),
			);
			run.require(
				&ctx,
				"guard overwritten",
				guards_ok(sp.as_ref()) && guards_ok(qp.as_ref()),
			);
		}
	}
}
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_tridiagonal_evd_cases() {
	let mut run = Run::new("tridiagonal_evd");
	for n in [0, 1, 2, 3, 5, 20, 7, 8, 9, 127, 128, 129] {
		for kind in 0..9 {
			for threshold in [8, 128] {
				tridiagonal_evd_case(
					&mut run,
					n,
					kind,
					(n + kind) % 4,
					threshold,
					Par::Seq,
				);
			}
		}
	}
	run.finish(12 * 9 * 2 * 4);
}
#[cfg(feature = "rayon")]
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_tridiagonal_evd_parallel() {
	let mut run = Run::new("tridiagonal_evd_parallel");
	tridiagonal_evd_case(&mut run, 257, 3, 0, 128, Par::Seq);
	for threads in [2, 4] {
		rayon::ThreadPoolBuilder::new()
			.num_threads(threads)
			.build()
			.unwrap()
			.install(|| {
				tridiagonal_evd_case(
					&mut run,
					257,
					3,
					threads / 2,
					128,
					Par::rayon(threads),
				)
			});
	}
	run.finish(12);
}
