// DRVLS/QRT16/QRT17 residuals; QRT14 row-space property adapted to literal SVD minimum-norm solutions.
// Adapted LAPACK LIN numerical properties, LAPACK 3.12.0, OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Deterministic Rust fixtures replace LATMS/LATMR.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD.

use crate::linalg::{
	qr::{col_pivoting as cp, no_pivoting as np},
	solvers::{SolveLstsqCore, Svd},
};
fn normal_ratio(
	a: MatRef<'_, Scalar>,
	x: MatRef<'_, Scalar>,
	b: MatRef<'_, Scalar>,
	inconsistent: bool,
) -> Real {
	if a.nrows() == 0 || a.ncols() == 0 || b.ncols() == 0 {
		return 0.0;
	}
	let scale = max_entry(a);
	if !scale.is_finite()
		|| !max_entry(x).is_finite()
		|| !max_entry(b).is_finite()
	{
		return Real::INFINITY;
	}
	if scale == 0.0 {
		return 0.0;
	}
	let an = Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)] / scale);
	let bn = Mat::from_fn(b.nrows(), b.ncols(), |i, j| b[(i, j)] / scale);
	let ax = product(an.as_ref(), x);
	let r = Mat::from_fn(b.nrows(), b.ncols(), |i, j| bn[(i, j)] - ax[(i, j)]);
	let rn = max_entry(r.as_ref());
	if !rn.is_finite() {
		return Real::INFINITY;
	}
	if rn == 0.0 {
		return 0.0;
	}
	let rnorm = Mat::from_fn(r.nrows(), r.ncols(), |i, j| r[(i, j)] / rn);
	let ra = product(adjoint(rnorm.as_ref()).as_ref(), an.as_ref());
	let error = norm_one(ra.as_ref()) / norm_one(an.as_ref());
	let alpha = if inconsistent {
		rn
	} else {
		norm_one(bn.as_ref())
	};
	if alpha == 0.0 {
		return if error == 0.0 { 0.0 } else { Real::INFINITY };
	}
	error * (rn / alpha)
		/ lin_eps()
		/ a.nrows().max(a.ncols()).max(b.ncols()).max(1) as Real
}
#[test]
fn lapack_lstsq_checkers() {
	let a = Mat::from_fn(3, 2, |i, j| if i == j { phase(j) } else { z(0.0) });
	let x = Mat::from_fn(2, 1, |i, _| z((i + 1) as Real));
	let mut b = product(a.as_ref(), x.as_ref());
	b[(2, 0)] = phase(1);
	crate::assert!(
		normal_ratio(a.as_ref(), x.as_ref(), b.as_ref(), true) == 0.0
	);
	let mut wrong = x.clone();
	wrong[(0, 0)] += z(1.0);
	crate::assert!(
		normal_ratio(a.as_ref(), wrong.as_ref(), b.as_ref(), true) > 30.0
	);
}
fn qr_lstsq_case(
	run: &mut Run,
	m: usize,
	n: usize,
	kind: usize,
	layout: usize,
	pivot: bool,
	mut par: Par,
) {
	let q = unitary(m);
	let sigma: Vec<_> = (0..n)
		.map(|i| {
			if kind == 1 {
				Real::EPSILON.sqrt().powf(i as Real / n.max(1) as Real)
			} else {
				1.0 + i as Real / n.max(1) as Real
			}
		})
		.collect();
	let scale = match kind {
		2 => Real::MAX.sqrt() / 16.0,
		3 => Real::MIN_POSITIVE.sqrt() * 16.0,
		_ => 1.0,
	};
	let a = Mat::from_fn(m, n, |i, j| q[(i, j)] * (sigma[j] * scale));
	let want = known_solution(n, 3);
	let mut ap = guarded(m, n, layout);
	let mut xp = guarded(m, 3, layout);
	let mut coeff = Mat::full(4, n, guard());
	let mut p = vec![usize::MAX; n];
	let mut ip = p.clone();
	let np = Spec::new(np::factor::QrParams { ..auto!(Scalar) });
	let cp = Spec::new(cp::factor::ColPivQrParams { ..auto!(Scalar) });
	let mut mem = MemBuffer::new(if pivot {
		cp::factor::qr_in_place_scratch::<usize, Scalar>(m, n, 4, par.shape(), cp)
	} else {
		np::factor::qr_in_place_scratch::<Scalar>(m, n, 4, par.shape(), np)
	});
	let mut sm = MemBuffer::new(if pivot {
		cp::solve::solve_lstsq_in_place_scratch::<usize, Scalar>(
			m, n, 4, 3, par.shape(),
		)
	} else {
		np::solve::solve_lstsq_in_place_scratch::<Scalar>(m, n, 4, 3, par.shape())
	});
	for pass in 0..2 {
		let input = scaled(a.as_ref(), if pass == 0 { 1.0 } else { 0.5 });
		interior_mut(ap.as_mut(), layout).copy_from(&input);
		let ctx = format!(
			"LS m={m} n={n} kind={kind} pivot={pivot} layout={layout} pass={pass} par={par:?}"
		);
		if run.call(&ctx, || {
			if pivot {
				cp::factor::qr_in_place(
					interior_mut(ap.as_mut(), layout),
					coeff.as_mut(),
					&mut p,
					&mut ip,
					par.reborrow(),
					MemStack::new(&mut mem),
					cp,
				);
			} else {
				np::factor::qr_in_place(
					interior_mut(ap.as_mut(), layout),
					coeff.as_mut(),
					par.reborrow(),
					MemStack::new(&mut mem),
					np,
				);
				for i in 0..n {
					p[i] = i;
					ip[i] = i;
				}
			}
			Ok::<_, ()>(())
		}) {
			run.require(&ctx, "invalid permutation", permutation_ok(&p, &ip));
			if permutation_ok(&p, &ip) {
				let before = ap.clone();
				let packed = interior(ap.as_ref(), layout);
				let perm = crate::perm::PermRef::new_checked(&p, &ip, n);
				for conjugate in [false, true] {
					for inconsistent in [false, true] {
						let op = Mat::from_fn(m, n, |i, j| {
							if conjugate {
								conj(&input[(i, j)])
							} else {
								input[(i, j)]
							}
						});
						let mut b = product(op.as_ref(), want.as_ref());
						if inconsistent && m > n {
							for j in 0..3 {
								for i in 0..m {
									b[(i, j)] += (if conjugate {
										conj(&q[(i, n)])
									} else {
										q[(i, n)]
									}) * (scale
										* if pass == 0 { 1.0 } else { 0.5 })
										* (j + 1) as Real;
								}
							}
						}
						interior_mut(xp.as_mut(), layout).copy_from(&b);
						if run.aux(&ctx, || {
							let c =
								if conjugate { Conj::Yes } else { Conj::No };
							if pivot {
								cp::solve::solve_lstsq_in_place_with_conj(
									packed,
									coeff.as_ref(),
									packed.subrows(0, n),
									perm,
									c,
									interior_mut(xp.as_mut(), layout),
									par.reborrow(),
									MemStack::new(&mut sm),
								);
							} else {
								np::solve::solve_lstsq_in_place_with_conj(
									packed,
									coeff.as_ref(),
									packed.subrows(0, n),
									c,
									interior_mut(xp.as_mut(), layout),
									par.reborrow(),
									MemStack::new(&mut sm),
								);
							}
							Ok::<_, ()>(())
						}) {
							let x = interior(xp.as_ref(), layout).subrows(0, n);
							run.ratio(
								&ctx,
								"normal residual",
								normal_ratio(
									op.as_ref(),
									x,
									b.as_ref(),
									inconsistent && m > n,
								),
								30.0,
							);
							run.ratio(
								&ctx,
								"condition-scaled known solution",
								lin_difference(x, want.as_ref(), m)
									* sigma
										.iter()
										.fold(1.0 as Real, |m, x| m.min(*x)),
								30.0,
							);
							if !inconsistent {
								run.ratio(
									&ctx,
									"QRT16 residual",
									solve_ratio(op.as_ref(), x, b.as_ref())
										/ m.max(n).max(1) as Real,
									30.0,
								);
							}
						}
					}
				}
				run.require(
					&ctx,
					"factor changed",
					same_bits(ap.as_ref(), before.as_ref()),
				);
			}
		}
		run.require(
			&ctx,
			"guards overwritten",
			guards_ok(ap.as_ref()) && guards_ok(xp.as_ref()),
		);
	}
}
fn svd_lstsq_case(
	run: &mut Run,
	m: usize,
	n: usize,
	kind: usize,
	layout: usize,
	mut par: Par,
) {
	let a = if kind == 1 {
		Mat::from_fn(m, n, |i, j| {
			if i == 0 {
				if j == 0 {
					z(1.0)
				} else if j == 1 {
					phase(0) * 2.0
				} else {
					z(0.0)
				}
			} else {
				z(0.0)
			}
		})
	} else {
		Mat::from_fn(m, n, |i, j| {
			if i == j && i + 1 < n {
				phase(j) * (i + 1) as Real
			} else {
				z(0.0)
			}
		})
	};
	let mut ap = guarded(m, n, layout);
	interior_mut(ap.as_mut(), layout).copy_from(&a);
	let before = ap.clone();
	let mut xp = guarded(m, 3, layout);
	let ctx = format!("SVD LS m={m} n={n} kind={kind} layout={layout}");
	let mut svd = None;
	if run.call(&ctx, || {
		svd = Some(Svd::new_thin(
			interior(ap.as_ref(), layout),
			par.reborrow(),
		)?);
		Ok::<_, crate::linalg::svd::SvdError>(())
	}) {
		let svd = svd.unwrap();
		for pass in 0..2 {
			for conjugate in [false, true] {
				let b = Mat::from_fn(m, 3, |i, j| {
					if i == 0 {
						z((j + 1) as Real * if pass == 0 { 1.0 } else { 0.5 })
					} else if i + 1 == m {
						phase(j)
					} else {
						z(0.0)
					}
				});
				interior_mut(xp.as_mut(), layout).copy_from(&b);
				let mut want = Mat::zeros(n, 3);
				for j in 0..3 {
					if kind == 1 {
						want[(0, j)] = b[(0, j)] / 5.0;
						if n > 1 {
							want[(1, j)] = (if conjugate {
								phase(0)
							} else {
								conj(&phase(0))
							}) * (2.0 / 5.0) * b[(0, j)];
						}
					} else if n > 1 {
						want[(0, j)] = b[(0, j)]
							/ if conjugate {
								conj(&phase(0))
							} else {
								phase(0)
							};
					}
				}
				if run.aux(&ctx, || {
					svd.solve_lstsq_in_place_with_conj(
						if conjugate { Conj::Yes } else { Conj::No },
						interior_mut(xp.as_mut(), layout),
						par.reborrow(),
					);
					Ok::<_, ()>(())
				}) {
					let x = interior(xp.as_ref(), layout).subrows(0, n);
					let op = Mat::from_fn(m, n, |i, j| {
						if conjugate {
							conj(&a[(i, j)])
						} else {
							a[(i, j)]
						}
					});
					run.ratio(
						&ctx,
						"literal minimum-norm solution",
						lin_difference(x, want.as_ref(), n),
						30.0,
					);
					run.ratio(
						&ctx,
						"normal residual",
						normal_ratio(op.as_ref(), x, b.as_ref(), true),
						30.0,
					);
					if kind == 0 && n > 0 {
						run.require(
							&ctx,
							"nonzero nullspace component",
							(0..3).all(|j| {
								magnitude(x[(n - 1, j)]) < 30.0 * Real::EPSILON
							}),
						);
					}
				}
			}
		}
		run.require(
			&ctx,
			"input changed",
			same_bits(ap.as_ref(), before.as_ref()),
		);
	}
	run.require(&ctx, "guards overwritten", guards_ok(xp.as_ref()));
}
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_lstsq_cases() {
	let mut run = Run::new("lstsq");
	for (m, n) in [
		(0, 0),
		(1, 0),
		(1, 1),
		(3, 2),
		(5, 3),
		(10, 5),
		(50, 10),
		(63, 32),
		(64, 32),
		(65, 32),
	] {
		for kind in 0..4 {
			for pivot in [false, true] {
				qr_lstsq_case(
					&mut run,
					m,
					n,
					kind,
					(m + kind) % 4,
					pivot,
					Par::Seq,
				);
			}
		}
	}
	for (m, n) in [(3, 2), (5, 3), (10, 5)] {
		for kind in [0, 1] {
			for layout in 0..4 {
				svd_lstsq_case(&mut run, m, n, kind, layout, Par::Seq);
			}
		}
	}
	run.finish(10 * 4 * 2 * 2 + 3 * 2 * 4);
}
#[cfg(feature = "rayon")]
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_lstsq_parallel() {
	let mut run = Run::new("lstsq_parallel");
	for pivot in [false, true] {
		qr_lstsq_case(&mut run, 257, 193, 0, 0, pivot, Par::Seq);
		for threads in [2, 4] {
			rayon::ThreadPoolBuilder::new()
				.num_threads(threads)
				.build()
				.unwrap()
				.install(|| {
					qr_lstsq_case(
						&mut run,
						257,
						193,
						0,
						threads / 2,
						pivot,
						Par::rayon(threads),
					)
				});
		}
	}
	run.finish(12);
}
