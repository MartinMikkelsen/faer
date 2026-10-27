// S/DCHKSY, C/ZCHKHE and SYT01/HET01; GET02/03 solve/inverse properties adapted to separate D blocks.
// Adapted LAPACK LIN numerical properties, LAPACK 3.12.0, OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Deterministic Rust fixtures replace LATMS/LATMR.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD.

use crate::linalg::cholesky::{lblt, ldlt};
fn indefinite_fixture(n: usize, kind: usize) -> Mat<Scalar> {
	if kind == 0 {
		return Mat::zeros(n, n);
	}
	let scale = match kind {
		6 => Real::MAX.sqrt() / 16.0,
		7 => Real::MIN_POSITIVE.sqrt() * 16.0,
		_ => 1.0,
	};
	if kind == 4 {
		return Mat::from_fn(n, n, |i, j| {
			if i == j {
				z(if n % 2 == 1 && i + 1 == n { scale } else { 0.0 })
			} else if i / 2 == j / 2 {
				if i > j {
					phase(j) * scale
				} else {
					conj(&phase(i)) * scale
				}
			} else {
				z(0.0)
			}
		});
	}
	if kind == 5 {
		return Mat::from_fn(n, n, |i, j| {
			if i == j {
				z(if i + 1 == n {
					0.0
				} else {
					if i % 2 == 0 { 1.0 } else { -1.0 }
				})
			} else {
				z(0.0)
			}
		});
	}
	let q = unitary(n);
	let d = Mat::from_fn(n, n, |i, j| {
		if i == j {
			z(scale
				* match kind {
					1 => 1.0 + i as Real / n.max(1) as Real,
					2 => -1.0 - i as Real / n.max(1) as Real,
					3 => {
						if i % 2 == 0 {
							1.0
						} else {
							-Real::EPSILON.sqrt()
						}
					},
					_ => {
						if i % 2 == 0 {
							1.0
						} else {
							-1.0
						}
					},
				})
		} else {
			z(0.0)
		}
	});
	product(
		product(q.as_ref(), d.as_ref()).as_ref(),
		adjoint(q.as_ref()).as_ref(),
	)
}
fn indefinite_reference(
	l: MatRef<'_, Scalar>,
	d: ColRef<'_, Scalar>,
	sub: ColRef<'_, Scalar>,
	p: &[usize],
) -> Mat<Scalar> {
	let n = l.nrows();
	let l = Mat::from_fn(n, n, |i, j| {
		if i == j {
			z(1.0)
		} else if i > j {
			l[(i, j)]
		} else {
			z(0.0)
		}
	});
	let d = Mat::from_fn(n, n, |i, j| {
		if i == j {
			d[i]
		} else if i == j + 1 {
			sub[j]
		} else if j == i + 1 {
			conj(&sub[i])
		} else {
			z(0.0)
		}
	});
	let rec = product(
		product(l.as_ref(), d.as_ref()).as_ref(),
		adjoint(l.as_ref()).as_ref(),
	);
	let mut a = Mat::zeros(n, n);
	for j in 0..n {
		for i in 0..n {
			a[(p[i], p[j])] = rec[(i, j)];
		}
	}
	a
}
#[test]
fn lapack_indefinite_checkers_and_separate_d() {
	let mut l = Mat::from_fn(3, 3, |i, j| {
		if i > j {
			if (i, j) == (1, 0) {
				z(0.0)
			} else {
				phase(i + j) * 0.5
			}
		} else {
			guard()
		}
	});
	let d = Col::from_fn(3, |i| z([0.0, 0.0, 4.0][i]));
	let sub = Col::from_fn(3, |i| if i == 0 { phase(0) } else { z(0.0) });
	let p = [1, 2, 0];
	let ip = [2, 0, 1];
	let want = indefinite_reference(l.as_ref(), d.as_ref(), sub.as_ref(), &p);
	assert_eq!(want[(1, 1)], z(0.0));
	assert_eq!(want[(2, 1)], phase(0));
	let mut out = Mat::full(3, 3, guard());
	let mut mem = MemBuffer::new(lblt::reconstruct::reconstruct_scratch::<
		usize,
		Scalar,
	>(3, ParShape::Seq));
	lblt::reconstruct::reconstruct(
		out.as_mut(),
		l.as_ref(),
		d.as_ref().as_diagonal(),
		sub.as_ref().as_diagonal(),
		crate::perm::PermRef::new_checked(&p, &ip, 3),
		Par::Seq,
		MemStack::new(&mut mem),
	);
	let rec = complete_lower(out.as_ref());
	crate::assert!(lin_difference(rec.as_ref(), want.as_ref(), 3) < 30.0);
	l[(2, 1)] += z(1.0);
	let bad = indefinite_reference(l.as_ref(), d.as_ref(), sub.as_ref(), &p);
	crate::assert!(lin_difference(bad.as_ref(), want.as_ref(), 3) > 30.0);
}
fn indefinite_case(
	run: &mut Run,
	n: usize,
	kind: usize,
	layout: usize,
	pivot: bool,
	mut par: Par,
	prefill: bool,
) {
	let a = indefinite_fixture(n, kind);
	let mut ap = guarded(n, n, layout);
	let mut out = guarded(n, n, layout);
	let mut rhs = guarded(n, 3, layout);
	let mut sub = Col::full(n, if prefill { guard() } else { z(0.0) });
	let mut p = vec![usize::MAX; n];
	let mut ip = p.clone();
	let lp = Spec::new(ldlt::factor::LdltParams { ..auto!(Scalar) });
	let bp = Spec::new(lblt::factor::LbltParams { ..auto!(Scalar) });
	let mut mem = MemBuffer::new(if pivot {
		lblt::factor::cholesky_in_place_scratch::<usize, Scalar>(n, par.shape(), bp)
	} else {
		ldlt::factor::cholesky_in_place_scratch::<Scalar>(n, par.shape(), lp)
	});
	let mut rm = MemBuffer::new(if pivot {
		lblt::reconstruct::reconstruct_scratch::<usize, Scalar>(n, par.shape())
	} else {
		ldlt::reconstruct::reconstruct_scratch::<Scalar>(n, par.shape())
	});
	let mut sm = MemBuffer::new(if pivot {
		lblt::solve::solve_in_place_scratch::<usize, Scalar>(n, 3, par.shape())
	} else {
		ldlt::solve::solve_in_place_scratch::<Scalar>(n, 3, par.shape())
	});
	let mut im = MemBuffer::new(if pivot {
		lblt::inverse::inverse_scratch::<usize, Scalar>(n, par.shape())
	} else {
		ldlt::inverse::inverse_scratch::<Scalar>(n, par.shape())
	});
	for pass in 0..2 {
		let input = scaled(a.as_ref(), if pass == 0 { 1.0 } else { 0.5 });
		interior_mut(ap.as_mut(), layout).copy_from(&Mat::from_fn(
			n,
			n,
			|i, j| {
				if i >= j {
					input[(i, j)]
				} else {
					scalar(Real::NAN, Real::NAN)
				}
			},
		));
		let ctx = format!(
			"SY/HE n={n} kind={kind} layout={layout} pivot={pivot} prefill={prefill} pass={pass} par={par:?}"
		);
		let mut accepted = false;
		if run.call(&ctx, || {
			if pivot {
				lblt::factor::cholesky_in_place(
					interior_mut(ap.as_mut(), layout),
					sub.as_mut().as_diagonal_mut(),
					&mut p,
					&mut ip,
					par.reborrow(),
					MemStack::new(&mut mem),
					bp,
				);
				accepted = true;
			} else {
				for i in 0..n {
					p[i] = i;
					ip[i] = i;
					sub[i] = z(0.0);
				}
				accepted = ldlt::factor::cholesky_in_place(
					interior_mut(ap.as_mut(), layout),
					default(),
					par.reborrow(),
					MemStack::new(&mut mem),
					lp,
				)
				.is_ok();
			}
			Ok::<_, ()>(())
		}) {
			let expected_error =
				!pivot
					&& n > 0 && (kind == 0 || kind == 4 && n > 1 || kind == 5);
			run.require(
				&ctx,
				"unexpected factor status",
				accepted != expected_error,
			);
			if accepted {
				let valid = permutation_ok(&p, &ip);
				run.require(&ctx, "invalid permutation", valid);
				if valid {
					let d = Col::from_fn(n, |i| {
						interior(ap.as_ref(), layout)[(i, i)]
					});
					for i in 0..n {
						interior_mut(ap.as_mut(), layout)[(i, i)] = guard();
					}
					let before = ap.clone();
					let l = interior(ap.as_ref(), layout);
					let metadata = (0..n).all(|i| {
						imag(&d[i]) == 0.0
							&& real(&d[i]).is_finite()
							&& is_finite(&sub[i]) && (sub[i] == z(0.0)
							|| i + 1 < n
								&& (i == 0 || sub[i - 1] == z(0.0))
								&& sub[i + 1] == z(0.0))
					});
					run.require(
						&ctx,
						"invalid separate block metadata",
						metadata,
					);
					if metadata {
						let rec = indefinite_reference(
							l,
							d.as_ref(),
							sub.as_ref(),
							&p,
						);
						run.ratio(
							&ctx,
							"LDLH reconstruction",
							lin_difference(rec.as_ref(), input.as_ref(), n),
							30.0,
						);
						let perm =
							crate::perm::PermRef::new_checked(&p, &ip, n);
						if run.aux(&ctx, || {
							if pivot {
								lblt::reconstruct::reconstruct(
									interior_mut(out.as_mut(), layout),
									l,
									d.as_ref().as_diagonal(),
									sub.as_ref().as_diagonal(),
									perm,
									par.reborrow(),
									MemStack::new(&mut rm),
								);
							} else {
								ldlt::reconstruct::reconstruct(
									interior_mut(out.as_mut(), layout),
									l,
									d.as_ref().as_diagonal(),
									par.reborrow(),
									MemStack::new(&mut rm),
								);
							}
							Ok::<_, ()>(())
						}) {
							let rec =
								complete_lower(interior(out.as_ref(), layout));
							run.ratio(
								&ctx,
								"API reconstruction",
								lin_difference(rec.as_ref(), input.as_ref(), n),
								30.0,
							);
						}
						if kind != 0 && kind != 5 || n == 0 {
							let want = known_solution(n, 3);
							for conjugate in [false, true] {
								let op = if conjugate {
									Mat::from_fn(n, n, |i, j| {
										conj(&input[(i, j)])
									})
								} else {
									input.clone()
								};
								let b = product(op.as_ref(), want.as_ref());
								interior_mut(rhs.as_mut(), layout)
									.copy_from(&b);
								if run.aux(&ctx, || {
									let conj = if conjugate {
										Conj::Yes
									} else {
										Conj::No
									};
									if pivot {
										lblt::solve::solve_in_place_with_conj(
											l,
											d.as_ref().as_diagonal(),
											sub.as_ref().as_diagonal(),
											conj,
											perm,
											interior_mut(rhs.as_mut(), layout),
											par.reborrow(),
											MemStack::new(&mut sm),
										);
									} else {
										ldlt::solve::solve_in_place_with_conj(
											l,
											d.as_ref().as_diagonal(),
											conj,
											interior_mut(rhs.as_mut(), layout),
											par.reborrow(),
											MemStack::new(&mut sm),
										);
									}
									Ok::<_, ()>(())
								}) {
									run.ratio(
										&ctx,
										"solve",
										solve_ratio(
											op.as_ref(),
											interior(rhs.as_ref(), layout),
											b.as_ref(),
										),
										30.0,
									);
								}
							}
							if run.aux(&ctx, || {
								if pivot {
									lblt::inverse::inverse(
										interior_mut(out.as_mut(), layout),
										l,
										d.as_ref().as_diagonal(),
										sub.as_ref().as_diagonal(),
										perm,
										par.reborrow(),
										MemStack::new(&mut im),
									);
								} else {
									ldlt::inverse::inverse(
										interior_mut(out.as_mut(), layout),
										l,
										d.as_ref().as_diagonal(),
										par.reborrow(),
										MemStack::new(&mut im),
									);
								}
								Ok::<_, ()>(())
							}) {
								let inv = complete_lower(interior(
									out.as_ref(),
									layout,
								));
								run.ratio(
									&ctx,
									"inverse",
									inverse_ratio(input.as_ref(), inv.as_ref()),
									30.0,
								);
							}
						}
						run.require(
							&ctx,
							"factor changed",
							same_bits(ap.as_ref(), before.as_ref()),
						);
					}
				}
			}
		}
		run.require(
			&ctx,
			"guards overwritten",
			guards_ok(ap.as_ref())
				&& guards_ok(out.as_ref())
				&& guards_ok(rhs.as_ref()),
		);
	}
}
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_indefinite_cases() {
	let mut run = Run::new("indefinite");
	let mut planned = 0;
	for n in [0, 1, 2, 3, 5, 10, 50, 63, 64, 65, 127, 128, 129] {
		for kind in 0..8 {
			for pivot in [false, true] {
				for prefill in [false, true] {
					indefinite_case(
						&mut run,
						n,
						kind,
						(n + kind) % 4,
						pivot,
						Par::Seq,
						prefill,
					);
					planned += 2;
				}
			}
		}
	}
	run.finish(planned);
}
#[cfg(feature = "rayon")]
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_indefinite_parallel() {
	let mut run = Run::new("indefinite_parallel");
	for pivot in [false, true] {
		indefinite_case(&mut run, 257, 6, 0, pivot, Par::Seq, false);
		for threads in [2, 4] {
			rayon::ThreadPoolBuilder::new()
				.num_threads(threads)
				.build()
				.unwrap()
				.install(|| {
					indefinite_case(
						&mut run,
						257,
						6,
						threads / 2,
						pivot,
						Par::rayon(threads),
						false,
					)
				});
		}
	}
	run.finish(12);
}
