// CHKPO/DRVPO/CHKPS and POT01/02/03: lower-triangle LLT, rank-aware pivoted LLT, inverse and solves.
// Adapted LAPACK LIN numerical properties, LAPACK 3.12.0, OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Deterministic Rust fixtures replace LATMS/LATMR.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD.

use crate::linalg::cholesky::{llt, llt_pivoting as pllt};
fn positive_fixture(n: usize, kind: usize) -> Mat<Scalar> {
	let scale = match kind {
		6 => Real::MAX.sqrt() / 16.0,
		7 => Real::MIN_POSITIVE.sqrt() * 16.0,
		_ => 1.0,
	};
	let s: Vec<_> = (0..n)
		.map(|i| {
			scale
				* match kind {
					0 => 0.0,
					1 => 1.0,
					2 => 1.0 + i as Real / n.max(1) as Real,
					3 => {
						Real::EPSILON.sqrt().powf(i as Real / n.max(1) as Real)
					},
					4 => {
						if i < n / 2 {
							1.0
						} else {
							0.0
						}
					},
					5 => {
						if i == 0 {
							-1.0
						} else {
							1.0
						}
					},
					_ => 1.0 + i as Real / n.max(1) as Real,
				}
		})
		.collect();
	if kind == 5 {
		return Mat::from_fn(n, n, |i, j| {
			if i == j {
				z(if i == 0 { -1.0 } else { 1.0 })
			} else {
				z(0.0)
			}
		});
	}
	if kind == 4 {
		return Mat::from_fn(n, n, |i, j| {
			if i == j && i % 2 == 1 {
				z(scale)
			} else {
				z(0.0)
			}
		});
	}
	let q = unitary(n);
	let d = Mat::from_fn(n, n, |i, j| if i == j { z(s[i]) } else { z(0.0) });
	product(
		product(q.as_ref(), d.as_ref()).as_ref(),
		adjoint(q.as_ref()).as_ref(),
	)
}
fn cholesky_reference(l: MatRef<'_, Scalar>, p: &[usize]) -> Mat<Scalar> {
	let n = l.nrows();
	let l = Mat::from_fn(n, n, |i, j| if i >= j { l[(i, j)] } else { z(0.0) });
	let ll = product(l.as_ref(), adjoint(l.as_ref()).as_ref());
	let mut a = Mat::zeros(n, n);
	for j in 0..n {
		for i in 0..n {
			a[(p[i], p[j])] = ll[(i, j)];
		}
	}
	a
}
#[test]
fn lapack_cholesky_checkers() {
	let l = Mat::from_fn(3, 3, |i, j| {
		if i == j {
			z((i + 1) as Real)
		} else if i > j {
			phase(i + j) * 0.5
		} else {
			guard()
		}
	});
	let p = [1, 2, 0];
	let a = cholesky_reference(l.as_ref(), &p);
	assert_eq!(a[(1, 1)], z(1.0));
	assert_eq!(a[(0, 0)], z(9.5));
	let mut bad = l.clone();
	bad[(2, 2)] = z(0.0);
	let b = cholesky_reference(bad.as_ref(), &p);
	crate::assert!(lin_difference(b.as_ref(), a.as_ref(), 3) > 30.0);
	let b = cholesky_reference(l.as_ref(), &[2, 0, 1]);
	crate::assert!(lin_difference(b.as_ref(), a.as_ref(), 3) > 30.0);
}
fn cholesky_case(
	run: &mut Run,
	n: usize,
	kind: usize,
	layout: usize,
	pivot: bool,
	mut par: Par,
) {
	let a = positive_fixture(n, kind);
	let mut ap = guarded(n, n, layout);
	let mut out = guarded(n, n, layout);
	let mut rhs = guarded(n, 3, layout);
	let mut p = vec![usize::MAX; n];
	let mut ip = p.clone();
	let lp = Spec::new(llt::factor::LltParams { ..auto!(Scalar) });
	let pp = Spec::new(pllt::factor::PivLltParams { ..auto!(Scalar) });
	let mut mem = MemBuffer::new(if pivot {
		pllt::factor::cholesky_in_place_scratch::<usize, Scalar>(n, par.shape(), pp)
	} else {
		llt::factor::cholesky_in_place_scratch::<Scalar>(n, par.shape(), lp)
	});
	let mut rm = MemBuffer::new(if pivot {
		pllt::reconstruct::reconstruct_scratch::<usize, Scalar>(n, par.shape())
	} else {
		llt::reconstruct::reconstruct_scratch::<Scalar>(n, par.shape())
	});
	let mut sm = MemBuffer::new(if pivot {
		pllt::solve::solve_in_place_scratch::<usize, Scalar>(n, 3, par.shape())
	} else {
		llt::solve::solve_in_place_scratch::<Scalar>(n, 3, par.shape())
	});
	let mut im = MemBuffer::new(if pivot {
		pllt::inverse::inverse_scratch::<usize, Scalar>(n, par.shape())
	} else {
		llt::inverse::inverse_scratch::<Scalar>(n, par.shape())
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
			"PO/PS n={n} kind={kind} layout={layout} pivot={pivot} pass={pass} par={par:?}"
		);
		let mut result = None;
		if run.call(&ctx, || {
			result = Some(if pivot {
				pllt::factor::cholesky_in_place(
					interior_mut(ap.as_mut(), layout),
					&mut p,
					&mut ip,
					par.reborrow(),
					MemStack::new(&mut mem),
					pp,
				)
				.map(|x| x.0.rank)
			} else {
				for i in 0..n {
					p[i] = i;
					ip[i] = i;
				}
				llt::factor::cholesky_in_place(
					interior_mut(ap.as_mut(), layout),
					default(),
					par.reborrow(),
					MemStack::new(&mut mem),
					lp,
				)
				.map(|_| n)
			});
			Ok::<_, ()>(())
		}) {
			let result = result.unwrap();
			let nonsingular = kind != 0 && kind != 4 && kind != 5;
			let expected_error = n > 0 && kind == 5
				|| !pivot && n > 0 && (kind == 0 || kind == 4);
			run.require(
				&ctx,
				"unexpected factor status",
				result.is_err() == expected_error,
			);
			if let Ok(rank) = result {
				run.require(&ctx, "invalid rank", rank <= n);
				if pivot && kind == 4 {
					run.require(&ctx, "wrong semidefinite rank", rank == n / 2);
				}
				if pivot && kind == 0 {
					run.require(&ctx, "wrong zero rank", rank == 0 || n == 0);
				}
				let valid = permutation_ok(&p, &ip);
				run.require(&ctx, "invalid permutation", valid);
				if valid {
					let l = interior(ap.as_ref(), layout);
					let before = ap.clone();
					let rec = cholesky_reference(l, &p);
					run.ratio(
						&ctx,
						"LLH reconstruction",
						lin_difference(rec.as_ref(), input.as_ref(), n),
						30.0,
					);
					let perm = crate::perm::PermRef::new_checked(&p, &ip, n);
					if run.aux(&ctx, || {
						if pivot {
							pllt::reconstruct::reconstruct(
								interior_mut(out.as_mut(), layout),
								l,
								perm,
								par.reborrow(),
								MemStack::new(&mut rm),
							);
						} else {
							llt::reconstruct::reconstruct(
								interior_mut(out.as_mut(), layout),
								l,
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
					if nonsingular || n == 0 {
						let want = known_solution(n, 3);
						// Hermitian A: op(A, 1) = conj(A) exercises Conj::Yes.
						for mode in [0, 1] {
							let conj =
								if mode == 1 { Conj::Yes } else { Conj::No };
							let op = op(input.as_ref(), mode);
							let b = product(op.as_ref(), want.as_ref());
							interior_mut(rhs.as_mut(), layout).copy_from(&b);
							if run.aux(&ctx, || {
								if pivot {
									pllt::solve::solve_in_place_with_conj(
										l,
										perm,
										conj,
										interior_mut(rhs.as_mut(), layout),
										par.reborrow(),
										MemStack::new(&mut sm),
									);
								} else {
									llt::solve::solve_in_place_with_conj(
										l,
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
								pllt::inverse::inverse(
									interior_mut(out.as_mut(), layout),
									l,
									perm,
									par.reborrow(),
									MemStack::new(&mut im),
								);
							} else {
								llt::inverse::inverse(
									interior_mut(out.as_mut(), layout),
									l,
									par.reborrow(),
									MemStack::new(&mut im),
								);
							}
							Ok::<_, ()>(())
						}) {
							let inv =
								complete_lower(interior(out.as_ref(), layout));
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
fn lapack_cholesky_cases() {
	let mut run = Run::new("cholesky");
	let mut planned = 0;
	for n in [0, 1, 2, 3, 5, 10, 50, 63, 64, 65, 127, 128, 129] {
		for kind in 0..8 {
			for pivot in [false, true] {
				cholesky_case(
					&mut run,
					n,
					kind,
					(n + kind) % 4,
					pivot,
					Par::Seq,
				);
				planned += 2;
			}
		}
	}
	run.finish(planned);
}
#[cfg(feature = "rayon")]
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_cholesky_parallel() {
	let mut run = Run::new("cholesky_parallel");
	for pivot in [false, true] {
		cholesky_case(&mut run, 257, 2, 0, pivot, Par::Seq);
		for threads in [2, 4] {
			rayon::ThreadPoolBuilder::new()
				.num_threads(threads)
				.build()
				.unwrap()
				.install(|| {
					cholesky_case(
						&mut run,
						257,
						2,
						threads / 2,
						pivot,
						Par::rayon(threads),
					)
				});
		}
	}
	run.finish(12);
}
