// CHKGE/GET01/GET02/GET03: rectangular LU, inverse and solve residuals; full pivoting extends GE properties.
// Adapted LAPACK LIN numerical properties, LAPACK 3.12.0, OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Deterministic Rust fixtures replace LATMS/LATMR.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD.

use crate::linalg::lu::{full_pivoting as fp, partial_pivoting as pp};
fn lu_fixture(m: usize, n: usize, kind: usize) -> Mat<Scalar> {
	if kind == 9 {
		return Mat::from_fn(m, n, |i, j| {
			if i == j {
				z(if i == 0 {
					Real::MIN_POSITIVE / 2.0
				} else {
					1.0
				})
			} else if j == 0 {
				z(Real::MIN_POSITIVE / 8.0)
			} else {
				z(0.0)
			}
		});
	}
	let scale = match kind {
		6 => Real::MAX.sqrt() / 16.0,
		7 => Real::MIN_POSITIVE.sqrt() * 16.0,
		_ => 1.0,
	};
	Mat::from_fn(m, n, |i, j| {
		scale
			* match kind {
				0 => z(0.0),
				1 => z(if i == j { 1.0 } else { 0.0 }),
				2 => {
					if j == 0 {
						z(0.0)
					} else {
						sample(i, j)
					}
				},
				3 => {
					if i == 0 {
						z(0.0)
					} else {
						sample(i, j)
					}
				},
				4 => phase(i) * z((j + 1) as Real),
				5 => {
					if i == j {
						phase(i)
							* Real::EPSILON
								.sqrt()
								.powf(i as Real / n.max(1) as Real)
					} else {
						z(0.0)
					}
				},
				_ => {
					if i == j {
						phase(i) * (m.max(n) + 1) as Real
					} else {
						sample(i, j) * 0.125
					}
				},
			}
	})
}
fn lu_reference(
	lu: MatRef<'_, Scalar>,
	row: &[usize],
	col: &[usize],
) -> Mat<Scalar> {
	let (m, n) = lu.shape();
	let k = m.min(n);
	let l = Mat::from_fn(m, k, |i, j| {
		if i == j {
			z(1.0)
		} else if i > j {
			lu[(i, j)]
		} else {
			z(0.0)
		}
	});
	let u = Mat::from_fn(k, n, |i, j| if i <= j { lu[(i, j)] } else { z(0.0) });
	let p = product(l.as_ref(), u.as_ref());
	let mut a = Mat::zeros(m, n);
	for j in 0..n {
		for i in 0..m {
			a[(row[i], col[j])] = p[(i, j)];
		}
	}
	a
}
#[test]
fn lapack_lu_checkers() {
	let lu = Mat::from_fn(3, 3, |i, j| {
		z([[2.0, 3.0, 4.0], [0.5, 5.0, 6.0], [0.25, 0.5, 7.0]][i][j])
	});
	let p = [1, 2, 0];
	let c = [2, 0, 1];
	let a = lu_reference(lu.as_ref(), &p, &c);
	assert_eq!(a[(1, 2)], z(2.0));
	assert_eq!(a[(2, 0)], z(6.5));
	assert_eq!(a[(0, 1)], z(11.0));
	let bad = lu_reference(lu.as_ref(), &[2, 0, 1], &c);
	crate::assert!(lin_difference(bad.as_ref(), a.as_ref(), 3) > 30.0);
}
fn lu_case(
	run: &mut Run,
	m: usize,
	n: usize,
	kind: usize,
	layout: usize,
	full: bool,
	mut par: Par,
) {
	let a = lu_fixture(m, n, kind);
	let mut ap = guarded(m, n, layout);
	let mut out = guarded(m, n, layout);
	let mut rhs = guarded(n, 3, layout);
	let mut inv = guarded(n, n, layout);
	let mut p = vec![usize::MAX; m];
	let mut ip = p.clone();
	let mut c = vec![usize::MAX; n];
	let mut ic = c.clone();
	let pp = Spec::new(pp::factor::PartialPivLuParams { ..auto!(Scalar) });
	let fp = Spec::new(fp::factor::FullPivLuParams { ..auto!(Scalar) });
	let req = if full {
		fp::factor::lu_in_place_scratch::<usize, Scalar>(m, n, par.shape(), fp)
	} else {
		pp::factor::lu_in_place_scratch::<usize, Scalar>(m, n, par.shape(), pp)
	};
	let mut mem = MemBuffer::new(req);
	let mut recmem = MemBuffer::new(if full {
		fp::reconstruct::reconstruct_scratch::<usize, Scalar>(m, n, par.shape())
	} else {
		pp::reconstruct::reconstruct_scratch::<usize, Scalar>(m, n, par.shape())
	});
	let mut smem = MemBuffer::new(if full {
		fp::solve::solve_in_place_scratch::<usize, Scalar>(n, 3, par.shape()).or(
			fp::solve::solve_transpose_in_place_scratch::<usize, Scalar>(
				n, 3, par.shape(),
			),
		)
	} else {
		pp::solve::solve_in_place_scratch::<usize, Scalar>(n, 3, par.shape()).or(
			pp::solve::solve_transpose_in_place_scratch::<usize, Scalar>(
				n, 3, par.shape(),
			),
		)
	});
	let mut imem = MemBuffer::new(if full {
		fp::inverse::inverse_scratch::<usize, Scalar>(n, par.shape())
	} else {
		pp::inverse::inverse_scratch::<usize, Scalar>(n, par.shape())
	});
	for pass in 0..2 {
		let input = scaled(a.as_ref(), if pass == 0 { 1.0 } else { 0.5 });
		interior_mut(ap.as_mut(), layout).copy_from(&input);
		let ctx = format!(
			"GE m={m} n={n} kind={kind} layout={layout} full={full} pass={pass} par={par:?}"
		);
		if run.call(&ctx, || {
			if full {
				fp::factor::lu_in_place(
					interior_mut(ap.as_mut(), layout),
					&mut p,
					&mut ip,
					&mut c,
					&mut ic,
					par.reborrow(),
					MemStack::new(&mut mem),
					fp,
				);
			} else {
				pp::factor::lu_in_place(
					interior_mut(ap.as_mut(), layout),
					&mut p,
					&mut ip,
					par.reborrow(),
					MemStack::new(&mut mem),
					pp,
				);
				for i in 0..n {
					c[i] = i;
					ic[i] = i;
				}
			}
			Ok::<_, ()>(())
		}) {
			let valid = permutation_ok(&p, &ip) && permutation_ok(&c, &ic);
			run.require(&ctx, "invalid permutation", valid);
			if valid {
				let lu = interior(ap.as_ref(), layout);
				let before = ap.clone();
				let rec = lu_reference(lu, &p, &c);
				run.ratio(
					&ctx,
					"independent LU reconstruction",
					lin_difference(rec.as_ref(), input.as_ref(), n),
					30.0,
				);
				let rp = crate::perm::PermRef::new_checked(&p, &ip, m);
				let cp = crate::perm::PermRef::new_checked(&c, &ic, n);
				if run.aux(&ctx, || {
					if full {
						fp::reconstruct::reconstruct(
							interior_mut(out.as_mut(), layout),
							lu,
							lu,
							rp,
							cp,
							par.reborrow(),
							MemStack::new(&mut recmem),
						);
					} else {
						pp::reconstruct::reconstruct(
							interior_mut(out.as_mut(), layout),
							lu,
							lu,
							rp,
							par.reborrow(),
							MemStack::new(&mut recmem),
						);
					}
					Ok::<_, ()>(())
				}) {
					run.ratio(
						&ctx,
						"API reconstruction",
						lin_difference(
							interior(out.as_ref(), layout),
							input.as_ref(),
							n,
						),
						30.0,
					);
				}
				if m == n && (5..=8).contains(&kind) || m == n && kind == 1 {
					let want = known_solution(n, 3);
					for mode in 0..3 {
						let op = op(input.as_ref(), mode);
						let b = product(op.as_ref(), want.as_ref());
						interior_mut(rhs.as_mut(), layout).copy_from(&b);
						if run.aux(&ctx,||{let conj=if mode==2{Conj::Yes}else{Conj::No};if full{if mode==0{fp::solve::solve_in_place_with_conj(lu,lu,rp,cp,conj,interior_mut(rhs.as_mut(),layout),par.reborrow(),MemStack::new(&mut smem));}else{fp::solve::solve_transpose_in_place_with_conj(lu,lu,rp,cp,conj,interior_mut(rhs.as_mut(),layout),par.reborrow(),MemStack::new(&mut smem));}}else{if mode==0{pp::solve::solve_in_place_with_conj(lu,lu,rp,conj,interior_mut(rhs.as_mut(),layout),par.reborrow(),MemStack::new(&mut smem));}else{pp::solve::solve_transpose_in_place_with_conj(lu,lu,rp,conj,interior_mut(rhs.as_mut(),layout),par.reborrow(),MemStack::new(&mut smem));}}Ok::<_,()>(())}){run.ratio(&ctx,&format!("solve mode {mode}"),solve_ratio(op.as_ref(),interior(rhs.as_ref(),layout),b.as_ref()),30.0);}
					}
					if run.aux(&ctx, || {
						if full {
							fp::inverse::inverse(
								interior_mut(inv.as_mut(), layout),
								lu,
								lu,
								rp,
								cp,
								par.reborrow(),
								MemStack::new(&mut imem),
							);
						} else {
							pp::inverse::inverse(
								interior_mut(inv.as_mut(), layout),
								lu,
								lu,
								rp,
								par.reborrow(),
								MemStack::new(&mut imem),
							);
						}
						Ok::<_, ()>(())
					}) {
						run.ratio(
							&ctx,
							"inverse",
							inverse_ratio(
								input.as_ref(),
								interior(inv.as_ref(), layout),
							),
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
		run.require(
			&ctx,
			"guards overwritten",
			guards_ok(ap.as_ref())
				&& guards_ok(out.as_ref())
				&& guards_ok(rhs.as_ref())
				&& guards_ok(inv.as_ref()),
		);
	}
}
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_lu_cases() {
	let mut run = Run::new("lu");
	let mut planned = 0;
	for (m, n) in [
		(0, 0),
		(0, 3),
		(3, 0),
		(1, 1),
		(2, 2),
		(3, 3),
		(5, 5),
		(10, 10),
		(50, 50),
		(5, 10),
		(10, 5),
		(15, 15),
		(16, 16),
		(17, 17),
		(63, 63),
		(64, 64),
		(65, 65),
	] {
		for kind in 0..8 {
			for full in [false, true] {
				lu_case(&mut run, m, n, kind, (m + kind) % 4, full, Par::Seq);
				planned += 2;
			}
		}
	}
	run.finish(planned);
}
#[cfg(feature = "rayon")]
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_lu_parallel() {
	let mut run = Run::new("lu_parallel");
	for full in [false, true] {
		lu_case(&mut run, 513, 513, 8, 0, full, Par::Seq);
		for threads in [2, 4] {
			rayon::ThreadPoolBuilder::new()
				.num_threads(threads)
				.build()
				.unwrap()
				.install(|| {
					lu_case(
						&mut run,
						513,
						513,
						8,
						threads / 2,
						full,
						Par::rayon(threads),
					)
				});
		}
	}
	run.finish(12);
}

#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_lu_subnormal() {
	let mut run = Run::new("lu_subnormal");
	for n in [1, 2] {
		for full in [false, true] {
			lu_case(&mut run, n, n, 9, 2, full, Par::Seq);
		}
	}
	run.finish(8);
}
