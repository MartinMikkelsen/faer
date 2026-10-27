// CHKTR/TRT02: lower/upper and implicit-unit TRS; no overflow-scaled LATRS contract.
// Adapted LAPACK LIN numerical properties, LAPACK 3.12.0, OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Deterministic Rust fixtures replace LATMS/LATMR.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD.

use crate::linalg::triangular_solve as ts;
fn triangular_case(
	run: &mut Run,
	n: usize,
	nrhs: usize,
	upper: bool,
	unit: bool,
	scale: Real,
	layout: usize,
	mut par: Par,
) {
	let actual = Mat::from_fn(n, n, |i, j| {
		if i == j {
			if unit {
				z(1.0)
			} else {
				phase(i) * (scale * (1.0 + i as Real / n.max(1) as Real))
			}
		} else if (i < j) == upper {
			phase(i + j) * (if unit { 1.0 } else { scale })
				/ (8.0 * (n + 1) as Real)
		} else {
			z(0.0)
		}
	});
	let mut ap = guarded(n, n, layout);
	let stored = Mat::from_fn(n, n, |i, j| {
		if i == j && unit || i != j && (i < j) != upper {
			scalar(Real::NAN, Real::NAN)
		} else {
			actual[(i, j)]
		}
	});
	interior_mut(ap.as_mut(), layout).copy_from(&stored);
	let before = ap.clone();
	let mut xp = guarded(n, nrhs, layout);
	let want = if scale < Real::MIN_POSITIVE {
		Mat::full(n, nrhs, z(1.0))
	} else {
		known_solution(n, nrhs)
	};
	for pass in 0..2 {
		for mode in 0..3 {
			let logical = op(actual.as_ref(), mode);
			let b = product(logical.as_ref(), want.as_ref());
			interior_mut(xp.as_mut(), layout).copy_from(&b);
			let ctx = format!(
				"TR n={n} rhs={nrhs} upper={upper} unit={unit} scale={scale:e} layout={layout} mode={mode} pass={pass} par={par:?}"
			);
			if run.call(&ctx, || {
				let a = interior(ap.as_ref(), layout);
				let a = if mode == 0 { a } else { a.transpose() };
				let upper = upper ^ (mode != 0);
				let conj = if mode == 2 { Conj::Yes } else { Conj::No };
				let x = interior_mut(xp.as_mut(), layout);
				match (upper, unit) {
					(false, false) => {
						ts::solve_lower_triangular_in_place_with_conj(
							a, conj, x, par.reborrow(),
						)
					},
					(false, true) => {
						ts::solve_unit_lower_triangular_in_place_with_conj(
							a, conj, x, par.reborrow(),
						)
					},
					(true, false) => {
						ts::solve_upper_triangular_in_place_with_conj(
							a, conj, x, par.reborrow(),
						)
					},
					(true, true) => {
						ts::solve_unit_upper_triangular_in_place_with_conj(
							a, conj, x, par.reborrow(),
						)
					},
				}
				Ok::<_, ()>(())
			}) {
				let x = interior(xp.as_ref(), layout);
				run.ratio(
					&ctx,
					"TRT02 residual",
					solve_ratio(logical.as_ref(), x, b.as_ref()),
					30.0,
				);
				run.ratio(
					&ctx,
					"known solution",
					lin_difference(x, want.as_ref(), n),
					30.0,
				);
			}
			run.require(
				&ctx,
				"input changed",
				same_bits(ap.as_ref(), before.as_ref()),
			);
			run.require(&ctx, "guard overwritten", guards_ok(xp.as_ref()));
		}
	}
}
#[test]
fn lapack_triangular_checkers() {
	let a = Mat::from_fn(2, 2, |i, j| {
		if i == j {
			z(2.0)
		} else if i == 1 {
			phase(0)
		} else {
			z(0.0)
		}
	});
	let x = Mat::full(2, 1, z(1.0));
	let b = product(a.as_ref(), x.as_ref());
	assert_eq!(b[(0, 0)], z(2.0));
	assert_eq!(b[(1, 0)], z(2.0) + phase(0));
	let mut bad = x.clone();
	bad[(1, 0)] = z(0.0);
	crate::assert!(solve_ratio(a.as_ref(), bad.as_ref(), b.as_ref()) > 30.0);
	if !<Scalar as ComplexField>::IS_REAL {
		let wrong = op(a.as_ref(), 1);
		let correct = op(a.as_ref(), 2);
		crate::assert!(
			lin_difference(wrong.as_ref(), correct.as_ref(), 2) > 30.0
		);
	}
}
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_triangular_cases() {
	let mut run = Run::new("triangular");
	let mut planned = 0;
	for n in [0, 1, 2, 3, 5, 10, 50, 63, 64, 65, 127, 128, 129] {
		for nrhs in [0, 1, 3] {
			for upper in [false, true] {
				for unit in [false, true] {
					for scale in [
						1.0,
						Real::MAX.sqrt() / 16.0,
						Real::MIN_POSITIVE.sqrt() * 16.0,
					] {
						triangular_case(
							&mut run,
							n,
							nrhs,
							upper,
							unit,
							scale,
							(n + nrhs) % 4,
							Par::Seq,
						);
						planned += 6;
					}
				}
			}
		}
	}
	for scale in [Real::MIN_POSITIVE / 2.0, Real::from_bits(1)] {
		for upper in [false, true] {
			triangular_case(&mut run, 1, 3, upper, false, scale, 2, Par::Seq);
			planned += 6;
		}
	}
	run.finish(planned);
}
#[cfg(feature = "rayon")]
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_triangular_parallel() {
	let mut run = Run::new("triangular_parallel");
	for upper in [false, true] {
		triangular_case(&mut run, 257, 65, upper, false, 1.0, 0, Par::Seq);
		for threads in [2, 4] {
			rayon::ThreadPoolBuilder::new()
				.num_threads(threads)
				.build()
				.unwrap()
				.install(|| {
					triangular_case(
						&mut run,
						257,
						65,
						upper,
						false,
						1.0,
						threads / 2,
						Par::rayon(threads),
					)
				});
		}
	}
	run.finish(36);
}
