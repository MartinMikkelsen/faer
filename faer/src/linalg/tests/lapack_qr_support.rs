// QRT01/02/03 reference products: native scalar elementary Householders replace ORGQR/UNGQR as oracle.
// Adapted LAPACK LIN numerical properties, LAPACK 3.12.0, OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Deterministic Rust fixtures replace LATMS/LATMR.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD.

// Independent elementary-reflector product; faer stores inverse tau on each
// block diagonal. No production Q-application/reconstruction defines this oracle.
fn scalar_q(
	packed: MatRef<'_, Scalar>,
	coeff: MatRef<'_, Scalar>,
) -> Mat<Scalar> {
	let m = packed.nrows();
	let k = coeff.ncols();
	let bs = coeff.nrows();
	let mut q = Mat::<Scalar>::identity(m, m);
	for r in (0..k).rev() {
		let tau = real(&coeff[(r % bs, r)]);
		let inv = 1.0 / tau;
		if inv == 0.0 {
			continue;
		}
		for j in 0..m {
			let mut dot = q[(r, j)];
			for i in r + 1..m {
				dot += conj(&packed[(i, r)]) * q[(i, j)];
			}
			dot *= inv;
			q[(r, j)] -= dot;
			for i in r + 1..m {
				q[(i, j)] -= packed[(i, r)] * dot;
			}
		}
	}
	q
}
fn qr_fixture(m: usize, n: usize, kind: usize) -> Mat<Scalar> {
	if kind == 0 {
		return Mat::zeros(m, n);
	}
	if kind == 4 {
		return Mat::from_fn(m, n, |i, j| phase(i) * (j + 1) as Real);
	}
	let scale = match kind {
		6 => Real::MAX.sqrt() / 16.0,
		7 => Real::MIN_POSITIVE.sqrt() * 16.0,
		_ => 1.0,
	};
	let q = unitary(m);
	let v = unitary(n);
	let d = Mat::from_fn(m, n, |i, j| {
		if i == j {
			z(scale
				* match kind {
					1 => 1.0,
					2 => 1.0 + i as Real / m.min(n).max(1) as Real,
					3 => Real::EPSILON
						.sqrt()
						.powf(i as Real / m.min(n).max(1) as Real),
					5 => {
						if i + 1 == m.min(n) {
							0.0
						} else {
							1.0
						}
					},
					_ => 1.0 + i as Real / m.min(n).max(1) as Real,
				})
		} else {
			z(0.0)
		}
	});
	product(
		product(q.as_ref(), d.as_ref()).as_ref(),
		adjoint(v.as_ref()).as_ref(),
	)
}
#[test]
fn lapack_qr_reflector_checkers() {
	let packed =
		Mat::from_fn(2, 1, |i, _| if i == 0 { guard() } else { phase(0) });
	let coeff = Mat::full(1, 1, z(1.0));
	let q = scalar_q(packed.as_ref(), coeff.as_ref());
	crate::assert!(magnitude(q[(0, 0)]) < 4.0 * Real::EPSILON);
	assert_eq!(q[(0, 1)], -conj(&phase(0)));
	assert_eq!(q[(1, 0)], -phase(0));
	crate::assert!(unitary_ratio(q.as_ref(), lin_eps()) < 30.0);
	let bad = Mat::full(1, 1, z(2.0));
	crate::assert!(
		unitary_ratio(
			scalar_q(packed.as_ref(), bad.as_ref()).as_ref(),
			lin_eps()
		) > 30.0
	);
}
