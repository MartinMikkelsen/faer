// GET01/02/03, POT01/02/03, SYT01/HET01, QRT01 and TRT02 residual helpers adapted to scalar loops.
// Adapted LAPACK LIN numerical properties, LAPACK 3.12.0, OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Deterministic Rust fixtures replace LATMS/LATMR.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD.

// LIN uses LAMCH('E'), half of Rust EPSILON under round-to-nearest.
fn lin_eps() -> Real {
	Real::EPSILON * 0.5
}
fn abs1(x: Scalar) -> Real {
	real(&x).abs() + imag(&x).abs()
}
fn lin_difference(
	a: MatRef<'_, Scalar>,
	b: MatRef<'_, Scalar>,
	dim: usize,
) -> Real {
	difference_ratio(a, b, lin_eps()) * b.nrows().max(b.ncols()).max(1) as Real
		/ dim.max(1) as Real
}
fn op(a: MatRef<'_, Scalar>, mode: usize) -> Mat<Scalar> {
	Mat::from_fn(
		if mode == 0 { a.nrows() } else { a.ncols() },
		if mode == 0 { a.ncols() } else { a.nrows() },
		|i, j| match mode {
			0 => a[(i, j)],
			1 => a[(j, i)],
			_ => conj(&a[(j, i)]),
		},
	)
}
fn solve_ratio(
	a: MatRef<'_, Scalar>,
	x: MatRef<'_, Scalar>,
	b: MatRef<'_, Scalar>,
) -> Real {
	if a.nrows() == 0 || a.ncols() == 0 || x.ncols() == 0 {
		return 0.0;
	}
	let scale = max_entry(a);
	if !scale.is_finite()
		|| scale == 0.0
		|| !max_entry(x).is_finite()
		|| !max_entry(b).is_finite()
	{
		return Real::INFINITY;
	}
	let an = Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)] / scale);
	let ax = product(an.as_ref(), x);
	let mut result = 0.0 as Real;
	for j in 0..x.ncols() {
		let xn = (0..x.nrows()).map(|i| abs1(x[(i, j)])).sum::<Real>();
		let rn = (0..b.nrows())
			.map(|i| abs1(b[(i, j)] / scale - ax[(i, j)]))
			.sum::<Real>();
		let r = rn / norm_one(an.as_ref()) / xn / lin_eps();
		if !r.is_finite() {
			return Real::INFINITY;
		}
		result = result.max(r);
	}
	result
}
fn inverse_ratio(a: MatRef<'_, Scalar>, inv: MatRef<'_, Scalar>) -> Real {
	let n = a.nrows();
	if n == 0 {
		return 0.0;
	}
	let scale = max_entry(a);
	if scale == 0.0 || !scale.is_finite() || !max_entry(inv).is_finite() {
		return Real::INFINITY;
	}
	let an = Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)] / scale);
	let xn = scaled(inv, scale);
	let ax = product(xn.as_ref(), an.as_ref());
	let err = Mat::from_fn(n, n, |i, j| {
		z(if i == j { 1.0 } else { 0.0 }) - ax[(i, j)]
	});
	norm_one(err.as_ref())
		/ norm_one(an.as_ref())
		/ norm_one(xn.as_ref())
		/ lin_eps()
		/ n as Real
}
fn permutation_ok(p: &[usize], ip: &[usize]) -> bool {
	p.len() == ip.len()
		&& p.iter()
			.enumerate()
			.all(|(i, &j)| j < p.len() && ip[j] == i)
}
fn complete_lower(a: MatRef<'_, Scalar>) -> Mat<Scalar> {
	Mat::from_fn(a.nrows(), a.ncols(), |i, j| {
		if i >= j { a[(i, j)] } else { conj(&a[(j, i)]) }
	})
}
fn known_solution(n: usize, nrhs: usize) -> Mat<Scalar> {
	Mat::from_fn(n, nrhs, |i, j| {
		phase(i + 2 * j) * (1.0 + (i + j) as Real / 8.0)
	})
}
#[test]
fn lapack_lin_checkers() {
	let a = Mat::<Scalar>::identity(2, 2);
	let x = known_solution(2, 2);
	let b = x.clone();
	crate::assert!(solve_ratio(a.as_ref(), x.as_ref(), b.as_ref()) == 0.0);
	let mut wrong = x.clone();
	wrong[(0, 0)] += scalar(1.0, 2.0);
	crate::assert!(solve_ratio(a.as_ref(), wrong.as_ref(), b.as_ref()) > 30.0);
	crate::assert!(inverse_ratio(a.as_ref(), a.as_ref()) == 0.0);
	let bad = scaled(a.as_ref(), 2.0);
	crate::assert!(inverse_ratio(a.as_ref(), bad.as_ref()) > 30.0);
	crate::assert!(!permutation_ok(&[1, 1], &[1, 0]));
	let mut nan = a.clone();
	nan[(0, 0)] = scalar(Real::NAN, Real::NAN);
	crate::assert!(!inverse_ratio(a.as_ref(), nan.as_ref()).is_finite());
}
