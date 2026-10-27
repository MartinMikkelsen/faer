// Independent complex fixtures for the LAPACK SVD pilot.
// The shared checks retain LAPACK/OpenBLAS provenance in lapack.rs.
// Copyright (c) tensor4all contributors
// SPDX-License-Identifier: MIT

use super::super::super::*;
use super::lapack::{
	check_driver_output, diagonal_case, orthogonality_ratio,
	reconstruction_ratio, rotate,
};
use super::{Real, Scalar};
use crate::assert;

#[test]
fn checkers_require_conjugation() {
	let a = Mat::from_fn(1, 1, |_, _| Scalar::new(0.0, -2.0));
	let u = Mat::<Scalar>::identity(1, 1);
	let v = Mat::from_fn(1, 1, |_, _| Scalar::new(0.0, 1.0));
	assert_eq!(
		reconstruction_ratio(a.as_ref(), &[2.0], u.as_ref(), v.as_ref()),
		0.0
	);
	assert_eq!(orthogonality_ratio(v.as_ref()), 0.0);
	let wrong_v = Mat::from_fn(1, 1, |_, _| Scalar::new(0.0, -1.0));
	assert!(
		reconstruction_ratio(a.as_ref(), &[2.0], u.as_ref(), wrong_v.as_ref())
			>= 50.0
	);
}

#[test]
fn checkers_reject_imaginary_corruption() {
	let identity = Mat::<Scalar>::identity(1, 1);
	let invalid = Mat::from_fn(1, 1, |_, _| Scalar::new(1.0, Real::NAN));
	assert!(!orthogonality_ratio(invalid.as_ref()).is_finite());
	assert!(
		!reconstruction_ratio(
			identity.as_ref(),
			&[1.0],
			invalid.as_ref(),
			identity.as_ref()
		)
		.is_finite()
	);
	for im in [1.0, Real::NAN, Real::INFINITY] {
		let mut failures = Vec::new();
		check_driver_output(
			&mut failures,
			"imaginary spectrum",
			identity.as_ref(),
			&[Scalar::new(1.0, im)],
			identity.as_ref(),
			identity.as_ref(),
			Some(&[1.0]),
			None,
		);
		assert!(failures.iter().any(|failure| failure.contains("imaginary")));
	}
}

#[test]
fn fixtures_have_known_complex_phases() {
	let (mut a, s) = diagonal_case(2, 2, 2);
	assert_eq!(s, [1.0, 1.0]);
	assert_eq!(a[(0, 0)], Scalar::new(0.0, 1.0));
	assert_eq!(a[(1, 1)], Scalar::new(0.6, 0.8));
	rotate(&mut a);
	for (i, j, expected) in [
		(0, 0, Scalar::new(-0.288, 0.096)),
		(0, 1, Scalar::new(-0.872, 0.384)),
		(1, 0, Scalar::new(-0.928, 0.216)),
		(1, 1, Scalar::new(0.288, -0.096)),
	] {
		assert!((a[(i, j)] - expected).norm() <= 8.0 * Real::EPSILON);
	}
	assert!(orthogonality_ratio(a.as_ref()) < 50.0);
}
