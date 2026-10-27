// Selected fixtures from LAPACK 3.12.0 TESTING/{s,d,c,z}ec.in,
// distributed in OpenBLAS 791b02a1a00c996b66a894f6df93421c53b62d80.
// S/D input line 3 (GET36 n=8, IFST=2, ILST=7); C/Z line 142
// (GET36 n=4, IFST=1, ILST=4). The scalar variants have identical literal values.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD. Indices below are zero-based.
fn get36_fixture() -> (Mat<Scalar>, usize, usize) {
	if <Scalar as ComplexField>::IS_REAL {
		let data: &[(Real, Real)] = &[
			(1.0, 0.0),
			(1.0, 0.0),
			(1.1, 0.0),
			(1.3, 0.0),
			(2.0, 0.0),
			(3.0, 0.0),
			(-4.7, 0.0),
			(3.3, 0.0),
			(-1.0, 0.0),
			(1.0, 0.0),
			(3.7, 0.0),
			(7.9, 0.0),
			(4.0, 0.0),
			(5.3, 0.0),
			(3.3, 0.0),
			(-0.9, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(2.0, 0.0),
			(-3.0, 0.0),
			(3.4, 0.0),
			(6.5, 0.0),
			(5.2, 0.0),
			(1.8, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(4.0, 0.0),
			(2.0, 0.0),
			(-5.3, 0.0),
			(-8.9, 0.0),
			(-0.2, 0.0),
			(-0.5, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(4.2, 0.0),
			(2.0, 0.0),
			(3.3, 0.0),
			(2.3, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(-3.7, 0.0),
			(4.2, 0.0),
			(9.9, 0.0),
			(8.8, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(9.9, 0.0),
			(8.8, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(-9.9, 0.0),
			(9.9, 0.0),
		];
		(
			Mat::from_fn(8, 8, |i, j| {
				let (r, im) = data[i * 8 + j];
				scalar(r, im)
			}),
			1,
			6,
		)
	} else {
		let data: &[(Real, Real)] = &[
			(12.0, 0.0),
			(0.0, 20.0),
			(-2.0, 0.0),
			(10.0, 0.0),
			(0.0, 0.0),
			(20.0, 0.0),
			(2.0, -1.0),
			(0.0, 0.9),
			(0.0, 0.0),
			(0.0, 0.0),
			(3.0, 0.0),
			(0.8, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(0.0, 0.0),
			(8.0, 0.0),
		];
		(
			Mat::from_fn(4, 4, |i, j| {
				let (r, im) = data[i * 4 + j];
				scalar(r, im)
			}),
			0,
			3,
		)
	}
}
