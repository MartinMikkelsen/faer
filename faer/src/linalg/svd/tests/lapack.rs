//! Selected LAPACK SVD test families and numerical checks, adapted to faer.
//!
//! Reference: LAPACK 3.12.0, distributed in OpenMathLib/OpenBLAS commit
//! 791b02a1a00c996b66a894f6df93421c53b62d80, lapack-netlib/TESTING/EIG/
//! {dchkbd.f,ddrvbd.f}, and TESTING/svd.in (threshold 50).
//! Matrix construction and safely scaled checks are implemented in Rust;
//! this is a bounded adaptation, not a translation of the Fortran drivers.
//!
//! Copyright (c) 1992-2023 The University of Tennessee and The University
//! of Tennessee Research Foundation. All rights reserved.
//! Copyright (c) 2000-2023 The University of California Berkeley.
//! All rights reserved.
//! Copyright (c) 2006-2023 The University of Colorado Denver.
//! All rights reserved.
//! SPDX-License-Identifier: BSD-3-Clause
//! See COPYING.LAPACK.BSD in the crate root for the license terms.

use super::super::*;
use crate::assert;
use crate::stats::prelude::*;
use dyn_stack::MemBuffer;

const THRESHOLD: f64 = 50.0;

fn reconstruction_ratio(
	a: MatRef<'_, f64>,
	s: &[f64],
	u: MatRef<'_, f64>,
	v: MatRef<'_, f64>,
) -> f64 {
	let (m, n) = a.shape();
	let mut scale = 0.0f64;
	for j in 0..n {
		for i in 0..m {
			if !a[(i, j)].is_finite() {
				return f64::INFINITY;
			}
			scale = scale.max(a[(i, j)].abs());
		}
	}
	if scale == 0.0 {
		scale = 1.0;
	}
	let mut anorm = 0.0f64;
	let mut residual = 0.0f64;
	// Normalize before multiplying: neither extreme-scale A nor S is squared.
	for j in 0..n {
		let mut a_column = 0.0;
		let mut error_column = 0.0;
		for i in 0..m {
			let mut reconstructed = 0.0;
			for k in 0..s.len() {
				reconstructed += u[(i, k)] * (s[k] / scale) * v[(j, k)];
			}
			let value = a[(i, j)] / scale;
			if !reconstructed.is_finite() {
				return f64::INFINITY;
			}
			a_column += value.abs();
			error_column += (value - reconstructed).abs();
		}
		anorm = anorm.max(a_column);
		residual = residual.max(error_column);
	}
	if anorm == 0.0 {
		if residual == 0.0 { 0.0 } else { f64::INFINITY }
	} else {
		(residual / anorm) / (m.max(n).max(1) as f64 * f64::EPSILON)
	}
}

fn orthogonality_ratio(q: MatRef<'_, f64>) -> f64 {
	let mut error = 0.0f64;
	for j in 0..q.ncols() {
		let mut column = 0.0;
		for i in 0..q.ncols() {
			let mut dot = 0.0;
			for k in 0..q.nrows() {
				dot += q[(k, i)] * q[(k, j)];
			}
			if !dot.is_finite() {
				return f64::INFINITY;
			}
			column += (dot - if i == j { 1.0 } else { 0.0 }).abs();
		}
		error = error.max(column);
	}
	error / (q.nrows().max(1) as f64 * f64::EPSILON)
}

fn valid_spectrum(s: &[f64]) -> bool {
	s.iter().all(|x| x.is_finite() && *x >= 0.0)
		&& s.windows(2).all(|pair| pair[0] >= pair[1])
}

fn spectrum_ratio(s: &[f64], expected: &[f64], dimension: usize) -> f64 {
	assert_eq!(s.len(), expected.len());
	if !s.iter().chain(expected).all(|x| x.is_finite()) {
		return f64::INFINITY;
	}
	let scale = expected.iter().fold(0.0f64, |max, x| max.max(x.abs()));
	if scale == 0.0 {
		return if s.iter().all(|x| *x == 0.0) {
			0.0
		} else {
			f64::INFINITY
		};
	}
	let mut error = 0.0f64;
	for (&actual, &reference) in s.iter().zip(expected) {
		let difference = (actual / scale - reference / scale).abs();
		if !difference.is_finite() {
			return f64::INFINITY;
		}
		error = error.max(difference);
	}
	error / (dimension.max(1) as f64 * f64::EPSILON)
}

fn check_ratio(
	failures: &mut Vec<String>,
	context: &str,
	check: &str,
	ratio: f64,
) {
	if !ratio.is_finite() || ratio >= THRESHOLD {
		failures.push(format!(
			"{context}: {check} ratio={ratio:e}, threshold={THRESHOLD}"
		));
	}
}

fn run_case(
	failures: &mut Vec<String>,
	name: &str,
	a: MatRef<'_, f64>,
	expected: Option<&[f64]>,
	threshold: usize,
	prefill: f64,
) {
	let (m, n) = a.shape();
	let rank = m.min(n);
	let params = Spec::new(SvdParams {
		recursion_threshold: threshold,
		..auto!(f64)
	});
	use ComputeSvdVectors::{Full, No, Thin};
	let mut reference: Option<Vec<f64>> = None;
	for (left, right) in
		[(Full, Full), (Thin, Thin), (Thin, No), (No, Thin), (No, No)]
	{
		let context = format!(
			"{name}, {m}x{n}, threshold={threshold}, U={left:?}, V={right:?}, prefill={prefill}"
		);
		let columns = |mode, full| match mode {
			Full => full,
			Thin => rank,
			No => 0,
		};
		let mut u = Mat::<f64>::full(m, columns(left, m), prefill);
		let mut v = Mat::<f64>::full(n, columns(right, n), prefill);
		let mut s = Col::<f64>::zeros(rank);
		let mut memory = MemBuffer::new(svd_scratch::<f64>(
			m,
			n,
			left,
			right,
			Par::Seq,
			params,
		));
		if let Err(error) = svd(
			a,
			s.as_mut().as_diagonal_mut(),
			if left == No { None } else { Some(u.as_mut()) },
			if right == No { None } else { Some(v.as_mut()) },
			Par::Seq,
			MemStack::new(&mut memory),
			params,
		) {
			failures.push(format!("{context}: {error:?}"));
			continue;
		}
		let s: Vec<f64> = s.iter().copied().collect();
		if !valid_spectrum(&s) {
			failures.push(format!(
				"{context}: spectrum is not finite, nonnegative and decreasing: {s:?}"
			));
		}
		if let Some(expected) = expected {
			check_ratio(
				failures,
				&context,
				"known spectrum",
				spectrum_ratio(&s, expected, m.max(n)),
			);
		}
		if let Some(reference) = &reference {
			check_ratio(
				failures,
				&context,
				"output-mode agreement",
				spectrum_ratio(&s, reference, rank),
			);
		} else {
			reference = Some(s.clone());
		}
		if left != No {
			check_ratio(
				failures,
				&context,
				"U orthogonality",
				orthogonality_ratio(u.as_ref()),
			);
		}
		if right != No {
			check_ratio(
				failures,
				&context,
				"V orthogonality",
				orthogonality_ratio(v.as_ref()),
			);
		}
		if left != No && right != No {
			check_ratio(
				failures,
				&context,
				"reconstruction",
				reconstruction_ratio(a, &s, u.as_ref(), v.as_ref()),
			);
		}
	}
}

fn diagonal_case(m: usize, n: usize, kind: usize) -> (Mat<f64>, Vec<f64>) {
	let rank = m.min(n);
	let scale = match kind {
		6 => f64::MAX.sqrt(),
		7 => f64::MIN_POSITIVE.sqrt(),
		_ => 1.0,
	};
	let spectrum: Vec<_> = (0..rank)
		.map(|i| {
			let fraction = i as f64 / rank.saturating_sub(1).max(1) as f64;
			scale
				* match kind {
					0 => {
						if i == 0 {
							1.0
						} else {
							0.0
						}
					},
					1 => 0.0,
					2 => 1.0,
					3 | 6 | 7 => 1.0 - fraction * (1.0 - f64::EPSILON),
					4 => f64::EPSILON.powf(fraction),
					5 => {
						if i == 0 {
							1.0
						} else {
							f64::EPSILON
						}
					},
					_ => unreachable!(),
				}
		})
		.collect();
	let a = Mat::from_fn(m, n, |i, j| {
		if i == j {
			spectrum[i] * if i % 2 == 0 { 1.0 } else { -1.0 }
		} else {
			0.0
		}
	});
	(a, spectrum)
}

fn rotate(a: &mut Mat<f64>) {
	// Fixed orthogonal rotations replace DLATMS; no faer factorization generates inputs.
	for i in 1..a.nrows() {
		for j in 0..a.ncols() {
			let (x, y) = (a[(i - 1, j)], a[(i, j)]);
			a[(i - 1, j)] = 0.6 * x - 0.8 * y;
			a[(i, j)] = 0.8 * x + 0.6 * y;
		}
	}
	for j in 1..a.ncols() {
		for i in 0..a.nrows() {
			let (x, y) = (a[(i, j - 1)], a[(i, j)]);
			a[(i, j - 1)] = 0.8 * x - 0.6 * y;
			a[(i, j)] = 0.6 * x + 0.8 * y;
		}
	}
}

#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_svd_f64_diagonal_families() {
	let mut failures = Vec::new();
	for (m, n) in [
		(0, 0),
		(0, 3),
		(3, 0),
		(1, 1),
		(3, 3),
		(7, 7),
		(8, 8),
		(9, 9),
		(10, 16),
		(16, 10),
	] {
		for kind in 1..=7 {
			let (a, expected) = diagonal_case(m, n, kind);
			for threshold in [8, 128] {
				run_case(
					&mut failures,
					&format!("DCHKBD type {kind}"),
					a.as_ref(),
					Some(&expected),
					threshold,
					0.0,
				);
			}
		}
	}
	assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_svd_f64_rotated_families() {
	let mut failures = Vec::new();
	for (m, n) in [(3, 3), (7, 7), (8, 8), (9, 9), (10, 16), (16, 10)] {
		for (kind, source) in [(3, 8), (4, 9), (5, 10), (6, 11), (7, 12)] {
			let (mut a, expected) = diagonal_case(m, n, kind);
			rotate(&mut a);
			for threshold in [8, 128] {
				run_case(
					&mut failures,
					&format!("DCHKBD type {source}, fixed rotations"),
					a.as_ref(),
					Some(&expected),
					threshold,
					0.0,
				);
			}
		}
	}
	assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_svd_f64_bidiagonal_family() {
	const SEED: u64 = 0x4C41_5041_434B;
	let mut failures = Vec::new();
	for n in [7, 8, 9, 127, 128, 129] {
		let mut rng = StdRng::seed_from_u64(SEED);
		let a = Mat::from_fn(n, n, |i, j| {
			if i == j || i == j + 1 {
				f64::EPSILON.powf(rng.random_range(-2.0..2.0))
			} else {
				0.0
			}
		});
		for threshold in [8, 128] {
			run_case(
				&mut failures,
				&format!("DCHKBD type 16, lower bidiagonal, seed={SEED:#x}"),
				a.as_ref(),
				None,
				threshold,
				0.0,
			);
			run_case(
				&mut failures,
				&format!("DCHKBD type 16, upper bidiagonal, seed={SEED:#x}"),
				a.transpose(),
				None,
				threshold,
				0.0,
			);
		}
	}
	assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_svd_f64_rank_one_at_default_boundary() {
	let mut failures = Vec::new();
	for n in [127, 128, 129] {
		let (a, expected) = diagonal_case(n, n, 0);
		run_case(
			&mut failures,
			"rank one (#33)",
			a.as_ref(),
			Some(&expected),
			128,
			0.0,
		);
	}
	assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_svd_f64_prefilled_outputs() {
	let mut failures = Vec::new();
	for (m, n) in [(4, 2), (2, 4), (4, 4)] {
		let (a, expected) = diagonal_case(m, n, 2);
		run_case(
			&mut failures,
			"identity, reused outputs (#41)",
			a.as_ref(),
			Some(&expected),
			128,
			1.0,
		);
	}
	assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn checkers_handle_zero_and_extreme_scales() {
	let q = Mat::<f64>::identity(2, 2);
	for scale in [0.0, 1.0, 1e-300, 1e300] {
		let a =
			Mat::<f64>::from_fn(2, 2, |i, j| if i == j { scale } else { 0.0 });
		assert_eq!(
			reconstruction_ratio(
				a.as_ref(),
				&[scale, scale],
				q.as_ref(),
				q.as_ref()
			),
			0.0,
		);
	}
	assert_eq!(orthogonality_ratio(q.as_ref()), 0.0);
	for scale in [0.0, 1.0, 1e-300, 1e300] {
		assert_eq!(spectrum_ratio(&[scale, scale], &[scale, scale], 2), 0.0);
	}
}

#[test]
fn checkers_reject_corrupted_outputs() {
	let a = Mat::<f64>::identity(4, 2);
	let v = Mat::<f64>::identity(2, 2);
	let mut u = Mat::<f64>::identity(4, 4);
	assert!(
		reconstruction_ratio(a.as_ref(), &[2.0, 1.0], u.as_ref(), v.as_ref())
			>= THRESHOLD
	);
	// Reconstruction cannot see corruption in the unused full-factor columns.
	u[(0, 3)] = 1.0;
	assert_eq!(
		reconstruction_ratio(a.as_ref(), &[1.0, 1.0], u.as_ref(), v.as_ref()),
		0.0
	);
	assert!(orthogonality_ratio(u.as_ref()) >= THRESHOLD);
	u[(0, 0)] = f64::NAN;
	assert!(!orthogonality_ratio(u.as_ref()).is_finite());
	assert!(
		!reconstruction_ratio(a.as_ref(), &[1.0, 1.0], u.as_ref(), v.as_ref())
			.is_finite()
	);
	assert!(valid_spectrum(&[2.0, 1.0, 0.0]));
	for s in [
		[1.0, 2.0],
		[1.0, -1.0],
		[1.0, f64::NAN],
		[f64::INFINITY, 1.0],
	] {
		assert!(!valid_spectrum(&s));
	}
	assert!(spectrum_ratio(&[1.0, 0.0], &[1.0, 1.0], 2) >= THRESHOLD);
	assert!(!spectrum_ratio(&[f64::NAN], &[1.0], 1).is_finite());
	assert!(!spectrum_ratio(&[1.0], &[0.0], 1).is_finite());
}
