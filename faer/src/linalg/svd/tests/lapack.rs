// Selected LAPACK SVD test families and numerical checks, adapted to faer.
// Heavy opt-in validation; the scheduled SVD workflow includes ignored tests.
//
// Reference: LAPACK 3.12.0, distributed in OpenMathLib/OpenBLAS commit
// 791b02a1a00c996b66a894f6df93421c53b62d80, lapack-netlib/TESTING/EIG/
// {dchkbd.f,ddrvbd.f,dbdt03.f,schkbd.f,sdrvbd.f,sbdt03.f,
// zchkbd.f,zdrvbd.f,zbdt03.f,zunt01.f,cchkbd.f,cdrvbd.f,cbdt03.f,cunt01.f},
// and TESTING/svd.in (threshold 50).
// Scalar/Real are the enclosing f32/f64/c32/c64 storage/precision aliases.
// Complex phase adapters extend DCHKBD 15-19; C/ZCHKBD have tests 1-14.
// Matrix construction and safely scaled checks are implemented in Rust;
// this is a bounded adaptation, not a translation of the Fortran drivers.
//
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// All rights reserved.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD in the crate root for the license terms.

use super::super::super::*;
use super::{
	ROTATION_PHASE, Real, Scalar, bidiagonal_scratch, compute_bidiagonal,
	diagonal_phase,
};
use crate::assert;
use crate::stats::prelude::*;
use dyn_stack::MemBuffer;
use std::panic::{AssertUnwindSafe, catch_unwind};

const THRESHOLD: Real = 50.0;
const BIDIAGONAL_SEED: u64 = 0x4C41_5041_434B;

pub(super) fn check_svd_call(
	failures: &mut Vec<String>,
	context: &str,
	call: impl FnOnce() -> Result<(), SvdError>,
) -> bool {
	let message = match catch_unwind(AssertUnwindSafe(call)) {
		Ok(Ok(())) => return true,
		Ok(Err(error)) => format!("{error:?}"),
		Err(payload) => {
			let message = payload
				.downcast_ref::<&str>()
				.copied()
				.or_else(|| {
					payload.downcast_ref::<String>().map(String::as_str)
				})
				.unwrap_or("non-string panic");
			format!("panic: {message}")
		},
	};
	failures.push(format!("call failure: {context}: {message}"));
	false
}

#[test]
fn panics_and_errors_do_not_skip_later_cases() {
	let mut failures = Vec::new();
	let mut successes = 0;
	for case in 0..4 {
		if check_svd_call(&mut failures, &format!("case {case}"), || match case
		{
			0 => Err(SvdError::NoConvergence),
			1 => panic!("scratch panic"),
			2 => std::panic::panic_any(String::from("owned panic")),
			_ => Ok(()),
		}) {
			successes += 1;
		}
	}
	assert_eq!(successes, 1);
	assert_eq!(failures.len(), 3);
	assert!(failures[0].contains("case 0: NoConvergence"));
	assert!(failures[1].contains("case 1: panic: scratch panic"));
	assert!(failures[2].contains("case 2: panic: owned panic"));
}

fn scalar_norm(value: Scalar) -> Real {
	real(&value).hypot(imag(&value))
}

pub(super) fn reconstruction_ratio(
	a: MatRef<'_, Scalar>,
	s: &[Real],
	u: MatRef<'_, Scalar>,
	v: MatRef<'_, Scalar>,
) -> Real {
	let (m, n) = a.shape();
	let mut scale = 0.0 as Real;
	for j in 0..n {
		for i in 0..m {
			let magnitude = scalar_norm(a[(i, j)]);
			if !magnitude.is_finite() {
				return Real::INFINITY;
			}
			scale = scale.max(magnitude);
		}
	}
	if scale == 0.0 {
		scale = 1.0;
	}
	let mut anorm = 0.0 as Real;
	let mut residual = 0.0 as Real;
	// Normalize before multiplying: neither extreme-scale A nor S is squared.
	for j in 0..n {
		let mut a_column = 0.0;
		let mut error_column = 0.0;
		for i in 0..m {
			let mut reconstructed = zero::<Scalar>();
			for k in 0..s.len() {
				reconstructed += u[(i, k)] * (s[k] / scale) * conj(&v[(j, k)]);
			}
			let value = a[(i, j)] / scale;
			if !is_finite(&reconstructed) {
				return Real::INFINITY;
			}
			a_column += scalar_norm(value);
			error_column += scalar_norm(value - reconstructed);
		}
		anorm = anorm.max(a_column);
		residual = residual.max(error_column);
	}
	if anorm == 0.0 {
		if residual == 0.0 { 0.0 } else { Real::INFINITY }
	} else {
		(residual / anorm) / (m.max(n).max(1) as Real * Real::EPSILON)
	}
}

pub(super) fn orthogonality_ratio(q: MatRef<'_, Scalar>) -> Real {
	let mut error = 0.0 as Real;
	for j in 0..q.ncols() {
		let mut column = 0.0;
		for i in 0..q.ncols() {
			let mut dot = zero::<Scalar>();
			for k in 0..q.nrows() {
				dot += conj(&q[(k, i)]) * q[(k, j)];
			}
			if !is_finite(&dot) {
				return Real::INFINITY;
			}
			column += scalar_norm(
				dot - if i == j {
					one::<Scalar>()
				} else {
					zero::<Scalar>()
				},
			);
		}
		error = error.max(column);
	}
	error / (q.nrows().max(1) as Real * Real::EPSILON)
}

fn valid_spectrum(s: &[Real]) -> bool {
	s.iter().all(|x| x.is_finite() && *x >= 0.0)
		&& s.windows(2).all(|pair| pair[0] >= pair[1])
}

fn spectrum_ratio(s: &[Real], expected: &[Real], dimension: usize) -> Real {
	assert_eq!(s.len(), expected.len());
	if !s.iter().chain(expected).all(|x| x.is_finite()) {
		return Real::INFINITY;
	}
	let scale = expected.iter().fold(0.0 as Real, |max, x| max.max(x.abs()));
	if scale == 0.0 {
		return if s.iter().all(|x| *x == 0.0) {
			0.0
		} else {
			Real::INFINITY
		};
	}
	let mut error = 0.0 as Real;
	for (&actual, &reference) in s.iter().zip(expected) {
		let difference = (actual / scale - reference / scale).abs();
		if !difference.is_finite() {
			return Real::INFINITY;
		}
		error = error.max(difference);
	}
	error / (dimension.max(1) as Real * Real::EPSILON)
}

fn dchkbd_spectrum_ratio(
	with_vectors: &[Real],
	without_vectors: &[Real],
) -> Real {
	assert_eq!(with_vectors.len(), without_vectors.len());
	if !valid_spectrum(with_vectors) || !valid_spectrum(without_vectors) {
		return Real::INFINITY;
	}
	let Some(&largest) = with_vectors.first() else {
		return 0.0;
	};
	// DCHKBD test 19 uses a safe-minimum floor and no dimension factor.
	let denominator = (Real::MIN_POSITIVE.sqrt() * largest.max(1.0))
		.max(Real::EPSILON * largest.max(without_vectors[0]));
	with_vectors
		.iter()
		.zip(without_vectors)
		.fold(0.0, |error, (&a, &b)| {
			error.max((a - b).abs() / denominator)
		})
}

fn check_ratio(
	failures: &mut Vec<String>,
	context: &str,
	check: &str,
	ratio: Real,
) {
	if !ratio.is_finite() || ratio >= THRESHOLD {
		failures.push(format!(
			"{context}: {check} ratio={ratio:e}, threshold={THRESHOLD}"
		));
	}
}

pub(super) fn real_spectrum(
	failures: &mut Vec<String>,
	context: &str,
	s: &[Scalar],
) -> Vec<Real> {
	if s.iter().any(|value| imag(value) != 0.0) {
		failures.push(format!(
			"{context}: singular values have nonzero or nonfinite imaginary parts"
		));
	}
	s.iter().map(real).collect()
}

pub(super) fn check_driver_output(
	failures: &mut Vec<String>,
	context: &str,
	a: MatRef<'_, Scalar>,
	s: &[Scalar],
	u: MatRef<'_, Scalar>,
	v: MatRef<'_, Scalar>,
	expected: Option<&[Real]>,
	reference: Option<&[Real]>,
) {
	let real_s = real_spectrum(failures, context, s);
	let s = real_s.as_slice();
	if !valid_spectrum(s) {
		failures.push(format!(
			"{context}: spectrum is not finite, nonnegative and decreasing: {s:?}"
		));
	}
	if let Some(expected) = expected {
		check_ratio(
			failures,
			context,
			"known spectrum",
			spectrum_ratio(s, expected, a.nrows().max(a.ncols())),
		);
	}
	if let Some(reference) = reference {
		check_ratio(
			failures,
			context,
			"output-mode agreement",
			spectrum_ratio(s, reference, s.len()),
		);
	}
	if u.ncols() > 0 {
		check_ratio(
			failures,
			context,
			"U orthogonality",
			orthogonality_ratio(u),
		);
	}
	if v.ncols() > 0 {
		check_ratio(
			failures,
			context,
			"V orthogonality",
			orthogonality_ratio(v),
		);
	}
	if u.ncols() > 0 && v.ncols() > 0 {
		check_ratio(
			failures,
			context,
			"reconstruction",
			reconstruction_ratio(a, s, u, v),
		);
	}
}

fn run_case(
	failures: &mut Vec<String>,
	attempted: &mut usize,
	name: &str,
	a: MatRef<'_, Scalar>,
	expected: Option<&[Real]>,
	threshold: usize,
	prefill: Real,
) {
	let (m, n) = a.shape();
	let rank = m.min(n);
	let params = Spec::new(SvdParams {
		recursion_threshold: threshold,
		..auto!(Scalar)
	});
	use ComputeSvdVectors::{Full, No, Thin};
	let mut reference: Option<Vec<Real>> = None;
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
		let mut u =
			Mat::<Scalar>::full(m, columns(left, m), Scalar::from(prefill));
		let mut v =
			Mat::<Scalar>::full(n, columns(right, n), Scalar::from(prefill));
		let mut s = Col::<Scalar>::zeros(rank);
		let mut memory = MemBuffer::new(svd_scratch::<Scalar>(
			m,
			n,
			left,
			right,
			ParShape::Seq,
			params,
		));
		*attempted += 1;
		if !check_svd_call(failures, &context, || {
			svd(
				a,
				s.as_mut().as_diagonal_mut(),
				if left == No { None } else { Some(u.as_mut()) },
				if right == No { None } else { Some(v.as_mut()) },
				Par::Seq,
				MemStack::new(&mut memory),
				params,
			)
		}) {
			continue;
		}
		let s: Vec<Scalar> = s.iter().copied().collect();
		check_driver_output(
			failures,
			&context,
			a,
			&s,
			u.as_ref(),
			v.as_ref(),
			expected,
			reference.as_deref(),
		);
		if reference.is_none() {
			reference = Some(s.iter().map(real).collect());
		}
	}
}

pub(super) fn diagonal_case(
	m: usize,
	n: usize,
	kind: usize,
) -> (Mat<Scalar>, Vec<Real>) {
	// Kinds 1–7 follow DCHKBD; 8/9 add repeated and tightly clustered spectra.
	let rank = m.min(n);
	let scale = match kind {
		6 => Real::MAX.sqrt(),
		7 => Real::MIN_POSITIVE.sqrt(),
		_ => 1.0,
	};
	let spectrum: Vec<_> = (0..rank)
		.map(|i| {
			let fraction = i as Real / rank.saturating_sub(1).max(1) as Real;
			scale
				* match kind {
					1 => 0.0,
					2 => 1.0,
					3 | 6 | 7 => 1.0 - fraction * (1.0 - Real::EPSILON),
					4 => Real::EPSILON.powf(fraction),
					5 => {
						if i == 0 {
							1.0
						} else {
							Real::EPSILON
						}
					},
					8 => {
						if i < rank / 2 {
							4.0
						} else {
							1.0
						}
					},
					9 => 1.0 - 8.0 * Real::EPSILON * i as Real,
					_ => unreachable!(),
				}
		})
		.collect();
	let a = Mat::from_fn(m, n, |i, j| {
		if i == j {
			Scalar::from(spectrum[i]) * diagonal_phase(i)
		} else {
			zero::<Scalar>()
		}
	});
	(a, spectrum)
}

pub(super) fn rotate(a: &mut Mat<Scalar>) {
	// Fixed orthogonal/unitary rotations replace DLATMS without a faer factorization.
	for i in 1..a.nrows() {
		for j in 0..a.ncols() {
			let (x, y) = (a[(i - 1, j)], a[(i, j)]);
			a[(i - 1, j)] = 0.6 * x - conj(&(0.8 * ROTATION_PHASE)) * y;
			a[(i, j)] = (0.8 * ROTATION_PHASE) * x + 0.6 * y;
		}
	}
	for j in 1..a.ncols() {
		for i in 0..a.nrows() {
			let (x, y) = (a[(i, j - 1)], a[(i, j)]);
			a[(i, j - 1)] = 0.8 * x - conj(&(0.6 * ROTATION_PHASE)) * y;
			a[(i, j)] = (0.6 * ROTATION_PHASE) * x + 0.8 * y;
		}
	}
}

fn complex_phase(index: usize) -> Scalar {
	if <Scalar as ComplexField>::IS_REAL {
		one::<Scalar>()
	} else {
		diagonal_phase(index)
	}
}

fn logarithmic_bidiagonal(n: usize) -> Mat<Scalar> {
	logarithmic_bidiagonal_seeded(n, BIDIAGONAL_SEED)
}

fn logarithmic_bidiagonal_seeded(n: usize, seed: u64) -> Mat<Scalar> {
	let mut rng = StdRng::seed_from_u64(seed);
	Mat::from_fn(n, n, |i, j| {
		if i == j || i == j + 1 {
			Scalar::from(Real::EPSILON.powf(rng.random_range(-2.0..2.0)))
				* complex_phase(i + 2 * j)
		} else {
			zero::<Scalar>()
		}
	})
}

/// #75: the divide-and-conquer merge forms its boundary rotation from
/// `alpha * lambda` and `beta * phi`. Both are subnormal for a deeply split
/// bidiagonal, so their squares underflowed and the rotation stopped being
/// orthogonal; the returned V lost orthogonality while U and the
/// reconstruction stayed inside their bounds.
#[test]
fn bidiagonal_merge_boundary_rotation_is_orthogonal() {
	// The reported case and the two neighbouring seeds that failed harder, the
	// last one at the default recursion threshold.
	for (seed, n, threshold) in [
		(BIDIAGONAL_SEED, 127, 8),
		(0x4C41_5041_4351, 126, 8),
		(0x4C41_5041_434D, 128, 128),
	] {
		let context = format!(
			"merge rotation, seed={seed:#x}, n={n}, threshold={threshold}"
		);
		let a = logarithmic_bidiagonal_seeded(n, seed);
		let params = Spec::new(SvdParams {
			recursion_threshold: threshold,
			..auto!(Scalar)
		});
		let mut u = Mat::<Scalar>::zeros(n, n);
		let mut v = Mat::<Scalar>::zeros(n, n);
		let mut s = Col::<Scalar>::zeros(n);
		let mut memory = MemBuffer::new(svd_scratch::<Scalar>(
			n,
			n,
			ComputeSvdVectors::Full,
			ComputeSvdVectors::Full,
			ParShape::Seq,
			params,
		));
		let result = svd(
			a.as_ref(),
			s.as_mut().as_diagonal_mut(),
			Some(u.as_mut()),
			Some(v.as_mut()),
			Par::Seq,
			MemStack::new(&mut memory),
			params,
		);
		assert!(result.is_ok(), "{context}: {result:?}");
		let spectrum: Vec<Real> = s.iter().copied().map(|s| real(&s)).collect();
		let v_ratio = orthogonality_ratio(v.as_ref());
		let u_ratio = orthogonality_ratio(u.as_ref());
		let residual =
			reconstruction_ratio(a.as_ref(), &spectrum, u.as_ref(), v.as_ref());
		assert!(
			v_ratio < THRESHOLD,
			"{context}: V orthogonality ratio={v_ratio:e}, \
			 threshold={THRESHOLD}"
		);
		assert!(
			u_ratio < THRESHOLD,
			"{context}: U orthogonality ratio={u_ratio:e}, \
			 threshold={THRESHOLD}"
		);
		assert!(
			residual < THRESHOLD,
			"{context}: reconstruction ratio={residual:e}, \
			 threshold={THRESHOLD}"
		);
	}
}

fn run_bidiagonal_case(
	failures: &mut Vec<String>,
	attempted: &mut usize,
	name: &str,
	b: MatRef<'_, Scalar>,
	expected: Option<&[Real]>,
	threshold: usize,
) {
	let n = b.ncols();
	assert_eq!(b.nrows(), n);
	assert!(n > 0);
	// The private solver stores a square lower bidiagonal with an extra zero row.
	let padded = Mat::from_fn(n + 1, n, |i, j| {
		if i < n { b[(i, j)] } else { zero::<Scalar>() }
	});
	let params = SvdParams {
		recursion_threshold: threshold,
		..auto!(Scalar)
	};
	let mut reference: Option<Vec<Real>> = None;
	for (want_u, want_v) in
		[(true, true), (true, false), (false, true), (false, false)]
	{
		let context = format!(
			"direct bidiagonal {name}, n={n}, threshold={threshold}, U={want_u}, V={want_v}"
		);
		let mut d = Col::from_fn(n, |i| b[(i, i)]);
		let mut e = Col::from_fn(n, |i| {
			if i + 1 < n {
				b[(i + 1, i)]
			} else {
				zero::<Scalar>()
			}
		});
		let mut u =
			Mat::full(n + 1, if want_u { n + 1 } else { 0 }, one::<Scalar>());
		let mut v = Mat::full(n, if want_v { n } else { 0 }, one::<Scalar>());
		let mut memory = MemBuffer::new(bidiagonal_scratch::<Scalar>(
			n,
			want_u,
			want_v,
			ParShape::Seq,
			params,
		));
		*attempted += 1;
		if !check_svd_call(failures, &context, || {
			compute_bidiagonal(
				d.as_mut().try_as_col_major_mut().unwrap(),
				e.as_mut().try_as_col_major_mut().unwrap(),
				if want_u { Some(u.as_mut()) } else { None },
				if want_v { Some(v.as_mut()) } else { None },
				params,
				Par::Seq,
				MemStack::new(&mut memory),
			)
		}) {
			continue;
		}
		let scalar_s: Vec<_> = d.iter().copied().collect();
		let s = real_spectrum(failures, &context, &scalar_s);
		if !valid_spectrum(&s) {
			failures.push(format!(
				"{context}: DCHKBD test 18: invalid spectrum {s:?}"
			));
		}
		if let Some(expected) = expected {
			check_ratio(
				failures,
				&context,
				"known spectrum",
				spectrum_ratio(&s, expected, n),
			);
		}
		if let Some(reference) = &reference {
			check_ratio(
				failures,
				&context,
				"DCHKBD test 19 (mode agreement)",
				dchkbd_spectrum_ratio(reference, &s),
			);
		} else {
			reference = Some(s.clone());
		}
		// Keep DCHKBD's denominator n despite the extra zero row in B.
		let padding_correction = (n + 1) as Real / n as Real;
		if want_u {
			check_ratio(
				failures,
				&context,
				"DCHKBD test 16 (U orthogonality)",
				orthogonality_ratio(u.as_ref()) * padding_correction,
			);
		}
		if want_v {
			check_ratio(
				failures,
				&context,
				"DCHKBD test 17 (V orthogonality)",
				orthogonality_ratio(v.as_ref()),
			);
		}
		if want_u && want_v {
			check_ratio(
				failures,
				&context,
				"DCHKBD test 15 (reconstruction)",
				reconstruction_ratio(
					padded.as_ref(),
					&s,
					u.as_ref(),
					v.as_ref(),
				) * padding_correction,
			);
		}
	}
}

#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn dchkbd_15_to_19() {
	let mut failures = Vec::new();
	let mut attempted = 0;
	for n in [1, 7, 8, 9, 127, 128, 129] {
		for threshold in [8, 128] {
			for kind in [1, 2, 4, 5, 6, 7] {
				let (b, expected) = diagonal_case(n, n, kind);
				run_bidiagonal_case(
					&mut failures,
					&mut attempted,
					&format!("DCHKBD type {kind}"),
					b.as_ref(),
					Some(&expected),
					threshold,
				);
			}
			let b = logarithmic_bidiagonal(n);
			run_bidiagonal_case(
				&mut failures,
				&mut attempted,
				&format!("DCHKBD type 16, seed={BIDIAGONAL_SEED:#x}"),
				b.as_ref(),
				None,
				threshold,
			);
		}
	}
	let mut split = Mat::<Scalar>::zeros(8, 8);
	split[(4, 4)] = complex_phase(4);
	run_bidiagonal_case(
		&mut failures,
		&mut attempted,
		"rank one (#33), pivot=4",
		split.as_ref(),
		Some(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
		4,
	);
	eprintln!(
		"LAPACK_CASES family=svd_dchkbd_15_to_19 dtype={} planned=396 attempted={attempted} call_failures={} diagnostics={}",
		core::any::type_name::<Scalar>(),
		failures
			.iter()
			.filter(|f| f.starts_with("call failure:"))
			.count(),
		failures.len()
	);
	assert_eq!(attempted, 396, "SVD case execution incomplete");
	assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn diagonal_families() {
	let mut failures = Vec::new();
	let mut attempted = 0;
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
					&mut attempted,
					&format!("DCHKBD type {kind}"),
					a.as_ref(),
					Some(&expected),
					threshold,
					0.0,
				);
			}
		}
	}
	eprintln!(
		"LAPACK_CASES family=svd_diagonal_families dtype={} planned=700 attempted={attempted} call_failures={} diagnostics={}",
		core::any::type_name::<Scalar>(),
		failures
			.iter()
			.filter(|f| f.starts_with("call failure:"))
			.count(),
		failures.len()
	);
	assert_eq!(attempted, 700, "SVD case execution incomplete");
	assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn rotated_families() {
	let mut failures = Vec::new();
	let mut attempted = 0;
	for (m, n) in [(3, 3), (7, 7), (8, 8), (9, 9), (10, 16), (16, 10)] {
		for (kind, source) in [(3, 8), (4, 9), (5, 10), (6, 11), (7, 12)] {
			let (mut a, expected) = diagonal_case(m, n, kind);
			rotate(&mut a);
			for threshold in [8, 128] {
				run_case(
					&mut failures,
					&mut attempted,
					&format!("DCHKBD type {source}, fixed rotations"),
					a.as_ref(),
					Some(&expected),
					threshold,
					0.0,
				);
			}
		}
	}
	eprintln!(
		"LAPACK_CASES family=svd_rotated_families dtype={} planned=300 attempted={attempted} call_failures={} diagnostics={}",
		core::any::type_name::<Scalar>(),
		failures
			.iter()
			.filter(|f| f.starts_with("call failure:"))
			.count(),
		failures.len()
	);
	assert_eq!(attempted, 300, "SVD case execution incomplete");
	assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn bidiagonal_family() {
	let mut failures = Vec::new();
	let mut attempted = 0;
	for n in [7, 8, 9, 127, 128, 129] {
		let a = logarithmic_bidiagonal(n);
		for threshold in [8, 128] {
			run_case(
				&mut failures,
				&mut attempted,
				&format!(
					"DCHKBD type 16, lower bidiagonal, seed={BIDIAGONAL_SEED:#x}"
				),
				a.as_ref(),
				None,
				threshold,
				0.0,
			);
			run_case(
				&mut failures,
				&mut attempted,
				&format!(
					"DCHKBD type 16, upper bidiagonal, seed={BIDIAGONAL_SEED:#x}"
				),
				a.transpose(),
				None,
				threshold,
				0.0,
			);
		}
	}
	eprintln!(
		"LAPACK_CASES family=svd_bidiagonal_family dtype={} planned=120 attempted={attempted} call_failures={} diagnostics={}",
		core::any::type_name::<Scalar>(),
		failures
			.iter()
			.filter(|f| f.starts_with("call failure:"))
			.count(),
		failures.len()
	);
	assert_eq!(attempted, 120, "SVD case execution incomplete");
	assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn rank_one_at_recursion_splits() {
	let mut failures = Vec::new();
	let mut attempted = 0;
	for (n, threshold) in [(8, 4), (127, 128), (128, 128), (129, 128)] {
		let mut expected = vec![0.0; n];
		expected[0] = 1.0;
		for pivot in [0, n / 2, n - 1] {
			let mut a = Mat::<Scalar>::zeros(n, n);
			a[(pivot, pivot)] = complex_phase(pivot);
			run_case(
				&mut failures,
				&mut attempted,
				&format!("rank one (#33), pivot={pivot}"),
				a.as_ref(),
				Some(&expected),
				threshold,
				0.0,
			);
		}
	}
	eprintln!(
		"LAPACK_CASES family=svd_rank_one_at_recursion_splits dtype={} planned=60 attempted={attempted} call_failures={} diagnostics={}",
		core::any::type_name::<Scalar>(),
		failures
			.iter()
			.filter(|f| f.starts_with("call failure:"))
			.count(),
		failures.len()
	);
	assert_eq!(attempted, 60, "SVD case execution incomplete");
	assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn prefilled_outputs() {
	let mut failures = Vec::new();
	let mut attempted = 0;
	for (m, n) in [(4, 2), (2, 4), (4, 4)] {
		let (a, expected) = diagonal_case(m, n, 2);
		run_case(
			&mut failures,
			&mut attempted,
			"identity, reused outputs (#41)",
			a.as_ref(),
			Some(&expected),
			128,
			1.0,
		);
	}
	eprintln!(
		"LAPACK_CASES family=svd_prefilled_outputs dtype={} planned=15 attempted={attempted} call_failures={} diagnostics={}",
		core::any::type_name::<Scalar>(),
		failures
			.iter()
			.filter(|f| f.starts_with("call failure:"))
			.count(),
		failures.len()
	);
	assert_eq!(attempted, 15, "SVD case execution incomplete");
	assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn checkers_dchkbd_mode_agreement_has_no_dimension_factor() {
	let with_vectors = [1.0; 128];
	let mut without_vectors = with_vectors;
	without_vectors[127] -= 64.0 * Real::EPSILON;
	assert_eq!(dchkbd_spectrum_ratio(&with_vectors, &without_vectors), 64.0);
	assert_eq!(
		dchkbd_spectrum_ratio(&[0.0], &[Real::MIN_POSITIVE.sqrt()]),
		1.0,
	);
	assert_eq!(dchkbd_spectrum_ratio(&[], &[]), 0.0);
	for scale in [0.0, Real::MIN_POSITIVE, 1.0, Real::MAX] {
		assert_eq!(dchkbd_spectrum_ratio(&[scale], &[scale]), 0.0);
	}
	for invalid in [Real::NAN, Real::INFINITY, -1.0] {
		assert!(!dchkbd_spectrum_ratio(&[1.0], &[invalid]).is_finite());
		assert!(!dchkbd_spectrum_ratio(&[invalid], &[1.0]).is_finite());
	}
}

#[test]
fn checkers_handle_zero_and_extreme_scales() {
	let q = Mat::<Scalar>::identity(2, 2);
	for scale in [0.0, 1.0, Real::MIN_POSITIVE, Real::MAX] {
		let a = Mat::<Scalar>::from_fn(2, 2, |i, j| {
			Scalar::from(if i == j { scale } else { 0.0 })
		});
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
	for scale in [0.0, 1.0, Real::MIN_POSITIVE, Real::MAX] {
		assert_eq!(spectrum_ratio(&[scale, scale], &[scale, scale], 2), 0.0);
	}
}

#[test]
fn checkers_reject_corrupted_outputs() {
	let a = Mat::<Scalar>::identity(4, 2);
	let v = Mat::<Scalar>::identity(2, 2);
	let mut u = Mat::<Scalar>::identity(4, 4);
	assert!(
		reconstruction_ratio(a.as_ref(), &[2.0, 1.0], u.as_ref(), v.as_ref())
			>= THRESHOLD
	);
	// Reconstruction cannot see corruption in the unused full-factor columns.
	u[(0, 3)] = one::<Scalar>();
	assert_eq!(
		reconstruction_ratio(a.as_ref(), &[1.0, 1.0], u.as_ref(), v.as_ref()),
		0.0
	);
	assert!(orthogonality_ratio(u.as_ref()) >= THRESHOLD);
	u[(0, 0)] = nan::<Scalar>();
	assert!(!orthogonality_ratio(u.as_ref()).is_finite());
	assert!(
		!reconstruction_ratio(a.as_ref(), &[1.0, 1.0], u.as_ref(), v.as_ref())
			.is_finite()
	);
	assert!(valid_spectrum(&[2.0, 1.0, 0.0]));
	for s in [
		[1.0, 2.0],
		[1.0, -1.0],
		[1.0, Real::NAN],
		[Real::INFINITY, 1.0],
	] {
		assert!(!valid_spectrum(&s));
	}
	assert!(spectrum_ratio(&[1.0, 0.0], &[1.0, 1.0], 2) >= THRESHOLD);
	assert!(!spectrum_ratio(&[Real::NAN], &[1.0], 1).is_finite());
	assert!(!spectrum_ratio(&[1.0], &[0.0], 1).is_finite());
}
