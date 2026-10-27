// Faer-specific layout, buffer-reuse, and parallel extensions to the LAPACK
// SVD pilot. Known-spectrum generators and numerical checks are shared with
// `super::lapack`, which records the pinned LAPACK/OpenBLAS source revision.
// Heavy opt-in validation; the scheduled SVD workflow includes ignored tests.
// Reference: OpenMathLib/OpenBLAS 791b02a1a00c996b66a894f6df93421c53b62d80,
// lapack-netlib/TESTING/EIG/{dchkbd.f,ddrvbd.f,schkbd.f,sdrvbd.f,
// zchkbd.f,zdrvbd.f,cchkbd.f,cdrvbd.f};
// cases here extend that port. Scalar/Real are the enclosing storage/precision aliases.
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
use super::lapack::{
	check_driver_output, check_svd_call, diagonal_case, rotate,
};
use super::{GUARD, Real, Scalar, diagonal_phase};
use crate::assert;
use dyn_stack::MemBuffer;

fn guards_unchanged(parent: MatRef<'_, Scalar>) -> bool {
	(0..parent.ncols()).all(|j| {
		(0..parent.nrows()).all(|i| {
			(i > 0 && i + 1 < parent.nrows() && j > 0 && j + 1 < parent.ncols())
				|| parent[(i, j)] == GUARD
		})
	})
}

fn output_view(
	parent: MatMut<'_, Scalar>,
	row_major: bool,
) -> MatMut<'_, Scalar> {
	let (m, n) = parent.shape();
	let inner = parent.submatrix_mut(1, 1, m - 2, n - 2);
	if row_major {
		inner.transpose_mut()
	} else {
		inner
	}
}

fn run_view_case(
	failures: &mut Vec<String>,
	attempted: &mut usize,
	name: &str,
	a: MatRef<'_, Scalar>,
	expected: &[Real],
	threshold: usize,
	mut par: Par,
	mode: (ComputeSvdVectors, ComputeSvdVectors),
	row_major: bool,
) {
	let (m, n) = a.shape();
	let rank = m.min(n);
	let (left, right) = mode;
	use ComputeSvdVectors::{Full, No, Thin};
	let columns = |mode, full| match mode {
		Full => full,
		Thin => rank,
		No => 0,
	};
	let uc = columns(left, m);
	let vc = columns(right, n);
	let parent = |rows, cols| {
		if row_major {
			Mat::full(cols + 2, rows + 2, GUARD)
		} else {
			Mat::full(rows + 2, cols + 2, GUARD)
		}
	};
	let mut u_parent = parent(m, uc);
	let mut v_parent = parent(n, vc);
	output_view(u_parent.as_mut(), row_major).fill(one::<Scalar>());
	output_view(v_parent.as_mut(), row_major).fill(one::<Scalar>());
	let mut s_parent = Col::full(rank + 2, GUARD);
	s_parent.as_mut().subrows_mut(1, rank).fill(nan::<Scalar>());
	let params = Spec::new(SvdParams {
		recursion_threshold: threshold,
		..auto!(Scalar)
	});
	let mut memory =
		MemBuffer::new(svd_scratch::<Scalar>(m, n, left, right, par.shape(), params));
	let before = Mat::from_fn(m, n, |i, j| a[(i, j)]);
	let next = Mat::from_fn(m, n, |i, j| -0.5 * a[(i, j)]);
	let next_spectrum: Vec<_> = expected.iter().map(|s| 0.5 * s).collect();
	for (pass, input, spectrum) in [
		(0, a, expected),
		(1, next.as_ref(), next_spectrum.as_slice()),
	] {
		let context = format!(
			"{name}, {m}x{n}, threshold={threshold}, par={par:?}, U={left:?}, V={right:?}, row-major outputs={row_major}, pass={pass}, input strides=({}, {})",
			input.row_stride(),
			input.col_stride()
		);
		let u = output_view(u_parent.as_mut(), row_major);
		let v = output_view(v_parent.as_mut(), row_major);
		let s = s_parent.as_mut().subrows_mut(1, rank).as_dyn_stride_mut();
		let s = if row_major { s.reverse_rows_mut() } else { s };
		*attempted += 1;
		if check_svd_call(failures, &context, || {
			svd(
				input,
				s.as_diagonal_mut(),
				if left == No { None } else { Some(u) },
				if right == No { None } else { Some(v) },
				par.reborrow(),
				MemStack::new(&mut memory),
				params,
			)
		}) {
			let s = s_parent.as_ref().subrows(1, rank).as_dyn_stride();
			let s = if row_major { s.reverse_rows() } else { s };
			let s: Vec<_> = s.iter().copied().collect();
			check_driver_output(
				failures,
				&context,
				input,
				&s,
				output_view(u_parent.as_mut(), row_major).as_ref(),
				output_view(v_parent.as_mut(), row_major).as_ref(),
				Some(spectrum),
				None,
			);
		}
		if !guards_unchanged(u_parent.as_ref())
			|| !guards_unchanged(v_parent.as_ref())
			|| s_parent[0] != GUARD
			|| s_parent[rank + 1] != GUARD
		{
			failures
				.push(format!("{context}: output parent guard overwritten"));
		}
		let input_scale = if pass == 0 { 1.0 } else { -0.5 };
		if !(0..n).all(|j| {
			(0..m).all(|i| {
				real(&input[(i, j)]).to_bits()
					== real(&(input_scale * before[(i, j)])).to_bits()
					&& imag(&input[(i, j)]).to_bits()
						== imag(&(input_scale * before[(i, j)])).to_bits()
			})
		}) {
			failures.push(format!("{context}: input was modified"));
		}
	}
}

#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn layouts_and_reused_buffers() {
	use ComputeSvdVectors::{Full, No, Thin};
	let mut failures = Vec::new();
	let mut attempted = 0;
	for (m, n) in [(4, 2), (17, 7), (17, 8), (17, 9), (29, 16), (30, 16)] {
		for kind in [4, 5, 6, 7, 8, 9] {
			let (mut a, expected) = diagonal_case(m, n, kind);
			rotate(&mut a);
			let mut input_parent = Mat::full(m + 2, n + 2, GUARD);
			input_parent
				.as_mut()
				.submatrix_mut(1, 1, m, n)
				.copy_from(&a);
			let padded = input_parent.as_ref().submatrix(1, 1, m, n);
			for (layout, input) in [
				("padded column-major", padded),
				("padded row-major", padded.transpose()),
				("reversed column-major", padded.reverse_rows_and_cols()),
				(
					"reversed row-major",
					padded.transpose().reverse_rows_and_cols(),
				),
			] {
				for row_major in [false, true] {
					for mode in [
						(Full, Full),
						(Thin, Thin),
						(Thin, No),
						(No, Thin),
						(No, No),
					] {
						run_view_case(
							&mut failures,
							&mut attempted,
							&format!("spectrum kind={kind}, {layout}"),
							input,
							&expected,
							8,
							Par::Seq,
							mode,
							row_major,
						);
					}
				}
			}
			assert!(guards_unchanged(input_parent.as_ref()));
		}
	}
	eprintln!(
		"LAPACK_CASES family=svd_layouts_and_reused_buffers dtype={} planned=2880 attempted={attempted} call_failures={} diagnostics={}",
		core::any::type_name::<Scalar>(),
		failures
			.iter()
			.filter(|f| f.starts_with("call failure:"))
			.count(),
		failures.len()
	);
	assert_eq!(attempted, 2880, "SVD case execution incomplete");
	assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[cfg(feature = "rayon")]
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn parallel_paths() {
	use ComputeSvdVectors::{No, Thin};
	let mut failures = Vec::new();
	let mut attempted = 0;
	for (m, n) in [(257, 256), (512, 128), (128, 512)] {
		let (mut a, expected) = diagonal_case(m, n, 3);
		rotate(&mut a);
		let modes = [(Thin, Thin), (Thin, No), (No, Thin), (No, No)];
		for mode in modes {
			run_view_case(
				&mut failures,
				&mut attempted,
				"parallel-path sequential reference",
				a.as_ref(),
				&expected,
				128,
				Par::Seq,
				mode,
				false,
			);
		}
		for threads in [2, 4] {
			let pool = rayon::ThreadPoolBuilder::new()
				.num_threads(threads)
				.build()
				.unwrap();
			pool.install(|| {
				assert_eq!(rayon::current_num_threads(), threads);
				for mode in modes {
					run_view_case(
						&mut failures,
						&mut attempted,
						"parallel path",
						a.as_ref(),
						&expected,
						128,
						Par::rayon(threads),
						mode,
						false,
					);
				}
			});
		}
	}
	eprintln!(
		"LAPACK_CASES family=svd_parallel_paths dtype={} planned=72 attempted={attempted} call_failures={} diagnostics={}",
		core::any::type_name::<Scalar>(),
		failures
			.iter()
			.filter(|f| f.starts_with("call failure:"))
			.count(),
		failures.len()
	);
	assert_eq!(attempted, 72, "SVD case execution incomplete");
	assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn guards_detect_writes_outside_output_views() {
	let mut parent = Mat::<Scalar>::full(5, 5, GUARD);
	parent
		.as_mut()
		.submatrix_mut(1, 1, 3, 3)
		.fill(one::<Scalar>());
	assert!(guards_unchanged(parent.as_ref()));
	for (i, j) in [(0, 2), (4, 2), (2, 0), (2, 4)] {
		parent[(i, j)] = zero::<Scalar>();
		assert!(!guards_unchanged(parent.as_ref()));
		parent[(i, j)] = GUARD;
	}
}

#[test]
fn known_spectra_include_repeated_and_tight_clusters() {
	for (kind, want) in [
		(8, [4.0, 4.0, 1.0, 1.0]),
		(
			9,
			[
				1.0,
				1.0 - 8.0 * Real::EPSILON,
				1.0 - 16.0 * Real::EPSILON,
				1.0 - 24.0 * Real::EPSILON,
			],
		),
	] {
		let (a, s) = diagonal_case(4, 4, kind);
		assert_eq!(s, want);
		for i in 0..4 {
			assert_eq!(a[(i, i)], Scalar::from(want[i]) * diagonal_phase(i));
		}
	}
}
