//! Faer-specific layout, buffer-reuse, and parallel extensions to the LAPACK
//! SVD pilot. The cases here are faer's own; the known-spectrum generators and
//! numerical checks come from `super::lapack`, which records the LAPACK/OpenBLAS
//! provenance and license notices of the adapted material.
//! The validation groups are ignored in ordinary test runs; the scheduled SVD
//! workflow runs them.

use super::super::*;
use super::lapack::{
	check_driver_output, check_ratio, diagonal_case, rotate, spectrum_ratio,
};
use crate::assert;
use dyn_stack::MemBuffer;
use std::panic::{AssertUnwindSafe, catch_unwind};

const GUARD: f64 = -12345.0;

fn guards_unchanged(parent: MatRef<'_, f64>) -> bool {
	(0..parent.ncols()).all(|j| {
		(0..parent.nrows()).all(|i| {
			(i > 0 && i + 1 < parent.nrows() && j > 0 && j + 1 < parent.ncols())
				|| parent[(i, j)] == GUARD
		})
	})
}

fn output_view(parent: MatMut<'_, f64>, row_major: bool) -> MatMut<'_, f64> {
	let (m, n) = parent.shape();
	let inner = parent.submatrix_mut(1, 1, m - 2, n - 2);
	if row_major {
		inner.transpose_mut()
	} else {
		inner
	}
}

fn guarded_input(a: MatRef<'_, f64>) -> Mat<f64> {
	let (m, n) = a.shape();
	let mut parent = Mat::full(m + 2, n + 2, GUARD);
	parent.as_mut().submatrix_mut(1, 1, m, n).copy_from(a);
	parent
}

fn input_layouts(
	parent: MatRef<'_, f64>,
) -> [(&'static str, MatRef<'_, f64>); 4] {
	let (m, n) = parent.shape();
	let padded = parent.submatrix(1, 1, m - 2, n - 2);
	[
		("padded column-major", padded),
		("padded row-major", padded.transpose()),
		("reversed column-major", padded.reverse_rows_and_cols()),
		(
			"reversed row-major",
			padded.transpose().reverse_rows_and_cols(),
		),
	]
}

/// Runs `inputs[0]` and then `inputs[1]`, which must be `-inputs[0] / 2`, through
/// the same output and scratch buffers. Returns each call's singular values.
fn run_view_case(
	failures: &mut Vec<String>,
	name: &str,
	inputs: [MatRef<'_, f64>; 2],
	expected: &[f64],
	threshold: usize,
	par: Par,
	mode: (ComputeSvdVectors, ComputeSvdVectors),
	row_major: bool,
	sequential: Option<&[Option<Vec<f64>>; 2]>,
) -> [Option<Vec<f64>>; 2] {
	let (m, n) = inputs[0].shape();
	assert_eq!(inputs[1].shape(), (m, n));
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
	output_view(u_parent.as_mut(), row_major).fill(f64::NAN);
	output_view(v_parent.as_mut(), row_major).fill(f64::NAN);
	let mut s_parent = Col::full(rank + 2, GUARD);
	s_parent.as_mut().subrows_mut(1, rank).fill(f64::NAN);
	let params = Spec::new(SvdParams {
		recursion_threshold: threshold,
		..auto!(f64)
	});
	let mut memory =
		MemBuffer::new(svd_scratch::<f64>(m, n, left, right, par, params));
	let half: Vec<_> = expected.iter().map(|s| 0.5 * s).collect();
	let mut spectra = [None, None];
	for (pass, (input, spectrum)) in inputs
		.into_iter()
		.zip([expected, half.as_slice()])
		.enumerate()
	{
		let context = format!(
			"{name}, {m}x{n}, threshold={threshold}, par={par:?}, U={left:?}, V={right:?}, row-major outputs={row_major}, pass={pass}, input strides=({}, {})",
			input.row_stride(),
			input.col_stride()
		);
		let before = Mat::from_fn(m, n, |i, j| input[(i, j)]);
		let u = output_view(u_parent.as_mut(), row_major);
		let v = output_view(v_parent.as_mut(), row_major);
		let s = s_parent.as_mut().subrows_mut(1, rank).as_dyn_stride_mut();
		let s = if row_major { s.reverse_rows_mut() } else { s };
		let result = catch_unwind(AssertUnwindSafe(|| {
			svd(
				input,
				s.as_diagonal_mut(),
				if left == No { None } else { Some(u) },
				if right == No { None } else { Some(v) },
				par,
				MemStack::new(&mut memory),
				params,
			)
		}));
		match result {
			Ok(Ok(())) => {
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
				if let Some(reference) = sequential
					.and_then(|sequential| sequential[pass].as_deref())
				{
					check_ratio(
						failures,
						&context,
						"agreement with Par::Seq",
						spectrum_ratio(&s, reference, rank),
					);
				}
				spectra[pass] = Some(s);
			},
			Ok(Err(error)) => failures.push(format!("{context}: {error:?}")),
			Err(payload) => {
				let message = payload
					.downcast_ref::<&str>()
					.copied()
					.or_else(|| {
						payload.downcast_ref::<String>().map(String::as_str)
					})
					.unwrap_or("non-string panic");
				failures.push(format!("{context}: panic: {message}"));
			},
		}
		if !guards_unchanged(u_parent.as_ref())
			|| !guards_unchanged(v_parent.as_ref())
			|| s_parent[0] != GUARD
			|| s_parent[rank + 1] != GUARD
		{
			failures
				.push(format!("{context}: output parent guard overwritten"));
		}
		if !(0..n).all(|j| {
			(0..m).all(|i| input[(i, j)].to_bits() == before[(i, j)].to_bits())
		}) {
			failures.push(format!("{context}: input was modified"));
		}
	}
	spectra
}

#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_svd_f64_layouts_and_reused_buffers() {
	use ComputeSvdVectors::{Full, No, Thin};
	let mut failures = Vec::new();
	for (m, n) in [(4, 2), (17, 7), (17, 8), (17, 9), (29, 16), (30, 16)] {
		for kind in [4, 5, 6, 7, 8, 9] {
			let (mut a, expected) = diagonal_case(m, n, kind);
			rotate(&mut a);
			let first = guarded_input(a.as_ref());
			let second = guarded_input(
				Mat::from_fn(m, n, |i, j| -0.5 * a[(i, j)]).as_ref(),
			);
			for ((layout, input), (_, next)) in input_layouts(first.as_ref())
				.into_iter()
				.zip(input_layouts(second.as_ref()))
			{
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
							&format!("spectrum kind={kind}, {layout}"),
							[input, next],
							&expected,
							8,
							Par::Seq,
							mode,
							row_major,
							None,
						);
					}
				}
			}
			assert!(guards_unchanged(first.as_ref()));
			assert!(guards_unchanged(second.as_ref()));
		}
	}
	assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[cfg(feature = "rayon")]
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_svd_f64_parallel_paths() {
	use ComputeSvdVectors::{No, Thin};
	let mut failures = Vec::new();
	for (m, n) in [(257, 256), (512, 128), (128, 512)] {
		let (mut a, expected) = diagonal_case(m, n, 3);
		rotate(&mut a);
		let next = Mat::from_fn(m, n, |i, j| -0.5 * a[(i, j)]);
		let inputs = [a.as_ref(), next.as_ref()];
		let modes = [(Thin, Thin), (Thin, No), (No, Thin), (No, No)];
		let sequential = modes.map(|mode| {
			run_view_case(
				&mut failures,
				"parallel-path sequential reference",
				inputs,
				&expected,
				128,
				Par::Seq,
				mode,
				false,
				None,
			)
		});
		for threads in [2, 4] {
			let pool = rayon::ThreadPoolBuilder::new()
				.num_threads(threads)
				.build()
				.unwrap();
			pool.install(|| {
				assert_eq!(rayon::current_num_threads(), threads);
				for (mode, sequential) in modes.into_iter().zip(&sequential) {
					run_view_case(
						&mut failures,
						"parallel path",
						inputs,
						&expected,
						128,
						Par::rayon(threads),
						mode,
						false,
						Some(sequential),
					);
				}
			});
		}
	}
	assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn guards_detect_writes_outside_output_views() {
	let mut parent = Mat::<f64>::full(5, 5, GUARD);
	parent.as_mut().submatrix_mut(1, 1, 3, 3).fill(1.0);
	assert!(guards_unchanged(parent.as_ref()));
	for (i, j) in [(0, 2), (4, 2), (2, 0), (2, 4)] {
		parent[(i, j)] = 0.0;
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
				1.0 - 8.0 * f64::EPSILON,
				1.0 - 16.0 * f64::EPSILON,
				1.0 - 24.0 * f64::EPSILON,
			],
		),
	] {
		let (a, s) = diagonal_case(4, 4, kind);
		assert_eq!(s, want);
		for i in 0..4 {
			assert_eq!(a[(i, i)], if i % 2 == 0 { want[i] } else { -want[i] });
		}
	}
}
