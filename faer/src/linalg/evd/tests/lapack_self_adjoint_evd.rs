// Adapted CHKST tests 9-12, 22-26 / SYT21-HET21, LAPACK 3.12.0, OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Deterministic Rust fixtures replace LATMS/LATMR.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD.

fn dense_self_adjoint_case(
	run: &mut Run,
	n: usize,
	kind: usize,
	layout: usize,
	threshold: usize,
	mut par: Par,
) {
	let (a, mut known) = if kind == 8 {
		assert_eq!(n, 2);
		(
			Mat::from_fn(2, 2, |i, j| {
				if i == j {
					z(Real::MAX / 2.0)
				} else if i > j {
					phase(0) * (Real::MAX / 4.0)
				} else {
					conj(&phase(0)) * (Real::MAX / 4.0)
				}
			}),
			vec![Real::MAX / 4.0, (Real::MAX / 4.0) * 3.0],
		)
	} else if kind == 9 {
		assert_eq!(n, 2);
		(
			Mat::from_fn(2, 2, |i, j| {
				if i == j {
					z(Real::MIN_POSITIVE / 4.0)
				} else {
					z(0.0)
				}
			}),
			vec![Real::MIN_POSITIVE / 4.0; 2],
		)
	} else {
		hermitian(n, kind)
	};
	known.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let bs = 3;
	let mut ap = guarded(n, n, layout);
	let mut qp = guarded(n, n, layout);
	let mut sp = guarded(n, 1, layout);
	let mut coeff = Mat::full(bs, n.saturating_sub(1), guard());
	let tp = Spec::new(tridiag::TridiagParams { ..auto!(Scalar) });
	let ep = Spec::new(SelfAdjointEvdParams {
		recursion_threshold: threshold,
		..auto!(Scalar)
	});
	let mut mem =
		MemBuffer::new(tridiag::tridiag_in_place_scratch::<Scalar>(n, par.shape(), tp));
	let mut qm=MemBuffer::new(householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<Scalar>(n.saturating_sub(1),bs,n));
	let mut em = MemBuffer::new(
		self_adjoint_evd_scratch::<Scalar>(
			n,
			ComputeEigenvectors::Yes,
			par.shape(),
			ep,
		)
		.or(self_adjoint_evd_scratch::<Scalar>(
			n,
			ComputeEigenvectors::No,
			par.shape(),
			ep,
		)),
	);
	for pass in 0..2 {
		let f = if pass == 0 { 1.0 } else { 0.5 };
		let input = scaled(a.as_ref(), f);
		let stored = Mat::from_fn(n, n, |i, j| {
			if i >= j {
				input[(i, j)]
			} else {
				scalar(Real::NAN, Real::NAN)
			}
		});
		interior_mut(ap.as_mut(), layout).copy_from(&stored);
		let ctx = format!(
			"CHKST reduction n={n} kind={kind} layout={layout} pass={pass} par={par:?}"
		);
		if run.call(&ctx, || {
			tridiag::tridiag_in_place(
				interior_mut(ap.as_mut(), layout),
				coeff.as_mut(),
				par.reborrow(),
				MemStack::new(&mut mem),
				tp,
			);
			Ok::<_, ()>(())
		}) {
			let packed = interior(ap.as_ref(), layout);
			let t = Mat::from_fn(n, n, |i, j| {
				if i == j || i == j + 1 {
					packed[(i, j)]
				} else if j == i + 1 {
					conj(&packed[(j, i)])
				} else {
					z(0.0)
				}
			});
			let mut q = interior_mut(qp.as_mut(), layout);
			q.fill(z(0.0));
			for i in 0..n {
				q[(i, i)] = z(1.0);
			}
			if run.aux(&ctx, || {
				if n > 1 {
					householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(packed.submatrix(1,0,n-1,n-1),coeff.as_ref(),Conj::No,q.rb_mut().subrows_mut(1,n-1),par.reborrow(),MemStack::new(&mut qm));
				}
				Ok::<_, ()>(())
			}) {
				run.ratio(
					&ctx,
					"A-QTQH",
					similarity_ratio(
						input.as_ref(),
						q.as_ref(),
						t.as_ref(),
						Real::EPSILON,
					),
					60.0,
				);
				run.ratio(
					&ctx,
					"unitary",
					unitary_ratio(q.as_ref(), Real::EPSILON),
					60.0,
				);
			}
		}
		interior_mut(ap.as_mut(), layout).copy_from(&stored);
		let before = ap.clone();
		let mut values_only: Option<Vec<Real>> = None;
		for vectors in [false, true] {
			let ctx = format!(
				"dense EVD n={n} kind={kind} layout={layout} threshold={threshold} vectors={vectors} pass={pass} par={par:?}"
			);
			if run.call(&ctx, || {
				self_adjoint_evd(
					interior(ap.as_ref(), layout),
					interior_mut(sp.as_mut(), layout)
						.col_mut(0)
						.as_diagonal_mut(),
					if vectors {
						Some(interior_mut(qp.as_mut(), layout))
					} else {
						None
					},
					par.reborrow(),
					MemStack::new(&mut em),
					ep,
				)
			}) {
				let s = interior(sp.as_ref(), layout);
				let values: Vec<_> = (0..n).map(|i| real(&s[(i, 0)])).collect();
				run.require(
					&ctx,
					"spectrum not ordered real finite",
					valid_real_spectrum(s),
				);
				let expected: Vec<_> = known.iter().map(|x| x * f).collect();
				run.ratio(
					&ctx,
					"known spectrum",
					ordered_spectrum_ratio(&values, &expected),
					60.0,
				);
				// The ratio's safe-minimum floor accepts any error below
				// MIN_POSITIVE; this diagonal spectrum is exactly representable.
				if kind == 9 {
					run.require(
						&ctx,
						"subnormal spectrum not exact",
						values == expected,
					);
				}
				if let Some(ref v) = values_only {
					run.ratio(
						&ctx,
						"mode agreement",
						ordered_spectrum_ratio(&values, v),
						60.0,
					);
				} else {
					values_only = Some(values);
				}
				if vectors {
					let q = interior(qp.as_ref(), layout);
					let d = Mat::from_fn(n, n, |i, j| {
						if i == j { s[(i, 0)] } else { z(0.0) }
					});
					run.ratio(
						&ctx,
						"AV-VD",
						self_adjoint_residual(input.as_ref(), q, s),
						60.0,
					);
					run.ratio(
						&ctx,
						"A-VDVH",
						similarity_ratio(
							input.as_ref(),
							q,
							d.as_ref(),
							Real::EPSILON,
						),
						60.0,
					);
					run.ratio(
						&ctx,
						"unitary",
						unitary_ratio(q, Real::EPSILON),
						60.0,
					);
				}
			}
			run.require(
				&ctx,
				"input changed",
				same_bits(ap.as_ref(), before.as_ref()),
			);
		}
		run.require(
			&ctx,
			"guards overwritten",
			guards_ok(ap.as_ref())
				&& guards_ok(qp.as_ref())
				&& guards_ok(sp.as_ref()),
		);
	}
}
#[test]
fn lapack_self_adjoint_evd_checkers() {
	let (a, s) = hermitian(2, 5);
	let q = unitary(2);
	let d = Mat::from_fn(2, 2, |i, j| if i == j { z(s[i]) } else { z(0.0) });
	crate::assert!(
		similarity_ratio(a.as_ref(), q.as_ref(), d.as_ref(), Real::EPSILON)
			< 60.0
	);
	let mut bad = a.clone();
	bad[(0, 1)] = z(4.0);
	crate::assert!(
		similarity_ratio(bad.as_ref(), q.as_ref(), d.as_ref(), Real::EPSILON)
			> 60.0
	);
	if !<Scalar as ComplexField>::IS_REAL {
		let mut bad = a.clone();
		bad[(0, 1)] = conj(&bad[(0, 1)]);
		crate::assert!(
			similarity_ratio(
				bad.as_ref(),
				q.as_ref(),
				d.as_ref(),
				Real::EPSILON
			) > 60.0
		);
	}
}
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_self_adjoint_evd_cases() {
	let mut run = Run::new("self_adjoint_evd");
	for n in [0, 1, 2, 3, 5, 10, 16, 7, 8, 9, 127, 128, 129] {
		for kind in 0..8 {
			for threshold in [8, 128] {
				dense_self_adjoint_case(
					&mut run,
					n,
					kind,
					(n + kind) % 4,
					threshold,
					Par::Seq,
				);
			}
		}
	}
	run.finish(13 * 8 * 2 * 6);
}
#[cfg(feature = "rayon")]
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_self_adjoint_evd_parallel() {
	let mut run = Run::new("self_adjoint_evd_parallel");
	dense_self_adjoint_case(&mut run, 257, 4, 0, 128, Par::Seq);
	for threads in [2, 4] {
		rayon::ThreadPoolBuilder::new()
			.num_threads(threads)
			.build()
			.unwrap()
			.install(|| {
				dense_self_adjoint_case(
					&mut run,
					257,
					4,
					threads / 2,
					128,
					Par::rayon(threads),
				)
			});
	}
	run.finish(18);
}

#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_self_adjoint_evd_range() {
	let mut run = Run::new("self_adjoint_evd_range");
	for kind in [8, 9] {
		for layout in 0..4 {
			dense_self_adjoint_case(&mut run, 2, kind, layout, 8, Par::Seq);
		}
	}
	run.finish(48);
}

// faer#76. The values-only path, and any vector-producing block that is not
// divided, solve the tridiagonal eigenproblem with the QR fallback. Its
// diagonal update used to re-round every diagonal of size `|d|` on each
// rotation, so this fixture — `s[i] = 1 - 2 i / (n - 1)` behind adjacent
// 0.6/0.8 rotations, which puts the intended eigenvalues on an even grid of
// spacing `2/126` — came out about 150 f32 epsilons wrong, while the
// divide-and-conquer path on the same input stayed within 5. Both recursion
// thresholds and both compute-eigenvectors modes answer to the same CHKST
// bound of 60 epsilons.
#[test]
fn dense_self_adjoint_qr_spectrum_accuracy() {
	let n = 127;
	let (a, mut known) = hermitian(n, 2);
	known.sort_by(|x, y| x.partial_cmp(y).unwrap());
	for threshold in [8, 128] {
		let params = Spec::new(SelfAdjointEvdParams {
			recursion_threshold: threshold,
			..auto!(Scalar)
		});
		let mut mem = MemBuffer::new(
			self_adjoint_evd_scratch::<Scalar>(
				n,
				ComputeEigenvectors::Yes,
				ParShape::Seq,
				params,
			)
			.or(self_adjoint_evd_scratch::<Scalar>(
				n,
				ComputeEigenvectors::No,
				ParShape::Seq,
				params,
			)),
		);
		for vectors in [false, true] {
			let mut s = Mat::<Scalar>::zeros(n, 1);
			let mut u = Mat::<Scalar>::zeros(n, n);
			self_adjoint_evd(
				a.as_ref(),
				s.as_mut().col_mut(0).as_diagonal_mut(),
				if vectors { Some(u.as_mut()) } else { None },
				Par::Seq,
				MemStack::new(&mut mem),
				params,
			)
			.unwrap();
			let values: Vec<_> = (0..n).map(|i| real(&s[(i, 0)])).collect();
			let ctx = format!(
				"dense self-adjoint n={n} threshold={threshold} \
				 vectors={vectors}"
			);
			let ratio = ordered_spectrum_ratio(&values, &known);
			crate::assert!(
				ratio < 60.0,
				"{ctx}: known spectrum ratio={ratio:e}, threshold=60"
			);
			if vectors {
				let residual =
					self_adjoint_residual(a.as_ref(), u.as_ref(), s.as_ref());
				crate::assert!(
					residual < 60.0,
					"{ctx}: AV-VD ratio={residual:e}, threshold=60"
				);
			}
		}
	}
}
