// Adapted CHKHS tests 1-2 / HST01, LAPACK 3.12.0, OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Deterministic Rust fixtures replace LATMS/LATMR.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD.
fn general_case(n: usize, kind: usize) -> Mat<Scalar> {
	if kind <= 2 || kind == 4 || kind == 6 || kind == 7 {
		return hermitian(n, kind).0;
	}
	Mat::from_fn(n, n, |i, j| match kind {
		3 => z(if i == j || i == j + 1 { 1.0 } else { 0.0 }),
		5 => {
			if i == j {
				z(1.0 + 8.0 * Real::EPSILON * i as Real)
			} else if i < j {
				phase(i + j) * 0.25
			} else {
				z(0.0)
			}
		},
		_ => sample(i, j),
	})
}
fn hessenberg_case(
	run: &mut Run,
	a: MatRef<'_, Scalar>,
	layout: usize,
	mut par: Par,
) {
	let n = a.nrows();
	let bs = 3;
	let mut parent = guarded(n, n, layout);
	let mut qp = guarded(n, n, layout);
	let mut coeff = Mat::full(bs, n.saturating_sub(1), guard());
	let params = Spec::new(hessenberg::HessenbergParams { ..auto!(Scalar) });
	let mut mem = MemBuffer::new(hessenberg::hessenberg_in_place_scratch::<
		Scalar,
	>(n, bs, par.shape(), params));
	let mut qmem=MemBuffer::new(householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<Scalar>(n.saturating_sub(1),bs,n));
	for pass in 0..2 {
		let input = scaled(a, if pass == 0 { 1.0 } else { 0.5 });
		interior_mut(parent.as_mut(), layout).copy_from(&input);
		let ctx =
			format!("Hessenberg n={n} layout={layout} pass={pass} par={par:?}");
		if run.call(&ctx, || {
			hessenberg::hessenberg_in_place(
				interior_mut(parent.as_mut(), layout),
				coeff.as_mut(),
				par.reborrow(),
				MemStack::new(&mut mem),
				params,
			);
			Ok::<_, ()>(())
		}) {
			let packed = interior(parent.as_ref(), layout);
			let h = Mat::from_fn(n, n, |i, j| {
				if i <= j + 1 { packed[(i, j)] } else { z(0.0) }
			});
			let mut q = interior_mut(qp.as_mut(), layout);
			q.fill(z(0.0));
			for i in 0..n {
				q[(i, i)] = z(1.0);
			}
			if run.aux(&ctx, || {
				if n > 1 {
					householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(packed.submatrix(1,0,n-1,n-1),coeff.as_ref(),Conj::No,q.rb_mut().subrows_mut(1,n-1),par.reborrow(),MemStack::new(&mut qmem));
				}
				Ok::<_, ()>(())
			}) {
				run.ratio(
					&ctx,
					"A-QHQH",
					similarity_ratio(
						input.as_ref(),
						q.as_ref(),
						h.as_ref(),
						Real::EPSILON,
					),
					20.0,
				);
				run.ratio(
					&ctx,
					"Q unitary",
					unitary_ratio(q.as_ref(), Real::EPSILON),
					20.0,
				);
			}
		}
		run.require(
			&ctx,
			"guard overwritten",
			guards_ok(parent.as_ref()) && guards_ok(qp.as_ref()),
		);
	}
}
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_hessenberg_cases() {
	let mut run = Run::new("hessenberg");
	for n in [0, 1, 2, 3, 5, 10, 16, 74, 75, 76] {
		for kind in 0..9 {
			let a = general_case(n, kind);
			for layout in 0..4 {
				hessenberg_case(&mut run, a.as_ref(), layout, Par::Seq);
			}
		}
	}
	run.finish(10 * 9 * 4 * 2);
}
#[cfg(feature = "rayon")]
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_hessenberg_parallel() {
	let mut run = Run::new("hessenberg_parallel");
	let a = general_case(257, 4);
	hessenberg_case(&mut run, a.as_ref(), 0, Par::Seq);
	for threads in [2, 4] {
		rayon::ThreadPoolBuilder::new()
			.num_threads(threads)
			.build()
			.unwrap()
			.install(|| {
				hessenberg_case(&mut run, a.as_ref(), 0, Par::rayon(threads))
			});
	}
	run.finish(6);
}
