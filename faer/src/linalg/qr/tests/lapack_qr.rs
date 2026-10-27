// CHKQR/CHKQ3 and QRT01/02/03: free-column pivoting, explicit scalar Q, reconstruction and Q/QH application.
// Adapted LAPACK LIN numerical properties, LAPACK 3.12.0, OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. Deterministic Rust fixtures replace LATMS/LATMR.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD.

use crate::linalg::{
	householder as hh,
	qr::{col_pivoting as cp, no_pivoting as np},
};
fn qr_case(
	run: &mut Run,
	m: usize,
	n: usize,
	kind: usize,
	layout: usize,
	bs: usize,
	pivot: bool,
	mut par: Par,
) {
	let a = qr_fixture(m, n, kind);
	let k = m.min(n);
	let mut ap = guarded(m, n, layout);
	let mut out = guarded(m, n, layout);
	let mut coeff = Mat::full(bs, k, guard());
	let mut p = vec![usize::MAX; n];
	let mut ip = p.clone();
	let np = Spec::new(np::factor::QrParams { ..auto!(Scalar) });
	let cp = Spec::new(cp::factor::ColPivQrParams { ..auto!(Scalar) });
	let mut mem = MemBuffer::new(if pivot {
		cp::factor::qr_in_place_scratch::<usize, Scalar>(m, n, bs, par.shape(), cp)
	} else {
		np::factor::qr_in_place_scratch::<Scalar>(m, n, bs, par.shape(), np)
	});
	let mut rm = MemBuffer::new(if pivot {
		cp::reconstruct::reconstruct_scratch::<usize, Scalar>(m, n, bs, par.shape())
	} else {
		np::reconstruct::reconstruct_scratch::<Scalar>(m, n, bs, par.shape())
	});
	for pass in 0..2 {
		let input = scaled(a.as_ref(), if pass == 0 { 1.0 } else { 0.5 });
		interior_mut(ap.as_mut(), layout).copy_from(&input);
		let ctx = format!(
			"QR m={m} n={n} kind={kind} layout={layout} block={bs} pivot={pivot} pass={pass} par={par:?}"
		);
		let mut rank = None;
		if run.call(&ctx, || {
			if pivot {
				cp::factor::qr_in_place(
					interior_mut(ap.as_mut(), layout),
					coeff.as_mut(),
					&mut p,
					&mut ip,
					par.reborrow(),
					MemStack::new(&mut mem),
					cp,
				);
			} else {
				rank = Some(
					np::factor::qr_in_place(
						interior_mut(ap.as_mut(), layout),
						coeff.as_mut(),
						par.reborrow(),
						MemStack::new(&mut mem),
						np,
					)
					.rank,
				);
				for i in 0..n {
					p[i] = i;
					ip[i] = i;
				}
			}
			Ok::<_, ()>(())
		}) {
			run.require(&ctx, "invalid permutation", permutation_ok(&p, &ip));
			if let Some(rank) = rank {
				run.require(
					&ctx,
					"invalid reported rank",
					rank <= k && (kind != 0 || rank == 0),
				);
			}
			if permutation_ok(&p, &ip) {
				let packed = interior(ap.as_ref(), layout);
				let before = ap.clone();
				let cb = coeff.clone();
				let q = scalar_q(packed, coeff.as_ref());
				let r = Mat::from_fn(m, n, |i, j| {
					if i <= j && i < k {
						packed[(i, j)]
					} else {
						z(0.0)
					}
				});
				let permuted = Mat::from_fn(m, n, |i, j| input[(i, p[j])]);
				let qha =
					product(adjoint(q.as_ref()).as_ref(), permuted.as_ref());
				run.ratio(
					&ctx,
					"R-QH AP",
					lin_difference(qha.as_ref(), r.as_ref(), m),
					30.0,
				);
				run.ratio(
					&ctx,
					"Q unitary",
					unitary_ratio(q.as_ref(), lin_eps()),
					30.0,
				);
				run.ratio(
					&ctx,
					"thin Q unitary",
					unitary_ratio(q.as_ref().subcols(0, k), lin_eps()),
					30.0,
				);
				let perm = crate::perm::PermRef::new_checked(&p, &ip, n);
				if run.aux(&ctx, || {
					if pivot {
						cp::reconstruct::reconstruct(
							interior_mut(out.as_mut(), layout),
							packed.subcols(0, k),
							coeff.as_ref(),
							packed.subrows(0, k),
							perm,
							par.reborrow(),
							MemStack::new(&mut rm),
						);
					} else {
						np::reconstruct::reconstruct(
							interior_mut(out.as_mut(), layout),
							packed.subcols(0, k),
							coeff.as_ref(),
							packed.subrows(0, k),
							par.reborrow(),
							MemStack::new(&mut rm),
						);
					}
					Ok::<_, ()>(())
				}) {
					run.ratio(
						&ctx,
						"API reconstruction",
						lin_difference(
							interior(out.as_ref(), layout),
							input.as_ref(),
							m,
						),
						30.0,
					);
				}
				let req=hh::apply_block_householder_sequence_on_the_left_in_place_scratch::<Scalar>(m,bs,3).or(hh::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<Scalar>(m,bs,3)).or(hh::apply_block_householder_sequence_on_the_right_in_place_scratch::<Scalar>(m,bs,3)).or(hh::apply_block_householder_sequence_transpose_on_the_right_in_place_scratch::<Scalar>(m,bs,3));
				let mut am = MemBuffer::new(req);
				for side in [false, true] {
					let (mr, nr) = if side { (3, m) } else { (m, 3) };
					let mut xp = guarded(mr, nr, layout);
					let original = Mat::from_fn(mr, nr, |i, j| sample(i, j));
					for adj in [false, true] {
						let oq =
							if adj { adjoint(q.as_ref()) } else { q.clone() };
						let expected = if side {
							product(original.as_ref(), oq.as_ref())
						} else {
							product(oq.as_ref(), original.as_ref())
						};
						interior_mut(xp.as_mut(), layout).copy_from(&original);
						if run.aux(&ctx,||{let basis=packed.subcols(0,k);let c=coeff.as_ref();let x=interior_mut(xp.as_mut(),layout);match(side,adj){(false,false)=>hh::apply_block_householder_sequence_on_the_left_in_place_with_conj(basis,c,Conj::No,x,par.reborrow(),MemStack::new(&mut am)),(false,true)=>hh::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(basis,c,Conj::Yes,x,par.reborrow(),MemStack::new(&mut am)),(true,false)=>hh::apply_block_householder_sequence_on_the_right_in_place_with_conj(basis,c,Conj::No,x,par.reborrow(),MemStack::new(&mut am)),(true,true)=>hh::apply_block_householder_sequence_transpose_on_the_right_in_place_with_conj(basis,c,Conj::Yes,x,par.reborrow(),MemStack::new(&mut am))};Ok::<_,()>(())}){run.ratio(&ctx,&format!("Q application right={side} adjoint={adj}"),lin_difference(interior(xp.as_ref(),layout),expected.as_ref(),m),30.0);}
						run.require(
							&ctx,
							"application guards",
							guards_ok(xp.as_ref()),
						);
					}
				}
				run.require(
					&ctx,
					"factors changed",
					same_bits(ap.as_ref(), before.as_ref())
						&& same_bits(coeff.as_ref(), cb.as_ref()),
				);
			}
		}
		run.require(
			&ctx,
			"guards overwritten",
			guards_ok(ap.as_ref()) && guards_ok(out.as_ref()),
		);
	}
}
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_qr_cases() {
	let mut run = Run::new("qr");
	for (m, n) in [
		(0, 0),
		(0, 3),
		(3, 0),
		(1, 1),
		(2, 2),
		(3, 5),
		(5, 3),
		(10, 10),
		(50, 50),
		(47, 47),
		(48, 48),
		(49, 49),
		(3, 3),
		(4, 4),
		(5, 5),
		(63, 63),
		(64, 64),
		(65, 65),
	] {
		for kind in 0..8 {
			for pivot in [false, true] {
				for bs in [1, 4] {
					qr_case(
						&mut run,
						m,
						n,
						kind,
						(m + kind) % 4,
						bs,
						pivot,
						Par::Seq,
					);
				}
			}
		}
	}
	run.finish(18 * 8 * 2 * 2 * 2);
}
#[cfg(feature = "rayon")]
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_qr_parallel() {
	let mut run = Run::new("qr_parallel");
	for pivot in [false, true] {
		qr_case(&mut run, 257, 257, 2, 0, 8, pivot, Par::Seq);
		for threads in [2, 4] {
			rayon::ThreadPoolBuilder::new()
				.num_threads(threads)
				.build()
				.unwrap()
				.install(|| {
					qr_case(
						&mut run,
						257,
						257,
						2,
						threads / 2,
						8,
						pivot,
						Par::rayon(threads),
					)
				});
		}
	}
	run.finish(12);
}
