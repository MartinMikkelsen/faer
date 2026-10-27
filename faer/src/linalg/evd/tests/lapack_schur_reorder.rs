// GET34/GET36 properties adapted from LAPACK 3.12.0, OpenBLAS
// 791b02a1a00c996b66a894f6df93421c53b62d80. See lapack_fixtures.rs for literal records.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD.
fn block_values(t: MatRef<'_, Scalar>) -> Vec<(Real, Real)> {
	let mut out = Vec::new();
	let mut i = 0;
	while i < t.nrows() {
		if <Scalar as ComplexField>::IS_REAL
			&& i + 1 < t.nrows()
			&& t[(i + 1, i)] != z(0.0)
		{
			let re = real(&t[(i, i)]);
			let im = real(&t[(i, i + 1)]).abs().sqrt()
				* real(&t[(i + 1, i)]).abs().sqrt();
			out.push((re, im));
			out.push((re, -im));
			i += 2;
		} else {
			out.push((real(&t[(i, i)]), imag(&t[(i, i)])));
			i += 1;
		}
	}
	out
}
fn spectrum_matches_scaled(
	a: &[(Real, Real)],
	b: &[(Real, Real)],
	scale: Real,
	threshold: Real,
) -> bool {
	if a.len() != b.len()
		|| a.iter()
			.chain(b)
			.any(|x| !x.0.is_finite() || !x.1.is_finite())
	{
		return false;
	}
	let edge: Vec<Vec<bool>> = a
		.iter()
		.map(|x| {
			b.iter()
				.map(|y| {
					if scale == 0.0 {
						x == y
					} else {
						(x.0 / scale - y.0 / scale)
							.hypot(x.1 / scale - y.1 / scale)
							< threshold * Real::EPSILON
					}
				})
				.collect()
		})
		.collect();
	fn augment(
		i: usize,
		edge: &[Vec<bool>],
		seen: &mut [bool],
		mate: &mut [Option<usize>],
	) -> bool {
		for j in 0..mate.len() {
			if edge[i][j] && !seen[j] {
				seen[j] = true;
				if mate[j].is_none()
					|| augment(mate[j].unwrap(), edge, seen, mate)
				{
					mate[j] = Some(i);
					return true;
				}
			}
		}
		false
	}
	let mut mate = vec![None; b.len()];
	(0..a.len())
		.all(|i| augment(i, &edge, &mut vec![false; b.len()], &mut mate))
}
fn spectrum_matches(
	a: &[(Real, Real)],
	b: &[(Real, Real)],
	threshold: Real,
) -> bool {
	spectrum_matches_scaled(
		a,
		b,
		b.iter().fold(0.0 as Real, |m, x| m.max(x.0.hypot(x.1))),
		threshold,
	)
}
fn block_start(t: MatRef<'_, Scalar>, i: usize) -> usize {
	if <Scalar as ComplexField>::IS_REAL && i > 0 && t[(i, i - 1)] != z(0.0) {
		i - 1
	} else {
		i
	}
}
fn block_len(t: MatRef<'_, Scalar>, i: usize) -> usize {
	if <Scalar as ComplexField>::IS_REAL
		&& i + 1 < t.nrows()
		&& t[(i + 1, i)] != z(0.0)
	{
		2
	} else {
		1
	}
}
fn swap_input(n1: usize, n2: usize, close: bool) -> Mat<Scalar> {
	let n = n1 + n2 + 2;
	let mut t = Mat::from_fn(n, n, |i, j| {
		if i == j {
			z((i + 4) as Real)
		} else if i < j {
			phase(i + j) * 0.125
		} else {
			z(0.0)
		}
	});
	let centre1 = 1.0;
	let centre2 = if close {
		1.0 + 16.0 * Real::EPSILON
	} else {
		3.0
	};
	for (start, size, centre) in [(1, n1, centre1), (1 + n1, n2, centre2)] {
		for i in start..start + size {
			t[(i, i)] = scalar(
				centre,
				if <Scalar as ComplexField>::IS_REAL {
					0.0
				} else {
					0.25 * i as Real
				},
			);
		}
		if size == 2 {
			t[(start, start + 1)] = z(1.0);
			t[(start + 1, start)] = z(-1.0);
		}
	}
	t
}
// Public real positions name a block in the original T, not the final row.
fn reorder_destination(t: MatRef<'_, Scalar>, from: usize, to: usize) -> usize {
	let from = block_start(t, from);
	let to = block_start(t, to);
	if from < to {
		to + block_len(t, to) - block_len(t, from)
	} else {
		to
	}
}
fn reorder_case(
	run: &mut Run,
	input: MatRef<'_, Scalar>,
	from: usize,
	to: usize,
	layout: usize,
	scale: Real,
) {
	let n = input.nrows();
	let input = scaled(input, scale);
	let original = block_values(input.as_ref());
	let start = block_start(input.as_ref(), from);
	let len = block_len(input.as_ref(), start);
	let moving = block_values(input.as_ref().submatrix(start, start, len, len));
	let mut tp = guarded(n, n, layout);
	let mut qp = guarded(n, n, layout);
	let mut mem = MemBuffer::new(reorder_schur_scratch::<Scalar>(n));
	for want_q in [true, false] {
		for pass in 0..2 {
			let a = scaled(input.as_ref(), if pass == 0 { 1.0 } else { 0.5 });
			interior_mut(tp.as_mut(), layout).copy_from(&a);
			let initial = if pass == 0 {
				Mat::<Scalar>::identity(n, n)
			} else {
				unitary(n)
			};
			interior_mut(qp.as_mut(), layout).copy_from(&initial);
			let full = product(
				product(initial.as_ref(), a.as_ref()).as_ref(),
				adjoint(initial.as_ref()).as_ref(),
			);
			let ctx = format!(
				"GET36 n={n} from={from} to={to} layout={layout} Q={want_q} scale={scale:e} pass={pass}"
			);
			let mut result = None;
			if run.call(&ctx, || {
				result = Some(reorder_call(
					interior_mut(tp.as_mut(), layout),
					if want_q {
						Some(interior_mut(qp.as_mut(), layout))
					} else {
						None
					},
					from,
					to,
					Par::Seq,
					MemStack::new(&mut mem),
				));
				Ok::<_, ()>(())
			}) {
				let t = interior(tp.as_ref(), layout);
				let result = result.unwrap();
				let (pos, accepted) = match result {
					Ok(i) => (i, true),
					Err(e) => (e.position, false),
				};
				run.require(
					&ctx,
					"invalid returned block position",
					pos < n && block_start(t, pos) == pos,
				);
				if pos < n {
					let expected: Vec<_> = moving
						.iter()
						.map(|x| {
							let f = if pass == 0 { 1.0 } else { 0.5 };
							(x.0 * f, x.1 * f)
						})
						.collect();
					let size = block_len(t, pos);
					let found = block_values(t.submatrix(pos, pos, size, size));
					run.require(&ctx,&format!("wrong moving block pos={pos} accepted={accepted} actual={found:?} expected={expected:?}"),spectrum_matches_scaled(&found,&expected,norm_one(a.as_ref()),20.0));
				}
				run.require(&ctx, "invalid Schur form", schur_form(t));
				let expected: Vec<_> = original
					.iter()
					.map(|x| {
						let f = if pass == 0 { 1.0 } else { 0.5 };
						(x.0 * f, x.1 * f)
					})
					.collect();
				run.require(
					&ctx,
					"spectrum changed",
					spectrum_matches(&block_values(t), &expected, 20.0),
				);
				if accepted {
					run.require(
						&ctx,
						"wrong destination",
						pos == if <Scalar as ComplexField>::IS_REAL {
							reorder_destination(input.as_ref(), from, to)
						} else {
							to
						},
					);
				}
				if want_q {
					let q = interior(qp.as_ref(), layout);
					run.ratio(
						&ctx,
						"similarity",
						similarity_ratio(full.as_ref(), q, t, Real::EPSILON),
						20.0,
					);
					run.ratio(
						&ctx,
						"Q unitary",
						unitary_ratio(q, Real::EPSILON),
						20.0,
					);
				}
			}
			run.require(
				&ctx,
				"guard overwritten",
				guards_ok(tp.as_ref()) && guards_ok(qp.as_ref()),
			);
		}
	}
}
fn adjacent_case(
	run: &mut Run,
	t: MatRef<'_, Scalar>,
	n1: usize,
	n2: usize,
	scale: Real,
) {
	let mut t = scaled(t, scale);
	let before = t.clone();
	let mut q = Mat::<Scalar>::identity(t.nrows(), t.nrows());
	let ctx = format!("GET34 adjacent {n1}x{n2} scale={scale:e}");
	let mut status = 0;
	if run.call(&ctx, || {
		status = adjacent_call(t.as_mut(), Some(q.as_mut()), 1, n1, n2);
		Ok::<_, ()>(())
	}) {
		run.require(&ctx, "invalid Schur form", schur_form(t.as_ref()));
		run.require(
			&ctx,
			"spectrum changed",
			spectrum_matches(
				&block_values(t.as_ref()),
				&block_values(before.as_ref()),
				20.0,
			),
		);
		run.ratio(
			&ctx,
			"similarity",
			similarity_ratio(
				before.as_ref(),
				q.as_ref(),
				t.as_ref(),
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
		if status == 0 {
			let wanted =
				block_values(before.as_ref().submatrix(1 + n1, 1 + n1, n2, n2));
			run.require(
				&ctx,
				"blocks not swapped",
				spectrum_matches_scaled(
					&block_values(t.as_ref().submatrix(1, 1, n2, n2)),
					&wanted,
					norm_one(before.as_ref()),
					20.0,
				),
			);
		} else {
			// DLAEXC: a rejected swap leaves T and Q unchanged.
			let identity = Mat::<Scalar>::identity(q.nrows(), q.ncols());
			run.require(
				&ctx,
				"rejected swap changed T or Q",
				same_bits(t.as_ref(), before.as_ref())
					&& same_bits(q.as_ref(), identity.as_ref()),
			);
		}
	}
}
#[test]
fn lapack_schur_reorder_checkers() {
	assert_eq!(
		reorder_destination(swap_input(1, 1, false).as_ref(), 1, 2),
		2
	);
	if <Scalar as ComplexField>::IS_REAL {
		assert_eq!(
			reorder_destination(swap_input(1, 2, false).as_ref(), 1, 2),
			3
		);
		assert_eq!(
			reorder_destination(swap_input(2, 1, false).as_ref(), 1, 3),
			2
		);
		assert_eq!(
			reorder_destination(swap_input(2, 1, false).as_ref(), 3, 1),
			1
		);
	}
	crate::assert!(!spectrum_matches(
		&[(1.0, 0.0), (1.0, 0.0)],
		&[(1.0, 0.0), (2.0, 0.0)],
		20.0
	));
	crate::assert!(spectrum_matches(
		&[(2.0, 0.0), (1.0, 0.0)],
		&[(1.0, 0.0), (2.0, 0.0)],
		20.0
	));
	// GET36 is normwise: a small moving block inherits the full T scale.
	crate::assert!(spectrum_matches_scaled(
		&[(1.0 + 32.0 * Real::EPSILON, 0.0)],
		&[(1.0, 0.0)],
		10.0,
		20.0
	));
	crate::assert!(!spectrum_matches_scaled(
		&[(2.0, 0.0)],
		&[(1.0, 0.0)],
		10.0,
		20.0
	));
	let (t, from, to) = get36_fixture();
	crate::assert!(schur_form(t.as_ref()));
	crate::assert!(from < t.nrows() && to < t.nrows());
}
#[test]
#[ignore = "opt-in LAPACK numerical validation (#31)"]
fn lapack_schur_reorder_cases() {
	let mut run = Run::new("schur_reorder");
	let combos = if <Scalar as ComplexField>::IS_REAL {
		vec![(1, 1), (1, 2), (2, 1), (2, 2)]
	} else {
		vec![(1, 1)]
	};
	for &(n1, n2) in &combos {
		for close in [false, true] {
			let t = swap_input(n1, n2, close);
			for scale in [1.0, Real::MAX.sqrt(), Real::MIN_POSITIVE.sqrt()] {
				adjacent_case(&mut run, t.as_ref(), n1, n2, scale);
				for (from, to) in [(1, 1 + n1), (1 + n1, 1), (1, 1)] {
					for layout in 0..4 {
						reorder_case(
							&mut run,
							t.as_ref(),
							from,
							to,
							layout,
							scale,
						);
					}
				}
			}
		}
	}
	let (t, from, to) = get36_fixture();
	for layout in 0..4 {
		for scale in [1.0, Real::MAX.sqrt(), Real::MIN_POSITIVE.sqrt()] {
			reorder_case(&mut run, t.as_ref(), from, to, layout, scale);
		}
	}
	run.finish(combos.len() * 2 * 3 * (1 + 3 * 4 * 4) + 4 * 3 * 4);
}
