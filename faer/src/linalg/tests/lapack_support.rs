// Numerical oracles adapted from LAPACK 3.12.0 TESTING/{EIG,LIN}, distributed
// in OpenBLAS 791b02a1a00c996b66a894f6df93421c53b62d80. See each suite's sources.
// Copyright (c) 1992-2023 The University of Tennessee and The University
// of Tennessee Research Foundation. All rights reserved.
// Copyright (c) 2000-2023 The University of California Berkeley.
// Copyright (c) 2006-2023 The University of Colorado Denver.
// SPDX-License-Identifier: BSD-3-Clause
// See COPYING.LAPACK.BSD. Scalar/Real are private native-precision aliases.

fn magnitude(x: Scalar) -> Real {
	real(&x).hypot(imag(&x))
}
fn z(x: Real) -> Scalar {
	scalar(x, 0.0)
}
fn max_entry(a: MatRef<'_, Scalar>) -> Real {
	let mut scale = 0.0 as Real;
	for j in 0..a.ncols() {
		for i in 0..a.nrows() {
			let x = magnitude(a[(i, j)]);
			if !x.is_finite() {
				return Real::INFINITY;
			}
			scale = scale.max(x);
		}
	}
	scale
}
fn norm_one(a: MatRef<'_, Scalar>) -> Real {
	let mut norm = 0.0 as Real;
	for j in 0..a.ncols() {
		let mut sum = 0.0;
		for i in 0..a.nrows() {
			sum += magnitude(a[(i, j)]);
		}
		if !sum.is_finite() {
			return Real::INFINITY;
		}
		norm = norm.max(sum);
	}
	norm
}
fn scaled(a: MatRef<'_, Scalar>, x: Real) -> Mat<Scalar> {
	Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)] * x)
}
fn adjoint(a: MatRef<'_, Scalar>) -> Mat<Scalar> {
	Mat::from_fn(a.ncols(), a.nrows(), |i, j| conj(&a[(j, i)]))
}
fn product(a: MatRef<'_, Scalar>, b: MatRef<'_, Scalar>) -> Mat<Scalar> {
	assert_eq!(a.ncols(), b.nrows());
	Mat::from_fn(a.nrows(), b.ncols(), |i, j| {
		let mut v = z(0.0);
		for k in 0..a.ncols() {
			v += a[(i, k)] * b[(k, j)];
		}
		v
	})
}
fn difference_ratio(
	a: MatRef<'_, Scalar>,
	b: MatRef<'_, Scalar>,
	eps: Real,
) -> Real {
	assert_eq!(a.shape(), b.shape());
	let scale = max_entry(b);
	if !scale.is_finite() || !max_entry(a).is_finite() {
		return Real::INFINITY;
	}
	if scale == 0.0 {
		return if max_entry(a) == 0.0 {
			0.0
		} else {
			Real::INFINITY
		};
	}
	let an = Mat::from_fn(a.nrows(), a.ncols(), |i, j| {
		a[(i, j)] / scale - b[(i, j)] / scale
	});
	let bn = Mat::from_fn(b.nrows(), b.ncols(), |i, j| b[(i, j)] / scale);
	norm_one(an.as_ref())
		/ norm_one(bn.as_ref())
		/ (b.nrows().max(b.ncols()).max(1) as Real * eps)
}
fn unitary_ratio(q: MatRef<'_, Scalar>, eps: Real) -> Real {
	let gram = Mat::from_fn(q.ncols(), q.ncols(), |i, j| {
		let mut v = z(if i == j { 1.0 } else { 0.0 });
		for k in 0..q.nrows() {
			v -= conj(&q[(k, i)]) * q[(k, j)];
		}
		v
	});
	norm_one(gram.as_ref()) / (q.nrows().max(1) as Real * eps)
}
fn similarity_ratio(
	a: MatRef<'_, Scalar>,
	q: MatRef<'_, Scalar>,
	t: MatRef<'_, Scalar>,
	eps: Real,
) -> Real {
	let scale = max_entry(a);
	if !scale.is_finite() || !max_entry(t).is_finite() {
		return Real::INFINITY;
	}
	let scale = if scale == 0.0 { 1.0 } else { scale };
	let tn = Mat::from_fn(t.nrows(), t.ncols(), |i, j| t[(i, j)] / scale);
	let rec = product(product(q, tn.as_ref()).as_ref(), adjoint(q).as_ref());
	let an = Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)] / scale);
	if max_entry(a) == 0.0 {
		return difference_ratio(rec.as_ref(), an.as_ref(), eps);
	}
	let err =
		Mat::from_fn(a.nrows(), a.ncols(), |i, j| an[(i, j)] - rec[(i, j)]);
	let n = a.nrows().max(1) as Real;
	let anorm = norm_one(an.as_ref());
	let floor = (Real::MIN_POSITIVE * n / eps) / scale;
	{
		let error = norm_one(err.as_ref());
		if !error.is_finite() {
			Real::INFINITY
		} else {
			error.min(anorm) / floor.max(anorm * eps) / n
		}
	}
}
fn phase(i: usize) -> Scalar {
	if <Scalar as ComplexField>::IS_REAL {
		z(if i % 2 == 0 { 1.0 } else { -1.0 })
	} else {
		match i % 4 {
			0 => scalar(0.0, 1.0),
			1 => scalar(0.6, 0.8),
			2 => scalar(-0.8, 0.6),
			_ => z(-1.0),
		}
	}
}
fn unitary(n: usize) -> Mat<Scalar> {
	let mut q = Mat::<Scalar>::identity(n, n);
	let phase = if <Scalar as ComplexField>::IS_REAL {
		z(1.0)
	} else {
		scalar(0.0, 1.0)
	};
	for i in 1..n {
		for j in 0..n {
			let (x, y) = (q[(i - 1, j)], q[(i, j)]);
			q[(i - 1, j)] = 0.6 * x - conj(&(0.8 * phase)) * y;
			q[(i, j)] = (0.8 * phase) * x + 0.6 * y;
		}
	}
	q
}
fn sample(i: usize, j: usize) -> Scalar {
	// Fixed stateless seed; an independent replacement for LATMR's random sequence.
	let draw = |k: u64| {
		let mut x = k.wrapping_add(0x4c415041434b);
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
		(x % 1009) as Real / 504.0 - 1.0
	};
	scalar(
		draw((31 * i + 1009 * j) as u64),
		if <Scalar as ComplexField>::IS_REAL {
			0.0
		} else {
			draw((29 * j + 997 * i + 1) as u64)
		},
	)
}
fn spectrum(n: usize, kind: usize) -> Vec<Real> {
	let scale = match kind {
		6 => Real::MAX.sqrt(),
		7 => Real::MIN_POSITIVE.sqrt(),
		_ => 1.0,
	};
	(0..n)
		.map(|i| {
			let f = i as Real / n.saturating_sub(1).max(1) as Real;
			scale
				* match kind {
					0 => 0.0,
					1 => 1.0,
					3 => Real::EPSILON.powf(f),
					4 => 1.0 + 8.0 * Real::EPSILON * i as Real,
					5 => {
						if i < n / 2 {
							-1.0
						} else {
							1.0
						}
					},
					_ => 1.0 - 2.0 * f,
				}
		})
		.collect()
}
fn hermitian(n: usize, kind: usize) -> (Mat<Scalar>, Vec<Real>) {
	let s = spectrum(n, kind);
	let q = unitary(n);
	let d = Mat::from_fn(n, n, |i, j| if i == j { z(s[i]) } else { z(0.0) });
	let a = product(
		product(q.as_ref(), d.as_ref()).as_ref(),
		adjoint(q.as_ref()).as_ref(),
	);
	(a, s)
}
const GUARD_RE: Real = -12345.0;
fn guard() -> Scalar {
	scalar(GUARD_RE, 6789.0)
}
fn interior(a: MatRef<'_, Scalar>, layout: usize) -> MatRef<'_, Scalar> {
	let (m, n) = a.shape();
	let v = a.submatrix(1, 1, m - 2, n - 2);
	let v = if layout % 2 == 1 { v.transpose() } else { v };
	if layout >= 2 {
		v.reverse_rows_and_cols()
	} else {
		v
	}
}
fn interior_mut(a: MatMut<'_, Scalar>, layout: usize) -> MatMut<'_, Scalar> {
	let (m, n) = a.shape();
	let v = a.submatrix_mut(1, 1, m - 2, n - 2);
	let v = if layout % 2 == 1 {
		v.transpose_mut()
	} else {
		v
	};
	if layout >= 2 {
		v.reverse_rows_and_cols_mut()
	} else {
		v
	}
}
fn guarded(m: usize, n: usize, layout: usize) -> Mat<Scalar> {
	if layout % 2 == 1 {
		Mat::full(n + 2, m + 2, guard())
	} else {
		Mat::full(m + 2, n + 2, guard())
	}
}
fn guards_ok(a: MatRef<'_, Scalar>) -> bool {
	(0..a.ncols()).all(|j| {
		(0..a.nrows()).all(|i| {
			(i > 0 && i + 1 < a.nrows() && j > 0 && j + 1 < a.ncols())
				|| a[(i, j)] == guard()
		})
	})
}
fn same_bits(a: MatRef<'_, Scalar>, b: MatRef<'_, Scalar>) -> bool {
	a.shape() == b.shape()
		&& (0..a.ncols()).all(|j| {
			(0..a.nrows()).all(|i| {
				real(&a[(i, j)]).to_bits() == real(&b[(i, j)]).to_bits()
					&& imag(&a[(i, j)]).to_bits() == imag(&b[(i, j)]).to_bits()
			})
		})
}
struct Run {
	name: &'static str,
	calls: usize,
	aux_calls: usize,
	errors: usize,
	failures: Vec<String>,
}
impl Run {
	fn new(name: &'static str) -> Self {
		Self {
			name,
			calls: 0,
			aux_calls: 0,
			errors: 0,
			failures: Vec::new(),
		}
	}
	fn call<E: core::fmt::Debug>(
		&mut self,
		ctx: &str,
		call: impl FnOnce() -> Result<(), E>,
	) -> bool {
		self.calls += 1;
		let result =
			std::panic::catch_unwind(std::panic::AssertUnwindSafe(call));
		let error = match result {
			Ok(Ok(())) => return true,
			Ok(Err(e)) => format!("{e:?}"),
			Err(e) => format!(
				"panic: {}",
				e.downcast_ref::<&str>()
					.copied()
					.or_else(|| e.downcast_ref::<String>().map(String::as_str))
					.unwrap_or("non-string panic")
			),
		};
		self.errors += 1;
		self.failures.push(format!("{ctx}: {error}"));
		false
	}
	// Dependent operations are attempted only after a usable factor exists.
	fn aux<E: core::fmt::Debug>(
		&mut self,
		ctx: &str,
		call: impl FnOnce() -> Result<(), E>,
	) -> bool {
		let ok = self.call(ctx, call);
		self.calls -= 1;
		self.aux_calls += 1;
		ok
	}
	fn ratio(&mut self, ctx: &str, name: &str, value: Real, threshold: Real) {
		if !value.is_finite() || value >= threshold {
			self.failures.push(format!(
				"{ctx}: {name} ratio={value:e}, threshold={threshold}"
			));
		}
	}
	fn require(&mut self, ctx: &str, name: &str, ok: bool) {
		if !ok {
			self.failures.push(format!("{ctx}: {name}"));
		}
	}
	fn finish(self, planned: usize) {
		eprintln!(
			"LAPACK_CASES family={} dtype={} planned={} attempted={} call_failures={} diagnostics={} auxiliary_attempted={}",
			self.name,
			core::any::type_name::<Scalar>(),
			planned,
			self.calls,
			self.errors,
			self.failures.len(),
			self.aux_calls
		);
		if self.calls != planned {
			eprintln!("{}", self.failures.join("\n"));
		}
		assert_eq!(self.calls, planned, "case execution incomplete");
		crate::assert!(
			self.failures.is_empty(),
			"{}",
			self.failures.join("\n")
		);
	}
}
#[test]
fn lapack_common_fixtures_and_guards() {
	let q = unitary(2);
	crate::assert!(unitary_ratio(q.as_ref(), Real::EPSILON) < 20.0);
	assert_eq!(q[(0, 0)], z(0.6));
	let off = if <Scalar as ComplexField>::IS_REAL {
		z(-0.8)
	} else {
		scalar(0.0, 0.8)
	};
	assert_eq!(q[(0, 1)], off);
	assert_eq!(spectrum(2, 6)[0], Real::MAX.sqrt());
	assert_eq!(spectrum(2, 7)[0], Real::MIN_POSITIVE.sqrt());
	let a = Mat::from_fn(2, 3, |i, j| {
		scalar((i + 3 * j) as Real, (2 * i + j) as Real)
	});
	for layout in 0..4 {
		let mut p = guarded(2, 3, layout);
		interior_mut(p.as_mut(), layout).copy_from(&a);
		crate::assert!(same_bits(interior(p.as_ref(), layout), a.as_ref()));
		crate::assert!(guards_ok(p.as_ref()));
		p[(0, 1)] = z(0.0);
		crate::assert!(!guards_ok(p.as_ref()));
	}
}

#[test]
fn lapack_common_checkers_reject_corruption() {
	let a = Mat::<Scalar>::identity(2, 2);
	assert_eq!(difference_ratio(a.as_ref(), a.as_ref(), Real::EPSILON), 0.0);
	let mut wrong = a.clone();
	wrong[(0, 1)] = scalar(1.0, 1.0);
	crate::assert!(
		difference_ratio(wrong.as_ref(), a.as_ref(), Real::EPSILON) > 60.0
	);
	crate::assert!(unitary_ratio(wrong.as_ref(), Real::EPSILON) > 60.0);
	wrong[(0, 0)] = scalar(Real::NAN, Real::NAN);
	crate::assert!(
		!difference_ratio(wrong.as_ref(), a.as_ref(), Real::EPSILON)
			.is_finite()
	);
	crate::assert!(!unitary_ratio(wrong.as_ref(), Real::EPSILON).is_finite());
}

#[test]
fn lapack_common_continues_after_errors_and_panics() {
	let mut run = Run::new("calibration");
	let mut reached = false;
	crate::assert!(!run.call("error", || Err::<(), _>("expected")));
	crate::assert!(
		!run.call::<()>("panic", || panic!("expected calibration panic"))
	);
	crate::assert!(run.call("next", || {
		reached = true;
		Ok::<_, ()>(())
	}));
	crate::assert!(run.aux("dependent", || Ok::<_, ()>(())));
	crate::assert!(reached);
	assert_eq!(
		(run.calls, run.aux_calls, run.errors, run.failures.len()),
		(3, 1, 2, 2)
	);
}
