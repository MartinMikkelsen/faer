mod common;
use common::*;
use faer::linalg::svd::*;

fn check<T: ComplexField>(make: fn(f64, f64) -> T) {
	for n in SIZES {
		for kind in 0..3 {
			for upper in [false, true] {
				let original_d: Vec<T> = (0..n)
					.map(|i| match kind {
						0 => make(sample(i, 0), sample(0, i)),
						1 => zero(),
						_ => from_f64(2.0),
					})
					.collect();
				let e: Vec<T> = (0..n - 1)
					.map(|i| {
						if kind == 0 {
							make(sample(i, 1), sample(1, i))
						} else {
							zero()
						}
					})
					.collect();
				let original = Mat::from_fn(n, n, |i, j| {
					if i == j {
						original_d[i].copy()
					} else if upper && i + 1 == j {
						e[i].copy()
					} else if !upper && j + 1 == i {
						e[j].copy()
					} else {
						zero()
					}
				});
				let mut d = original_d;
				let (mut u, mut v) = (Mat::zeros(n, n), Mat::zeros(n, n));
				let mut params = faer::Spec::<SvdParams, T>::default();
				params.recursion_threshold = 8;
				let mut mem = MemBuffer::new(bidiagonal_svd_scratch::<T>(
					n,
					ParShape::Seq,
					params,
				));
				bidiagonal_svd(
					&mut d,
					&e,
					upper,
					u.as_mut(),
					v.as_mut(),
					Par::Seq,
					MemStack::new(&mut mem),
					params,
				)
				.unwrap();
				for i in 0..n {
					assert!(d[i].real() >= zero() && d[i].imag() == zero());
					if i > 0 {
						assert!(d[i - 1].real() >= d[i].real());
					}
				}
				unitary(u.as_ref());
				unitary(v.as_ref());
				let s = Mat::from_fn(n, n, |i, j| {
					if i == j { d[i].copy() } else { zero() }
				});
				close((&u * s * v.adjoint()).as_ref(), original.as_ref());
			}
		}
	}
}
#[test]
fn reconstruction_upper_lower_zero_and_repeated() {
	check::<f64>(|r, _| r);
	check::<f32>(|r, _| r as f32);
	check::<c64>(c64::new);
	check::<c32>(|r, i| c32::new(r as f32, i as f32));
}
#[test]
fn nonfinite_and_empty() {
	let mut d = [f64::NAN];
	let (mut u, mut v) = (Mat::zeros(1, 1), Mat::zeros(1, 1));
	let mut mem = MemBuffer::new(bidiagonal_svd_scratch::<f64>(
		1,
		ParShape::Seq,
		Default::default(),
	));
	assert_eq!(
		bidiagonal_svd(
			&mut d,
			&[],
			true,
			u.as_mut(),
			v.as_mut(),
			Par::Seq,
			MemStack::new(&mut mem),
			Default::default()
		),
		Err(SvdError::NoConvergence)
	);
	let (mut u, mut v) = (Mat::<f64>::zeros(0, 0), Mat::zeros(0, 0));
	bidiagonal_svd(
		&mut [],
		&[],
		false,
		u.as_mut(),
		v.as_mut(),
		Par::Seq,
		MemStack::new(&mut mem),
		Default::default(),
	)
	.unwrap();
}

/// Extreme finite magnitudes: subnormal and near-overflow entries must give
/// finite singular values and unitary factors, or never a non-finite `Ok`.
fn extreme<T: ComplexField>(make: fn(f64, f64) -> T) {
	for scale in [1e-310, 1e300] {
		for n in [1usize, 2, 3, 8] {
			for upper in [false, true] {
				let original_d: Vec<T> = (0..n)
					.map(|i| {
						make(scale * (1.0 + i as f64), scale * sample(i, 2))
					})
					.collect();
				let e: Vec<T> = (0..n.saturating_sub(1))
					.map(|i| make(scale * sample(i, 3), -scale * 0.5))
					.collect();
				let original = Mat::from_fn(n, n, |i, j| {
					if i == j {
						original_d[i].copy()
					} else if upper && i + 1 == j {
						e[i].copy()
					} else if !upper && j + 1 == i {
						e[j].copy()
					} else {
						zero()
					}
				});
				let mut d = original_d;
				let (mut u, mut v) = (Mat::zeros(n, n), Mat::zeros(n, n));
				let mut mem = MemBuffer::new(bidiagonal_svd_scratch::<T>(
					n,
					ParShape::Seq,
					Default::default(),
				));
				bidiagonal_svd(
					&mut d,
					&e,
					upper,
					u.as_mut(),
					v.as_mut(),
					Par::Seq,
					MemStack::new(&mut mem),
					Default::default(),
				)
				.unwrap();
				assert!(d.iter().all(|x| x.is_finite()));
				assert!(u.is_all_finite() && v.is_all_finite());
				unitary(u.as_ref());
				unitary(v.as_ref());
				let s = Mat::from_fn(n, n, |i, j| {
					if i == j { d[i].copy() } else { zero() }
				});
				// Subnormal reconstruction is limited by absolute, not
				// relative, precision; compare after rescaling by the input
				// magnitude. `1 / 1e-310` overflows, so rescale in two
				// finite steps.
				let (a, b) = if scale < 1.0 {
					(from_f64::<T::Real>(1e300), from_f64::<T::Real>(1e10))
				} else {
					(from_f64::<T::Real>(1e-300), one())
				};
				let rescale = |m: Mat<T>| {
					Mat::from_fn(n, n, |i, j| {
						m[(i, j)].mul_real(&a).mul_real(&b)
					})
				};
				let reconstructed = rescale(&u * s * v.adjoint());
				let expected = rescale(original);
				if scale < 1.0 {
					let err = (&reconstructed - &expected).norm_l2();
					assert!(err <= from_f64(1e-3), "error {err:?}");
				} else {
					close(reconstructed.as_ref(), expected.as_ref());
				}
			}
		}
	}
}
#[test]
fn tiny_and_large_magnitudes() {
	extreme::<f64>(|r, _| r);
	extreme::<c64>(c64::new);
}
#[test]
fn reviewer_subnormal_one_by_one() {
	for (value, upper) in [
		(c64::new(1e-310, 0.0), false),
		(c64::new(1e-310, 0.0), true),
		(c64::new(1e-310, 1e-310), false),
		(c64::new(1e-310, 1e-310), true),
	] {
		let (mut u, mut v) = (Mat::zeros(1, 1), Mat::zeros(1, 1));
		let mut mem = MemBuffer::new(bidiagonal_svd_scratch::<c64>(
			1,
			ParShape::Seq,
			Default::default(),
		));
		let mut d = [value];
		bidiagonal_svd(
			&mut d,
			&[],
			upper,
			u.as_mut(),
			v.as_mut(),
			Par::Seq,
			MemStack::new(&mut mem),
			Default::default(),
		)
		.unwrap();
		assert_eq!(d[0].im, 0.0);
		assert!(d[0].re.is_finite());
		// A value near 1e-310 carries only about 13 significant digits.
		let tol = 1e-10 * value.norm();
		assert!((d[0].re - value.norm()).abs() <= tol, "{d:?} {value:?}");
		assert!((u[(0, 0)].norm() - 1.0).abs() <= 1e-15);
		assert!((v[(0, 0)].norm() - 1.0).abs() <= 1e-15);
		assert!(((u[(0, 0)] * d[0] * v[(0, 0)].conj()) - value).norm() <= tol);
		let mut d = [value.re];
		let (mut u, mut v) = (Mat::zeros(1, 1), Mat::zeros(1, 1));
		bidiagonal_svd(
			&mut d,
			&[],
			upper,
			u.as_mut(),
			v.as_mut(),
			Par::Seq,
			MemStack::new(&mut mem),
			Default::default(),
		)
		.unwrap();
		assert_eq!(d[0], value.re);
		assert_eq!(u[(0, 0)] * v[(0, 0)], 1.0);
	}
}

/// 1x1 magnitude boundaries: the singular value is `hypot(re, im)` and must be
/// returned, not flushed to zero or reported as nonconvergence.
fn one_by_one_boundary<T: ComplexField>(value: T, upper: bool) {
	let (mut u, mut v) = (Mat::zeros(1, 1), Mat::zeros(1, 1));
	let mut mem = MemBuffer::new(bidiagonal_svd_scratch::<T>(
		1,
		ParShape::Seq,
		Default::default(),
	));
	let mut d = [value.copy()];
	bidiagonal_svd(
		&mut d,
		&[],
		upper,
		u.as_mut(),
		v.as_mut(),
		Par::Seq,
		MemStack::new(&mut mem),
		Default::default(),
	)
	.unwrap();
	let (re, im) = (value.real().abs(), value.imag().abs());
	let (big, small) = if re > im { (re, im) } else { (im, re) };
	let ratio = &small / &big;
	let expected = big * (one::<T::Real>() + &ratio * &ratio).sqrt();
	let tolerance =
		from_f64::<T::Real>(4.0) * min_positive::<T::Real>() * eps::<T::Real>()
			+ from_f64::<T::Real>(32.0) * eps::<T::Real>() * &expected;
	assert!(d[0].imag() == zero());
	assert!(
		(d[0].real() - &expected).abs() <= tolerance,
		"sigma {:?}, expected {expected:?}",
		d[0].real()
	);
	assert!(
		(u[(0, 0)].abs() - one::<T::Real>()).abs()
			<= from_f64::<T::Real>(4.0) * eps::<T::Real>()
	);
	assert!(
		(v[(0, 0)].abs() - one::<T::Real>()).abs()
			<= from_f64::<T::Real>(4.0) * eps::<T::Real>()
	);
	assert!(d[0].is_finite() && u.is_all_finite() && v.is_all_finite());
	let reconstructed = &u[(0, 0)] * &d[0] * v[(0, 0)].conj();
	assert!(
		(reconstructed - &value).abs() <= from_f64::<T::Real>(2.0) * &tolerance
	);
}
/// The maintainer's eight cases from the second review of PR #27: before the
/// fix, each returned `Ok` with an infinite singular value and NaN in a factor.
#[test]
fn subnormal_one_by_one_in_every_scalar_type() {
	for upper in [false, true] {
		one_by_one_boundary(1e-40f32, upper);
		one_by_one_boundary(1e-310f64, upper);
		one_by_one_boundary(c32::new(1e-40, 1e-40), upper);
		one_by_one_boundary(c64::new(1e-310, 1e-310), upper);
	}
}
#[test]
fn reviewer_magnitude_boundaries() {
	for upper in [false, true] {
		one_by_one_boundary(c64::new(1e-320, 1e-320), upper);
		one_by_one_boundary(c64::new(f64::MAX / 2.0, f64::MAX / 2.0), upper);
		one_by_one_boundary(c64::new(f64::MAX, 0.0), upper);
		one_by_one_boundary(c64::new(5e-324, 0.0), upper);
		one_by_one_boundary(c32::new(1e-44, 1e-44), upper);
		one_by_one_boundary(c32::new(f32::MAX / 2.0, f32::MAX / 2.0), upper);
		one_by_one_boundary(f64::MAX, upper);
		one_by_one_boundary(-1e-320f64, upper);
		one_by_one_boundary(f32::MAX, upper);
		one_by_one_boundary(-1e-44f32, upper);
	}
}
