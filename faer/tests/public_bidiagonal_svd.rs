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
					Par::Seq,
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
		Par::Seq,
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
