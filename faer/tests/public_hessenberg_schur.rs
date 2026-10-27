mod common;
use common::*;
use faer::linalg::evd::*;

fn real<T: RealField>() {
	for n in SIZES {
		for structured in [false, true] {
			let original = Mat::<T>::from_fn(n, n, |i, j| {
				if i > j + 1 {
					zero()
				} else if structured {
					from_f64(if i == j {
						2.0
					} else if j == i + 1 {
						1.0
					} else {
						0.0
					})
				} else {
					from_f64(sample(i, j))
				}
			});
			let mut t = original.clone();
			let mut z = Mat::identity(n, n);
			let (mut wr, mut wi) = (Col::zeros(n), Col::zeros(n));
			let mut mem = MemBuffer::new(hessenberg_schur_scratch::<T>(
				n,
				true,
				ParShape::Seq,
				Default::default(),
			));
			hessenberg_schur_real(
				t.as_mut(),
				Some(z.as_mut()),
				wr.as_mut(),
				wi.as_mut(),
				Par::Seq,
				MemStack::new(&mut mem),
				Default::default(),
			)
			.unwrap();
			close((&z * &t * z.adjoint()).as_ref(), original.as_ref());
			unitary(z.as_ref());
			schur_form(t.as_ref(), true);
			let mut j = 0;
			while j < n {
				assert!((&wr[j] - &t[(j, j)]).abs() <= tolerance::<T>(n));
				if j + 1 < n && t[(j + 1, j)] != zero() {
					let im =
						t[(j, j + 1)].abs().sqrt() * t[(j + 1, j)].abs().sqrt();
					assert!(wi[j] > zero() && wi[j + 1] < zero());
					assert!((&wi[j] - im).abs() <= tolerance::<T>(n));
					assert_eq!(wi[j], -&wi[j + 1]);
					j += 2;
				} else {
					assert_eq!(wi[j], zero());
					j += 1;
				}
			}
		}
	}
}
fn complex<T: ComplexField>(make: fn(f64, f64) -> T) {
	for n in SIZES {
		for structured in [false, true] {
			let original = Mat::<T>::from_fn(n, n, |i, j| {
				if i > j + 1 {
					zero()
				} else if structured {
					make(
						if i == j {
							2.0
						} else if j == i + 1 {
							1.0
						} else {
							0.0
						},
						0.0,
					)
				} else {
					make(sample(i, j), sample(j, i + 1))
				}
			});
			let mut t = original.clone();
			let mut z = Mat::identity(n, n);
			let mut w = Col::zeros(n);
			let mut mem = MemBuffer::new(hessenberg_schur_scratch::<T>(
				n,
				true,
				ParShape::Seq,
				Default::default(),
			));
			hessenberg_schur_cplx(
				t.as_mut(),
				Some(z.as_mut()),
				w.as_mut(),
				Par::Seq,
				MemStack::new(&mut mem),
				Default::default(),
			)
			.unwrap();
			close((&z * &t * z.adjoint()).as_ref(), original.as_ref());
			unitary(z.as_ref());
			schur_form(t.as_ref(), false);
			for j in 0..n {
				assert!((&w[j] - &t[(j, j)]).abs() <= tolerance::<T>(n));
			}
		}
	}
}
#[test]
fn reconstruction_and_spectrum() {
	real::<f64>();
	real::<f32>();
	complex::<c64>(c64::new);
	complex::<c32>(|r, i| c32::new(r as f32, i as f32));
}
#[test]
fn accumulator_empty_and_nonfinite() {
	let mut h = faer::mat![[1.0f64, 2.0], [-3.0, 4.0]];
	let original = h.clone();
	let mut z = faer::mat![[0.0, 1.0], [1.0, 0.0]];
	let initial = z.clone();
	let (mut wr, mut wi) = (Col::zeros(2), Col::zeros(2));
	let mut mem = MemBuffer::new(hessenberg_schur_scratch::<f64>(
		2,
		true,
		ParShape::Seq,
		Default::default(),
	));
	hessenberg_schur_real(
		h.as_mut(),
		Some(z.as_mut()),
		wr.as_mut(),
		wi.as_mut(),
		Par::Seq,
		MemStack::new(&mut mem),
		Default::default(),
	)
	.unwrap();
	close(
		(&z * &h * z.adjoint()).as_ref(),
		(&initial * original * initial.adjoint()).as_ref(),
	);
	h[(0, 0)] = f64::NAN;
	assert_eq!(
		hessenberg_schur_real(
			h.as_mut(),
			None,
			wr.as_mut(),
			wi.as_mut(),
			Par::Seq,
			MemStack::new(&mut mem),
			Default::default()
		),
		Err(EvdError::NoConvergence)
	);
	let mut empty = Mat::<c64>::zeros(0, 0);
	let mut w = Col::zeros(0);
	hessenberg_schur_cplx(
		empty.as_mut(),
		None,
		w.as_mut(),
		Par::Seq,
		MemStack::new(&mut mem),
		Default::default(),
	)
	.unwrap();
}
