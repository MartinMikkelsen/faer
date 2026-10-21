mod common;
use common::*;
use faer::linalg::evd::*;

fn real<T: RealField>() {
	for n in SIZES {
		for structured in [false, true] {
			let mut t = Mat::<T>::from_fn(n, n, |i, j| {
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
			let original = t.clone();
			let mut q = Mat::identity(n, n);
			let (mut wr, mut wi) = (Col::zeros(n), Col::zeros(n));
			let mut qr_mem = MemBuffer::new(hessenberg_schur_scratch::<T>(
				n,
				true,
				Par::Seq,
				Default::default(),
			));
			hessenberg_schur_real(
				t.as_mut(),
				Some(q.as_mut()),
				wr.as_mut(),
				wi.as_mut(),
				Par::Seq,
				MemStack::new(&mut qr_mem),
				Default::default(),
			)
			.unwrap();
			for back in [false, true] {
				let a = if back { original.as_ref() } else { t.as_ref() };
				let mut x = Mat::zeros(n, n);
				let mut mem =
					MemBuffer::new(eigenvectors_from_schur_scratch::<T>(n));
				eigenvectors_from_schur_real(
					t.as_ref(),
					if back { Some(q.as_ref()) } else { None },
					x.as_mut(),
					Par::Seq,
					MemStack::new(&mut mem),
					Default::default(),
				)
				.unwrap();
				let ax = a * &x;
				let mut j = 0;
				while j < n {
					let paired = wi[j] > zero();
					let norm = if paired {
						x.col(j).norm_l2().hypot(&x.col(j + 1).norm_l2())
					} else {
						x.col(j).norm_l2()
					};
					assert!((norm - one::<T>()).abs() <= tolerance::<T>(n));
					for i in 0..n {
						let re = &wr[j] * &x[(i, j)]
							- if paired {
								&wi[j] * &x[(i, j + 1)]
							} else {
								zero()
							};
						assert!(
							(&ax[(i, j)] - re).abs()
								<= tolerance::<T>(n)
									* a.norm_l2().fmax(one::<T>())
						);
						if paired {
							let im =
								&wr[j] * &x[(i, j + 1)] + &wi[j] * &x[(i, j)];
							assert!(
								(&ax[(i, j + 1)] - im).abs()
									<= tolerance::<T>(n)
										* a.norm_l2().fmax(one::<T>())
							);
						}
					}
					j += if paired { 2 } else { 1 };
				}
			}
		}
	}
}
fn complex<T: ComplexField>(make: fn(f64, f64) -> T) {
	for n in SIZES {
		for structured in [false, true] {
			let mut t = Mat::<T>::from_fn(n, n, |i, j| {
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
			let original = t.clone();
			let mut q = Mat::identity(n, n);
			let mut w = Col::zeros(n);
			let mut qr_mem = MemBuffer::new(hessenberg_schur_scratch::<T>(
				n,
				true,
				Par::Seq,
				Default::default(),
			));
			hessenberg_schur_cplx(
				t.as_mut(),
				Some(q.as_mut()),
				w.as_mut(),
				Par::Seq,
				MemStack::new(&mut qr_mem),
				Default::default(),
			)
			.unwrap();
			for back in [false, true] {
				let a = if back { original.as_ref() } else { t.as_ref() };
				let mut x = Mat::zeros(n, n);
				let mut mem =
					MemBuffer::new(eigenvectors_from_schur_scratch::<T>(n));
				eigenvectors_from_schur_cplx(
					t.as_ref(),
					if back { Some(q.as_ref()) } else { None },
					x.as_mut(),
					Par::Seq,
					MemStack::new(&mut mem),
					Default::default(),
				)
				.unwrap();
				let mut xs = x.clone();
				for j in 0..n {
					assert!(
						(x.col(j).norm_l2() - one::<T::Real>()).abs()
							<= tolerance::<T>(n)
					);
					for i in 0..n {
						xs[(i, j)] *= &w[j];
					}
				}
				close((a * x).as_ref(), xs.as_ref());
			}
		}
	}
}
#[test]
fn normalized_right_residuals_in_diagonal_order() {
	real::<f64>();
	real::<f32>();
	complex::<c64>(c64::new);
	complex::<c32>(|r, i| c32::new(r as f32, i as f32));
}
#[test]
fn nonfinite_is_an_error() {
	let t = faer::mat![[f64::INFINITY]];
	let mut x = Mat::zeros(1, 1);
	let mut mem = MemBuffer::new(eigenvectors_from_schur_scratch::<f64>(1));
	assert_eq!(
		eigenvectors_from_schur_real(
			t.as_ref(),
			None,
			x.as_mut(),
			Par::Seq,
			MemStack::new(&mut mem),
			Default::default()
		),
		Err(EvdError::NoConvergence)
	);
}
#[test]
fn tridiagonal_dedicated_scratch() {
	for n in SIZES {
		let d = Col::from_fn(n, |i| c64::new(i as f64, 0.0));
		let e = Col::from_fn(n - 1, |_| c64::new(0.25, -0.5));
		let original = Mat::from_fn(n, n, |i, j| {
			if i == j {
				d[i]
			} else if i == j + 1 {
				e[j]
			} else if j == i + 1 {
				e[i].conj()
			} else {
				zero()
			}
		});
		for compute in [ComputeEigenvectors::No, ComputeEigenvectors::Yes] {
			let mut s = faer::diag::Diag::zeros(n);
			let mut u = Mat::zeros(n, n);
			let mut params = faer::Spec::<SelfAdjointEvdParams, c64>::default();
			params.recursion_threshold = 8;
			let mut mem =
				MemBuffer::new(tridiagonal_self_adjoint_evd_scratch::<c64>(
					n,
					compute,
					Par::Seq,
					params,
				));
			tridiagonal_self_adjoint_evd(
				d.as_ref().as_diagonal(),
				e.as_ref().as_diagonal(),
				s.as_mut(),
				if compute == ComputeEigenvectors::Yes {
					Some(u.as_mut())
				} else {
					None
				},
				Par::Seq,
				MemStack::new(&mut mem),
				params,
			)
			.unwrap();
			for i in 1..n {
				assert!(s[i].re >= s[i - 1].re);
			}
			if compute == ComputeEigenvectors::Yes {
				unitary(u.as_ref());
				close(
					(&u * s.as_ref() * u.adjoint()).as_ref(),
					original.as_ref(),
				);
			}
		}
	}
}
