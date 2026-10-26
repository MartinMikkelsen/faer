// Copyright (c) tensor4all contributors
// SPDX-License-Identifier: MIT
use super::*;
use crate::linalg::householder;
use dyn_stack::MemBuffer;
macro_rules! adapters {
	(real) => {
		include!("real_adapters.rs");
	};
	(complex) => {
		include!("complex_adapters.rs");
	};
}
macro_rules! suite {
	($name:ident, $ty:ty, $real:ty, $make:expr, $kind:ident) => {
		#[allow(dead_code, unused_imports)]
		mod $name {
			use super::*;
			type Scalar = $ty;
			type Real = $real;
			fn scalar(re: Real, im: Real) -> Scalar {
				($make)(re, im)
			}
			include!("../../tests/lapack_support.rs");
			adapters!($kind);
			include!("lapack_hessenberg.rs");
			include!("lapack_schur.rs");
			include!("lapack_nonsymmetric_evd.rs");
			include!("lapack_fixtures.rs");
			include!("lapack_schur_reorder.rs");
			include!("lapack_balance.rs");
			include!("lapack_tridiagonal_evd.rs");
			include!("lapack_self_adjoint_evd.rs");
		}
	};
}
suite!(f64, f64, f64, |re, _im| re, real);
suite!(f32, f32, f32, |re, _im| re, real);
suite!(c64, crate::c64, f64, crate::c64::new, complex);
suite!(c32, crate::c32, f32, crate::c32::new, complex);
