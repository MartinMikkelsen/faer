// Copyright (c) tensor4all contributors
// SPDX-License-Identifier: MIT
use crate::internal_prelude::*;
macro_rules! suite {
	($name:ident,$ty:ty,$real:ty,$make:expr) => {
		#[allow(dead_code, unused_imports)]
		mod $name {
			use super::*;
			type Scalar = $ty;
			type Real = $real;
			fn scalar(re: Real, im: Real) -> Scalar {
				($make)(re, im)
			}
			include!("lapack_support.rs");
			include!("lapack_lin_support.rs");
			include!("lapack_triangular_cases.rs");
		}
	};
}
suite!(f64, f64, f64, |re, _im| re);
suite!(f32, f32, f32, |re, _im| re);
suite!(c64, crate::c64, f64, crate::c64::new);
suite!(c32, crate::c32, f32, crate::c32::new);
