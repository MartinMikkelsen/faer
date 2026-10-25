//! The parallel rule: a region runs in parallel when its serial time is at least `PAYS` times
//! the dispatch cost of every section it opens (design record: `threadteam`
//! `docs/design/thresholds.md`).
//!
//! The dispatch cost belongs to the token ([`ParShape::dispatch_cost`]); the serial time of a
//! kernel family is its own throughput constant, below. With the `threadteam` feature the rule
//! and the costs are threadteam's; without it the Rayon cost and the rule are the same numbers
//! kept here, so a build without the feature decides identically.

use crate::ParShape;
use core::num::NonZeroUsize;
use core::time::Duration;

#[cfg(feature = "threadteam")]
pub use threadteam::{PAYS, pays_at, width_for_by};

/// The multiple of the stack (mirrors `threadteam::PAYS`).
#[cfg(not(feature = "threadteam"))]
pub const PAYS: f64 = 3.0;

/// `estimate >= PAYS * sections * dispatch_cost` (mirrors `threadteam::pays_at`).
#[cfg(not(feature = "threadteam"))]
pub fn pays_at(estimate: Duration, sections: u32, dispatch_cost: Duration) -> bool {
	if dispatch_cost == Duration::MAX {
		return false;
	}
	if sections == 0 {
		return true;
	}
	estimate.as_secs_f64() >= PAYS * f64::from(sections) * dispatch_cost.as_secs_f64()
}

/// Cost of one section started from outside the pool (`threadteam::DispatchKind::Outside`) at
/// width `w`: the Rayon route of faer enters the global or installed pool from the calling
/// thread. Apple M5 Max (measured, microseconds at widths 2, 4, 6: 14, 18, 19), the conservative
/// 40 us elsewhere; grown in proportion to the width above 6 (mirrors threadteam's model).
#[cfg(all(feature = "rayon", not(feature = "threadteam")))]
pub(crate) fn outside_cost(w: NonZeroUsize) -> Duration {
	let w = w.get() as u64;
	let ns: u64 = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
		match w {
			0..=2 => 14_000,
			3 => 16_000,
			4 => 18_000,
			5 => 18_500,
			6 => 19_000,
			_ => 19_000 * w / 6,
		}
	} else if w <= 6 {
		40_000
	} else {
		40_000 * w / 6
	};
	Duration::from_nanos(ns)
}

/// Serial throughput of the GEMM kernel, nanoseconds per multiply-add of the dtype (a complex
/// one for `c32`/`c64`). Measured on an Apple M5 Max with the `seq` mode of the threshold sweeps
/// (`threadteam/docs/design/thresholds-data`), L2 resident shapes.
pub mod ns_per_mac {
	/// `f64`: 0.037, range 0.034 to 0.040.
	pub const F64: f64 = 0.037;
	/// `c64`: 0.165, range 0.162 to 0.172.
	pub const C64: f64 = 0.165;
	/// `f32`: not measured, scaled from `f64` by the SIMD width (two lanes per register).
	pub const F32: f64 = F64 / 2.0;
	/// `c32`: not measured, scaled from `c64` by the SIMD width.
	pub const C32: f64 = C64 / 2.0;
}

/// Serial time of one row swap of one column in the LU swap stage, nanoseconds: random pivot
/// rows, columns of 64 to 1024 `f64`, cache resident, median of 0.31 to 1.0 over
/// `m` 128 to 1024 and 16 to 128 swaps per column (Apple M5 Max, one thread; memory bound, so
/// the same for every dtype of the same element width).
pub const NS_PER_SWAP: f64 = 0.55;

/// Nanoseconds per multiply-add of `T` for a GEMM, by the table above. Non-native scalars use
/// the `f64` rate for real types and the `c64` rate for complex ones, a lower bound of their
/// time, so such a region stays serial a little longer rather than dispatching too early.
pub(crate) fn gemm_ns_per_mac<T: crate::ComplexField>() -> f64 {
	if const { T::IS_NATIVE_F32 } {
		ns_per_mac::F32
	} else if const { T::IS_NATIVE_C32 } {
		ns_per_mac::C32
	} else if const { T::IS_REAL } {
		ns_per_mac::F64
	} else {
		ns_per_mac::C64
	}
}

/// Serial time of `count` units at `ns` nanoseconds each.
pub(crate) fn serial_time(count: usize, ns: f64) -> Duration {
	Duration::from_secs_f64(count as f64 * ns * 1e-9)
}

impl ParShape {
	/// The cost of one parallel section of `width` participants on this shape, in the steady
	/// state (sections back to back); [`Duration::MAX`] for a sequential shape or `width < 2`.
	/// `width` is clamped to the degree of the shape.
	///
	/// * `Team`: the cost of the team token the shape was taken from (threadteam's
	///   `DispatchKind::Team`, with the costs of that token's pool).
	/// * `Rayon`: faer reaches the pool from the calling thread, so the cost is the
	///   `Outside` cost of the host table (the default table of threadteam, or the same numbers
	///   without the `threadteam` feature).
	pub fn dispatch_cost(self, width: NonZeroUsize) -> Duration {
		let width = width.min(NonZeroUsize::new(self.degree().max(1)).unwrap());
		if width.get() < 2 {
			return Duration::MAX;
		}
		match self {
			Self::Seq => Duration::MAX,
			#[cfg(feature = "rayon")]
			Self::Rayon(_) => {
				#[cfg(feature = "threadteam")]
				{
					threadteam::DispatchCosts::host_default()
						.cost(threadteam::DispatchKind::Outside, width)
				}
				#[cfg(not(feature = "threadteam"))]
				{
					outside_cost(width)
				}
			},
			#[cfg(feature = "threadteam")]
			Self::Team(_, costs) => costs.cost(threadteam::DispatchKind::Team, width),
		}
	}

	/// Whether a region of serial time `estimate` that opens `sections` sections of `width`
	/// participants pays on this shape: `estimate >= PAYS * sections * dispatch_cost(width)`.
	/// `sections` is the number of dispatches the region really performs.
	pub fn pays(self, estimate: Duration, sections: u32, width: NonZeroUsize) -> bool {
		pays_at(estimate, sections, self.dispatch_cost(width))
	}

	/// [`pays`](Self::pays) at the full degree of the shape.
	pub fn pays_full(self, estimate: Duration, sections: u32) -> bool {
		match NonZeroUsize::new(self.degree()) {
			Some(width) => self.pays(estimate, sections, width),
			None => false,
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn nz(n: usize) -> NonZeroUsize {
		NonZeroUsize::new(n).unwrap()
	}

	#[test]
	fn seq_never_pays() {
		assert_eq!(ParShape::Seq.dispatch_cost(nz(4)), Duration::MAX);
		assert!(!ParShape::Seq.pays(Duration::from_secs(10), 1, nz(4)));
		assert!(!ParShape::Seq.pays_full(Duration::from_secs(10), 1));
	}

	#[test]
	fn pays_is_the_multiple_of_the_cost() {
		let d = Duration::from_micros(2);
		assert!(pays_at(Duration::from_micros(6), 1, d));
		assert!(!pays_at(Duration::from_nanos(5999), 1, d));
		assert!(!pays_at(Duration::from_micros(11), 2, d));
		assert!(pays_at(Duration::from_micros(12), 2, d));
		assert!(!pays_at(Duration::from_secs(1), 1, Duration::MAX));
	}

	#[cfg(feature = "rayon")]
	#[test]
	fn rayon_uses_the_outside_cost() {
		let s = ParShape::Rayon(nz(6));
		assert_eq!(s.dispatch_cost(nz(1)), Duration::MAX);
		let d6 = s.dispatch_cost(nz(6));
		assert!(d6 >= Duration::from_micros(19));
		// width is clamped to the degree
		assert_eq!(s.dispatch_cost(nz(60)), d6);
		// 3 times the cost is the floor: the cost itself does not pay, four times does
		assert!(!s.pays(d6, 1, nz(6)));
		assert!(s.pays(d6 * 4, 1, nz(6)));
	}

	#[cfg(feature = "threadteam")]
	#[test]
	fn team_shape_carries_the_costs_of_its_token() {
		use threadteam::{DispatchCosts, DispatchKind, WidthCosts};
		let costs = DispatchCosts::generic()
			.with(DispatchKind::Team, WidthCosts::flat(Duration::from_micros(5)));
		let s = ParShape::Team(nz(4), costs);
		assert_eq!(s.dispatch_cost(nz(4)), Duration::from_micros(5));
		assert_eq!(s.dispatch_cost(nz(1)), Duration::MAX);
		assert!(s.pays(Duration::from_micros(15), 1, nz(4)));
		assert!(!s.pays(Duration::from_micros(14), 1, nz(4)));
	}

	#[test]
	fn gemm_rates_are_per_dtype() {
		assert_eq!(gemm_ns_per_mac::<f64>(), ns_per_mac::F64);
		assert_eq!(gemm_ns_per_mac::<crate::c64>(), ns_per_mac::C64);
		assert_eq!(gemm_ns_per_mac::<f32>(), ns_per_mac::F32);
		assert_eq!(gemm_ns_per_mac::<crate::c32>(), ns_per_mac::C32);
		// the complex rate is 4.5 times the real one, not 1: the floor bug of the team route
		assert!(ns_per_mac::C64 > 4.0 * ns_per_mac::F64);
	}
}
