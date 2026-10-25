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

/// Floor of the parallel regions of a factorization family: the one type of every
/// `par_threshold` field (LU, full-pivoting LU, Bunch-Kaufman, QR with column pivoting,
/// bidiagonalization, Hessenberg and tridiagonal reductions).
///
/// The default is [`Derived`](Self::Derived): each region is decided by [`ParShape::pays`], its
/// serial time (units of work times the family's serial throughput, [`rate`]) against the
/// dispatch cost of the parallelism it would use and the number of sections it opens.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ParThreshold {
	/// Derive the floor of every region from its serial time and the dispatch cost.
	#[default]
	Derived,
	/// A fixed floor in the work unit of the family (documented where the family declares its
	/// gate: elements of the updated trailing block for the reductions, `f64` elements for LU), a
	/// region runs in parallel from that many units. `Fixed(0)` and `Fixed(1)` make every region
	/// parallel that has any work.
	Fixed(usize),
	/// Never run a region in parallel.
	Never,
}

/// The integer form used by the C API: `0` is [`Derived`](ParThreshold::Derived),
/// `usize::MAX` is [`Never`](ParThreshold::Never), any other value is
/// [`Fixed`](ParThreshold::Fixed).
impl From<usize> for ParThreshold {
	fn from(value: usize) -> Self {
		match value {
			0 => ParThreshold::Derived,
			usize::MAX => ParThreshold::Never,
			n => ParThreshold::Fixed(n),
		}
	}
}

/// See [`From<usize>`](ParThreshold#impl-From<usize>-for-ParThreshold); `Fixed(0)` is
/// written as `1`, the same behaviour, because `0` means derived.
impl From<ParThreshold> for usize {
	fn from(value: ParThreshold) -> Self {
		match value {
			ParThreshold::Derived => 0,
			ParThreshold::Fixed(n) => n.max(1),
			ParThreshold::Never => usize::MAX,
		}
	}
}

impl ParThreshold {
	/// Whether a region of `units` of work in the unit of `rate` that opens `sections`
	/// sections runs in parallel on `shape`, at the full degree of the shape.
	///
	/// [`Derived`](Self::Derived) is [`ParShape::pays_full`] of `units * rate`; an explicit
	/// [`Fixed`](Self::Fixed) floor keeps the previous rule, `units >= floor`.
	pub(crate) fn region_pays<T: crate::ComplexField>(
		self,
		shape: ParShape,
		units: usize,
		rate: Rate,
		sections: u32,
	) -> bool {
		match self {
			Self::Never => false,
			Self::Fixed(floor) => units >= floor,
			Self::Derived => {
				shape.pays_full(serial_time(units, rate.of::<T>()), sections)
			},
		}
	}
}

/// Serial throughput of a kernel family, nanoseconds per unit of work for the real and the
/// complex native scalars. Measured with one thread on an Apple M5 Max, QoS `UserInitiated`,
/// L2 resident shapes (`faer-benchmark` `data/probes/20261026-threshold-recalibration`); `f32`
/// and `c32` take half of the `f64` and `c64` value (twice the lanes per register and half the
/// bytes; checked for the GEMM only), a scalar that is not native takes the `f64` or `c64` value.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rate {
	/// `f64` (and, halved, `f32`).
	pub real: f64,
	/// `c64` (and, halved, `c32`).
	pub cplx: f64,
}

impl Rate {
	/// Nanoseconds per unit for the scalar `T`.
	pub fn of<T: crate::ComplexField>(self) -> f64 {
		if const { T::IS_NATIVE_F32 } {
			self.real / 2.0
		} else if const { T::IS_NATIVE_C32 } {
			self.cplx / 2.0
		} else if const { T::IS_REAL } {
			self.real
		} else {
			self.cplx
		}
	}
}

/// The serial throughput constants of the kernel families, one per family (design record:
/// `threadteam` `docs/design/thresholds.md`, section 2.3; one thread, Apple M5 Max, QoS
/// `UserInitiated`, f64 and c64, median over the shapes 96 to 768 and two processes; the raw
/// numbers and the host log are in `faer-benchmark` `data/probes/20261026-threshold-recalibration`).
/// The unit is what the gate of the family counts. A step of a factorization is timed inside the
/// whole serial factorization, so the constant includes the serial parts of a step (pivot
/// search, Householder generation) and slightly overestimates the time of the part that is
/// split; the Hessenberg value comes from processes the host log marks busy.
pub mod rate {
	use super::{Rate, ns_per_mac};
	/// GEMM and every kernel counted in multiply-adds (triangular products and solves, the
	/// block Householder apply), per multiply-add; the values of [`ns_per_mac`].
	pub const GEMM_MAC: Rate = Rate { real: ns_per_mac::F64, cplx: ns_per_mac::C64 };
	/// Full-pivoting LU, one step: rank-one update fused with the pivot search, per element of
	/// the trailing block `(m-k-1)*(n-k-1)`.
	pub const LU_FULL_PIVOT: Rate = Rate { real: 0.216, cplx: 0.6 };
	/// Bunch-Kaufman with full pivoting, one step: rank-1 or rank-2 update fused with the pivot
	/// search, per element of the trailing block `rem*rem`.
	pub const BUNCH_KAUFMAN: Rate = Rate { real: 0.238, cplx: 0.38 };
	/// QR with column pivoting, one step: fused update of the trailing block and the norms, per
	/// element `(m-k-1)*(n-k-1)`.
	pub const QR_COL_PIVOT: Rate = Rate { real: 0.202, cplx: 0.713 };
	/// Bidiagonalization, one step: the fused two-sided update and the matrix-vector product, per
	/// element `(m-k-1)*(n-k-1)`.
	pub const BIDIAG: Rate = Rate { real: 0.48, cplx: 1.36 };
	/// Hessenberg reduction (unblocked), one step: two matrix-vector products and a rank-one
	/// update, per element `n*n` of the trailing block.
	pub const HESSENBERG: Rate = Rate { real: 0.451, cplx: 1.57 };
	/// Tridiagonalization, one step: the symmetric fused update, per element `rem*rem/2` of the
	/// triangle.
	pub const TRIDIAG: Rate = Rate { real: 0.339, cplx: 1.16 };
	/// Matrix-vector product, per matrix element.
	pub const MATVEC: Rate = Rate { real: 0.1, cplx: 0.34 };
	/// Rank-one Householder apply (`N == 1`), per element of the target `M*K`.
	pub const HOUSEHOLDER_RANK_ONE: Rate = Rate { real: 0.212, cplx: 0.725 };
}

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

/// Whether a region of `units` of work at `rate` that opens `sections` sections pays on `par`,
/// at the full degree of `par` ([`ParShape::pays_full`]).
pub(crate) fn par_pays<T: crate::ComplexField>(
	par: &crate::Par,
	units: usize,
	rate: Rate,
	sections: u32,
) -> bool {
	par.shape().pays_full(serial_time(units, rate.of::<T>()), sections)
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
	use crate::c64;

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

	/// Smallest `units` (doubling, then bisection) for which the region pays on `shape`.
	fn floor_units<T: crate::ComplexField>(
		shape: ParShape,
		r: Rate,
		sections: u32,
	) -> usize {
		let pays = |u: usize| {
			ParThreshold::Derived.region_pays::<T>(shape, u, r, sections)
		};
		let mut hi = 1usize;
		while !pays(hi) {
			hi *= 2;
			assert!(hi < 1 << 40, "never pays");
		}
		let mut lo = hi / 2;
		while hi - lo > 1 {
			let mid = (lo + hi) / 2;
			if pays(mid) {
				hi = mid;
			} else {
				lo = mid;
			}
		}
		hi
	}

	/// (name, rate, sections as counted in the code of the gate)
	const GATES: &[(&str, Rate, u32)] = &[
		("lu_full_pivot", rate::LU_FULL_PIVOT, 1),
		("bunch_kaufman", rate::BUNCH_KAUFMAN, 1),
		("qr_col_pivot", rate::QR_COL_PIVOT, 1),
		("bidiag", rate::BIDIAG, 2),
		("hessenberg", rate::HESSENBERG, 2),
		("tridiag", rate::TRIDIAG, 1),
		("matvec", rate::MATVEC, 1),
		("householder_rank_one", rate::HOUSEHOLDER_RANK_ONE, 1),
		("gemm_mac", rate::GEMM_MAC, 1),
	];

	#[test]
	fn fixed_and_never_keep_their_meaning() {
		let s = ParShape::Rayon(nz(6));
		let r = rate::LU_FULL_PIVOT;
		assert!(!ParThreshold::Never.region_pays::<f64>(s, usize::MAX, r, 1));
		assert!(ParThreshold::Fixed(100).region_pays::<f64>(ParShape::Seq, 100, r, 1));
		assert!(!ParThreshold::Fixed(100).region_pays::<f64>(s, 99, r, 1));
		assert!(ParThreshold::Fixed(100).region_pays::<f64>(s, 100, r, 1));
		// the integer form of the C API
		assert_eq!(ParThreshold::from(0), ParThreshold::Derived);
		assert_eq!(ParThreshold::from(usize::MAX), ParThreshold::Never);
		assert_eq!(ParThreshold::from(7), ParThreshold::Fixed(7));
		assert_eq!(usize::from(ParThreshold::Fixed(0)), 1);
	}

	#[test]
	fn derived_gates_never_pay_on_a_sequential_shape() {
		for &(_, r, sections) in GATES {
			assert!(!ParThreshold::Derived.region_pays::<f64>(
				ParShape::Seq,
				usize::MAX / 2,
				r,
				sections
			));
			assert!(!ParThreshold::Derived.region_pays::<c64>(
				ParShape::Seq,
				usize::MAX / 2,
				r,
				sections
			));
		}
	}

	#[cfg(feature = "threadteam")]
	#[test]
	fn derived_floors_on_a_team_follow_the_rule() {
		use threadteam::DispatchCosts;
		let team = ParShape::Team(nz(6), DispatchCosts::apple_m5());
		let d = team.dispatch_cost(nz(6)).as_secs_f64() * 1e9;
		for &(name, r, sections) in GATES {
			for (ty, ns) in [("f64", r.real), ("c64", r.cplx)] {
				let floor = if ty == "f64" {
					floor_units::<f64>(team, r, sections)
				} else {
					floor_units::<c64>(team, r, sections)
				};
				// the floor is PAYS * sections * D / rate, within the rounding of a `Duration` (nanoseconds)
				let want = PAYS * f64::from(sections) * d / ns;
				assert!(
					(floor as f64 - want).abs() <= 4.0 + want * 2e-3,
					"{name} {ty}: floor {floor}, rule {want}"
				);
			}
		}
		// representative sizes at width 6 on the M5 table (D = 1.4 us, 4.2 us per section)
		let pays = |u: usize, r: Rate, f: u32| {
			ParThreshold::Derived.region_pays::<f64>(team, u, r, f)
		};
		// full pivoting LU: a 100 x 100 trailing block stays serial, 200 x 200 is parallel
		assert!(!pays(100 * 100, rate::LU_FULL_PIVOT, 1));
		assert!(pays(200 * 200, rate::LU_FULL_PIVOT, 1));
		// bidiagonalization counts two sections: it needs twice the work of a one-section step
		assert!(floor_units::<f64>(team, rate::BIDIAG, 2) >= 2 * floor_units::<f64>(team, rate::BIDIAG, 1) - 2);
		// a 256 x 256 matrix-vector product (65536 elements) pays on a team, not below 100 x 100
		assert!(pays(256 * 256, rate::MATVEC, 1));
		assert!(!pays(64 * 64, rate::MATVEC, 1));
		// the old fixed floors were 192*256 = 49152 elements for every one of these steps; the
		// derived floors differ per family (the faster the kernel, the larger the floor)
		let lu = floor_units::<f64>(team, rate::LU_FULL_PIVOT, 1);
		let bidiag = floor_units::<f64>(team, rate::BIDIAG, 1);
		assert!(lu > bidiag);
	}

	#[cfg(all(feature = "rayon", target_os = "macos", target_arch = "aarch64"))]
	#[test]
	fn derived_floors_outside_the_pool_are_larger_than_on_a_team() {
		let rayon = ParShape::Rayon(nz(6));
		for &(name, r, sections) in GATES {
			let outside = floor_units::<f64>(rayon, r, sections);
			#[cfg(feature = "threadteam")]
			{
				use threadteam::DispatchCosts;
				let team = ParShape::Team(nz(6), DispatchCosts::apple_m5());
				let t = floor_units::<f64>(team, r, sections);
				assert!(outside > 8 * t, "{name}: outside {outside}, team {t}");
			}
			// 57 us of serial work per section at width 6
			let want = PAYS * f64::from(sections) * 19_000.0 / r.real;
			assert!((outside as f64 - want).abs() <= 4.0 + want * 2e-3, "{name}");
		}
		// the old fixed floor of the reductions (192 * 256) would have been 10 us of
		// bidiagonalization and runs 5 times slower than serial from outside the pool
		assert!(floor_units::<f64>(rayon, rate::BIDIAG, 2) > 192 * 256);
	}

	#[cfg(feature = "rayon")]
	#[test]
	fn par_pays_reads_the_degree_of_the_token() {
		let seq = crate::Par::Seq;
		assert!(!par_pays::<f64>(&seq, usize::MAX / 2, rate::MATVEC, 1));
		let par = crate::Par::rayon(6);
		assert!(par_pays::<f64>(&par, 10_000_000, rate::MATVEC, 1));
		assert!(!par_pays::<f64>(&par, 1_000, rate::MATVEC, 1));
		let one = crate::Par::rayon(1);
		assert!(!par_pays::<f64>(&one, 10_000_000, rate::MATVEC, 1));
	}

	#[test]
	fn rates_are_per_dtype() {
		let r = rate::BIDIAG;
		assert_eq!(r.of::<f64>(), r.real);
		assert_eq!(r.of::<c64>(), r.cplx);
		assert_eq!(r.of::<f32>(), r.real / 2.0);
		assert_eq!(r.of::<crate::c32>(), r.cplx / 2.0);
		assert_eq!(rate::GEMM_MAC.of::<f64>(), ns_per_mac::F64);
		assert_eq!(rate::GEMM_MAC.of::<c64>(), ns_per_mac::C64);
	}

	/// `Fixed(1)` makes every step of every reduction parallel, and the factors must equal the
	/// serial ones: the `Derived` default keeps the sizes of the other tests below the floors, so
	/// the parallel branches of the gated steps are exercised here.
	#[cfg(feature = "rayon")]
	mod forced_parallel {
		use super::*;
		use crate::dyn_stack::{MemBuffer, MemStack};
		use crate::internal_prelude::*;
		use crate::linalg::cholesky::lblt::factor as bk;
		use crate::linalg::evd::{hessenberg, tridiag};
		use crate::linalg::lu::full_pivoting::factor as fplu;
		use crate::linalg::qr::col_pivoting::factor as cpqr;
		use crate::linalg::svd::bidiag;
		use crate::{Auto, Mat, MatRef, Par};

		const N: usize = 96;

		fn fill<T: crate::ComplexField>(n: usize, seed: usize) -> Mat<T> {
			Mat::from_fn(n, n, |i, j| {
				let h = (i * 2654435761usize)
					.wrapping_add(j * 40503)
					.wrapping_add(seed * 977)
					% 1009;
				from_f64::<T>(h as f64 / 504.0 - 1.0)
			})
		}
		fn close<T: crate::ComplexField<Real = f64>>(a: MatRef<'_, T>, b: MatRef<'_, T>) {
			let mut err = 0.0f64;
			for j in 0..a.ncols() {
				for i in 0..a.nrows() {
					err = err.max((a[(i, j)].clone() - b[(i, j)].clone()).abs());
				}
			}
			core::assert!(err <= 1e-9, "error {err}");
		}

		fn check<T: crate::ComplexField<Real = f64>>() {
			let a0 = fill::<T>(N, 1);
			let sym = &a0 + a0.adjoint();
			// results with a given par and `Fixed(1)`
			let lu = |par: Par| {
				let mut a = a0.clone();
				let mut p = vec![0usize; 4 * N];
				let (r, rest) = p.split_at_mut(N);
				let (ri, rest) = rest.split_at_mut(N);
				let (c, ci) = rest.split_at_mut(N);
				let mut params: fplu::FullPivLuParams = Auto::<T>::auto();
				params.par_threshold = ParThreshold::Fixed(1);
				let mut mem = MemBuffer::new(fplu::lu_in_place_scratch::<usize, T>(
					N,
					N,
					par.shape(),
					params.into(),
				));
				fplu::lu_in_place(
					a.as_mut(), r, ri, c, ci, par, MemStack::new(&mut mem), params.into(),
				);
				a
			};
			close(lu(Par::Seq).as_ref(), lu(Par::rayon(4)).as_ref());
			let bunch = |par: Par| {
				let mut a = sym.clone();
				let mut perm = vec![0usize; N];
				let mut perm_inv = vec![0usize; N];
				let mut sub = Mat::<T>::zeros(N, 1);
				let mut params: bk::LbltParams = Auto::<T>::auto();
				params.pivoting = bk::PivotingStrategy::Full;
				params.par_threshold = ParThreshold::Fixed(1);
				let mut mem = MemBuffer::new(bk::cholesky_in_place_scratch::<usize, T>(
					N,
					par.shape(),
					params.into(),
				));
				bk::cholesky_in_place(
					a.as_mut(),
					sub.as_mut().col_mut(0).as_diagonal_mut(),
					&mut perm,
					&mut perm_inv,
					par,
					MemStack::new(&mut mem),
					params.into(),
				);
				a
			};
			close(bunch(Par::Seq).as_ref(), bunch(Par::rayon(4)).as_ref());
			let qr = |par: Par| {
				let mut a = a0.clone();
				let mut h = Mat::<T>::zeros(16, N);
				let mut p = vec![0usize; N];
				let mut pi = vec![0usize; N];
				let mut params: cpqr::ColPivQrParams = Auto::<T>::auto();
				params.par_threshold = ParThreshold::Fixed(1);
				let mut mem = MemBuffer::new(cpqr::qr_in_place_scratch::<usize, T>(
					N,
					N,
					16,
					par.shape(),
					params.into(),
				));
				cpqr::qr_in_place(
					a.as_mut(), h.as_mut(), &mut p, &mut pi, par, MemStack::new(&mut mem), params.into(),
				);
				a
			};
			close(qr(Par::Seq).as_ref(), qr(Par::rayon(4)).as_ref());
			let bid = |par: Par| {
				let mut a = a0.clone();
				let mut hl = Mat::<T>::zeros(8, N);
				let mut hr = Mat::<T>::zeros(8, N - 1);
				let mut params: bidiag::BidiagParams = Auto::<T>::auto();
				params.par_threshold = ParThreshold::Fixed(1);
				let mut mem = MemBuffer::new(bidiag::bidiag_in_place_scratch::<T>(
					N,
					N,
					par.shape(),
					params.into(),
				));
				bidiag::bidiag_in_place(
					a.as_mut(), hl.as_mut(), hr.as_mut(), par, MemStack::new(&mut mem), params.into(),
				);
				a
			};
			close(bid(Par::Seq).as_ref(), bid(Par::rayon(4)).as_ref());
			let hess = |par: Par| {
				let mut a = a0.clone();
				let mut h = Mat::<T>::zeros(8, N - 1);
				let mut params: hessenberg::HessenbergParams = Auto::<T>::auto();
				params.par_threshold = ParThreshold::Fixed(1);
				params.blocking_threshold = usize::MAX;
				let mut mem = MemBuffer::new(hessenberg::hessenberg_in_place_scratch::<T>(
					N,
					8,
					par.shape(),
					params.into(),
				));
				hessenberg::hessenberg_in_place(
					a.as_mut(), h.as_mut(), par, MemStack::new(&mut mem), params.into(),
				);
				a
			};
			close(hess(Par::Seq).as_ref(), hess(Par::rayon(4)).as_ref());
			let tri = |par: Par| {
				let mut a = sym.clone();
				let mut h = Mat::<T>::zeros(8, N - 1);
				let mut params: tridiag::TridiagParams = Auto::<T>::auto();
				params.par_threshold = ParThreshold::Fixed(1);
				let mut mem = MemBuffer::new(tridiag::tridiag_in_place_scratch::<T>(
					N,
					par.shape(),
					params.into(),
				));
				tridiag::tridiag_in_place(
					a.as_mut(), h.as_mut(), par, MemStack::new(&mut mem), params.into(),
				);
				a
			};
			close(tri(Par::Seq).as_ref(), tri(Par::rayon(4)).as_ref());
		}

		#[test]
		fn forced_parallel_steps_equal_serial_f64() {
			check::<f64>();
		}

		#[test]
		fn forced_parallel_steps_equal_serial_c64() {
			check::<c64>();
		}
	}
}
