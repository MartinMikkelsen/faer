//! The route of the parallel regions onto a threadteam team (feature `threadteam`).
//!
//! A `Par::Team(TeamExec)` is the exclusive authority over the reserved members of a team. The
//! functions here consume that authority (or lend it for the call), so the type system, not a
//! thread local, decides who may dispatch to the members:
//!
//! * flat sections ([`run`]): every participant runs a closure at once; the team is consumed;
//! * splits and forks ([`fork`], [`for_each_child`]): each child is a `TeamExec` of its own,
//!   handed to the callback that runs on its leader, concurrently with its siblings;
//! * a team of one runs everything inline on the caller.
//!
//! Nothing here starts a Rayon job: the team members are threadteam members that spin while the
//! reservation is held, and the section dispatch is a write to their slots.

use core::cell::UnsafeCell;
use rayon::iter::plumbing::{Producer, ProducerCallback};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};
use threadteam::TeamExec;

/// Optional counters of how the team regions were dispatched, for probes that ask "did the
/// parallel path engage at this size". Off by default (one relaxed load per region).
pub mod stats {
	use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
	pub(super) static ON: AtomicBool = AtomicBool::new(false);
	const W: usize = 17;
	#[allow(clippy::declare_interior_mutable_const)]
	const Z: AtomicUsize = AtomicUsize::new(0);
	/// flat sections by participant count (index 1 = ran inline on the caller)
	pub(super) static RUN: [AtomicUsize; W] = [Z; W];
	/// splits and forks by number of parts (index 1 = ran inline on the caller)
	pub(super) static SPLIT: [AtomicUsize; W] = [Z; W];
	pub(super) static GEMM_BELOW_FLOOR: AtomicUsize = AtomicUsize::new(0);
	pub(super) static GEMM_ONE_TILE: AtomicUsize = AtomicUsize::new(0);
	pub(super) static GEMM_TILED: AtomicUsize = AtomicUsize::new(0);
	pub(super) static GEMM_TEAM_KERNEL: AtomicUsize = AtomicUsize::new(0);
	pub(super) fn hist(h: &[AtomicUsize; W], n: usize) {
		if ON.load(Relaxed) {
			h[n.min(W - 1)].fetch_add(1, Relaxed);
		}
	}
	pub(super) fn bump(c: &AtomicUsize) {
		if ON.load(Relaxed) {
			c.fetch_add(1, Relaxed);
		}
	}
	/// Counters since the last `take`.
	#[derive(Debug, Clone, Default)]
	pub struct Stats {
		/// `run` sections, index = participants
		pub run: Vec<usize>,
		/// splits and forks, index = parts
		pub split: Vec<usize>,
		/// GEMM calls decided sequential by the work floor
		pub gemm_below_floor: usize,
		/// GEMM calls whose tile grid had a single tile
		pub gemm_one_tile: usize,
		/// GEMM calls cut into tiles and run on the team
		pub gemm_tiled: usize,
		/// products (plain, triangular or indexed) run by the x86 kernel on the team's own
		/// members (`gemm_with` on a [`Fork`](private_gemm_x86::Fork) of the team)
		pub gemm_team_kernel: usize,
	}
	/// Enables or disables counting.
	pub fn enable(on: bool) {
		ON.store(on, Relaxed);
	}
	/// Returns and resets the counters.
	pub fn take() -> Stats {
		let t = |h: &[AtomicUsize; W]| h.iter().map(|c| c.swap(0, Relaxed)).collect();
		Stats {
			run: t(&RUN),
			split: t(&SPLIT),
			gemm_below_floor: GEMM_BELOW_FLOOR.swap(0, Relaxed),
			gemm_one_tile: GEMM_ONE_TILE.swap(0, Relaxed),
			gemm_tiled: GEMM_TILED.swap(0, Relaxed),
			gemm_team_kernel: GEMM_TEAM_KERNEL.swap(0, Relaxed),
		}
	}
}

/// Records a split of the team into `parts` concurrent children (probes only).
pub(crate) fn note_split(parts: usize) {
	stats::hist(&stats::SPLIT, parts);
}

/// Runs `f(i, m)` on `m = min(n, team size)` participants at once; inline `f(0, 1)` when
/// `m == 1`. The team is consumed; members the section does not use stay parked.
pub(crate) fn run(team: TeamExec<'_>, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
	let size = team.size().get();
	let m = n.min(size);
	if m <= 1 {
		stats::hist(&stats::RUN, 1);
		return f(0, 1);
	}
	stats::hist(&stats::RUN, m);
	if m == size {
		team.run(|i, k| f(i, k));
	} else {
		let _ = team.split(&[m], |_c, child| {
			child.run(|i, k| f(i, k));
		});
	}
}

struct Slots<T>(Vec<UnsafeCell<Option<T>>>);
// SAFETY: every slot is taken by exactly one participant.
unsafe impl<T: Send> Sync for Slots<T> {}
impl<T> Slots<T> {
	/// # Safety
	/// Slot `j` is taken by one participant only.
	unsafe fn take(&self, j: usize) -> T {
		unsafe { (*self.0[j].get()).take() }.unwrap()
	}
}

struct Collect<'a, T>(&'a mut Vec<T>);
impl<T: Send> ProducerCallback<T> for Collect<'_, T> {
	type Output = ();
	fn callback<P: Producer<Item = T>>(self, producer: P) {
		self.0.extend(producer.into_iter());
	}
}

/// The team version of `spindle::for_each`: the items of `iter` are dealt out in contiguous
/// static blocks over `min(n_jobs, items, team size)` parts (the same static distribution the
/// spindle region uses), each part processed by the leader of a sub-team.
pub(crate) fn for_each<T: Send, I>(
	team: TeamExec<'_>,
	n_jobs: usize,
	iter: I,
	f: impl Sync + Fn(T),
) where
	I: IntoParallelIterator<Item = T>,
	I::Iter: IndexedParallelIterator,
{
	for_each_child(team, n_jobs, iter, |item, _child| f(item))
}

/// [`for_each`] whose closure also receives the sub-team of its block, lent for the call.
pub(crate) fn for_each_child<T: Send, I>(
	team: TeamExec<'_>,
	n_jobs: usize,
	iter: I,
	f: impl Sync + for<'c> Fn(T, TeamExec<'c>),
) where
	I: IntoParallelIterator<Item = T>,
	I::Iter: IndexedParallelIterator,
{
	let mut items = Vec::new();
	// `with_producer` walks the splitting machinery without starting any Rayon job.
	iter.into_par_iter().with_producer(Collect(&mut items));
	let len = items.len();
	if len == 0 {
		return;
	}
	let slots = Slots(items.into_iter().map(|x| UnsafeCell::new(Some(x))).collect());
	let jobs = n_jobs.max(1).min(len).min(team.size().get());
	stats::hist(&stats::SPLIT, jobs);
	team.fork(jobs, |c, mut child| {
		let (start, n) = super::thread::par_split_indices(len, c, jobs);
		for j in start..start + n {
			// SAFETY: block `c` owns slots `start..start + n` alone.
			let item = unsafe { slots.take(j) };
			f(item, child.reborrow());
		}
	});
}

struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> Clone for SendPtr<T> {
	fn clone(&self) -> Self {
		*self
	}
}
impl<T> Copy for SendPtr<T> {}
impl<T> SendPtr<T> {
	fn get(self) -> *mut T {
		self.0
	}
}

/// Granularity of tile edges: a multiple of the register tile of the `gemm` micro kernels
/// (f64 on aarch64: 8 rows, 4 columns), so a tile does not end in a partial micro panel unless
/// the matrix does.
const ROW_GRAIN: usize = 8;
const COL_GRAIN: usize = 4;

/// The grid `(pm, pn)` with `pm * pn <= nt` that minimises the packed volume
/// `k * (m / pm + n / pn)` per member, among grids that use as many members as the shape
/// allows (every tile at least one grain wide).
pub(crate) fn tile_grid(m: usize, n: usize, nt: usize) -> (usize, usize) {
	let max_pm = (m / ROW_GRAIN).max(1);
	let max_pn = (n / COL_GRAIN).max(1);
	let mut best = (1, 1);
	let mut best_used = 1;
	let mut best_cost = f64::INFINITY;
	for pm in 1..=nt.min(max_pm) {
		for pn in 1..=(nt / pm).min(max_pn) {
			let used = pm * pn;
			let cost = m as f64 / pm as f64 + n as f64 / pn as f64;
			// more members first, then the smaller packed volume
			if used > best_used || (used == best_used && cost < best_cost) {
				best = (pm, pn);
				best_used = used;
				best_cost = cost;
			}
		}
	}
	best
}

/// `[start, end)` of part `idx` of `0..len` cut into `parts` blocks whose edges are multiples
/// of `grain` (the last block takes the remainder).
fn grain_range(len: usize, parts: usize, idx: usize, grain: usize) -> (usize, usize) {
	let units = len.div_ceil(grain);
	let edge = |i: usize| -> usize {
		let base = units / parts;
		let rem = units % parts;
		let u = if i < rem { i * (base + 1) } else { rem + i * base };
		(u * grain).min(len)
	};
	(edge(idx), edge(idx + 1))
}

/// The instruction set and dtype of the `private_gemm_x86` kernel for `E`, if it has one here.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn x86_kernel<E: 'static>() -> Option<(private_gemm_x86::InstrSet, private_gemm_x86::DType)> {
	use core::any::TypeId;
	use private_gemm_x86::*;
	let id = TypeId::of::<E>();
	let dtype = if id == TypeId::of::<f64>() {
		DType::F64
	} else if id == TypeId::of::<f32>() {
		DType::F32
	} else if id == TypeId::of::<num_complex::Complex<f64>>() {
		DType::C64
	} else if id == TypeId::of::<num_complex::Complex<f32>>() {
		DType::C32
	} else {
		return None;
	};
	if std::arch::is_x86_feature_detected!("avx512f") {
		Some((InstrSet::Avx512, dtype))
	} else if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
		Some((InstrSet::Avx256, dtype))
	} else {
		None
	}
}

/// The `private_gemm_x86` kernel on a team: the kernel's parallel stages (shared packing of the
/// right-hand side, then the dynamic job queue) run on the team's own members, through the
/// kernel's `Fork` executor. Nothing starts a thread outside the team, and the triangular and
/// indexed destinations of `spicy_matmul` and the triangular product need no tiling outside.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
pub(crate) mod kernel {
	use super::{stats, x86_kernel};
	use core::cell::{RefCell, UnsafeCell};
	use core::time::Duration;
	use private_gemm_x86::Fork;
	use threadteam::TeamExec;

	/// Adapter of a team token (all of it, or a sub-team of the gate's width) to the kernel's
	/// executor: `run(f)` is one team section, `f(j)` on member `j`, the calling thread being
	/// member 0. The token is exclusive authority, so the adapter holds it behind a `RefCell`
	/// and is not `Sync`, which `Fork` does not ask for.
	struct TeamFork<'t> {
		team: RefCell<TeamExec<'t>>,
		width: usize,
	}

	impl Fork for TeamFork<'_> {
		fn width(&self) -> usize {
			self.width
		}

		fn run(&self, f: &(dyn Fn(usize) + Sync)) {
			stats::hist(&stats::RUN, self.width);
			// the kernel never re-enters `run`, so the borrow is never contended
			self.team.borrow_mut().reborrow().run(|j, _| f(j));
		}
	}

	/// The width at which `team` runs a product of serial time `estimate` through the kernel, or
	/// `None` (run sequentially on the caller) when no width pays. The kernel forks twice per
	/// call (packing, then compute), so the gate is `pays(estimate, 2, width)` at the width
	/// `width_for` chooses.
	pub(crate) fn kernel_width(team: &TeamExec<'_>, estimate: Duration) -> Option<usize> {
		// the cheap rejection first: a product that does not pay at the cheapest width of the table
		// pays nowhere (whatever the shape of the table), so `width_for` need not search (a QR 128
		// asks for every product of its 59 block applies)
		if !crate::utils::cost::par_can_pay_time_team(team, estimate, 2) {
			stats::bump(&stats::GEMM_BELOW_FLOOR);
			return None;
		}
		let width = team.width_for(estimate, 2);
		if width.get() >= 2 && team.pays(estimate, 2, width) {
			Some(width.get())
		} else {
			stats::bump(&stats::GEMM_BELOW_FLOOR);
			None
		}
	}

	/// Calls `f` with an executor of `width` participants of `team` (a sub-team when `width` is
	/// narrower than the team; the members it leaves out stay parked). The team is consumed.
	pub(crate) fn with_fork<R>(team: TeamExec<'_>, width: usize, f: impl FnOnce(&dyn Fork) -> R) -> R {
		stats::bump(&stats::GEMM_TEAM_KERNEL);
		let size = team.size().get();
		let width = width.clamp(1, size);
		if width == size {
			return f(&TeamFork { team: RefCell::new(team), width });
		}
		struct Slot<F, R>(UnsafeCell<(Option<F>, Option<R>)>);
		// SAFETY: a split into the single child `[width]` runs the callback once, on the
		// calling thread (child 0 is led by the caller), so the slot is never shared.
		unsafe impl<F, R> Sync for Slot<F, R> {}
		impl<F, R> Slot<F, R> {
			fn get(&self) -> *mut (Option<F>, Option<R>) {
				self.0.get()
			}
		}
		let slot = Slot(UnsafeCell::new((Some(f), None)));
		team.split(&[width], |_, child| {
			// SAFETY: see `Slot`.
			let (f, out) = unsafe { &mut *slot.get() };
			let f = f.take().expect("one child");
			*out = Some(f(&TeamFork { team: RefCell::new(child), width }));
		})
		.expect("1 <= width <= size is a valid split");
		slot.0.into_inner().1.expect("the child ran")
	}

	/// The kernel route of [`gemm_team`](super::gemm_team): `None` when `E` has no kernel here.
	pub(crate) fn available<E: 'static>() -> Option<(private_gemm_x86::InstrSet, private_gemm_x86::DType)> {
		x86_kernel::<E>()
	}
}

/// A plain product on a team token (`dst := alpha * dst + beta * lhs * rhs`, the contract of
/// `gemm::gemm` as [`gemm_tiled`]): on x86_64 with AVX2 the `private_gemm_x86` kernel on the
/// team's members ([`kernel::with_fork`], width from the gate [`kernel::kernel_width`], two
/// sections); below the gate the same kernel on the calling thread. Other targets and dtypes
/// take [`gemm_tiled`].
///
/// # Safety
///
/// As `gemm::gemm`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemm_team<E: 'static + Copy + Send + Sync>(
	team: TeamExec<'_>,
	m: usize,
	n: usize,
	k: usize,
	ns_per_mac: f64,
	dst: *mut E,
	dst_cs: isize,
	dst_rs: isize,
	read_dst: bool,
	lhs: *const E,
	lhs_cs: isize,
	lhs_rs: isize,
	rhs: *const E,
	rhs_cs: isize,
	rhs_rs: isize,
	alpha: E,
	beta: E,
	conj_lhs: bool,
	conj_rhs: bool,
) {
	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	if let Some((feat, dtype)) = kernel::available::<E>() {
		use private_gemm_x86::*;
		// `gemm::gemm` is `dst := alpha * dst + beta * lhs * rhs`, the kernel's
		// `dst := (beta == Add ? dst : 0) + alpha * lhs * rhs`: `alpha` of the call is the
		// identity whenever `read_dst` holds (the plain entry only passes 0 or 1)
		macro_rules! go {
			($call:path, $last:expr) => {
				$call(
					dtype,
					IType::U64,
					feat,
					m,
					n,
					k,
					dst as *mut (),
					dst_rs,
					dst_cs,
					core::ptr::null(),
					core::ptr::null(),
					DstKind::Full,
					if read_dst { Accum::Add } else { Accum::Replace },
					lhs as *const (),
					lhs_rs,
					lhs_cs,
					conj_lhs,
					core::ptr::null(),
					0,
					rhs as *const (),
					rhs_rs,
					rhs_cs,
					conj_rhs,
					&raw const beta as *const (),
					$last,
				)
			};
		}
		let call = |fork: Option<&dyn Fork>| unsafe {
			match fork {
				Some(fork) => go!(gemm_with, fork),
				None => go!(gemm, 1),
			}
		};
		let estimate = crate::utils::cost::serial_time(m.saturating_mul(n).saturating_mul(k), ns_per_mac);
		match kernel::kernel_width(&team, estimate) {
			Some(width) => kernel::with_fork(team, width, |fork| call(Some(fork))),
			None => call(None),
		}
		return;
	}
	gemm_tiled::<E>(team, m, n, k, ns_per_mac, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs, rhs, rhs_cs, rhs_rs, alpha, beta, conj_lhs, conj_rhs)
}

/// `dst := alpha * dst + beta * lhs * rhs` (the contract of `gemm::gemm`) with the output cut
/// into a `pm x pn` grid of tiles, one sequential `gemm::gemm` per team member (the route of
/// targets without the x86 kernel, see [`gemm_team`]). The depth `k`
/// is never split, so no reduction and no partial buffer is needed. Each member's call blocks
/// for the caches by itself (the `gemm` crate chooses `kc`, `mc` and `nc` from the cache sizes),
/// so a tile is the unit of work of a member and the blocking inside it is unchanged.
///
/// Below the `gemm` crate's own threading floor, or on a single participant, the whole product
/// is one sequential call.
///
/// # Safety
///
/// As `gemm::gemm`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemm_tiled<E: 'static + Copy + Send + Sync>(
	team: TeamExec<'_>,
	m: usize,
	n: usize,
	k: usize,
	ns_per_mac: f64,
	dst: *mut E,
	dst_cs: isize,
	dst_rs: isize,
	read_dst: bool,
	lhs: *const E,
	lhs_cs: isize,
	lhs_rs: isize,
	rhs: *const E,
	rhs_cs: isize,
	rhs_rs: isize,
	alpha: E,
	beta: E,
	conj_lhs: bool,
	conj_rhs: bool,
) {
	// The generic route: one `gemm::gemm` per member with `Parallelism::None`. On x86_64 with
	// AVX2 the team runs the `private_gemm_x86` kernel itself instead ([`gemm_team`]); this is the
	// route of every other target.
	let seq = |dst: *mut E, lhs: *const E, rhs: *const E, m: usize, n: usize| unsafe {
		gemm::gemm(
			m,
			n,
			k,
			dst,
			dst_cs,
			dst_rs,
			read_dst,
			lhs,
			lhs_cs,
			lhs_rs,
			rhs,
			rhs_cs,
			rhs_rs,
			alpha,
			beta,
			false,
			conj_lhs,
			conj_rhs,
			gemm::Parallelism::None,
		)
	};
	let nt = team.size().get();
	// The floor is time, not a count: the serial time of the product (its MACs at the dtype's
	// rate) against `PAYS` times the cost of one section of this team. A complex MAC is 4.5
	// real ones, so the floor of a complex product is that much lower in MACs.
	let estimate = crate::utils::cost::serial_time(m.saturating_mul(n).saturating_mul(k), ns_per_mac);
	if nt <= 1 || !team.pays(estimate, 1, team.size()) {
		stats::bump(&stats::GEMM_BELOW_FLOOR);
		return seq(dst, lhs, rhs, m, n);
	}
	let (pm, pn) = tile_grid(m, n, nt);
	if pm * pn <= 1 {
		stats::bump(&stats::GEMM_ONE_TILE);
		return seq(dst, lhs, rhs, m, n);
	}
	stats::bump(&stats::GEMM_TILED);
	let (dst, lhs, rhs) = (
		SendPtr(dst),
		SendPtr(lhs as *mut E),
		SendPtr(rhs as *mut E),
	);
	run(team, pm * pn, &|i, parts| {
		// `parts == pm * pn` unless the team is narrower, then the tiles are dealt round robin
		let mut t = i;
		while t < pm * pn {
			let (ti, tj) = (t % pm, t / pm);
			let (i0, i1) = grain_range(m, pm, ti, ROW_GRAIN);
			let (j0, j1) = grain_range(n, pn, tj, COL_GRAIN);
			if i1 > i0 && j1 > j0 {
				let d = dst.get().wrapping_offset(i0 as isize * dst_rs + j0 as isize * dst_cs);
				let l = lhs.get().wrapping_offset(i0 as isize * lhs_rs) as *const E;
				let r = rhs.get().wrapping_offset(j0 as isize * rhs_cs) as *const E;
				seq(d, l, r, i1 - i0, j1 - j0);
			}
			t += parts;
		}
	});
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::linalg::matmul::matmul;
	use crate::{c64, Accum, Mat, MatRef, Par};
	use core::sync::atomic::{AtomicUsize, Ordering};
	use threadteam::Pool;

	fn team_of(width: usize) -> threadteam::Team {
		Pool::builder()
			.threads(width)
			.acquire_timeout(std::time::Duration::from_secs(30))
			.build()
			.unwrap()
			.into_team(width)
			.expect("an idle pool")
	}

	/// Team tests take turns: one pool at a time, and the dispatch counters are not shared.
	static TEAM_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

	/// Runs `f` with the authority over a team of `width` participants.
	fn with_par<R>(width: usize, f: impl FnOnce(Par<'_>) -> R) -> R {
		let _turn = TEAM_TESTS.lock().unwrap_or_else(|e| e.into_inner());
		let team = team_of(width);
		let mut exec = team.exec(width);
		exec.team(|team| f(Par::Team(team))).ok().expect("a team token")
	}

	fn rand_mat(m: usize, n: usize, seed: u64) -> Mat<f64> {
		let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		Mat::from_fn(m, n, |_, _| {
			x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
			((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
		})
	}

	fn max_diff(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
		let mut d: f64 = 0.0;
		for j in 0..a.ncols() {
			for i in 0..a.nrows() {
				d = d.max((a[(i, j)] - b[(i, j)]).abs());
			}
		}
		d
	}

	/// A factorization too small to pay for a section asks the gates of a team token as the
	/// sequential path does: the same factors to the last bit, and no section opened. The block
	/// apply decides once at its top (`householder.rs`, the floor of the whole apply) and runs its
	/// products sequentially, instead of asking each product and each triangular step.
	#[test]
	fn small_qr_on_a_team_is_the_sequential_one() {
		use crate::dyn_stack::{MemBuffer, MemStack};
		use crate::linalg::qr::no_pivoting::factor as qrf;
		let n = 32;
		let a0 = rand_mat(n, n, 11);
		let run = |mut par: Par<'_>| {
			let bs = qrf::recommended_block_size::<f64>(n, n);
			let mut a = a0.clone();
			let mut q = Mat::<f64>::zeros(bs, n);
			let mut mem = MemBuffer::new(qrf::qr_in_place_scratch::<f64>(n, n, bs, par.shape(), Default::default()));
			qrf::qr_in_place(a.as_mut(), q.as_mut(), par.reborrow(), MemStack::new(&mut mem), Default::default());
			(a, q)
		};
		let (a_seq, q_seq) = run(Par::Seq);
		let (a_team, q_team) = with_par(4, |par| {
			stats::take();
			stats::enable(true);
			let r = run(par);
			stats::enable(false);
			let s = stats::take();
			let sections: usize = s.run.iter().skip(2).sum::<usize>() + s.split.iter().skip(2).sum::<usize>();
			assert_eq!(sections, 0, "a QR {n} opened a section: {s:?}");
			assert_eq!(s.gemm_tiled + s.gemm_team_kernel, 0, "{s:?}");
			r
		});
		assert_eq!(max_diff(a_seq.as_ref(), a_team.as_ref()), 0.0);
		assert_eq!(max_diff(q_seq.as_ref(), q_team.as_ref()), 0.0);
	}

	/// The rejection before the width search is against the cheapest width of the table, so it is right
	/// for a table whose cost rises with the width: here 2 us at width 2 and 12 us at width 6, a
	/// 13 us product (two sections) pays at width 2 only, and a 11 us one nowhere.
	#[test]
	fn kernel_width_rejects_at_the_cheapest_width() {
		use threadteam::{DispatchCosts, DispatchKind, WidthCosts};
		use core::time::Duration;
		let _turn = TEAM_TESTS.lock().unwrap_or_else(|e| e.into_inner());
		let us = Duration::from_micros;
		let costs = DispatchCosts::for_class(threadteam::HostClass::Generic).with(DispatchKind::Team, WidthCosts::new([us(2), us(7), us(12)]));
		let team = Pool::builder()
			.threads(6)
			.acquire_timeout(std::time::Duration::from_secs(30))
			.dispatch_costs(costs)
			.build()
			.unwrap()
			.into_team(6)
			.expect("an idle pool");
		let mut exec = team.exec(6);
		exec.team(|team| {
			assert_eq!(kernel::kernel_width(&team, us(13)), Some(2));
			assert_eq!(kernel::kernel_width(&team, us(11)), None);
			let par = Par::Team(team);
			assert!(crate::utils::cost::par_can_pay_time(&par, us(6), 1));
			assert!(!crate::utils::cost::par_can_pay_time(&par, us(5), 1));
		})
		.ok()
		.expect("a team token");
	}

	/// `par_pays_time` is `ParShape::pays` without the shape.
	#[test]
	fn par_pays_time_is_the_shape_gate() {
		use crate::utils::cost::par_pays_time;
		use core::time::Duration;
		with_par(6, |par| {
			let shape = par.shape();
			for &us in &[0u64, 1, 3, 5, 8, 12, 30, 1000] {
				for sections in 1..=3 {
					for width in 1..=8usize {
						let est = Duration::from_micros(us);
						let w = core::num::NonZeroUsize::new(width.min(6)).unwrap();
						let want = width >= 2 && shape.pays(est, sections, w);
						assert_eq!(par_pays_time(&par, est, sections, width), want, "{us} us, {sections}, {width}");
					}
				}
			}
		});
		assert!(!par_pays_time(&Par::Seq, Duration::from_secs(1), 1, 4));
	}

	#[test]
	fn grid_uses_the_team_and_stays_in_bounds() {
		for nt in 1..=12 {
			for &(m, n) in &[(2048, 2048), (4096, 64), (64, 4096), (100, 7), (8, 4), (1, 1000)] {
				let (pm, pn) = tile_grid(m, n, nt);
				assert!(pm * pn <= nt && pm >= 1 && pn >= 1, "{m}x{n} nt={nt}");
				// the tiles partition the output
				let mut rows = 0;
				for i in 0..pm {
					let (a, b) = grain_range(m, pm, i, ROW_GRAIN);
					assert_eq!(a, rows);
					rows = b;
				}
				assert_eq!(rows, m);
				let mut cols = 0;
				for j in 0..pn {
					let (a, b) = grain_range(n, pn, j, COL_GRAIN);
					assert_eq!(a, cols);
					cols = b;
				}
				assert_eq!(cols, n);
			}
		}
		assert_eq!(tile_grid(2048, 2048, 6).0 * tile_grid(2048, 2048, 6).1, 6);
	}

	#[test]
	fn gemm_on_the_team_matches_sequential() {
		with_par(6, |mut par| {
			for &(m, n, k) in &[(512, 512, 512), (257, 301, 129), (4096, 64, 64), (64, 4096, 64), (1000, 33, 500)] {
				let a = rand_mat(m, k, 1);
				let b = rand_mat(k, n, 2);
				for beta in [Accum::Replace, Accum::Add] {
					let mut want = rand_mat(m, n, 3);
					let mut got = want.clone();
					matmul(want.as_mut(), beta, a.as_ref(), b.as_ref(), 1.5, Par::Seq);
					matmul(got.as_mut(), beta, a.as_ref(), b.as_ref(), 1.5, par.reborrow());
					assert!(max_diff(want.as_ref(), got.as_ref()) < 1e-10, "{m}x{n}x{k}");
				}
				// transposed and strided operands
				let mut want = Mat::<f64>::zeros(m, n);
				let mut got = Mat::<f64>::zeros(m, n);
				let at = a.transpose().to_owned();
				matmul(want.as_mut(), Accum::Replace, at.transpose(), b.as_ref(), 1.0, Par::Seq);
				matmul(got.as_mut(), Accum::Replace, at.transpose(), b.as_ref(), 1.0, par.reborrow());
				assert!(max_diff(want.as_ref(), got.as_ref()) < 1e-10);
			}
			// complex
			let (m, n, k) = (300, 280, 200);
			let a = Mat::<c64>::from_fn(m, k, |i, j| c64::new(i as f64 * 0.01 - j as f64 * 0.003, (i + j) as f64 * 0.002));
			let b = Mat::<c64>::from_fn(k, n, |i, j| c64::new((i * j % 7) as f64 * 0.1, i as f64 * 0.01 - j as f64 * 0.01));
			let mut want = Mat::<c64>::zeros(m, n);
			let mut got = Mat::<c64>::zeros(m, n);
			matmul(want.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), c64::new(1.0, 0.0), Par::Seq);
			matmul(got.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), c64::new(1.0, 0.0), par.reborrow());
			let mut d: f64 = 0.0;
			for j in 0..n {
				for i in 0..m {
					d = d.max((want[(i, j)] - got[(i, j)]).norm_sqr().sqrt());
				}
			}
			assert!(d < 1e-9);
		});
	}

	/// The floor is the serial time against the cost of a section, per dtype: a complex product
	/// reaches it at a quarter (4.5 times fewer MACs) of the real one. The old team floor counted
	/// MACs of every dtype against the real one and kept `c64` `64^3` (43 us serial) sequential.
	#[test]
	fn the_team_floor_is_per_dtype() {
		// the grid of `gemm_tiled` (no x86 kernel) or the kernel on the team's members
		let engaged = |st: &stats::Stats| st.gemm_tiled + st.gemm_team_kernel;
		let before = |m: usize, n: usize, k: usize, par: &mut Par<'_>, cplx: bool| {
			stats::enable(true);
			let _ = stats::take();
			if cplx {
				let a = Mat::<c64>::from_fn(m, k, |i, j| c64::new((i + j) as f64, 1.0));
				let b = Mat::<c64>::from_fn(k, n, |i, j| c64::new(i as f64, j as f64));
				let mut d = Mat::<c64>::zeros(m, n);
				matmul(d.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), c64::new(1.0, 0.0), par.reborrow());
			} else {
				let a = rand_mat(m, k, 1);
				let b = rand_mat(k, n, 2);
				let mut d = Mat::<f64>::zeros(m, n);
				matmul(d.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), 1.0, par.reborrow());
			}
			let st = stats::take();
			stats::enable(false);
			st
		};
		with_par(6, |mut par| {
			// c64 64^3: 43 us serial, far above 3 sections' worth on every host table
			let st = before(64, 64, 64, &mut par, true);
			assert_eq!(st.gemm_below_floor, 0, "{st:?}");
			assert_eq!(engaged(&st), 1, "{st:?}");
			// f64 20^3 (8 ki MACs, 0.3 us) is below any floor
			let st = before(20, 20, 20, &mut par, false);
			assert_eq!(engaged(&st), 0, "{st:?}");
			// f64 192^3 (7 M MACs, 262 us) is above every floor
			let st = before(192, 192, 192, &mut par, false);
			assert_eq!(engaged(&st), 1, "{st:?}");
		});
	}

	#[test]
	fn teams_of_odd_and_unit_width_match_sequential() {
		let (m, n, k) = (300, 200, 250);
		let a = rand_mat(m, k, 1);
		let b = rand_mat(k, n, 2);
		let mut want = Mat::<f64>::zeros(m, n);
		matmul(want.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), 1.0, Par::Seq);
		for width in [1usize, 3, 5] {
			let mut got = Mat::<f64>::zeros(m, n);
			with_par(width, |par| {
				matmul(got.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), 1.0, par)
			});
			assert!(max_diff(want.as_ref(), got.as_ref()) < 1e-12, "width = {width}");
		}
	}

	#[test]
	fn for_each_deals_every_item_once_and_nests() {
		with_par(6, |mut par| {
			for jobs in [1usize, 2, 3, 6, 9] {
				let hits: Vec<AtomicUsize> = (0..13).map(|_| AtomicUsize::new(0)).collect();
				crate::utils::thread::for_each_with_par(
					par.reborrow(),
					jobs,
					rayon::iter::IntoParallelIterator::into_par_iter(0..13usize),
					|i, child| {
						hits[i].fetch_add(1, Ordering::Relaxed);
						// a job receives a share of the team no wider than the team
						assert!(child.degree() <= 6);
					},
				);
				assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "jobs = {jobs}");
			}
		});
	}

	#[test]
	fn join_raw_leaves_match_the_width() {
		for n in [2usize, 3, 5, 6] {
			let widths = std::sync::Mutex::new(Vec::new());
			with_par(n, |par| {
				crate::utils::thread::join_raw(
					|p| widths.lock().unwrap().push(p.degree()),
					|p| widths.lock().unwrap().push(p.degree()),
					par,
				)
			});
			let mut w = widths.into_inner().unwrap();
			w.sort_unstable();
			let half = n / 2;
			assert_eq!(w, vec![half, n - half], "n = {n}");
		}
	}

	#[test]
	fn lu_qr_cholesky_on_the_team_match_sequential() {
		use crate::linalg::solvers::{Llt, Qr};
		use crate::Side;
		with_par(6, |mut par| {
			for n in [200usize, 513] {
				let a = rand_mat(n, n, 7);
				let b = rand_mat(n, 5, 8);
				// LU (partial pivoting) through the solver entry points with an explicit Par
				let lu = |par: Par| {
					let mut f = a.clone();
					let mut perm = vec![0usize; n];
					let mut perm_inv = vec![0usize; n];
					let mut mem = dyn_stack::MemBuffer::new(
						crate::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, f64>(n, n, par.shape(), Default::default()),
					);
					crate::linalg::lu::partial_pivoting::factor::lu_in_place(
						f.as_mut(),
						&mut perm,
						&mut perm_inv,
						par,
						dyn_stack::MemStack::new(&mut mem),
						Default::default(),
					);
					f
				};
				let want = lu(Par::Seq);
				let got = lu(par.reborrow());
				assert!(max_diff(want.as_ref(), got.as_ref()) < 1e-8, "lu n = {n}");
				// QR
				let mut r_want = a.clone();
				let mut r_got = a.clone();
				let qr = |m: &mut Mat<f64>, par: Par| {
					let bs = crate::linalg::qr::no_pivoting::factor::recommended_block_size::<f64>(n, n);
					let mut h = Mat::<f64>::zeros(bs, n);
					let mut mem = dyn_stack::MemBuffer::new(
						crate::linalg::qr::no_pivoting::factor::qr_in_place_scratch::<f64>(n, n, bs, par.shape(), Default::default()),
					);
					crate::linalg::qr::no_pivoting::factor::qr_in_place(
						m.as_mut(),
						h.as_mut(),
						par,
						dyn_stack::MemStack::new(&mut mem),
						Default::default(),
					);
				};
				qr(&mut r_want, Par::Seq);
				qr(&mut r_got, par.reborrow());
				assert!(max_diff(r_want.as_ref(), r_got.as_ref()) < 1e-8, "qr n = {n}");
				let _ = (&b, Side::Lower);
				let _: Option<(Llt<f64>, Qr<f64>)> = None;
				// Cholesky of a well conditioned SPD matrix
				let spd = {
					let mut s = Mat::<f64>::zeros(n, n);
					matmul(s.as_mut(), Accum::Replace, a.transpose(), a.as_ref(), 1.0, Par::Seq);
					for i in 0..n {
						s[(i, i)] += n as f64;
					}
					s
				};
				let ch = |par: Par| {
					let mut f = spd.clone();
					let mut mem = dyn_stack::MemBuffer::new(
						crate::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<f64>(n, par.shape(), Default::default()),
					);
					crate::linalg::cholesky::llt::factor::cholesky_in_place(
						f.as_mut(),
						Default::default(),
						par,
						dyn_stack::MemStack::new(&mut mem),
						Default::default(),
					)
					.unwrap();
					f
				};
				let want = ch(Par::Seq);
				let got = ch(par.reborrow());
				// compare the lower triangles
				let mut d: f64 = 0.0;
				for j in 0..n {
					for i in j..n {
						d = d.max((want[(i, j)] - got[(i, j)]).abs());
					}
				}
				assert!(d < 1e-8, "cholesky n = {n}: {d}");
			}
		});
	}

	/// Dispatch counters of `f`, which runs on a team of `width`.
	fn counted<R>(width: usize, f: impl FnOnce(Par<'_>) -> R) -> (R, stats::Stats) {
		with_par(width, |par| {
			stats::enable(true);
			let _ = stats::take();
			let r = f(par);
			let st = stats::take();
			stats::enable(false);
			(r, st)
		})
	}

	#[test]
	fn sparse_matmul_on_a_team_runs_on_the_team() {
		use crate::sparse::{SparseColMat, Triplet};
		let n = 3000;
		let triplets: Vec<_> = (0..n)
			.flat_map(|j| {
				(0..4).map(move |t| {
					let i = (j * 7 + t * 131) % n;
					Triplet::new(i, j, ((i + 2 * j) as f64).sin())
				})
			})
			.collect();
		let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();
		let want = crate::sparse::linalg::matmul::sparse_sparse_matmul(
			a.as_ref(), a.as_ref(), 1.0, Par::Seq,
		)
		.unwrap();
		let (got, st) = counted(4, |par| {
			crate::sparse::linalg::matmul::sparse_sparse_matmul(a.as_ref(), a.as_ref(), 1.0, par).unwrap()
		});
		// the numeric phase forked the team into its four jobs; raw Rayon would record nothing
		assert!(st.split.get(4).copied().unwrap_or(0) >= 1, "{:?}", st.split);
		assert!(max_diff(want.to_dense().as_ref(), got.to_dense().as_ref()) < 1e-12);
	}

	#[test]
	fn lu_swap_stage_splits_the_team_it_is_handed() {
		let n = 513;
		let a = rand_mat(n, n, 11);
		let lu = |par: Par| {
			let mut f = a.clone();
			let mut perm = vec![0usize; n];
			let mut perm_inv = vec![0usize; n];
			let mut mem = dyn_stack::MemBuffer::new(
				crate::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, f64>(
					n, n, par.shape(), Default::default(),
				),
			);
			crate::linalg::lu::partial_pivoting::factor::lu_in_place(
				f.as_mut(), &mut perm, &mut perm_inv, par,
				dyn_stack::MemStack::new(&mut mem), Default::default(),
			);
			f
		};
		let want = lu(Par::Seq);
		let (got, st) = counted(6, lu);
		// a swap stage with columns on both sides runs its two sides on a split of the team
		assert!(st.split.get(2).copied().unwrap_or(0) >= 1, "{:?}", st.split);
		assert!(max_diff(want.as_ref(), got.as_ref()) < 1e-8);
	}

	#[test]
	fn join_sized_on_a_team_hands_out_exactly_the_team() {
		use rayon::iter::IntoParallelIterator;
		for left in 0..=7usize {
			let active = AtomicUsize::new(0);
			let peak = AtomicUsize::new(0);
			let widths = [AtomicUsize::new(0), AtomicUsize::new(0)];
			with_par(6, |par| {
				let side = |which: usize, par: Par<'_>| {
					let jobs = par.degree();
					widths[which].store(jobs, Ordering::Relaxed);
					crate::utils::thread::for_each_with_par(par, jobs, (0..jobs).into_par_iter(), |_, _| {
						let now = active.fetch_add(1, Ordering::AcqRel) + 1;
						peak.fetch_max(now, Ordering::AcqRel);
						std::thread::sleep(std::time::Duration::from_millis(3));
						active.fetch_sub(1, Ordering::AcqRel);
					});
				};
				crate::utils::thread::join_sized(par, left, |p| side(0, p), |p| side(1, p));
			});
			assert_eq!(
				widths[0].load(Ordering::Relaxed) + widths[1].load(Ordering::Relaxed),
				6,
				"left = {left}"
			);
			assert!(peak.load(Ordering::Relaxed) <= 6, "left = {left}");
		}
	}
	#[test]
	fn a_team_of_one_behaves_exactly_like_seq() {
		use crate::utils::cost::{par_pays, rate, ParThreshold};
		threadteam::Exec::serial()
			.team(|team| {
				let par = Par::Team(team);
				assert_eq!(par.degree(), 1);
				// no gate pays on a single participant, however large the work
				assert!(!par_pays::<f64>(&par, 1 << 40, rate::gemm_mac(), 1));
				assert!(!crate::utils::cost::join_pays::<f64>(&par, 1 << 40, rate::gemm_mac(), 1));
				assert!(!ParThreshold::Derived.region_pays::<f64>(par.shape(), 1 << 40, rate::gemm_mac(), 1));
				// and the product equals the sequential one, with no team section at all
				let a = rand_mat(200, 150, 1);
				let b = rand_mat(150, 180, 2);
				let mut want = Mat::<f64>::zeros(200, 180);
				let mut got = Mat::<f64>::zeros(200, 180);
				matmul(want.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), 1.0, Par::Seq);
				stats::enable(true);
				let _ = stats::take();
				matmul(got.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), 1.0, par);
				let st = stats::take();
				stats::enable(false);
				assert_eq!(max_diff(want.as_ref(), got.as_ref()), 0.0);
				assert_eq!(st.gemm_team_kernel + st.gemm_tiled, 0, "{st:?}");
			})
			.unwrap();
	}

	/// The triangular products (the Cholesky trailing update) used to pass the team size to the
	/// `private_gemm_x86` kernel, outside the team. A team that pays now recurses on the team,
	/// its rectangular blocks reaching `gemm_tiled`.
	#[test]
	fn triangular_products_on_the_team_match_sequential_and_stay_on_it() {
		use crate::linalg::matmul::triangular::{matmul as tri_matmul, BlockStructure};
		let (n, k) = (768, 768);
		let a = rand_mat(n, k, 5);
		let b = rand_mat(k, n, 6);
		let lower = |par: Par| {
			let mut c = rand_mat(n, n, 7);
			tri_matmul(c.as_mut(), BlockStructure::TriangularLower, Accum::Add, a.as_ref(), BlockStructure::Rectangular, b.as_ref(), BlockStructure::Rectangular, 0.5, par);
			c
		};
		let upper = |par: Par| {
			let mut c = rand_mat(n, n, 8);
			tri_matmul(c.as_mut(), BlockStructure::TriangularUpper, Accum::Replace, a.as_ref(), BlockStructure::Rectangular, b.as_ref(), BlockStructure::Rectangular, 1.0, par);
			c
		};
		let (want_l, want_u) = (lower(Par::Seq), upper(Par::Seq));
		let ((got_l, got_u), st) = counted(6, |mut par| (lower(par.reborrow()), upper(par.reborrow())));
		let tri = |m: &Mat<f64>, w: &Mat<f64>, lo: bool| {
			let mut d: f64 = 0.0;
			for j in 0..n {
				for i in 0..n {
					if (lo && i >= j) || (!lo && i <= j) {
						d = d.max((m[(i, j)] - w[(i, j)]).abs());
					}
				}
			}
			d
		};
		assert!(tri(&got_l, &want_l, true) < 1e-9);
		assert!(tri(&got_u, &want_u, false) < 1e-9);
		// the work ran as team sections, not as an outside job
		assert!(st.gemm_tiled > 0 || st.gemm_team_kernel > 0 || st.split.iter().skip(2).any(|&c| c > 0), "{st:?}");
	}

	/// `spicy_matmul` (LDLT, supernodal Cholesky) on a team token: one thread per participant
	/// through the x86 kernel, the same result as sequential.
	#[test]
	fn spicy_matmul_on_a_team_matches_sequential() {
		use crate::linalg::matmul::internal::spicy_matmul;
		use crate::linalg::matmul::triangular::BlockStructure;
		if !crate::linalg::matmul::internal::has_spicy_matmul::<f64>() {
			return;
		}
		let (n, k) = (768, 512);
		let a = rand_mat(n, k, 9);
		let b = rand_mat(k, n, 10);
		let run = |par: Par| {
			let mut c = rand_mat(n, n, 11);
			spicy_matmul::<usize, f64>(c.as_mut(), BlockStructure::TriangularLower, None, None, Accum::Add, a.as_ref(), crate::Conj::No, b.as_ref(), crate::Conj::No, None, 0.5, par, dyn_stack::MemStack::new(&mut []));
			c
		};
		let want = run(Par::Seq);
		let got = with_par(6, |par| run(par));
		let mut d: f64 = 0.0;
		for j in 0..n {
			for i in j..n {
				d = d.max((want[(i, j)] - got[(i, j)]).abs());
			}
		}
		assert!(d < 1e-9, "{d}");
	}

	/// `gemm_tiled` (cut into tiles, through the x86 kernel where there is one) against the `gemm` crate, for the four dtypes and
	/// both accumulation modes.
	#[test]
	fn gemm_tiled_matches_the_gemm_crate_for_every_dtype() {
		use num_complex::Complex;
		#[cfg(all(target_arch = "x86_64", feature = "std"))]
		assert_eq!(
			x86_kernel::<f64>().is_some(),
			std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
		);
		fn check<E: 'static + Copy + Send + Sync + core::fmt::Debug>(
			par: &mut Par<'_>,
			one: E,
			from: impl Fn(f64) -> E,
			err: impl Fn(E, E) -> f64,
			cplx: bool,
		) {
			let (m, n, k) = (301, 263, 190);
			let gen = |len: usize, seed: u64| -> Vec<E> {
				let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
				(0..len)
					.map(|_| {
						x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
						from(((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5)
					})
					.collect()
			};
			let (a, b) = (gen(m * k, 1), gen(k * n, 2));
			let zero = from(0.0);
			let alpha = from(1.25);
			for read_dst in [false, true] {
				for conj in [false, true] {
					if conj && !cplx {
						continue;
					}
					let base = gen(m * n, 3);
					let (mut want, mut got) = (base.clone(), base);
					let acc = if read_dst { one } else { zero };
					unsafe {
						gemm::gemm(
							m, n, k, want.as_mut_ptr(), m as isize, 1, read_dst,
							a.as_ptr(), m as isize, 1, b.as_ptr(), k as isize, 1,
							acc, alpha, false, conj, conj, gemm::Parallelism::None,
						);
						let Par::Team(team) = par.reborrow() else { unreachable!() };
						gemm_tiled::<E>(
							team, m, n, k, 1e3, got.as_mut_ptr(), m as isize, 1, read_dst,
							a.as_ptr(), m as isize, 1, b.as_ptr(), k as isize, 1,
							acc, alpha, conj, conj,
						);
					}
					let d = want.iter().zip(&got).map(|(&w, &g)| err(w, g)).fold(0.0, f64::max);
					assert!(d < 1e-3, "read_dst {read_dst} conj {conj}: {d}");
				}
			}
		}
		with_par(5, |mut par| {
			check::<f64>(&mut par, 1.0, |x| x, |w, g| (w - g).abs(), false);
			check::<f32>(&mut par, 1.0, |x| x as f32, |w, g| (w - g).abs() as f64, false);
			check::<Complex<f64>>(&mut par, Complex::new(1.0, 0.0), |x| Complex::new(x, 0.5 - x), |w, g| (w - g).norm(), true);
			check::<Complex<f32>>(&mut par, Complex::new(1.0, 0.0), |x| Complex::new(x as f32, 0.5 - x as f32), |w, g| (w - g).norm() as f64, true);
		});
		let (_, st) = counted(5, |mut par| check::<f64>(&mut par, 1.0, |x| x, |w, g| (w - g).abs(), false));
		assert!(st.gemm_tiled >= 2 && st.gemm_below_floor == 0, "{st:?}");
	}

	/// Records the thread of every participant of the fork it wraps.
	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	struct Recording<'a> {
		inner: &'a dyn private_gemm_x86::Fork,
		seen: std::sync::Mutex<Vec<std::thread::ThreadId>>,
		runs: AtomicUsize,
	}

	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	impl private_gemm_x86::Fork for Recording<'_> {
		fn width(&self) -> usize {
			self.inner.width()
		}
		fn run(&self, f: &(dyn Fn(usize) + Sync)) {
			self.runs.fetch_add(1, Ordering::Relaxed);
			let seen = &self.seen;
			self.inner.run(&|j| {
				seen.lock().unwrap().push(std::thread::current().id());
				f(j)
			});
		}
	}

	/// The team route starts no thread outside the team: every participant the kernel forks on
	/// is the caller or a member of the team token (whole team and a narrower sub-team), the
	/// kernel forks twice for a product that packs (once for one that does not), and the result
	/// is the sequential one.
	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	#[test]
	fn the_kernel_forks_only_on_members_of_the_team() {
		use private_gemm_x86::{gemm_with, Accum as KAccum, DType, DstKind, IType, InstrSet};
		let Some((feat, _)) = super::kernel::available::<f64>() else { return };
		let _ = InstrSet::Avx256;
		let (m, n, k) = (400usize, 360usize, 300usize);
		let a = rand_mat(m, k, 1);
		let b = rand_mat(k, n, 2);
		let mut want = Mat::<f64>::zeros(m, n);
		matmul(want.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), 1.0, Par::Seq);
		let _turn = TEAM_TESTS.lock().unwrap_or_else(|e| e.into_inner());
		let team = team_of(6);
		let mut exec = team.exec(6);
		exec.team(|mut token| {
			// the threads of the members
			let members = std::sync::Mutex::new(Vec::new());
			token.reborrow().run(|_, _| members.lock().unwrap().push(std::thread::current().id()));
			let members = members.into_inner().unwrap();
			assert_eq!(members.len(), 6);
			for width in [6usize, 3, 2] {
				let mut got = Mat::<f64>::zeros(m, n);
				let alpha = 1.0f64;
				let rec = std::cell::RefCell::new(None);
				super::kernel::with_fork(token.reborrow(), width, |fork| {
					assert_eq!(fork.width(), width);
					let recording = Recording { inner: fork, seen: Default::default(), runs: AtomicUsize::new(0) };
					unsafe {
						gemm_with(
							DType::F64, IType::U64, feat, m, n, k,
							got.as_mut().as_ptr_mut() as *mut (), 1, got.col_stride(),
							core::ptr::null(), core::ptr::null(), DstKind::Full, KAccum::Replace,
							a.as_ref().as_ptr() as *const (), 1, a.col_stride(), false,
							core::ptr::null(), 0,
							b.as_ref().as_ptr() as *const (), 1, b.col_stride(), false,
							&raw const alpha as *const (),
							&recording,
						);
					}
					*rec.borrow_mut() = Some((recording.seen.into_inner().unwrap(), recording.runs.into_inner()));
				});
				let (seen, runs) = rec.into_inner().unwrap();
				assert!(runs >= 1, "width {width}: the kernel never forked");
				assert_eq!(seen.len(), runs * width, "width {width}");
				assert!(seen.iter().all(|t| members.contains(t)), "width {width}: a participant outside the team");
				let distinct: std::collections::HashSet<_> = seen.iter().collect();
				assert_eq!(distinct.len(), width, "width {width}: seen {} entries over {runs} runs", seen.len());
				assert!(max_diff(want.as_ref(), got.as_ref()) < 1e-10, "width {width}");
			}
		})
		.ok()
		.expect("a team token");
	}

	/// Every route that reaches the kernel on a team agrees with sequential, and the counters
	/// show the kernel ran on the team (not `gemm_tiled`, not an outside runtime).
	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	#[test]
	fn team_kernel_products_match_sequential_for_every_dtype() {
		use num_complex::Complex;
		if super::kernel::available::<f64>().is_none() {
			return;
		}
		fn check<T: crate::ComplexField + Copy>(from: impl Fn(f64, f64) -> T, err: impl Fn(T, T) -> f64) {
			let (m, n, k) = (333usize, 287usize, 211usize);
			let gen = |r: usize, c: usize, seed: u64| {
				let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
				Mat::<T>::from_fn(r, c, |_, _| {
					x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
					let u = (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
					from(u, 0.25 - u)
				})
			};
			let (a, b) = (gen(m, k, 1), gen(k, n, 2));
			let alpha = from(0.75, -0.25);
			for beta in [Accum::Replace, Accum::Add] {
				let base = gen(m, n, 3);
				let (mut want, mut got) = (base.clone(), base);
				matmul(want.as_mut(), beta, a.as_ref(), b.as_ref(), alpha, Par::Seq);
				let ((), st) = counted(6, |par| matmul(got.as_mut(), beta, a.as_ref(), b.as_ref(), alpha, par));
				let mut d = 0.0f64;
				for j in 0..n {
					for i in 0..m {
						d = d.max(err(want[(i, j)], got[(i, j)]));
					}
				}
				assert!(d < 1e-3, "{beta:?}: {d}");
				assert_eq!(st.gemm_team_kernel, 1, "{st:?}");
				assert_eq!(st.gemm_tiled, 0, "{st:?}");
			}
		}
		check::<f64>(|x, _| x, |w, g| (w - g).abs());
		check::<f32>(|x, _| x as f32, |w, g| (w - g).abs() as f64);
		check::<Complex<f64>>(|x, y| Complex::new(x, y), |w, g| (w - g).norm());
		check::<Complex<f32>>(|x, y| Complex::new(x as f32, y as f32), |w, g| (w - g).norm() as f64);
	}

	/// `spicy_matmul` and the triangular product on a team (kernel, lower and upper, indexed
	/// destination) and the factorizations built on them: Cholesky (both), LDLT, QR and LU on a
	/// team equal their sequential results, and the kernel ran on the team.
	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	#[test]
	fn team_kernel_triangular_indexed_and_factorizations_match_sequential() {
		use crate::linalg::matmul::internal::spicy_matmul;
		use crate::linalg::matmul::triangular::{matmul as tri_matmul, BlockStructure};
		if !crate::linalg::matmul::internal::has_spicy_matmul::<f64>() {
			return;
		}
		let (n, k) = (700usize, 500usize);
		let a = rand_mat(n, k, 21);
		let b = rand_mat(k, n, 22);
		let tri_diff = |x: &Mat<f64>, y: &Mat<f64>, lower: bool| {
			let mut d: f64 = 0.0;
			for j in 0..n {
				for i in 0..n {
					if (lower && i >= j) || (!lower && i <= j) {
						d = d.max((x[(i, j)] - y[(i, j)]).abs());
					}
				}
			}
			d
		};
		// spicy_matmul: both triangles, plain
		for (block, lower) in [(BlockStructure::TriangularLower, true), (BlockStructure::TriangularUpper, false)] {
			let run = |par: Par| {
				let mut c = rand_mat(n, n, 23);
				spicy_matmul::<usize, f64>(c.as_mut(), block, None, None, Accum::Add, a.as_ref(), crate::Conj::No, b.as_ref(), crate::Conj::No, None, 0.5, par, dyn_stack::MemStack::new(&mut []));
				c
			};
			let want = run(Par::Seq);
			let (got, st) = counted(6, run);
			assert!(tri_diff(&want, &got, lower) < 1e-9);
			assert_eq!(st.gemm_team_kernel, 1, "{st:?}");
		}
		// spicy_matmul: gathered rows and columns into a larger destination (a scatter)
		{
			let rows: Vec<usize> = (0..n).map(|i| (i * 3) % (3 * n - 1) + (i % 2)).collect::<std::collections::BTreeSet<_>>().into_iter().take(n).collect();
			let cols: Vec<usize> = (0..n).map(|j| 2 * j).collect();
			let (nr, nc) = (rows.iter().max().unwrap() + 1, 2 * n);
			let m_rows = rows.len();
			let a2 = rand_mat(m_rows, k, 24);
			let run = |par: Par| {
				let mut c = rand_mat(nr, nc, 25);
				spicy_matmul::<usize, f64>(c.as_mut(), BlockStructure::Rectangular, Some(&rows), Some(&cols), Accum::Add, a2.as_ref(), crate::Conj::No, b.as_ref(), crate::Conj::No, None, 0.5, par, dyn_stack::MemStack::new(&mut []));
				c
			};
			let want = run(Par::Seq);
			let (got, st) = counted(6, run);
			assert!(max_diff(want.as_ref(), got.as_ref()) < 1e-9);
			assert_eq!(st.gemm_team_kernel, 1, "{st:?}");
		}
		// the triangular product into a triangle
		{
			let run = |par: Par| {
				let mut c = rand_mat(n, n, 26);
				tri_matmul(c.as_mut(), BlockStructure::TriangularLower, Accum::Add, a.as_ref(), BlockStructure::Rectangular, b.as_ref(), BlockStructure::Rectangular, 0.5, par);
				c
			};
			let want = run(Par::Seq);
			let (got, st) = counted(6, run);
			assert!(tri_diff(&want, &got, true) < 1e-9);
			assert!(st.gemm_team_kernel >= 1, "{st:?}");
		}
		// factorizations
		let n = 640usize;
		let g = rand_mat(n, n, 30);
		let spd = {
			let mut s = Mat::<f64>::zeros(n, n);
			matmul(s.as_mut(), Accum::Replace, g.transpose(), g.as_ref(), 1.0, Par::Seq);
			for i in 0..n {
				s[(i, i)] += n as f64;
			}
			s
		};
		let lower_diff = |x: &Mat<f64>, y: &Mat<f64>| {
			let mut d: f64 = 0.0;
			for j in 0..n {
				for i in j..n {
					d = d.max((x[(i, j)] - y[(i, j)]).abs());
				}
			}
			d
		};
		let llt = |par: Par| {
			let mut f = spd.clone();
			let mut mem = dyn_stack::MemBuffer::new(crate::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<f64>(n, par.shape(), Default::default()));
			crate::linalg::cholesky::llt::factor::cholesky_in_place(f.as_mut(), Default::default(), par, dyn_stack::MemStack::new(&mut mem), Default::default()).unwrap();
			f
		};
		let (got, st) = counted(6, llt);
		assert!(lower_diff(&llt(Par::Seq), &got) < 1e-8, "llt");
		assert!(st.gemm_team_kernel > 0, "llt: {st:?}");
		let ldlt = |par: Par| {
			let mut f = spd.clone();
			let mut mem = dyn_stack::MemBuffer::new(crate::linalg::cholesky::ldlt::factor::cholesky_in_place_scratch::<f64>(n, par.shape(), Default::default()));
			crate::linalg::cholesky::ldlt::factor::cholesky_in_place(
				f.as_mut(),
				crate::linalg::cholesky::ldlt::factor::LdltRegularization { dynamic_regularization_signs: None, dynamic_regularization_delta: 0.0, dynamic_regularization_epsilon: 0.0 },
				par,
				dyn_stack::MemStack::new(&mut mem),
				Default::default(),
			)
			.unwrap();
			f
		};
		let (got, st) = counted(6, ldlt);
		assert!(lower_diff(&ldlt(Par::Seq), &got) < 1e-8, "ldlt");
		assert!(st.gemm_team_kernel > 0, "ldlt: {st:?}");
		let qr = |par: Par| {
			let mut f = g.clone();
			let bs = crate::linalg::qr::no_pivoting::factor::recommended_block_size::<f64>(n, n);
			let mut h = Mat::<f64>::zeros(bs, n);
			let mut mem = dyn_stack::MemBuffer::new(crate::linalg::qr::no_pivoting::factor::qr_in_place_scratch::<f64>(n, n, bs, par.shape(), Default::default()));
			crate::linalg::qr::no_pivoting::factor::qr_in_place(f.as_mut(), h.as_mut(), par, dyn_stack::MemStack::new(&mut mem), Default::default());
			f
		};
		// the panel products of a 640 QR are below the floor, so no counter claim here: the
		// kernel's own coverage is the plain, triangular, indexed, Cholesky, LDLT and LU rows
		let got = with_par(6, qr);
		assert!(max_diff(qr(Par::Seq).as_ref(), got.as_ref()) < 1e-8, "qr");
		let lu = |par: Par| {
			let mut f = g.clone();
			let (mut perm, mut perm_inv) = (vec![0usize; n], vec![0usize; n]);
			let mut mem = dyn_stack::MemBuffer::new(crate::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, f64>(n, n, par.shape(), Default::default()));
			crate::linalg::lu::partial_pivoting::factor::lu_in_place(f.as_mut(), &mut perm, &mut perm_inv, par, dyn_stack::MemStack::new(&mut mem), Default::default());
			f
		};
		let (got, st) = counted(6, lu);
		assert!(max_diff(lu(Par::Seq).as_ref(), got.as_ref()) < 1e-8, "lu");
		assert!(st.gemm_team_kernel > 0, "lu: {st:?}");
	}

	/// QR at a size whose block-reflector products are above the gate: they run on the kernel.
	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	#[test]
	fn team_kernel_qr_matches_sequential() {
		if super::kernel::available::<f64>().is_none() {
			return;
		}
		let n = 1400usize;
		let g = rand_mat(n, n, 31);
		let qr = |par: Par| {
			let mut f = g.clone();
			let bs = crate::linalg::qr::no_pivoting::factor::recommended_block_size::<f64>(n, n);
			let mut h = Mat::<f64>::zeros(bs, n);
			let mut mem = dyn_stack::MemBuffer::new(crate::linalg::qr::no_pivoting::factor::qr_in_place_scratch::<f64>(n, n, bs, par.shape(), Default::default()));
			crate::linalg::qr::no_pivoting::factor::qr_in_place(f.as_mut(), h.as_mut(), par, dyn_stack::MemStack::new(&mut mem), Default::default());
			f
		};
		let (got, st) = counted(6, qr);
		assert!(max_diff(qr(Par::Seq).as_ref(), got.as_ref()) < 1e-7, "qr");
		assert!(st.gemm_team_kernel > 0, "qr: {st:?}");
	}

	/// The high level methods take the parallelism as an argument (there is no process wide
	/// setting): `singular_values(Par::Team(..))` runs on the team and matches the sequential one.
	#[test]
	fn high_level_singular_values_run_on_the_team() {
		let n = 384;
		let a = rand_mat(n, n, 21);
		let want = a.as_ref().singular_values(Par::Seq).unwrap();
		let (got, st) = counted(4, |par| a.as_ref().singular_values(par).unwrap());
		assert_eq!(want.len(), got.len());
		for (w, g) in want.iter().zip(&got) {
			assert!((w - g).abs() < 1e-9 * want[0], "{w} {g}");
		}
		let opened = st.run.iter().skip(2).sum::<usize>() + st.split.iter().skip(2).sum::<usize>();
		assert!(opened > 0 || st.gemm_tiled > 0 || st.gemm_team_kernel > 0, "no team section: {st:?}");
	}

	/// A solver struct built with a team and used with a team: the work happens where `par` is given.
	#[test]
	fn high_level_qr_and_lu_solve_on_the_team() {
		use crate::linalg::solvers::Solve;
		let n = 320;
		let a = rand_mat(n, n, 22);
		let b = rand_mat(n, 8, 23);
		let want = a.as_ref().partial_piv_lu(Par::Seq).solve(b.as_ref(), Par::Seq);
		let (got, st) = counted(4, |mut par| {
			let lu = a.as_ref().partial_piv_lu(par.reborrow());
			lu.solve(b.as_ref(), par)
		});
		assert!(max_diff(want.as_ref(), got.as_ref()) < 1e-8);
		let opened = st.run.iter().skip(2).sum::<usize>() + st.split.iter().skip(2).sum::<usize>();
		assert!(opened > 0 || st.gemm_tiled > 0 || st.gemm_team_kernel > 0, "no team section: {st:?}");
		let want = a.as_ref().qr(Par::Seq).solve(b.as_ref(), Par::Seq);
		let got = with_par(4, |mut par| {
			let qr = a.as_ref().qr(par.reborrow());
			qr.solve(b.as_ref(), par)
		});
		assert!(max_diff(want.as_ref(), got.as_ref()) < 1e-8);
	}
}
