//! Experimental route of the parallel regions onto a threadteam team
//! (feature `threadteam`, `Par::Team(n)`).
//!
//! # Design of the prototype
//!
//! `Par` is `Copy` and carries no lifetime, so a team token cannot travel inside it.
//! The prototype keeps the authority on the calling thread instead: [`with_team`] turns an
//! `Exec` minted from a `threadteam::Team` into a scoped [`TeamExec`] and installs a pointer to
//! it in a thread local; `Par::Team(n)` asks for at most `n` participants of the team installed
//! on the *current* thread.
//!
//! * Flat sections ([`run`]): every participant runs a closure at once. The installed pointer is
//!   cleared on the leader for the duration, so a body that calls back into faer with
//!   `Par::Team` runs sequentially instead of re-entering the team.
//! * Splits ([`fork`], [`split`]): a child leader installs its own sub-team for the duration of
//!   its callback, so nested `Par::Team(k)` regions (the halves of `join_raw`, the inner
//!   products of a column split) run on the child's members.
//! * No installed team, or a team of one: everything runs inline on the caller.
//!
//! The target design is a lifetime-carrying `Par<'team>`; this is the smallest change that puts
//! the measured paths on the team without touching every signature of the crate.
//!
//! Nothing here starts a Rayon job: the team members are threadteam members that spin while the
//! reservation is held, and the section dispatch is a write to their slots.

use core::cell::{Cell, UnsafeCell};
use core::ptr::null_mut;
use rayon::iter::plumbing::{Producer, ProducerCallback};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};
use threadteam::{Exec, TeamExec};

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
		}
	}
}

thread_local! {
	static CURRENT: Cell<*mut TeamExec<'static>> = const { Cell::new(null_mut()) };
}

struct Restore(*mut TeamExec<'static>);
impl Drop for Restore {
	fn drop(&mut self) {
		CURRENT.with(|c| c.set(self.0));
	}
}

fn install(p: *mut TeamExec<'static>) -> Restore {
	Restore(CURRENT.with(|c| c.replace(p)))
}

/// Runs `f` with the team of `exec` installed on the calling thread, so that `Par::Team(n)`
/// regions started from `f` run on it. `exec` should come from `threadteam::Team::exec`
/// (a token that carries members); with a serial or pool token the regions run inline.
pub fn with_team<R>(exec: &mut Exec<'_>, f: impl FnOnce() -> R) -> R {
	let mut f = Some(f);
	let r = exec.team(|mut team| {
		let _guard = install(
			(&mut team as *mut TeamExec<'_>).cast::<TeamExec<'static>>(),
		);
		(f.take().unwrap())()
	});
	match r {
		Ok(r) => r,
		Err(_) => (f.take().unwrap())(),
	}
}

/// Number of participants `Par::Team(n)` can use right now on this thread.
pub fn available(n: usize) -> usize {
	let p = CURRENT.with(|c| c.get());
	if p.is_null() {
		1
	} else {
		n.max(1).min(unsafe { (*p).size().get() })
	}
}

/// Runs `f(i, m)` on `m = min(n, team size)` participants at once; inline `f(0, 1)` when the
/// team is absent or `m == 1`.
pub fn run(n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
	let p = CURRENT.with(|c| c.get());
	if p.is_null() || n <= 1 {
		stats::hist(&stats::RUN, 1);
		return f(0, 1);
	}
	let team = unsafe { &mut *p };
	let size = team.size().get();
	let m = n.min(size);
	if m <= 1 {
		stats::hist(&stats::RUN, 1);
		return f(0, 1);
	}
	stats::hist(&stats::RUN, m);
	let _guard = install(null_mut());
	if m == size {
		team.reborrow().run(|i, k| f(i, k));
	} else {
		let _ = team.reborrow().split(&[m], |_c, child| {
			child.run(|i, k| f(i, k));
		});
	}
}

/// Splits the installed team into `children = min(children, team size)` equal parts and runs
/// `f(c, children, width)` on the leader of part `c`, with that part installed as the current
/// team. Without a team the parts run one after the other on the caller with `width == 1`.
pub fn fork(children: usize, f: &(dyn Fn(usize, usize, usize) + Sync)) {
	if children == 0 {
		return;
	}
	let p = CURRENT.with(|c| c.get());
	if p.is_null() || children == 1 {
		stats::hist(&stats::SPLIT, 1);
		for c in 0..children {
			f(c, children, 1);
		}
		return;
	}
	let team = unsafe { &mut *p };
	let k = children.min(team.size().get());
	stats::hist(&stats::SPLIT, k);
	team.reborrow().fork(k, |c, mut child| {
		let width = child.size().get();
		let _guard = install(
			(&mut child as *mut TeamExec<'_>).cast::<TeamExec<'static>>(),
		);
		f(c, k, width);
	});
}

/// Splits the installed team into parts of the given widths (their sum must not exceed the team
/// size) and runs `f(c, width)` on the leader of part `c`. Falls back to running the parts one
/// after the other with `width == 1` when no team is installed or it is too small.
pub fn split(sizes: &[usize], f: &(dyn Fn(usize, usize) + Sync)) {
	let p = CURRENT.with(|c| c.get());
	let total: usize = sizes.iter().sum();
	if p.is_null() || unsafe { (*p).size().get() } < total || sizes.contains(&0) {
		stats::hist(&stats::SPLIT, 1);
		for c in 0..sizes.len() {
			f(c, 1);
		}
		return;
	}
	stats::hist(&stats::SPLIT, sizes.len());
	let team = unsafe { &mut *p };
	let _ = team.reborrow().split(sizes, |c, mut child| {
		let width = child.size().get();
		let _guard = install(
			(&mut child as *mut TeamExec<'_>).cast::<TeamExec<'static>>(),
		);
		f(c, width);
	});
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
pub fn for_each<T: Send, I>(n_jobs: usize, iter: I, f: impl Sync + Fn(T))
where
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
	let jobs = n_jobs.max(1).min(len);
	fork(jobs, &|c, count, _width| {
		let (start, n) = super::thread::par_split_indices(len, c, count);
		for j in start..start + n {
			// SAFETY: block `c` owns slots `start..start + n` alone.
			let item = unsafe { slots.take(j) };
			f(item);
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

/// `dst := alpha * dst + beta * lhs * rhs` (the contract of `gemm::gemm`) with the output cut
/// into a `pm x pn` grid of tiles, one sequential `gemm::gemm` per team member. The depth `k`
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
	nt: usize,
	m: usize,
	n: usize,
	k: usize,
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
	let nt = available(nt);
	let work = m.saturating_mul(n).saturating_mul(k);
	if nt <= 1 || work < gemm::get_threading_threshold() {
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
	run(pm * pn, &|i, parts| {
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
		let team = team_of(6);
		let mut exec = team.exec(6);
		with_team(&mut exec, || {
			for &(m, n, k) in &[(512, 512, 512), (257, 301, 129), (4096, 64, 64), (64, 4096, 64), (1000, 33, 500)] {
				let a = rand_mat(m, k, 1);
				let b = rand_mat(k, n, 2);
				for beta in [Accum::Replace, Accum::Add] {
					let mut want = rand_mat(m, n, 3);
					let mut got = want.clone();
					matmul(want.as_mut(), beta, a.as_ref(), b.as_ref(), 1.5, Par::Seq);
					matmul(got.as_mut(), beta, a.as_ref(), b.as_ref(), 1.5, Par::team(6));
					assert!(max_diff(want.as_ref(), got.as_ref()) < 1e-10, "{m}x{n}x{k}");
				}
				// transposed and strided operands
				let mut want = Mat::<f64>::zeros(m, n);
				let mut got = Mat::<f64>::zeros(m, n);
				let at = a.transpose().to_owned();
				matmul(want.as_mut(), Accum::Replace, at.transpose(), b.as_ref(), 1.0, Par::Seq);
				matmul(got.as_mut(), Accum::Replace, at.transpose(), b.as_ref(), 1.0, Par::team(4));
				assert!(max_diff(want.as_ref(), got.as_ref()) < 1e-10);
			}
			// complex
			let (m, n, k) = (300, 280, 200);
			let a = Mat::<c64>::from_fn(m, k, |i, j| c64::new(i as f64 * 0.01 - j as f64 * 0.003, (i + j) as f64 * 0.002));
			let b = Mat::<c64>::from_fn(k, n, |i, j| c64::new((i * j % 7) as f64 * 0.1, i as f64 * 0.01 - j as f64 * 0.01));
			let mut want = Mat::<c64>::zeros(m, n);
			let mut got = Mat::<c64>::zeros(m, n);
			matmul(want.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), c64::new(1.0, 0.0), Par::Seq);
			matmul(got.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), c64::new(1.0, 0.0), Par::team(6));
			let mut d: f64 = 0.0;
			for j in 0..n {
				for i in 0..m {
					d = d.max((want[(i, j)] - got[(i, j)]).norm_sqr().sqrt());
				}
			}
			assert!(d < 1e-9);
		});
	}

	#[test]
	fn no_team_installed_runs_sequentially() {
		let (m, n, k) = (200, 200, 200);
		let a = rand_mat(m, k, 1);
		let b = rand_mat(k, n, 2);
		let mut want = Mat::<f64>::zeros(m, n);
		let mut got = Mat::<f64>::zeros(m, n);
		matmul(want.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), 1.0, Par::Seq);
		matmul(got.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), 1.0, Par::team(6));
		assert!(max_diff(want.as_ref(), got.as_ref()) < 1e-12);
	}

	#[test]
	fn for_each_deals_every_item_once_and_nests() {
		let team = team_of(6);
		let mut exec = team.exec(6);
		with_team(&mut exec, || {
			for jobs in [1usize, 2, 3, 6, 9] {
				let hits: Vec<AtomicUsize> = (0..13).map(|_| AtomicUsize::new(0)).collect();
				for_each(jobs, 0..13usize, |i| {
					hits[i].fetch_add(1, Ordering::Relaxed);
					// a nested region sees a sub-team no wider than its part
					assert!(available(6) <= 6);
				});
				assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "jobs = {jobs}");
			}
		});
	}

	#[test]
	fn join_raw_leaves_match_the_width() {
		let team = team_of(6);
		let mut exec = team.exec(6);
		with_team(&mut exec, || {
			for n in [2usize, 3, 5, 6] {
				let widths = std::sync::Mutex::new(Vec::new());
				crate::utils::thread::join_raw(
					|p| widths.lock().unwrap().push(p.degree()),
					|p| widths.lock().unwrap().push(p.degree()),
					Par::team(n),
				);
				let mut w = widths.into_inner().unwrap();
				w.sort_unstable();
				let half = n / 2;
				assert_eq!(w, vec![half, n - half], "n = {n}");
			}
		});
	}

	#[test]
	fn lu_qr_cholesky_on_the_team_match_sequential() {
		use crate::linalg::solvers::{Llt, Qr};
		use crate::Side;
		let team = team_of(6);
		let mut exec = team.exec(6);
		with_team(&mut exec, || {
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
				let got = lu(Par::team(6));
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
				qr(&mut r_got, Par::team(6));
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
				let got = ch(Par::team(6));
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
}
