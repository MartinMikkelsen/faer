#![allow(missing_docs)]
/// compile-time bound-checked indexing types
pub mod bound;
/// the parallel rule: dispatch cost of a shape against the serial time of a region
pub mod cost;
/// experimental route of the parallel regions onto a threadteam team
#[cfg(feature = "threadteam")]
pub mod team;
#[doc(hidden)]
pub mod thread {
	use crate::*;
	/// executes the two operations, possibly in parallel, while splitting the
	/// amount of parallelism between the two
	#[inline]
	pub fn join_raw(
		op_a: impl Send + FnOnce(Par),
		op_b: impl Send + FnOnce(Par),
		parallelism: Par,
	) {
		fn implementation<'a>(
			op_a: &'a mut (dyn Send + FnMut(Par)),
			op_b: &'a mut (dyn Send + FnMut(Par)),
			mut parallelism: Par,
		) {
			match parallelism {
				Par::Seq => {
					(op_a(parallelism.reborrow()), op_b(parallelism));
				},
				#[cfg(feature = "rayon")]
				Par::Rayon(n_threads) => {
					let n_threads = n_threads.get();
					if n_threads == 1 {
						(op_a(Par::Seq), op_b(Par::Seq));
					} else {
						// the two halves add up to `n_threads`, so the leaves of
						// a recursive split never exceed the requested width
						let par = |n: usize| {
							Par::Rayon(core::num::NonZeroUsize::new(n).unwrap())
						};
						let half = n_threads / 2;
						spindle::for_each(
							2,
							[(op_a, par(n_threads - half)), (op_b, par(half))],
							|(op, parallelism)| op(parallelism),
						);
					}
				},
				#[cfg(feature = "threadteam")]
				Par::Team(team) => {
					let n_threads = team.size().get();
					if n_threads == 1 {
						(op_a(Par::Team(team)), op_b(Par::Seq));
					} else {
						// same split as the Rayon route; each half runs on the leader of
						// a sub-team of its width and owns that sub-team
						let half = n_threads / 2;
						let ops = [
							std::sync::Mutex::new(op_a),
							std::sync::Mutex::new(op_b),
						];
						super::team::note_split(2);
						let _ = team.split(&[n_threads - half, half], |c, child| {
							(ops[c].lock().unwrap())(Par::Team(child))
						});
					}
				},
			};
		}
		let mut op_a = Some(op_a);
		let mut op_b = Some(op_b);
		implementation(
			&mut |parallelism| op_a.take().unwrap()(parallelism),
			&mut |parallelism| op_b.take().unwrap()(parallelism),
			parallelism,
		)
	}
	/// unsafe [`Send`] and [`Sync`] pointer type
	pub struct Ptr<T>(pub *mut T);
	unsafe impl<T> Send for Ptr<T> {}
	unsafe impl<T> Sync for Ptr<T> {}
	impl<T> Copy for Ptr<T> {}
	impl<T> Clone for Ptr<T> {
		#[inline]
		fn clone(&self) -> Self {
			*self
		}
	}
	/// the amount of threads that should ideally execute an operation with the
	/// given parallelism
	#[inline]
	pub fn parallelism_degree(parallelism: &Par) -> usize {
		match parallelism {
			Par::Seq => 1,
			#[cfg(feature = "rayon")]
			Par::Rayon(n_threads) => n_threads.get(),
			#[cfg(feature = "threadteam")]
			Par::Team(team) => team.size().get(),
			#[cfg(not(feature = "threadteam"))]
			Par::__Lifetime(never, _) => match *never {},
		}
	}

	/// whether `par` is a `Team` token
	#[inline]
	pub fn is_team(parallelism: &Par) -> bool {
		match parallelism {
			#[cfg(feature = "threadteam")]
			Par::Team(_) => true,
			_ => false,
		}
	}

	/// `spindle::for_each` on the backend of `par`: the spindle group for `Rayon`, the team
	/// itself for `Team`, the caller for `Seq`
	#[cfg(feature = "rayon")]
	pub fn for_each<
		T: Send,
		I: rayon::iter::IntoParallelIterator<Item = T>,
	>(
		par: Par,
		n_jobs: usize,
		iter: I,
		f: impl Sync + Fn(T),
	) where
		I::Iter: rayon::iter::IndexedParallelIterator,
	{
		match par {
			Par::Seq => {
				use rayon::iter::plumbing::{Producer, ProducerCallback};
				struct Each<F>(F);
				impl<T, F: Fn(T)> ProducerCallback<T> for Each<F> {
					type Output = ();
					fn callback<P: Producer<Item = T>>(self, p: P) {
						p.into_iter().for_each(self.0);
					}
				}
				rayon::iter::IndexedParallelIterator::with_producer(
					iter.into_par_iter(),
					Each(f),
				);
			},
			Par::Rayon(_) => spindle::for_each(n_jobs, iter, f),
			#[cfg(feature = "threadteam")]
			Par::Team(team) => super::team::for_each(team, n_jobs, iter, f),
		}
	}
	/// `for_each` whose closure also receives the parallelism of its job: the share of `par`
	/// that one of `n_jobs` concurrent jobs may use (sequential below two threads)
	#[cfg(feature = "rayon")]
	pub fn for_each_with_par<
		T: Send,
		I: rayon::iter::IntoParallelIterator<Item = T>,
	>(
		par: Par,
		n_jobs: usize,
		iter: I,
		f: impl Sync + Fn(T, Par),
	) where
		I::Iter: rayon::iter::IndexedParallelIterator,
	{
		match par {
			Par::Seq => for_each(Par::Seq, n_jobs, iter, |item| f(item, Par::Seq)),
			Par::Rayon(n) => {
				let width = n.get() / n_jobs.max(1);
				spindle::for_each(n_jobs, iter, |item| {
					f(item, if width >= 2 { Par::rayon(width) } else { Par::Seq })
				})
			},
			#[cfg(feature = "threadteam")]
			Par::Team(team) => {
				// the item runs on the leader of its own sub-team and receives it
				super::team::for_each_child(team, n_jobs, iter, |item, child| {
					f(item, Par::Team(child))
				})
			},
		}
	}

	/// runs the two operations concurrently on a split of `par` in which the first receives
	/// `size_a` threads (clamped so that both get at least one) and the second the rest
	#[cfg(feature = "rayon")]
	pub fn join_sized(
		par: Par,
		size_a: usize,
		op_a: impl Send + FnOnce(Par),
		op_b: impl Send + FnOnce(Par),
	) {
		let degree = par.degree();
		if degree < 2 {
			op_a(Par::Seq);
			op_b(Par::Seq);
			return;
		}
		let size_a = size_a.clamp(1, degree - 1);
		let op_a = std::sync::Mutex::new(Some(op_a));
		let op_b = std::sync::Mutex::new(Some(op_b));
		split(par, &[size_a, degree - size_a], &|c, child| {
			if c == 0 {
				(op_a.lock().unwrap().take().unwrap())(child)
			} else {
				(op_b.lock().unwrap().take().unwrap())(child)
			}
		});
	}

	/// runs `f(c, child)` for every `c < sizes.len()` concurrently, where `child` is the part of
	/// `par` of width `sizes[c]` (sequential for a width of one). The widths must be non-zero and
	/// add up to at most the degree of `par`; the children never use more than they are handed.
	#[cfg(feature = "rayon")]
	pub fn split(par: Par, sizes: &[usize], f: &(dyn Sync + Fn(usize, Par))) {
		core::assert!(sizes.iter().all(|&s| s > 0));
		match par {
			Par::Seq => {
				for c in 0..sizes.len() {
					f(c, Par::Seq);
				}
			},
			Par::Rayon(n) => {
				core::assert!(sizes.iter().sum::<usize>() <= n.get());
				use rayon::iter::IntoParallelIterator;
				spindle::for_each(sizes.len(), (0..sizes.len()).into_par_iter(), |c| {
					f(c, if sizes[c] >= 2 { Par::rayon(sizes[c]) } else { Par::Seq })
				})
			},
			#[cfg(feature = "threadteam")]
			Par::Team(team) => {
				if sizes.iter().sum::<usize>() <= team.size().get() {
					super::team::note_split(sizes.len());
					let _ = team.split(sizes, |c, child| f(c, Par::Team(child)));
				} else {
					// the team is too small for the requested widths: run the parts in turn
					for c in 0..sizes.len() {
						f(c, Par::Seq);
					}
				}
			},
		}
	}

	/// returns the start and length of a subsegment of `0..n`, split between
	/// `chunk_count` consumers, for the consumer at index `idx`
	///
	/// for the same `n` and `chunk_count`, different values of `idx` between in
	/// `0..chunk_count` will represent distinct subsegments
	#[inline]
	pub fn par_split_indices(
		n: usize,
		idx: usize,
		chunk_count: usize,
	) -> (usize, usize) {
		let chunk_size = n / chunk_count;
		let rem = n % chunk_count;
		let idx_to_col_start = move |idx| {
			if idx < rem {
				idx * (chunk_size + 1)
			} else {
				rem + idx * chunk_size
			}
		};
		let start = idx_to_col_start(idx);
		let end = idx_to_col_start(idx + 1);
		(start, end - start)
	}

	#[cfg(all(test, feature = "rayon"))]
	mod tests {
		use super::{for_each, for_each_with_par, join_raw, join_sized, parallelism_degree};
		use crate::Par;
		use core::sync::atomic::{AtomicUsize, Ordering};

		struct Leaves {
			count: AtomicUsize,
			active: AtomicUsize,
			peak: AtomicUsize,
		}

		// splits until each branch has a single thread, then holds the
		// leaf long enough for the leaves to overlap
		fn split(parallelism: Par, leaves: &Leaves) {
			if parallelism_degree(&parallelism) > 1 {
				join_raw(
					|par| split(par, leaves),
					|par| split(par, leaves),
					parallelism,
				);
			} else {
				leaves.count.fetch_add(1, Ordering::Relaxed);
				let now = leaves.active.fetch_add(1, Ordering::AcqRel) + 1;
				leaves.peak.fetch_max(now, Ordering::AcqRel);
				std::thread::sleep(std::time::Duration::from_millis(5));
				leaves.active.fetch_sub(1, Ordering::AcqRel);
			}
		}

		#[test]
		fn join_raw_leaves_do_not_exceed_requested_width() {
			// the pool is wider than every requested width, so only the
			// split itself can bound the concurrency
			let pool = rayon::ThreadPoolBuilder::new()
				.num_threads(16)
				.build()
				.unwrap();
			for n in [2usize, 3, 5, 6, 7, 8] {
				let leaves = Leaves {
					count: AtomicUsize::new(0),
					active: AtomicUsize::new(0),
					peak: AtomicUsize::new(0),
				};
				pool.install(|| {
					split(
						Par::Rayon(core::num::NonZeroUsize::new(n).unwrap()),
						&leaves,
					)
				});
				assert_eq!(leaves.count.into_inner(), n, "leaves for n = {n}");
				assert!(
					leaves.peak.into_inner() <= n,
					"concurrent leaves exceed n = {n}"
				);
			}
		}

		// a region handed a `Par::Rayon` budget runs on the caller's held worker group. under a held
		// group every thread of the pool is inside it, so a region dispatched through a plain Rayon
		// iterator instead runs on the caller alone and the budget buys nothing: the block
		// Householder apply's row split was dispatched that way (4096x256 f64 at eight threads: 14.6
		// ms against 5.9). the wait makes this a statement about the dispatch and not about
		// scheduling: the caller's chunk cannot finish until a worker has entered one, so a
		// caller-alone dispatch runs into the deadline instead of passing by luck on a quiet machine
		#[test]
		fn for_each_region_reaches_the_held_group() {
			use rayon::prelude::*;
			use std::sync::atomic::{AtomicUsize, Ordering::Acquire};
			let pool = rayon::ThreadPoolBuilder::new()
				.num_threads(4)
				.build()
				.unwrap();
			let workers = AtomicUsize::new(0);
			let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
			pool.install(|| {
				spindle::with_lock(4, || {
					for_each(
						Par::Rayon(core::num::NonZeroUsize::new(4).unwrap()),
						4,
						(0..4).into_par_iter(),
						|_| {
							if spindle::current_thread_index().is_some() {
								workers.fetch_add(1, Ordering::AcqRel);
								return;
							}
							// the caller's own chunk: hold it until a worker starts one
							while workers.load(Ordering::Acquire) == 0
								&& std::time::Instant::now() < deadline
							{
								std::hint::spin_loop();
							}
						},
					);
				});
			});
			let n = workers.into_inner();
			assert!(
				n > 0,
				"no worker of the held group of 4 entered the region: it ran on the caller alone"
			);
		}

		// the two halves of a sized join run on the widths they are handed: their children
		// never overlap beyond the budget, and each half sees the width it asked for
		#[test]
		fn join_sized_children_stay_within_their_share() {
			use rayon::prelude::*;
			let pool = rayon::ThreadPoolBuilder::new()
				.num_threads(16)
				.build()
				.unwrap();
			for n in [2usize, 3, 5, 8] {
				for left in 0..=n + 1 {
					let active = AtomicUsize::new(0);
					let peak = AtomicUsize::new(0);
					let widths = [AtomicUsize::new(0), AtomicUsize::new(0)];
					let side = |which: usize, par: Par| {
						widths[which].store(parallelism_degree(&par), Ordering::Relaxed);
						let jobs = parallelism_degree(&par);
						for_each_with_par(par, jobs, (0..jobs).into_par_iter(), |_, _| {
							let now = active.fetch_add(1, Ordering::AcqRel) + 1;
							peak.fetch_max(now, Ordering::AcqRel);
							std::thread::sleep(std::time::Duration::from_millis(3));
							active.fetch_sub(1, Ordering::AcqRel);
						});
					};
					pool.install(|| {
						join_sized(
							Par::Rayon(core::num::NonZeroUsize::new(n).unwrap()),
							left,
							|par| side(0, par),
							|par| side(1, par),
						)
					});
					let total = widths[0].load(Ordering::Relaxed)
						+ widths[1].load(Ordering::Relaxed);
					assert_eq!(total, n, "n = {n}, left = {left}");
					assert!(
						peak.load(Ordering::Relaxed) <= n,
						"concurrent jobs exceed n = {n} with left = {left}"
					);
				}
			}
		}
	}
}
/// approximate comparators for testing purposes
pub mod approx;
/// simd helper utilities based on lifetime-bound indices
pub mod simd;
