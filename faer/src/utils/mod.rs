#![allow(missing_docs)]
/// compile-time bound-checked indexing types
pub mod bound;
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
			parallelism: Par,
		) {
			match parallelism {
				Par::Seq => {
					(op_a(parallelism), op_b(parallelism));
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
				Par::Team(n_threads) => {
					let n_threads = super::team::available(n_threads.get());
					if n_threads == 1 {
						(op_a(Par::Seq), op_b(Par::Seq));
					} else {
						// same split as the Rayon route; each half runs on the leader of
						// a sub-team of its width
						let half = n_threads / 2;
						let ops = [
							std::sync::Mutex::new(op_a),
							std::sync::Mutex::new(op_b),
						];
						super::team::split(&[n_threads - half, half], &|c, width| {
							(ops[c].lock().unwrap())(Par::Team(
								core::num::NonZeroUsize::new(width).unwrap(),
							))
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
	pub fn parallelism_degree(parallelism: Par) -> usize {
		match parallelism {
			Par::Seq => 1,
			#[cfg(feature = "rayon")]
			Par::Rayon(n_threads) => n_threads.get(),
			#[cfg(feature = "threadteam")]
			Par::Team(n_threads) => n_threads.get(),
		}
	}

	/// the raw Rayon `iter.for_each(f)` of the regions that never used spindle, with `Team`
	/// routed to the installed team instead (items dealt in static blocks, one part per member)
	#[cfg(feature = "rayon")]
	pub fn par_for_each<
		T: Send,
		I: rayon::iter::IntoParallelIterator<Item = T>,
	>(
		par: Par,
		iter: I,
		f: impl Send + Sync + Fn(T),
	) where
		I::Iter: rayon::iter::IndexedParallelIterator,
	{
		match par {
			#[cfg(feature = "threadteam")]
			Par::Team(n) => super::team::for_each(n.get(), iter, f),
			_ => {
				use rayon::iter::ParallelIterator;
				iter.into_par_iter().for_each(f)
			},
		}
	}

	/// `spindle::for_each` on the backend of `par`: the spindle group for `Rayon`, the installed
	/// team for `Team`, the caller for `Seq`
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
			Par::Team(_) => super::team::for_each(n_jobs, iter, f),
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
		use super::{join_raw, parallelism_degree};
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
			if parallelism_degree(parallelism) > 1 {
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
	}
}
/// approximate comparators for testing purposes
pub mod approx;
/// simd helper utilities based on lifetime-bound indices
pub mod simd;
