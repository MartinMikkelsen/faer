use super::*;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

fn bits32(k: usize) -> u32 {
	match k % 8 {
		0 => 0,
		1 => 0x8000_0000,
		2 => 0x7F80_0000,
		3 => 0xFF80_0000,
		4 => 0x7FC0_0001 | (k as u32 & 0x003F_FFFF),
		5 => 0xFFC0_0001 | (k as u32 & 0x003F_FFFF),
		6 => 0x7F80_0001 | (k as u32 & 0x003F_FFFF),
		_ => 0x3F80_0000 | (k as u32 & 0x007F_FFFF),
	}
}
fn bits64(k: usize) -> u64 {
	match k % 8 {
		0 => 0,
		1 => 0x8000_0000_0000_0000,
		2 => 0x7FF0_0000_0000_0000,
		3 => 0xFFF0_0000_0000_0000,
		4 => 0x7FF8_0000_0000_0001 | k as u64,
		5 => 0xFFF8_0000_0000_0001 | k as u64,
		6 => 0x7FF0_0000_0000_0001 | k as u64,
		_ => 0x3FF0_0000_0000_0000 | k as u64,
	}
}

#[cfg(feature = "rayon")]
fn native_cases<T: ComplexField>(
	value: fn(usize, usize) -> T,
	bits: fn(&T) -> (u64, u64),
	par: Par,
) {
	for (m, n) in [
		(0, 0),
		(0, 7),
		(7, 0),
		(1, 1),
		(1, 33),
		(33, 1),
		(15, 17),
		(16, 16),
		(17, 15),
		(32, 64),
		(129, 65),
	] {
		for row_source in [false, true] {
			let storage = if row_source {
				Mat::from_fn(n, m, |j, i| value(i, j))
			} else {
				Mat::from_fn(m, n, value)
			};
			for reverse_src_rows in [false, true] {
				for reverse_src_cols in [false, true] {
					let mut src = if row_source {
						storage.as_ref().transpose().as_dyn_stride()
					} else {
						storage.as_ref().as_dyn_stride()
					};
					if reverse_src_rows {
						src = src.reverse_rows();
					}
					if reverse_src_cols {
						src = src.reverse_cols();
					}
					for reverse_dst_rows in [false, true] {
						for reverse_dst_cols in [false, true] {
							let (pm, pn) = if row_source {
								(m + 4, n + 6)
							} else {
								(n + 6, m + 4)
							};
							let mut parent = Mat::from_fn(pm, pn, |i, j| {
								value(i + 1000, j + 2000)
							});
							let mut expected = Vec::new();
							for j in 0..pn {
								for i in 0..pm {
									expected.push(bits(&parent[(i, j)]));
								}
							}
							for j in 0..n {
								for i in 0..m {
									let di = 2 + if reverse_dst_rows {
										m - 1 - i
									} else {
										i
									};
									let dj = 3 + if reverse_dst_cols {
										n - 1 - j
									} else {
										j
									};
									let (pi, pj) = if row_source {
										(di, dj)
									} else {
										(dj, di)
									};
									expected[pj * pm + pi] = bits(&src[(i, j)]);
								}
							}
							let mut dst = if row_source {
								parent.as_mut().as_dyn_stride_mut()
							} else {
								parent
									.as_mut()
									.transpose_mut()
									.as_dyn_stride_mut()
							};
							dst = dst.submatrix_mut(2, 3, m, n);
							if reverse_dst_rows {
								dst = dst.reverse_rows_mut();
							}
							if reverse_dst_cols {
								dst = dst.reverse_cols_mut();
							}
							copy_across_layouts(dst, src, par);
							for j in 0..pn {
								for i in 0..pm {
									std::assert_eq!(
										bits(&parent[(i, j)]),
										expected[j * pm + i]
									);
								}
							}
						}
					}
				}
			}
		}
	}
}

#[test]
#[cfg(feature = "rayon")]
fn builtin_copy_preserves_bits_and_parent_guards() {
	for threads in [1, 4, 8] {
		let pool = rayon::ThreadPoolBuilder::new()
			.num_threads(threads)
			.build()
			.unwrap();
		pool.install(|| {
			std::assert_eq!(rayon::current_num_threads(), threads);
			let par = if threads == 1 {
				Par::Seq
			} else {
				Par::rayon(threads)
			};
			native_cases::<f32>(
				|i, j| f32::from_bits(bits32(i * 31 + j * 17)),
				|v| (v.to_bits() as u64, 0),
				par,
			);
			native_cases::<f64>(
				|i, j| f64::from_bits(bits64(i * 31 + j * 17)),
				|v| (v.to_bits(), 0),
				par,
			);
			native_cases::<crate::c32>(
				|i, j| {
					crate::c32::new(
						f32::from_bits(bits32(i * 31 + j * 17)),
						f32::from_bits(bits32(i * 31 + j * 17 + 7)),
					)
				},
				|v| (v.re.to_bits() as u64, v.im.to_bits() as u64),
				par,
			);
			native_cases::<crate::c64>(
				|i, j| {
					crate::c64::new(
						f64::from_bits(bits64(i * 31 + j * 17)),
						f64::from_bits(bits64(i * 31 + j * 17 + 7)),
					)
				},
				|v| (v.re.to_bits(), v.im.to_bits()),
				par,
			);
		});
	}
}

// Deliberately Clone-only, with non-'static borrowed counters and an owned heap
// payload. Test the actual production dispatcher, not a separate classifier.
struct Counted<'a> {
	payload: Vec<usize>,
	clones: &'a AtomicUsize,
	drops: &'a AtomicUsize,
}
impl Clone for Counted<'_> {
	fn clone(&self) -> Self {
		self.clones.fetch_add(1, Relaxed);
		Self {
			payload: self.payload.clone(),
			clones: self.clones,
			drops: self.drops,
		}
	}
}
impl Drop for Counted<'_> {
	fn drop(&mut self) {
		self.drops.fetch_add(1, Relaxed);
	}
}

#[test]
fn generic_copy_preserves_clone_drop_and_heap_ownership() {
	for (m, n) in [
		(0, 0),
		(0, 17),
		(17, 0),
		(1, 1),
		(1, 33),
		(33, 1),
		(15, 17),
		(17, 15),
		(32, 64),
	] {
		for row_source in [false, true] {
			for reversed in [false, true] {
				let clones = AtomicUsize::new(0);
				let drops = AtomicUsize::new(0);
				let make = |i, j| Counted {
					payload: vec![i, j, i * 101 + j],
					clones: &clones,
					drops: &drops,
				};
				let storage = if row_source {
					Mat::from_fn(n, m, |j, i| make(i, j))
				} else {
					Mat::from_fn(m, n, make)
				};
				let mut src = if row_source {
					storage.as_ref().transpose().as_dyn_stride()
				} else {
					storage.as_ref().as_dyn_stride()
				};
				if reversed {
					src = src.reverse_rows().reverse_cols();
				}
				let (pm, pn) = if row_source {
					(m + 4, n + 6)
				} else {
					(n + 6, m + 4)
				};
				let mut parent =
					Mat::from_fn(pm, pn, |i, j| make(1000 + i, 2000 + j));
				let mut expected = Vec::new();
				for j in 0..pn {
					for i in 0..pm {
						expected.push(parent[(i, j)].payload.clone());
					}
				}
				for j in 0..n {
					for i in 0..m {
						let di = 2 + if reversed { m - 1 - i } else { i };
						let dj = 3 + if reversed { n - 1 - j } else { j };
						let (pi, pj) =
							if row_source { (di, dj) } else { (dj, di) };
						expected[pj * pm + pi] = src[(i, j)].payload.clone();
					}
				}
				{
					let mut dst = if row_source {
						parent.as_mut().as_dyn_stride_mut()
					} else {
						parent.as_mut().transpose_mut().as_dyn_stride_mut()
					};
					dst = dst.submatrix_mut(2, 3, m, n);
					if reversed {
						dst = dst.reverse_rows_mut().reverse_cols_mut();
					}
					copy_across_layouts_block(dst.rb_mut(), src);
					for j in 0..n {
						for i in 0..m {
							std::assert_ne!(
								dst[(i, j)].payload.as_ptr(),
								src[(i, j)].payload.as_ptr()
							);
						}
					}
				}
				std::assert_eq!(clones.load(Relaxed), m * n);
				std::assert_eq!(drops.load(Relaxed), m * n);
				for j in 0..pn {
					for i in 0..pm {
						std::assert_eq!(
							parent[(i, j)].payload,
							expected[j * pm + i]
						);
					}
				}
				drop(parent);
				drop(storage);
				std::assert_eq!(clones.load(Relaxed), m * n);
				std::assert_eq!(drops.load(Relaxed), pm * pn + 2 * m * n);
			}
		}
	}
}
