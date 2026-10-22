//! dqds: the singular values of a real bidiagonal matrix.
//!
//! The differential form of the implicit QL iteration with shifts, on the *squared* factors
//! `a_i = d_i^2` and `b_i = e_i^2`. One sweep is the transform
//!
//! ```text
//! delta = a_1 - tau
//! for i in 1..n-1:
//!     ahat_i = delta + b_i
//!     t      = a_{i+1} / ahat_i
//!     b_i    = b_i * t
//!     a_i    = ahat_i
//!     delta  = delta * t - tau
//! a_n = delta
//! ```
//!
//! which is the algorithm of Fernando & Parlett, *Accurate singular values and differential qd
//! algorithms* (Numer. Math. 67, 1994) and Parlett & Marques, *An implementation of the dqds
//! algorithm (positive case)* (LAPACK Working Note). It applies no rotations, so a sweep costs one
//! division and one multiplication per entry. Nothing here is translated from LAPACK or OpenBLAS
//! source.
//!
//! A sweep is accepted only if every factor it produced is positive and finite; otherwise the shift
//! is halved and the sweep is retried from the state it started with, and a block that cannot be
//! accepted at a zero shift is reported as non-convergence.

use crate::internal_prelude::*;
use crate::linalg::svd::SvdError;


/// How many sweeps one entry may take before the block gives up.
const SWEEPS_PER_ENTRY: usize = 200;

/// Singular values of the bidiagonal with diagonal `d` and off-diagonal `e`, in place.
///
/// On return `d` holds the singular values in **descending** order and `e` is destroyed. `d` may
/// hold negative entries on entry, since flipping the sign of a diagonal entry is an orthogonal
/// transform.
///
/// `d` must not contain a zero: the transform divides by the factors, and a rank-deficient
/// bidiagonal has a cheaper answer than this loop. The caller checks that and keeps its own
/// rotation-based path for it.
pub(crate) fn singular_values<T: RealField>(d: &mut [T], e: &mut [T]) -> Result<(), SvdError> {
    let n = d.len();
    core::assert!(e.len() + 1 >= n, "off-diagonal shorter than the diagonal");
    if n == 0 {
        return Ok(());
    }

    // Scale before squaring: a finite bidiagonal scaled by 1e200 has representable singular values
    // and unrepresentable squares.
    let mut scale = zero::<T>();
    for i in 0..n {
        scale = scale.fmax(d[i].clone().abs());
        if i < e.len() {
            scale = scale.fmax(e[i].clone().abs());
        }
    }
    if !scale.is_finite() {
        return Err(SvdError::NoConvergence);
    }
    if scale == zero() {
        for x in d.iter_mut() {
            *x = zero();
        }
        return Ok(());
    }

    let inv_scale = scale.recip();
    let mut a = vec![zero::<T>(); n];
    let mut b = vec![zero::<T>(); e.len()];
    for i in 0..n {
        let v = d[i].clone() * inv_scale.clone();
        a[i] = v.clone() * v.clone();
    }
    for i in 0..e.len() {
        let v = e[i].clone() * inv_scale.clone();
        b[i] = v.clone() * v.clone();
    }

    // One save buffer per call, sized once: a rejected sweep is retried from the state it started
    // with. The entry point that reaches this allocates its own workspace already.
    let mut save_a = vec![zero::<T>(); n];
    let mut save_b = vec![zero::<T>(); e.len()];

    let mut lo = 0usize;
    while lo < n {
        let mut hi = lo + 1;
        while hi < n && b[hi - 1] != zero::<T>() {
            hi += 1;
        }
        block(
            &mut a[lo..hi],
            &mut b[lo..hi - 1],
            &mut save_a[lo..hi],
            &mut save_b[lo..hi - 1],
        )?;
        lo = hi;
    }

    for i in 0..n {
        d[i] = a[i].clone().fmax(zero::<T>()).sqrt() * scale.clone();
    }
    // The public contract is descending order; the iteration produces none.
    d.sort_by(|x, y| y.partial_cmp(x).unwrap_or(core::cmp::Ordering::Equal));
    Ok(())
}

/// One contiguous block: deflate from the bottom until a single factor is left.
fn block<T: RealField>(
    a: &mut [T],
    b: &mut [T],
    save_a: &mut [T],
    save_b: &mut [T],
) -> Result<(), SvdError> {
    let eps = eps::<T>();
    let tol = eps.clone() * eps.clone();
    let half = from_f64::<T>(0.5);
    let mut live = a.len();
    let mut shift = zero::<T>();
    let mut sweeps_left = SWEEPS_PER_ENTRY * a.len();

    while live > 1 {
        let mut tau = shift_candidate(&a[..live], &b[..live - 1]);
        let mut accepted = false;
        for attempt in 0..5 {
            for i in 0..live {
                save_a[i] = a[i].clone();
            }
            for i in 0..live - 1 {
                save_b[i] = b[i].clone();
            }
            if sweep(&mut a[..live], &mut b[..live - 1], tau.clone()) {
                accepted = true;
                break;
            }
            for i in 0..live {
                a[i] = save_a[i].clone();
            }
            for i in 0..live - 1 {
                b[i] = save_b[i].clone();
            }
            tau = if attempt < 2 { tau.clone() * half.clone() } else { zero::<T>() };
        }
        if !accepted {
            return Err(SvdError::NoConvergence);
        }
        shift = shift + tau;
        sweeps_left = sweeps_left.saturating_sub(1);
        if sweeps_left == 0 {
            return Err(SvdError::NoConvergence);
        }

        // Li's test on the state the sweep produced: both sides are squared singular-value units.
        while live > 1
            && b[live - 2].clone()
                <= tol.clone() * (a[live - 2].clone() * a[live - 1].clone()).sqrt()
        {
            a[live - 1] = a[live - 1].clone() + shift.clone();
            live -= 1;
        }
    }
    a[0] = a[0].clone() + shift;
    Ok(())
}

/// One dqds sweep. Returns `false` when the transform produced a non-positive or non-finite
/// factor, in which case the shift was not admissible and the caller retries from the save.
fn sweep<T: RealField>(a: &mut [T], b: &mut [T], tau: T) -> bool {
    let n = a.len();
    if n == 0 {
        return true;
    }
    let mut delta = a[0].clone() - tau.clone();
    for i in 0..n - 1 {
        let ahat = delta.clone() + b[i].clone();
        if !(ahat > zero()) || !ahat.is_finite() {
            return false;
        }
        let t = a[i + 1].clone() / ahat.clone();
        if !t.is_finite() {
            return false;
        }
        b[i] = b[i].clone() * t.clone();
        a[i] = ahat;
        delta = delta * t - tau.clone();
    }
    if !(delta > zero()) {
        return false;
    }
    if !delta.is_finite() {
        return false;
    }
    a[n - 1] = delta;
    true
}

/// The shift: the smaller eigenvalue of the trailing 2x2 of `B B^T`, in squared-factor units.
///
/// That value is an upper bound on the block's smallest eigenvalue, not a lower one, so it is a
/// *candidate*: `sweep` rejects it when it is not admissible and the caller retries smaller.
fn shift_candidate<T: RealField>(a: &[T], b: &[T]) -> T {
    let n = a.len();
    if n < 2 {
        return zero();
    }
    // `B B^T`'s trailing 2x2: diagonal `(a_{n-2} + b_{n-3}, a_{n-1} + b_{n-2})`, off-diagonal
    // `sqrt(a_{n-2} b_{n-2})` -- all of it already in squared-factor units.
    let d1 = a[n - 2].clone() + if n >= 3 { b[n - 3].clone() } else { zero::<T>() };
    let d2 = a[n - 1].clone() + b[n - 2].clone();
    let f = (a[n - 2].clone() * b[n - 2].clone()).sqrt();
    let half_trace = (d1.clone() + d2.clone()) * from_f64::<T>(0.5);
    let det = d1 * d2 - f.clone() * f.clone();
    let disc = (half_trace.clone() * half_trace.clone() - det).fmax(zero::<T>()).sqrt();
    (half_trace - disc).fmax(zero::<T>()).fmin(a[n - 1].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mat;

    /// The kernel against `thin_svd` of the same bidiagonal, over spectra that make the iteration
    /// work: well separated, clustered, repeated, badly scaled, and with negative diagonals.
    #[test]
    fn matches_the_reference_on_bidiagonals() {
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let x = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            (x >> 11) as f64 / (1u64 << 53) as f64
        };
        for n in 2..41usize {
            for case in 0..4 {
                let mut d = vec![0.0f64; n];
                let mut e = vec![0.0f64; n - 1];
                for i in 0..n {
                    d[i] = match case {
                        0 => 0.5 + next(),
                        1 => if i % 3 == 0 { -1.0 } else { 0.5 + next() },
                        2 => 1.0,
                        _ => (0.5 + next()) * if i == 0 { 1e-6 } else { 1.0 },
                    };
                }
                for i in 0..n - 1 {
                    e[i] = match case {
                        0 => 0.2 * next(),
                        1 => 0.2 * next(),
                        2 => 1e-3 * next(),
                        _ => 0.2 * next(),
                    };
                }
                let mut mat = Mat::<f64>::zeros(n, n);
                for i in 0..n {
                    mat[(i, i)] = d[i];
                    if i + 1 < n {
                        mat[(i + 1, i)] = e[i];
                    }
                }
                let mut want: Vec<f64> = mat
                    .as_ref()
                    .thin_svd()
                    .unwrap()
                    .S()
                    .column_vector()
                    .iter()
                    .copied()
                    .collect();
                want.sort_by(|x, y| y.partial_cmp(x).unwrap());

                let mut dd = d.clone();
                let mut ee = e.clone();
                singular_values(&mut dd, &mut ee).unwrap();
                for i in 0..n {
                    let tol = 1e-10 * want[0].max(1.0);
                    core::assert!(
                        (dd[i] - want[i]).abs() <= tol,
                        "n={n} case={case} i={i}: {} vs {}",
                        dd[i],
                        want[i]
                    );
                }
            }
        }
    }

    /// A diagonal bidiagonal is its own answer, sorted.
    #[test]
    fn diagonal_is_its_own_answer() {
        let mut d = vec![3.0f64, 1.0, 2.0];
        let mut e: Vec<f64> = vec![0.0; 2];
        singular_values(&mut d, &mut e).unwrap();
        core::assert_eq!(d, vec![3.0, 2.0, 1.0]);
    }

    /// Scaling by a power of two must not change the answer's relative accuracy.
    #[test]
    fn scale_does_not_change_the_answer() {
        let d = vec![1.0f64, 2.0, 3.0, 4.0];
        let e = vec![0.3f64, 0.2, 0.1];
        let mut d1 = d.clone();
        let mut e1 = e.clone();
        singular_values(&mut d1, &mut e1).unwrap();

        let mut d2: Vec<f64> = d.iter().map(|x| x * 1e150).collect();
        let mut e2: Vec<f64> = e.iter().map(|x| x * 1e150).collect();
        singular_values(&mut d2, &mut e2).unwrap();
        for i in 0..d1.len() {
            let got = d2[i] * 1e-150;
            core::assert!((got - d1[i]).abs() <= 1e-10 * d1[0], "{got} vs {}", d1[i]);
        }
    }
}
