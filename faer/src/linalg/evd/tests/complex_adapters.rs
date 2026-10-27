// Private test adapters. Copyright (c) tensor4all contributors. SPDX-License-Identifier: MIT
fn schur_call(
	h: MatMut<'_, Scalar>,
	q: Option<MatMut<'_, Scalar>>,
	mut wr: ColMut<'_, Real>,
	mut wi: ColMut<'_, Real>,
	par: Par,
	stack: &mut MemStack,
	p: Spec<SchurParams, Scalar>,
) -> Result<(), EvdError> {
	let n = h.nrows();
	let mut w = Col::<Scalar>::full(n, scalar(Real::NAN, Real::NAN));
	hessenberg_schur_cplx(h, q, w.as_mut(), par, stack, p)?;
	for i in 0..n {
		wr[i] = real(&w[i]);
		wi[i] = imag(&w[i]);
	}
	Ok(())
}
fn vectors_call(
	t: MatRef<'_, Scalar>,
	q: Option<MatRef<'_, Scalar>>,
	x: MatMut<'_, Scalar>,
	par: Par,
	stack: &mut MemStack,
	p: Spec<EvdFromSchurParams, Scalar>,
) -> Result<(), EvdError> {
	eigenvectors_from_schur_cplx(t, q, x, par, stack, p)
}
fn evd_call(
	a: MatRef<'_, Scalar>,
	mut wr: ColMut<'_, Real>,
	mut wi: ColMut<'_, Real>,
	left: Option<MatMut<'_, Scalar>>,
	right: Option<MatMut<'_, Scalar>>,
	par: Par,
	stack: &mut MemStack,
	p: Spec<EvdParams, Scalar>,
) -> Result<(), EvdError> {
	let n = a.nrows();
	let mut w = Col::<Scalar>::full(n, scalar(Real::NAN, Real::NAN));
	evd_cplx(a, w.as_mut().as_diagonal_mut(), left, right, par, stack, p)?;
	for i in 0..n {
		wr[i] = real(&w[i]);
		wi[i] = imag(&w[i]);
	}
	Ok(())
}
fn reorder_call(
	t: MatMut<'_, Scalar>,
	q: Option<MatMut<'_, Scalar>>,
	from: usize,
	to: usize,
	par: Par,
	stack: &mut MemStack,
) -> Result<usize, SchurReorderError> {
	Ok(reorder_schur_cplx(t, q, from, to, par, stack))
}

fn adjacent_call(
	t: MatMut<'_, Scalar>,
	q: Option<MatMut<'_, Scalar>>,
	start: usize,
	_n1: usize,
	_n2: usize,
) -> isize {
	schur::complex_schur::schur_swap(t, q, start)
}
