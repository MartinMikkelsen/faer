// Private test adapters. Copyright (c) tensor4all contributors. SPDX-License-Identifier: MIT
fn schur_call(
	h: MatMut<'_, Scalar>,
	q: Option<MatMut<'_, Scalar>>,
	wr: ColMut<'_, Real>,
	wi: ColMut<'_, Real>,
	par: Par,
	stack: &mut MemStack,
	p: Spec<SchurParams, Scalar>,
) -> Result<(), EvdError> {
	hessenberg_schur_real(h, q, wr, wi, par, stack, p)
}
fn vectors_call(
	t: MatRef<'_, Scalar>,
	q: Option<MatRef<'_, Scalar>>,
	x: MatMut<'_, Scalar>,
	par: Par,
	stack: &mut MemStack,
	p: Spec<EvdFromSchurParams, Scalar>,
) -> Result<(), EvdError> {
	eigenvectors_from_schur_real(t, q, x, par, stack, p)
}
fn evd_call(
	a: MatRef<'_, Scalar>,
	wr: ColMut<'_, Real>,
	wi: ColMut<'_, Real>,
	left: Option<MatMut<'_, Scalar>>,
	right: Option<MatMut<'_, Scalar>>,
	par: Par,
	stack: &mut MemStack,
	p: Spec<EvdParams, Scalar>,
) -> Result<(), EvdError> {
	evd_real(
		a,
		wr.as_diagonal_mut(),
		wi.as_diagonal_mut(),
		left,
		right,
		par,
		stack,
		p,
	)
}
fn reorder_call(
	t: MatMut<'_, Scalar>,
	q: Option<MatMut<'_, Scalar>>,
	from: usize,
	to: usize,
	par: Par,
	stack: &mut MemStack,
) -> Result<usize, SchurReorderError> {
	reorder_schur_real(t, q, from, to, par, stack)
}

fn adjacent_call(
	t: MatMut<'_, Scalar>,
	q: Option<MatMut<'_, Scalar>>,
	start: usize,
	n1: usize,
	n2: usize,
) -> isize {
	schur::real_schur::schur_swap_checked(t, q, start, n1, n2, true)
}
