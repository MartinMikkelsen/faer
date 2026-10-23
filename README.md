> **A tensor4all fork used as an AI-driven, high-speed experiment bench.**
>
> This is a tensor4all fork of [faer](https://codeberg.org/sarah-quinones/faer)
> (the GitHub `sarah-ek/faer-rs` repository is a mirror). The `t4a-faer*` crates
> published from here are experiment builds for `tenferro-rs`, `tlinalg-rs` and
> `tprims-rs`.
>
> ### Stance
>
> - **An AI-driven, high-speed experiment bench.** Coding agents iterate here at
>   full speed. Commits, tests and reviews are optimized for fast iteration, not
>   for matching upstream's review standards.
> - **A place for destructive experiments.** Features shaped for the tensor4all /
>   tenferro workload — strided and batched entry points, typed uninitialized
>   overwrite destinations, caller-owned workspaces, an explicit thread policy —
>   get tried and broken without ceremony. API churn and removal are expected,
>   and compatibility is not promised.
> - **The experiments are recorded.** What was tried, kept, or dropped is
>   tracked in
>   [`PATCHSET.md`](https://github.com/tensor4all/faer/blob/main/PATCHSET.md).
>
> The crate body stays `MIT`, keeping the upstream copyright and attribution.
> Code ported from other projects keeps its own `COPYING` terms (see
> [`NOTICE`](https://github.com/tensor4all/faer/blob/main/NOTICE)). Use this fork
> only through a pinned git revision or the published `t4a-faer*` versions.

<p align="center">
  <img src="https://faer.veganb.tw/faer-logo-color.png" alt="faer logo"/ width="25%">
</p>

# faer

[![documentation](https://docs.rs/faer/badge.svg)](https://docs.rs/faer)
[![crate](https://img.shields.io/crates/v/faer.svg)](https://crates.io/crates/faer)

`faer` is a rust crate that implements low level linear algebra routines and a high level wrapper for ease of use, in pure rust.
the aim is to provide a fully featured library for linear algebra with focus on portability, correctness, and performance.

see the [official website](https://faer.veganb.tw) and the [docs.rs](https://docs.rs/faer/latest/faer) documentation for code examples and usage instructions.

questions about using the library, contributing, and future directions can be discussed in the [zulip server](https://faer.zulipchat.com).

# parallelism

operations take a `Par`; with `Par::rayon(n)` the parallel regions run on the global rayon pool. a
caller with a thread budget should hold the worker group around its calls,
`faer::spindle::with_lock(n, || { ... })`: without it every region is a plain rayon fork/join that
wakes the workers and partitions the work afresh, and an algorithm made of many short regions (the
bidiagonalization behind the SVD, two vector products per step) runs a third slower. see the
*parallelism* section of the crate documentation.

# contributing

if you'd like to contribute to `faer`, check out the list of "good first issue"
issues. these are all (or should be) issues that are suitable for getting
started, and they generally include a detailed set of instructions for what to
do. please ask questions on the zulip server or the issue itself if anything
is unclear!

# minimum supported rust version

the current msrv is rust 1.84.0.

# benchmarks

see [the benchmark page](https://faer.veganb.tw/benchmarks/) on the main website.

