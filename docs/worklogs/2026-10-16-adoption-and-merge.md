# Adoption and merge of the row-major QR/LU work

The campaign recorded the evidence; this is the fork-side record.

- **Measured**: `fe41bbdfb88c0528e0726dca3b1764f47222ad56`, C-order `4096x256` at 4T/8T against the pinned
  baseline: `qr_factor` f64 0.223/0.187, c64 0.356/0.283, `lu` f64 0.740/0.724, `qr` f64 0.906/0.897;
  F-order guards 1.00 +/- 0.01; the unchanged `faer-upstream` row reproduced the baseline in every pair;
  `cargo test -p t4a-faer --lib` 209 passed / 2 ignored.
- **Not fixed**: the wide `64x1024` loss from the historical page (1.38x) is unchanged (measured
  0.96-1.02) - the row-major adaptation applies to tall/square shapes only.
- **Merged**: `1d754763e7198692d5af0bbf282ac70d9d7e61b5` on `main`, verified to carry the measured
  revision's tree hashes for `faer/src`, `faer/Cargo.toml`, `faer-traits/**` and `Cargo.lock`, so no
  second benchmark was run after the merge.
- **Process note**: the earlier acceptance evidence was invalid until two defects were fixed - the
  campaign's lockfile silently linked the crates.io release instead of this checkout, and its candidate
  comparison dropped the memory order from its cell key. Both are recorded in the campaign
  (`docs/reviews/20261016-linked-crate-provenance.md` and the 2026-10-16 worklog entry).
