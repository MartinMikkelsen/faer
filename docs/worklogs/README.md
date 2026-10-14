# Work Logs

Keep lightweight records for nontrivial multi-phase changes, non-obvious design
choices, or performance experiments, following the shared
[Work Logs And Design Records policy](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/repository.md#work-logs-and-design-records).
Small fixes and AI assistance alone do not require one.

One file per change theme: `YYYY-MM-DD-<slug>.md`, linked from the PR that carries
the change. Start with a few lines:

```markdown
# <Change theme>

## Decisions
- Chosen approach and why; important rejected alternatives, if any.

## Verification conclusions and constraints
- What the final state establishes, and what remains unverified or limited.
- Links to detailed evidence or special reproduction conditions, when needed.
```

A performance experiment in this fork has one extra duty: an upstream revision was
measured, or it was not, and the record says which. `PATCHSET.md` is the ledger of
what the fork carries; a record here explains why it carries it and what lost along
the way.

Do not include commands run (Cargo or otherwise), files read, agent activity, or
edit/review chronology. Mention failed attempts only when they explain a choice or
limitation. Keep complete performance results in the linked evidence rather than
copying them here. Update the same record when its conclusions change, not after
every correction; do not rewrite historical records for this format.
