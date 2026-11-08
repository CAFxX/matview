# Agent Instructions

The implementation must prioritize correctness, reliability, maintainability, and explicit behavior. Features must be generic where their contract requires it; do not mistake a collection of examples or templates for general support.

## Testing standards

- Design tests from the documented behavior and supported SQL/engine contract, not from the implementation's current code paths. Avoid tautological tests that reproduce the implementation's assumptions.
- For each feature, cover focused hand-written edge cases, relevant interactions with other features, default behavior, and supported configuration variants. Include adversarial inputs and failure/rollback paths where applicable.
- Use property-based, randomized, and differential testing where they add value. Generators must create meaningful structural diversity: randomizing only values, identifiers, or a finite list of hardcoded templates is not broad grammar or feature coverage.
- For SQL/schema/DML fuzzing, generate from schema-aware ASTs or grammars. Make every supported production reachable, measure production and combination coverage, retain reproducible seeds, and minimize failures into permanent regression tests. Do not claim to generate every possible valid statement; state the actual grammar, depth, dialect, and coverage limits.
- Keep the differential oracle independent of the implementation under test. Preserve exact integer and decimal semantics; use explicit tolerances only for approximate numeric types. Compare result types and row multiplicities deliberately, and test precision boundaries.
- Distinguish native database execution from emulation, transpilation, or mocked behavior. Only native runs establish engine-specific semantics. Verify the actual server product and version in CI rather than trusting matrix labels.
- Concurrency tests must use independent connections or processes and deliberate synchronization. Sequential callbacks or interleaving operations on one connection are not evidence of concurrent behavior.
- Test safety and fallback contracts under default settings. Test opt-in overrides separately and assert their warnings, limitations, and consequences; do not suppress warnings in a way that makes an unsafe path appear supported.
- Test meaningful invariants and metamorphic properties (for example rollback restoration and rebuild-versus-incremental equivalence) where applicable. Use mutation testing selectively to check whether important tests detect representative faults.
- Report what was actually tested. Test counts, generated case counts, and CI job names are not substitutes for structural coverage, native-engine coverage, or demonstrated fault detection.

## Completion and review

A feature is not done merely because its happy-path tests pass. Before marking it complete, verify its documented contract, relevant edge cases and interactions, failure behavior, and supported engine/version combinations. Record known exclusions explicitly. Review tests for independence, sensitivity, and false confidence as well as reviewing production code.

Address identified technical debt proactively when it is relevant and safe to do so. Keep unrelated cleanup separate from feature changes; do not expand scope indiscriminately. Never claim a test, engine, version, configuration, or input class was covered unless it was actually exercised and verified.

## Git and push workflow

- Make changes on a dedicated topic branch, never directly on `master`. Use a short, descriptive branch name and keep each branch scoped to one coherent change.
- Before editing, fetch the latest repository refs and inspect the target branch and its open PR. Base new work on the intended branch, not automatically on `master`: when extending a PR stack, branch from the immediate predecessor and target that predecessor's branch. Keep independent work based on `master`.
- Prefer a local Git working tree for multi-file changes: fetch, check out the intended base, create or switch to the topic branch, make and review the complete change, run focused tests, then create coherent commits. Push the branch once the change is internally consistent. Avoid a series of GitHub Contents API writes for related files because each write creates a separate commit and can trigger redundant CI runs.
- Push with an explicit upstream, for example `git push --set-upstream origin <branch>`. Do not push to `master` or another PR's branch unless that is the explicitly intended target.
- Never force-push by default. If rewriting a branch is necessary (for example, to restack or remove accidental commits), first inspect the remote branch and all dependent PRs, coordinate the rewrite, and use `--force-with-lease` rather than `--force`. Do not rewrite a branch that others may be updating without checking its latest remote SHA.
- Keep PR stacks clean and linear: each stacked PR should contain only its own logical change relative to its immediate base PR. When a predecessor is merged or its base changes, restack descendants in dependency order, verify each diff contains only its intended changes, and update each PR's base branch accordingly. Do not merge a descendant before its prerequisites.
- After every push, verify the remote branch head SHA and the PR's head SHA match the commit just pushed. Check that the PR targets the intended base and that its diff contains only the intended changes. Treat a successful API write or local push response as insufficient proof that the PR was updated correctly.
- CI results apply only to the exact commit SHA tested. After any push, rebase, restack, or other head change, wait for CI on the new head; never use a green result from an older SHA as merge authorization.
- Avoid unnecessary pushes and CI runs. Batch related edits into one coherent commit before pushing; do not push intermediate or speculative states unless early review is specifically useful. Do not cancel useful CI runs merely to reduce queue length; cancel only demonstrably obsolete or duplicate runs after checking their head SHA and purpose.
- Before opening a PR, inspect the branch comparison against its intended base and confirm the commit range and file diff are correct. Include a concise summary, validation performed, remaining limitations, and explicit stack/base relationship in the PR description.
- Do not report work as pushed, updated, rebased, or ready until the remote ref, PR metadata, intended base, diff, and relevant CI state have been independently checked.

## Strict linear PR queue (required for auto-merge)

The repository must have **one globally ordered PR queue represented by one linear branch chain**. Do not create multiple roots, parallel stack forks, or sibling PRs in the queue, even when their work is logically independent. Parallel CI execution is fine; parallel branch ancestry is not.

For the current queue order `PR[1], PR[2], ..., PR[n]`, enforce all of these invariants:

- `PR[1]` is the only root and targets `master`.
- For every `i > 1`, `PR[i]` targets the head branch of exactly `PR[i-1]`. It must not target `master`, an earlier ancestor, or a separate root.
- The head commit of each PR is based on the exact current head of its immediate predecessor. Its comparison against that base must contain only that PR's own logical commits and file changes; no copied parent commits, duplicate diffs, or unrelated changes.
- The ordered PR queue and actual GitHub base-branch relationships must agree at all times. A list or report that looks ordered is not enough: verify every adjacent pair's base branch, head SHA, commit ancestry, and diff.
- If a new PR is added, insert it at an explicit position and restack the entire suffix so every next PR still targets its immediate predecessor. Reordering, removing, closing, or merging a PR likewise requires immediate reconciliation of the whole affected suffix.
- Do not skip a blocked PR by creating a new root or branching from a later ancestor. Resolve the blocker, or deliberately reorder the queue and restack all later PRs.
- Keep each PR's own diff isolated when rebasing/restacking. Never change only the PR base pointer when the head does not contain the intended predecessor ancestry.
- After a predecessor merges, rebase/retarget the next PR onto the actual resulting `master` state as appropriate for GitHub auto-merge, then rebuild and verify the remaining single chain from that new root. Do not assume a topic branch was merged just because a descendant targets it.
- Before enabling auto-merge, verify the PR is the next queue item, its immediate predecessor is already merged (or it is the root), its exact-head required checks are green, and the required base/ruleset safeguards apply. Never enable auto-merge on multiple sibling branches or on a PR that violates the chain.
- CI may run concurrently across the linear chain as soon as each child is based on its predecessor's current head. This does not authorize independent branch roots.

A queue is **not** a clean linear stack if two PRs share the same base branch, if any non-root PR targets `master`, or if a PR skips its immediate predecessor. Treat any such state as a queue-integrity defect and repair it before claiming the queue is ready for auto-merge.
