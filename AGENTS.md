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

## Native stacked PR workflow (required for optimistic auto-merge)

Use GitHub's native stacked pull-request feature, not merely a manually maintained list of dependent PRs. The desired workflow is optimistic: build and test every layer concurrently, then let GitHub merge layers automatically from the bottom upward as each layer and all prerequisites satisfy the trunk's rules.

- Maintain exactly one native GitHub stack with master as its trunk and one ordered linear branch chain. Explicitly link the chain as a stack using the GitHub website's Create stack action, gh stack submit/link, or the REST Stacks API. A sequence of PR base refs alone is not sufficient; verify native membership through the PR stack object or Stacks API.
- The bottom PR targets master. Every later PR targets the immediately preceding PR's head branch. Every head must descend from the exact current predecessor head, and each PR's diff against its immediate base must contain only that layer's logical change. All stack branches must be in this repository.
- Construct the entire stack optimistically. Create, link, push, and start CI for descendants without waiting for predecessors to merge. Run CI concurrently across all layers. A lower failure blocks merging above it, but does not prevent diagnosis and repair of other layers.
- GitHub evaluates branch protections and required checks against the stack trunk for every layer. Preserve the master ruleset, including the required stable check named CI success, strict freshness, and linear history. Do not create ad-hoc unprotected-topic-branch exceptions or weaken protections.
- Enable GitHub auto-merge using Rebase on every PR in the native stack as soon as GitHub permits it, not only on the bottom PR. Each layer should merge automatically once it and all lower layers satisfy trunk requirements. GitHub advances the stack bottom-up and automatically rebases/retargets the remaining layers after each merge. Keep auto-merge enabled on eligible layers and re-enable it if GitHub disables it after a head/base change.
- Do not interpret a clean or unstable response from the auto-merge API as evidence that topic branches need separate protection. Inspect exact-head CI, mergeability, and native stack membership. If the chain is not linked as a native stack, link it first. If a PR is already immediately mergeable, do not merge a mid-stack PR manually unless intentionally merging that contiguous portion, which also merges every lower PR.
- Prefer GitHub native auto-merge. If a merge must be initiated through the API for a native stack, use the asynchronous stack merge API and poll the result; never use the legacy synchronous merge endpoint for a stacked PR. Never bypass branch protections or required checks.
- After any lower PR merges, GitHub should automatically rebase and retarget the next layer to the trunk and maintain the remaining stack. Verify the stack remains linked, ordered, linear, and that each PR retains its isolated diff. If linearity is lost, use GitHub's Rebase stack action or gh stack rebase followed by gh stack push; do not repair it by changing only PR base refs.
- If connected tools cannot create/link a native stack or enable auto-merge on all eligible layers, identify the exact missing capability and the one-time GitHub UI/API action required. Do not silently fall back to manual merge-then-restack.
- Never merge unless all required checks are green for the exact current head SHA and all lower stack layers satisfy the trunk rules. Checks from older heads are irrelevant. Preserve rebase-only merges and linear history.
- On every run, inspect every open PR and the live stack object. Verify trunk, stack order, PR bases, head SHAs, ancestry, per-layer diffs, mergeability, exact-head CI, and auto-merge state. Reconcile the affected stack immediately after any change. Never claim the stack is clean or complete without verifying all links.

