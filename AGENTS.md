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
