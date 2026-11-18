#!/usr/bin/env python3
"""Summarize a Suite 12 fuzz campaign manifest (fuzz-campaign.json) after a run.

Used by .github/workflows/ci-extended.yml after the random-seed fuzz step. It
lists every batch that failed or never finished (status "failed" or
"running", the latter meaning the process died or timed out mid-batch) and
prints, for each failing seed, a replay command that pins the seed and every
campaign size parameter recorded in the manifest. A generated scenario depends
on (seed, scenario, dialect, views, mutations), and the local default scale
differs from CI's, so ``MATVIEW_FUZZ_SEEDS`` alone does not reproduce a CI
scenario.

Outputs (written to --out-dir):
  fuzz-failures.md   Markdown report (also appended to $GITHUB_STEP_SUMMARY).
  failing-seeds.txt  One failing seed per line (empty if none were recorded).

The exit status is always 0: this is a reporting step; the fuzz step's own
exit status decides the job result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

FAILED_STATUSES = {"failed": "failed", "running": "interrupted (process died or timed out)"}


def replay_env(manifest: Mapping[str, Any], seeds: Sequence[int]) -> Dict[str, str]:
    """Environment that reproduces the manifest's campaign for ``seeds``.

    Mirrors ``test_matview_sync.fuzz_replay_env``; the round trip is checked by
    test_p2_ci_matrix_verification.py.
    """
    return {
        "MATVIEW_FUZZ_SCALE": str(manifest["scale"]),
        "MATVIEW_FUZZ_SCENARIOS": str(manifest["scenarios_per_seed"]),
        "MATVIEW_FUZZ_VIEWS_PER_SCHEMA": str(manifest["views_per_schema"]),
        "MATVIEW_FUZZ_MUTATIONS": str(manifest["mutations_per_scenario"]),
        "MATVIEW_FUZZ_SEEDS": ",".join(str(int(s)) for s in seeds),
    }


def replay_command(manifest: Mapping[str, Any], seeds: Sequence[int], engine_env: str = "") -> str:
    parts = [engine_env.strip()] if engine_env.strip() else []
    parts += [f"{k}={v}" for k, v in replay_env(manifest, seeds).items()]
    parts.append("python3 test_matview_sync.py --fuzz")
    return " ".join(parts)


def failing_runs(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in manifest.get("runs") or [] if r.get("status") in FAILED_STATUSES]


def failing_seeds(manifest: Mapping[str, Any]) -> List[int]:
    return list(dict.fromkeys(int(r["seed"]) for r in failing_runs(manifest) if r.get("seed") is not None))


def _cell(text: Any, limit: int = 300) -> str:
    s = " ".join(str(text).split()).replace("|", "\\|")
    return s if len(s) <= limit else s[: limit - 3] + "..."


def render(manifest: Optional[Mapping[str, Any]], *, target: str, engine_env: str = "",
           run_url: str = "") -> str:
    lines = [f"### Random-seed fuzz: {target}", ""]
    if run_url:
        lines += [f"Run: {run_url}", ""]
    if manifest is None:
        lines.append(
            "No fuzz-campaign.json manifest was written: the job failed before Suite 12 started "
            "(setup, server start or engine verification). See the job log."
        )
        return "\n".join(lines) + "\n"

    random_seeds = set(int(s) for s in manifest.get("random_seeds") or [])
    lines.append(
        f"Campaign: scale={manifest.get('scale')}, scenarios/seed={manifest.get('scenarios_per_seed')}, "
        f"views/schema={manifest.get('views_per_schema')}, mutations/scenario={manifest.get('mutations_per_scenario')}, "
        f"seeds={manifest.get('seeds')} (random: {sorted(random_seeds)}), commit={manifest.get('commit')}, "
        f"python={manifest.get('python')}, default dialect={manifest.get('default_dialect')}."
    )
    lines.append("")
    runs = failing_runs(manifest)
    if not runs:
        lines.append(
            "No Suite 12 batch failed or was interrupted. If the job failed, the failure is outside "
            "the random-seed campaign (for example the Suite 13 permanent corpus); see the job log."
        )
        return "\n".join(lines) + "\n"

    lines += ["| Seed | Kind | Scenario | Batch | Status | Error |", "| --- | --- | --- | --- | --- | --- |"]
    for r in runs:
        seed = r.get("seed")
        kind = "random" if seed is not None and int(seed) in random_seeds else "fixed"
        error = f"{r.get('error_type', '')}: {r.get('error', '')}" if r.get("error_type") else ""
        lines.append(
            f"| {seed} | {kind} | {r.get('scenario')} | {_cell(r.get('suite', ''), 120)} | "
            f"{FAILED_STATUSES[r['status']]} | {_cell(error)} |"
        )
    lines += ["", "Replay (each command reruns every scenario of that seed with the CI campaign sizes):", "", "```sh"]
    for seed in failing_seeds(manifest):
        lines.append(replay_command(manifest, [seed], engine_env))
    lines += [
        "```",
        "",
        "Add a minimized reproduction of each confirmed failure to the permanent regression corpus "
        "(Suite 13, `get_fuzz_regression_corpus` in test_matview_sync.py) in the fix PR.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="fuzz-campaign.json")
    ap.add_argument("--target", default=os.environ.get("MATVIEW_FUZZ_TARGET", "unknown target"))
    ap.add_argument("--engine-env", default="",
                    help="environment prefix for the replay command, e.g. 'POSTGRES_DSN=...'")
    ap.add_argument("--run-url", default="")
    ap.add_argument("--out-dir", default="fuzz-report")
    args = ap.parse_args(argv)

    manifest: Optional[Dict[str, Any]] = None
    path = Path(args.manifest)
    if path.is_file():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"::warning::cannot read {path}: {exc}")

    report = render(manifest, target=args.target, engine_env=args.engine_env, run_url=args.run_url)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fuzz-failures.md").write_text(report, encoding="utf-8")
    seeds = failing_seeds(manifest) if manifest else []
    (out / "failing-seeds.txt").write_text("".join(f"{s}\n" for s in seeds), encoding="utf-8")
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
