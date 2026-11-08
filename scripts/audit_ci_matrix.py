#!/usr/bin/env python3
"""Audit CI matrix breadth against observed GitHub Actions job outcomes and cost.

This is an evidence collector, not an automatic matrix pruner. Different engine
images can intentionally execute the same test command to validate native
integration behavior; identical commands alone are not evidence of redundancy.
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import os
import re
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml


API = "https://api.github.com"
TOKEN = os.environ["GITHUB_TOKEN"]
REPOSITORY = os.environ["GITHUB_REPOSITORY"]
WORKFLOW = ".github/workflows/ci.yml"
DAYS = int(os.environ.get("AUDIT_DAYS", "30"))
MAX_RUNS = int(os.environ.get("AUDIT_MAX_RUNS", "50"))


def api_get(path: str) -> dict:
    request = urllib.request.Request(
        API + path,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {TOKEN}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def matrix_size(matrix: dict) -> int:
    if not isinstance(matrix, dict):
        return 1
    if isinstance(matrix.get("include"), list):
        return len(matrix["include"])
    size = 1
    for key, values in matrix.items():
        if key == "exclude":
            continue
        if isinstance(values, list):
            size *= len(values)
    return size


def run_commands(job: dict) -> list[str]:
    commands = []
    for step in job.get("steps", []) or []:
        command = step.get("run")
        if command:
            commands.append(" ".join(str(command).split()))
    return commands


def expanded_name_pattern(job_id: str, job: dict) -> re.Pattern[str]:
    template = str(job.get("name") or job_id)
    pattern = re.escape(template)
    pattern = re.sub(r"\\\$\\\{\\\{.*?\\\}\\\}", r".*?", pattern)
    return re.compile("^" + pattern + "$")


def minutes_between(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return (dt.datetime.fromisoformat(end.replace("Z", "+00:00")) -
            dt.datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds() / 60


def main() -> int:
    workflow_text = Path(WORKFLOW).read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)
    definitions = workflow.get("jobs", {})
    profiles = {}
    for job_id, definition in definitions.items():
        matrix = (definition.get("strategy") or {}).get("matrix", {})
        profiles[job_id] = {
            "name_template": definition.get("name", job_id),
            "matrix_entries": matrix_size(matrix),
            "matrix_axes": {
                key: len(value) if isinstance(value, list) else "include"
                for key, value in matrix.items()
            },
            "runs_on": definition.get("runs-on"),
            "commands": run_commands(definition),
            "name_pattern": expanded_name_pattern(job_id, definition),
        }

    cutoff = (dt.datetime.now(dt.timezone.utc) -
              dt.timedelta(days=DAYS)).date().isoformat()
    query = urllib.parse.urlencode({
        "per_page": min(MAX_RUNS, 100),
        "created": f">={cutoff}",
    })
    runs = api_get(
        f"/repos/{REPOSITORY}/actions/workflows/ci.yml/runs?{query}"
    ).get("workflow_runs", [])

    observations = collections.defaultdict(list)
    run_summaries = []
    for run in runs[:MAX_RUNS]:
        jobs = api_get(
            f"/repos/{REPOSITORY}/actions/runs/{run['id']}/jobs?per_page=100"
        ).get("jobs", [])
        run_summaries.append({
            "id": run["id"],
            "event": run.get("event"),
            "branch": run.get("head_branch"),
            "sha": run.get("head_sha"),
            "status": run.get("status"),
            "conclusion": run.get("conclusion"),
            "created_at": run.get("created_at"),
        })
        for job in jobs:
            duration = minutes_between(job.get("started_at"), job.get("completed_at"))
            matched = [
                job_id for job_id, profile in profiles.items()
                if profile["name_pattern"].match(job.get("name", ""))
            ]
            job_id = matched[0] if len(matched) == 1 else "__unmatched__"
            observations[job_id].append({
                "name": job.get("name"),
                "conclusion": job.get("conclusion") or job.get("status"),
                "minutes": duration,
                "run_id": run["id"],
            })

    rows = []
    for job_id, profile in profiles.items():
        samples = observations.get(job_id, [])
        durations = [item["minutes"] for item in samples if item["minutes"] is not None]
        counts = collections.Counter(item["conclusion"] for item in samples)
        completed = sum(counts[key] for key in ("success", "failure", "cancelled", "timed_out", "action_required"))
        success_rate = (counts["success"] / completed * 100) if completed else None
        rows.append({
            "job_id": job_id,
            "name_template": profile["name_template"],
            "matrix_entries": profile["matrix_entries"],
            "matrix_axes": profile["matrix_axes"],
            "runs_observed": len(samples),
            "success": counts["success"],
            "failure": counts["failure"],
            "cancelled": counts["cancelled"],
            "other": sum(v for k, v in counts.items()
                         if k not in {"success", "failure", "cancelled"}),
            "success_rate_pct": round(success_rate, 1) if success_rate is not None else None,
            "median_minutes": round(statistics.median(durations), 1) if durations else None,
            "p90_minutes": round(sorted(durations)[max(0, int(len(durations) * .9) - 1)], 1) if durations else None,
            "runs_on": profile["runs_on"],
            "commands": profile["commands"],
            "matrix_axes_detail": profile["matrix_axes"],
        })

    total_matrix_entries = sum(row["matrix_entries"] for row in rows)
    total_samples = sum(row["runs_observed"] for row in rows)
    report = {
        "repository": REPOSITORY,
        "workflow": WORKFLOW,
        "period_days": DAYS,
        "runs_sampled": len(run_summaries),
        "jobs_sampled": total_samples,
        "declared_job_groups": rows,
        "unmatched_jobs": observations.get("__unmatched__", []),
        "recent_runs": run_summaries,
        "matrix_entry_count_per_full_run": total_matrix_entries,
        "interpretation": [
            "A matrix entry is justified only by a distinct compatibility target, runtime, or test obligation.",
            "Repeated test commands are not automatically redundant: each live database image/driver is a distinct integration target.",
            "A low failure rate does not prove low value; successful compatibility coverage is expected to pass.",
            "Review duration, setup cost, target/version uniqueness, feature coverage, and whether a smaller representative version set preserves supported guarantees.",
            "Samples marked cancelled may be superseded runs; do not count them as test failures.",
            "No jobs are removed or skipped by this audit.",
        ],
    }

    lines = [
        "# CI matrix efficacy audit",
        "",
        f"- Repository: `{REPOSITORY}`",
        f"- Window: last {DAYS} days; sampled {len(run_summaries)} workflow runs",
        f"- Declared matrix entries per full run: **{total_matrix_entries}**",
        f"- Observed job executions: **{total_samples}**",
        "",
        "| Job group | Matrix entries | Observed | Pass | Fail | Cancel | Success % | Median min | P90 min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fmt = lambda value: "—" if value is None else str(value)
        lines.append(
            f"| {row['job_id']} | {row['matrix_entries']} | {row['runs_observed']} "
            f"| {row['success']} | {row['failure']} | {row['cancelled']} "
            f"| {fmt(row['success_rate_pct'])} | {fmt(row['median_minutes'])} "
            f"| {fmt(row['p90_minutes'])} |"
        )
    lines += [
        "",
        "## Per-group justification evidence",
        "",
    ]
    for row in rows:
        lines += [
            f"### {row['job_id']}: {row['name_template']}",
            f"- Runner: `{row['runs_on']}`",
            f"- Matrix axes/cardinalities: `{json.dumps(row['matrix_axes_detail'], sort_keys=True)}`",
            f"- Test/setup commands: `{' ; '.join(row['commands']) or '(none found)'}`",
            f"- Historical evidence: {row['runs_observed']} executions; "
            f"{row['median_minutes'] if row['median_minutes'] is not None else 'no'} median minutes; "
            f"{row['success_rate_pct'] if row['success_rate_pct'] is not None else 'no'}% success among completed jobs.",
            "- Review question: what unique engine/version/runtime/feature contract does each matrix entry protect, and can a representative subset preserve it?",
            "",
        ]
    lines += [
        "## Interpretation safeguards",
        "",
        *[f"- {item}" for item in report["interpretation"]],
        "",
        "This report is evidence for a human matrix review, not an automatic recommendation to delete jobs.",
    ]
    markdown = "\n".join(lines) + "\n"
    Path("ci-matrix-audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Path("ci-matrix-audit.md").write_text(markdown, encoding="utf-8")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (urllib.error.URLError, KeyError, ValueError, yaml.YAMLError) as error:
        print(f"CI matrix audit failed: {error}", file=sys.stderr)
        raise SystemExit(1)
