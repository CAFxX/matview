#!/usr/bin/env python3
"""Local pre-push gate derived from .github/workflows/ci.yml.

The single source of truth is the PR-gate workflow of the revision under test.
This script parses it and executes the same ``run:`` steps, with the same
``env:`` and in the same order, for every PR-gate job and matrix row.  Nothing
about the job list, the matrix, the test commands, or the documentation-only
whitelist is hand-copied here: if the workflow uses a construct this script
cannot reproduce faithfully, planning fails loudly (exit 2) instead of
guessing.

Exit codes
----------
0  GREEN      every requested step passed.  Only a ``--tier full`` run in which
              every PR-gate job/matrix row was executed natively counts as a
              CI-equivalence claim; only such a run writes an attestation stamp.
1  FAILED     at least one step (or container start) failed.
2  PLAN ERROR the workflow could not be interpreted faithfully, or bad usage.
3  INCOMPLETE nothing failed, but at least one required job/matrix row was not
              verified (missing interpreter, missing container runtime, a
              blocked ``sudo`` step, interpreter substitution, ``--only``).

See ``--help`` and the "Local presubmit" section of README.md.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import dataclasses
import datetime as _dt
import fnmatch
import hashlib
import itertools
import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

EXIT_GREEN = 0
EXIT_FAILED = 1
EXIT_PLAN_ERROR = 2
EXIT_INCOMPLETE = 3

WORKFLOW_PATH = ".github/workflows/ci.yml"
STAMP_SCHEMA = 1
REQUIRED_CHECK_NAME = "CI success"

# Fast-tier policy (not CI-derived): test files that always run in the fast
# tier regardless of which paths changed.  Each must still be a step of the
# CI sqlite job, otherwise planning fails.
FAST_CORE_TESTS = ("test_sqlite_runtime_boundaries.py",)
FAST_PREFLIGHT_INDEX = 900  # synthetic fast-tier step; never part of a CI-equivalence claim

# Steps whose test file matches this pattern (by source content) are repeated
# under ``--repeat N``.
DEFAULT_REPEAT_MATCH = r"\bthreading\b|\bmultiprocessing\b|\bBarrier\("

# Environment variables inherited from the developer's shell.  Everything else
# is dropped so that, e.g., a stray POSTGRES_DSN or MATVIEW_FUZZ_SEEDS in the
# developer's shell cannot silently change what a job tests.
_ENV_ALLOWLIST = {
    "PATH", "HOME", "USER", "LOGNAME", "TERM", "TMPDIR", "SHELL",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "XDG_CACHE_HOME",
    "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "CONTAINER_HOST", "CONTAINERS_CONF",
    "LIMA_HOME", "LIMA_INSTANCE",
}
_ENV_ALLOWLIST_PREFIXES = ("PIP_",)

KNOWN_LIMITATIONS = (
    "Runner OS/arch is the local machine, not GitHub-hosted ubuntu-latest x64.",
    "actions/setup-python is emulated with a venv from a local interpreter of the same "
    "major.minor; patch release and bundled SQLite library may differ from CI's build.",
    "actions/checkout is emulated by checking out the tested revision's tree; PR CI tests "
    "the merge ref (head merged into base). Equal only when the head contains the base tip.",
    "Dependencies resolve to the newest compatible releases at local run time; CI resolves "
    "at its own run time.",
    "Containers run under the local runtime (docker/podman/nerdctl, possibly in a VM) with "
    "renamed containers and remapped host ports; DSNs are rewritten to match.",
    "Jobs run in parallel on one machine; resource contention differs from isolated runners.",
    "Documentation-only classification lists renames as delete+add (more conservative than "
    "GitHub's rename detection).",
    "ci-deferred.yml, GitHub branch protection, merge queue and auto-merge are not exercised.",
)


class PlanError(Exception):
    """The workflow uses something this script cannot reproduce faithfully."""


# --------------------------------------------------------------------------
# Workflow loading
# --------------------------------------------------------------------------

def parse_workflow_text(text: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PlanError(
            "PyYAML is required to parse the workflow: pip install -r requirements-dev.txt"
        ) from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), dict):
        raise PlanError("workflow has no 'jobs' mapping")
    return data


# --------------------------------------------------------------------------
# Documentation-only classification (extracted from the classify job)
# --------------------------------------------------------------------------

def _find_classify_script(workflow: Mapping[str, Any]) -> str:
    job = workflow["jobs"].get("classify")
    if not isinstance(job, dict):
        raise PlanError("workflow has no 'classify' job")
    scripts = [
        s.get("with", {}).get("script")
        for s in job.get("steps", [])
        if str(s.get("uses", "")).startswith("actions/github-script@")
    ]
    if len(scripts) != 1 or not isinstance(scripts[0], str):
        raise PlanError("classify job must contain exactly one actions/github-script step with a script")
    return scripts[0]


def _parse_js_regex_literal(src: str, pos: int) -> Tuple[str, str, int]:
    """Parse a JS regex literal starting at src[pos] == '/'. Returns (body, flags, end)."""
    assert src[pos] == "/"
    i = pos + 1
    in_class = False
    while i < len(src):
        c = src[i]
        if c == "\\":
            i += 2
            continue
        if c == "\n":
            raise PlanError("unterminated regex literal in classify script")
        if in_class:
            if c == "]":
                in_class = False
        elif c == "[":
            in_class = True
        elif c == "/":
            body = src[pos + 1:i]
            j = i + 1
            while j < len(src) and src[j].isalpha():
                j += 1
            return body, src[i + 1:j], j
        i += 1
    raise PlanError("unterminated regex literal in classify script")


_PUNCT_ESCAPES = set("./-\\^$*+?()[]{}|")
_JS_LINE_TERMINATORS_CLASS = "[^\\n\\r\\u2028\\u2029]"


def js_regex_to_python(body: str, flags: str) -> "re.Pattern[str]":
    """Translate the restricted JS regex subset used by the whitelist.

    Only constructs with identical semantics are accepted; anything else raises
    PlanError.  Differences handled explicitly: JS ``.`` excludes all four line
    terminators; JS ``$`` (no m flag) matches only at the very end; JS ``/i``
    without ``u`` never folds non-ASCII characters onto ASCII ones (hence
    re.ASCII, so e.g. U+017F LONG S does not match ``s``).
    """
    if set(flags) - {"i"}:
        raise PlanError(f"unsupported regex flags {flags!r} in classify whitelist")
    out: List[str] = []
    i = 0
    in_class = False
    while i < len(body):
        c = body[i]
        if c == "\\":
            if i + 1 >= len(body):
                raise PlanError(f"dangling escape in regex /{body}/")
            nxt = body[i + 1]
            if nxt not in _PUNCT_ESCAPES:
                raise PlanError(f"unsupported regex escape \\{nxt} in /{body}/")
            out.append("/" if nxt == "/" else "\\" + nxt)
            i += 2
            continue
        if ord(c) > 127:
            raise PlanError(f"non-ASCII character in regex /{body}/")
        if in_class:
            if c == "[":
                raise PlanError(f"nested '[' in character class in /{body}/")
            if c == "]":
                in_class = False
            out.append(c)
        elif c == "[":
            if body.startswith("[]", i) or body.startswith("[^]", i):
                raise PlanError(f"empty character class in /{body}/")
            in_class = True
            out.append(c)
        elif c == ".":
            out.append(_JS_LINE_TERMINATORS_CLASS)
        elif c == "$":
            out.append(r"\Z")
        elif c == "^":
            out.append(r"\A")
        elif c == "(":
            if body.startswith("(?", i) and not body.startswith("(?:", i):
                raise PlanError(f"unsupported group syntax in /{body}/")
            out.append(c)
        elif c == "{":
            m = re.match(r"\{\d+(,\d*)?\}", body[i:])
            if not m:
                raise PlanError(f"unsupported brace in /{body}/")
            out.append(m.group(0))
            i += len(m.group(0))
            continue
        else:
            out.append(c)
        i += 1
    if in_class:
        raise PlanError(f"unterminated character class in /{body}/")
    re_flags = re.ASCII | (re.IGNORECASE if "i" in flags else 0)
    return re.compile("".join(out), re_flags)


_DOCS_ONLY_EXPR = re.compile(
    r"const docsOnly = files\.length > 0 && files\.length < (\d+) && "
    r"files\.every\(file => documentationPaths\.some\(pattern => pattern\.test\(file\.filename\)\)\);"
)


@dataclasses.dataclass(frozen=True)
class DocsClassifier:
    patterns: Tuple["re.Pattern[str]", ...]
    sources: Tuple[str, ...]
    max_files_exclusive: int

    def is_docs_path(self, path: str) -> bool:
        return any(p.search(path) for p in self.patterns)

    def docs_only(self, files: Sequence[str]) -> bool:
        return (
            len(files) > 0
            and len(files) < self.max_files_exclusive
            and all(self.is_docs_path(f) for f in files)
        )


def extract_docs_classifier(workflow: Mapping[str, Any]) -> DocsClassifier:
    script = _find_classify_script(workflow)
    # Strip // line comments (only whole-line comments; regex literals never start a line with //).
    lines = [ln for ln in script.splitlines() if not ln.strip().startswith("//")]
    code = "\n".join(lines)
    m = re.search(r"const\s+documentationPaths\s*=\s*\[", code)
    if not m:
        raise PlanError("cannot find 'const documentationPaths = [' in classify script")
    pos = m.end()
    patterns: List["re.Pattern[str]"] = []
    sources: List[str] = []
    while True:
        while pos < len(code) and code[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(code):
            raise PlanError("unterminated documentationPaths array")
        if code[pos] == "]":
            break
        if code[pos] != "/":
            raise PlanError(f"documentationPaths contains a non-regex-literal element near {code[pos:pos+30]!r}")
        body, flags, pos = _parse_js_regex_literal(code, pos)
        patterns.append(js_regex_to_python(body, flags))
        sources.append(f"/{body}/{flags}")
    if not patterns:
        raise PlanError("documentationPaths is empty")
    normalized = re.sub(r"\s+", " ", code)
    normalized = re.sub(r"\( ", "(", normalized)
    normalized = re.sub(r" \)", ")", normalized)
    em = _DOCS_ONLY_EXPR.search(normalized)
    if not em:
        raise PlanError("classify script's docsOnly expression changed shape; update presubmit.py")
    for required in (
        'if (context.eventName !== "pull_request") { core.setOutput("docs_only", "false"); return; }',
        'core.setOutput("docs_only", docsOnly ? "true" : "false");',
        "github.rest.pulls.listFiles",
    ):
        if required not in normalized:
            raise PlanError(f"classify script no longer contains {required!r}; update presubmit.py")
    return DocsClassifier(tuple(patterns), tuple(sources), int(em.group(1)))


# --------------------------------------------------------------------------
# Required jobs (extracted from the "CI success" aggregator)
# --------------------------------------------------------------------------

_DOCS_TRUE = "needs.classify.outputs.docs_only == 'true'"
_DOCS_NOT_TRUE = "needs.classify.outputs.docs_only != 'true'"


def _strip_expr(value: Any) -> str:
    s = str(value).strip()
    m = re.fullmatch(r"\$\{\{\s*(.*?)\s*\}\}", s, re.S)
    return m.group(1) if m else s


def extract_required_jobs(workflow: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (docs_only_required, full_required) job ids from the CI success job."""
    jobs = workflow["jobs"]
    agg = [jid for jid, j in jobs.items() if isinstance(j, dict) and j.get("name") == REQUIRED_CHECK_NAME]
    if len(agg) != 1:
        raise PlanError(f"expected exactly one job named {REQUIRED_CHECK_NAME!r}, found {agg}")
    job = jobs[agg[0]]
    needs = job.get("needs") or []
    needs = [needs] if isinstance(needs, str) else list(needs)
    scripts = [s.get("run", "") for s in job.get("steps", []) if "python3 - <<'PY'" in s.get("run", "")]
    if len(scripts) != 1:
        raise PlanError("CI success job must contain one embedded python3 heredoc step")
    m = re.search(r"python3 - <<'PY'\n(.*?)\n\s*PY\s*$", scripts[0], re.S)
    if not m:
        raise PlanError("cannot extract the CI success heredoc")
    tree = ast.parse(textwrap.dedent(m.group(1)))
    found: Optional[Tuple[List[str], List[str]]] = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "required_jobs"
        ):
            v = node.value
            if not (isinstance(v, ast.IfExp) and isinstance(v.test, ast.Name) and v.test.id == "docs_only"):
                raise PlanError("required_jobs is no longer 'A if docs_only else B'")
            try:
                found = (list(ast.literal_eval(v.body)), list(ast.literal_eval(v.orelse)))
            except ValueError as exc:
                raise PlanError("required_jobs lists are not literals") from exc
    if found is None:
        raise PlanError("CI success script has no required_jobs assignment")
    docs_req, full_req = found
    for jid in docs_req + full_req:
        if jid not in needs:
            raise PlanError(f"required job {jid!r} is not in CI success needs")
        if jid not in jobs:
            raise PlanError(f"required job {jid!r} does not exist")
    for jid in full_req:
        if jid == "classify":
            continue
        if _strip_expr(jobs[jid].get("if", "")) != _DOCS_NOT_TRUE:
            raise PlanError(f"job {jid!r} is required on the full path but its 'if' is not {_DOCS_NOT_TRUE!r}")
    for jid in docs_req:
        if jid == "classify":
            continue
        if _strip_expr(jobs[jid].get("if", "")) != _DOCS_TRUE:
            raise PlanError(f"job {jid!r} is required on the docs path but its 'if' is not {_DOCS_TRUE!r}")
    if "classify" not in docs_req or "classify" not in full_req:
        raise PlanError("classify must be required on both paths")
    return docs_req, full_req


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Service:
    name: str
    image: str
    env: Dict[str, str]
    ports: List[Tuple[int, int]]  # (ci_host_port, container_port)
    options: List[str]  # passthrough docker-run options
    health_cmd: Optional[str]
    health_interval: float
    health_retries: int


@dataclasses.dataclass
class Step:
    index: int
    name: str
    kind: str  # run | checkout | setup-python | upload-artifact
    condition: str  # success | always
    run: Optional[str] = None
    env: Dict[str, str] = dataclasses.field(default_factory=dict)  # merged workflow+job+step env
    python_version: Optional[str] = None


@dataclasses.dataclass
class Job:
    job_id: str
    key: str
    slug: str
    name: str
    matrix: Dict[str, Any]
    steps: List[Step]
    services: List[Service]
    python_version: Optional[str]
    needs_containers: bool
    timeout_s: Optional[float]


_EXPR_RE = re.compile(r"\$\{\{\s*(.*?)\s*\}\}", re.S)


def _fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    return str(v)


def evaluate(text: Any, *, job_id: str, matrix: Mapping[str, Any], index: int, where: str, strict: bool = True) -> str:
    def repl(m: "re.Match[str]") -> str:
        expr = m.group(1)
        mm = re.fullmatch(r"matrix\.([A-Za-z0-9_-]+)", expr)
        if mm:
            # GitHub renders a missing matrix property as the empty string.
            return _fmt_value(matrix.get(mm.group(1)))
        if expr == "github.job":
            return job_id
        if expr == "strategy.job-index":
            return str(index)
        if expr == "github.run_attempt":
            return "1"
        if strict:
            raise PlanError(f"unsupported expression ${{{{ {expr} }}}} in {where}")
        return m.group(0)

    return _EXPR_RE.sub(repl, _fmt_value(text))


def expand_matrix(job_id: str, strategy: Any) -> List[Dict[str, Any]]:
    if not strategy:
        return [{}]
    if not isinstance(strategy, dict) or set(strategy) - {"fail-fast", "matrix", "max-parallel"}:
        raise PlanError(f"{job_id}: unsupported strategy keys {sorted(strategy)}")
    matrix = strategy.get("matrix")
    if matrix is None:
        return [{}]
    if not isinstance(matrix, dict):
        raise PlanError(f"{job_id}: matrix must be a literal mapping (expressions unsupported)")
    if "exclude" in matrix:
        raise PlanError(f"{job_id}: matrix.exclude is unsupported")
    include = matrix.get("include") or []
    axes = {k: v for k, v in matrix.items() if k != "include"}
    for k, v in axes.items():
        if not isinstance(v, list) or not v:
            raise PlanError(f"{job_id}: matrix axis {k!r} must be a non-empty literal list")
    if axes and include:
        raise PlanError(f"{job_id}: combining matrix axes with include is unsupported")
    if axes:
        keys = list(axes)
        return [dict(zip(keys, combo)) for combo in itertools.product(*(axes[k] for k in keys))]
    if not all(isinstance(r, dict) for r in include):
        raise PlanError(f"{job_id}: matrix.include rows must be mappings")
    return [dict(r) for r in include]


def _parse_service(job_id: str, name: str, spec: Mapping[str, Any], ev: Callable[[Any, str], str]) -> Service:
    unknown = set(spec) - {"image", "env", "ports", "options"}
    if unknown:
        raise PlanError(f"{job_id}: service {name}: unsupported keys {sorted(unknown)}")
    env = {str(k): ev(v, f"services.{name}.env") for k, v in (spec.get("env") or {}).items()}
    ports: List[Tuple[int, int]] = []
    for p in spec.get("ports") or []:
        m = re.fullmatch(r"(\d+):(\d+)", ev(p, f"services.{name}.ports"))
        if not m:
            raise PlanError(f"{job_id}: service {name}: unsupported port spec {p!r} (need HOST:CONTAINER)")
        ports.append((int(m.group(1)), int(m.group(2))))
    options: List[str] = []
    health_cmd = None
    interval = 10.0
    retries = 3
    raw = shlex.split(ev(spec.get("options", ""), f"services.{name}.options"))
    i = 0
    while i < len(raw):
        opt = raw[i]
        key, eq, val = opt.partition("=")
        if key in ("--health-cmd", "--health-interval", "--health-retries", "--health-timeout"):
            if not eq:
                i += 1
                if i >= len(raw):
                    raise PlanError(f"{job_id}: service {name}: {key} without a value")
                val = raw[i]
            if key == "--health-cmd":
                health_cmd = val
            elif key == "--health-interval":
                interval = _parse_duration(val)
            elif key == "--health-retries":
                retries = int(val)
            options.append(f"{key}={val}")
        else:
            options.append(opt)
        i += 1
    image = ev(spec.get("image", ""), f"services.{name}.image")
    if not image:
        raise PlanError(f"{job_id}: service {name} has no image")
    return Service(name, image, env, ports, options, health_cmd, interval, retries)


def _parse_duration(val: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|m)?", val)
    if not m:
        raise PlanError(f"unsupported duration {val!r}")
    n = float(m.group(1))
    return {"ms": n / 1000, "s": n, "m": n * 60, None: n}[m.group(2)]


_ALLOWED_JOB_KEYS = {"name", "needs", "if", "runs-on", "strategy", "services", "steps", "outputs", "env", "timeout-minutes", "permissions"}
_ALLOWED_STEP_KEYS = {"name", "id", "if", "env", "run", "uses", "with", "shell"}
_DOCKER_RE = re.compile(r"(?<![\w-])docker(?![\w-])")


def _matrix_key(job_id: str, matrix: Mapping[str, Any]) -> str:
    if not matrix:
        return job_id
    return job_id + "[" + ",".join(f"{k}={_fmt_value(v)}" for k, v in matrix.items()) + "]"


def build_jobs(workflow: Mapping[str, Any], job_ids: Sequence[str]) -> List[Job]:
    wf_env = workflow.get("env") or {}
    out: List[Job] = []
    for job_id in job_ids:
        spec = workflow["jobs"][job_id]
        unknown = set(spec) - _ALLOWED_JOB_KEYS
        if unknown:
            raise PlanError(f"{job_id}: unsupported job keys {sorted(unknown)}")
        if "container" in spec:
            raise PlanError(f"{job_id}: job containers are unsupported")
        for idx, matrix in enumerate(expand_matrix(job_id, spec.get("strategy"))):
            def ev(v: Any, where: str, strict: bool = True) -> str:
                return evaluate(v, job_id=job_id, matrix=matrix, index=idx, where=f"{job_id}.{where}", strict=strict)

            base_env = {str(k): ev(v, "env") for k, v in wf_env.items()}
            base_env.update({str(k): ev(v, "env") for k, v in (spec.get("env") or {}).items()})
            services = [
                _parse_service(job_id, n, s, ev) for n, s in (spec.get("services") or {}).items()
            ]
            steps: List[Step] = []
            pyver: Optional[str] = None
            for sidx, s in enumerate(spec.get("steps") or []):
                unknown = set(s) - _ALLOWED_STEP_KEYS
                if unknown:
                    raise PlanError(f"{job_id} step {sidx}: unsupported keys {sorted(unknown)}")
                if s.get("shell") not in (None, "bash"):
                    raise PlanError(f"{job_id} step {sidx}: unsupported shell {s.get('shell')!r}")
                cond_raw = s.get("if")
                if cond_raw is None:
                    cond = "success"
                elif _strip_expr(cond_raw) == "always()":
                    cond = "always"
                else:
                    raise PlanError(f"{job_id} step {sidx}: unsupported if: {cond_raw!r}")
                name = ev(s.get("name") or s.get("run") or s.get("uses") or f"step {sidx}", "name", strict=False)
                env = dict(base_env)
                env.update({str(k): ev(v, f"steps[{sidx}].env") for k, v in (s.get("env") or {}).items()})
                if "run" in s and "uses" in s:
                    raise PlanError(f"{job_id} step {sidx}: both run and uses")
                if "run" in s:
                    steps.append(Step(sidx, name, "run", cond, run=ev(s["run"], f"steps[{sidx}].run"), env=env))
                    continue
                uses = str(s.get("uses", ""))
                action = uses.split("@", 1)[0]
                if action == "actions/checkout":
                    steps.append(Step(sidx, name, "checkout", cond, env=env))
                elif action == "actions/setup-python":
                    v = (s.get("with") or {}).get("python-version")
                    if v is None:
                        raise PlanError(f"{job_id} step {sidx}: setup-python without python-version")
                    pyver = ev(v, f"steps[{sidx}].with.python-version")
                    if not re.fullmatch(r"3\.\d+", pyver):
                        raise PlanError(f"{job_id}: unsupported python-version {pyver!r} (need 3.N)")
                    steps.append(Step(sidx, name, "setup-python", cond, env=env, python_version=pyver))
                elif action == "actions/upload-artifact":
                    steps.append(Step(sidx, name, "upload-artifact", cond, env=env))
                else:
                    raise PlanError(f"{job_id} step {sidx}: unsupported action {uses!r}")
            needs_containers = bool(services) or any(
                st.kind == "run" and _DOCKER_RE.search(st.run or "") for st in steps
            )
            tmo = spec.get("timeout-minutes")
            out.append(Job(
                job_id=job_id,
                key=_matrix_key(job_id, matrix),
                slug=f"{job_id}-{idx}",
                name=ev(spec.get("name", job_id), "name", strict=False),
                matrix=matrix,
                steps=steps,
                services=services,
                python_version=pyver,
                needs_containers=needs_containers,
                timeout_s=float(tmo) * 60 if tmo else None,
            ))
    return out


def step_test_file(step: Step) -> Optional[str]:
    if step.kind != "run" or not step.run:
        return None
    m = re.search(r"(?:python3?|pytest)\b[^\n]*?\b(test_[\w]+\.py)\b", step.run)
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# Script transformation (container runtime, names, ports, sudo)
# --------------------------------------------------------------------------

_HOSTPORT_RE = re.compile(r"\b(127\.0\.0\.1|localhost):(\d+)\b")


def rewrite_host_ports(text: str, port_map: Mapping[int, int], host: str) -> str:
    def repl(m: "re.Match[str]") -> str:
        p = int(m.group(2))
        return f"{host}:{port_map[p]}" if p in port_map else m.group(0)

    return _HOSTPORT_RE.sub(repl, text)


@dataclasses.dataclass
class ContainerPlan:
    port_map: Dict[int, int]  # CI host port -> local host port
    names: Dict[str, str]  # CI container name -> local container name


def plan_containers(job: Job, run_tag: str, alloc_port: Callable[[], int]) -> ContainerPlan:
    port_map: Dict[int, int] = {}
    names: Dict[str, str] = {}

    def add_port(ci_port: int) -> None:
        if ci_port in port_map:
            raise PlanError(f"{job.key}: CI host port {ci_port} published twice; DSN rewrite would be ambiguous")
        port_map[ci_port] = alloc_port()

    for svc in job.services:
        names[svc.name] = f"presubmit-{run_tag}-{job.slug}-{svc.name}"
        for host_port, _ in svc.ports:
            add_port(host_port)
    for st in job.steps:
        if st.kind != "run" or not _DOCKER_RE.search(st.run or ""):
            continue
        script = st.run or ""
        if re.search(r"(?<![\w-])docker\s+run\b", script):
            if re.search(r"\s(-P|--publish-all)\b", script):
                raise PlanError(f"{job.key}: docker run -P is unsupported")
            nm = re.findall(r"--name[ =](\S+)", script)
            if not nm:
                raise PlanError(f"{job.key}: docker run without --name cannot be isolated")
            for n in nm:
                names[n] = f"presubmit-{run_tag}-{job.slug}-{n}"
            pubs = re.findall(r"(?:\s-p|--publish)[ =](\S+)", script)
            for pub in pubs:
                m = re.fullmatch(r"(\d+):(\d+)", pub)
                if not m:
                    raise PlanError(f"{job.key}: unsupported publish spec {pub!r} (need HOST:CONTAINER)")
                add_port(int(m.group(1)))
    return ContainerPlan(port_map, names)


def transform_script(script: str, cplan: ContainerPlan, *, runtime: Optional[Sequence[str]], host: str,
                     bind: str, allow_sudo: bool) -> str:
    body = rewrite_host_ports(script, cplan.port_map, host)

    def pub(m: "re.Match[str]") -> str:
        ci_host, cport = int(m.group(2)), m.group(3)
        target = f"{bind}:{cplan.port_map[ci_host]}:{cport}" if bind else f"{cplan.port_map[ci_host]}:{cport}"
        return f"{m.group(1)}{target}"

    body = re.sub(r"((?:\s-p|--publish)[ =])(\d+):(\d+)\b", pub, body)
    for old, new in cplan.names.items():
        body = re.sub(rf"(?<![\w.-]){re.escape(old)}(?![\w.-])", new, body)
    prelude = []
    if runtime:
        prelude.append(f'docker() {{ {shlex.join(list(runtime))} "$@"; }}')
    if not allow_sudo:
        prelude.append(
            'sudo() { echo "presubmit: blocked sudo $*" >&2; : > "$PRESUBMIT_SUDO_MARKER"; exit 97; }'
        )
    return "\n".join(prelude + [body]) + "\n"


# --------------------------------------------------------------------------
# Process runner (injectable for tests)
# --------------------------------------------------------------------------

class ProcessRunner:
    def __init__(self) -> None:
        self._active: Set[int] = set()
        self._active_lock = threading.Lock()

    def kill_all(self) -> None:
        """Kill every running step's process group (used on Ctrl-C)."""
        with self._active_lock:
            pids = list(self._active)
        for pid in pids:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def run(self, argv: Sequence[str], *, env: Optional[Mapping[str, str]], cwd: Optional[str],
            log_path: Path, timeout: Optional[float]) -> int:
        with open(log_path, "ab") as log:
            log.write(f"$ {shlex.join(list(argv))}\n".encode())
            log.flush()
            try:
                proc = subprocess.Popen(list(argv), env=dict(env) if env is not None else None, cwd=cwd,
                                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True)
            except FileNotFoundError:
                log.write(b"presubmit: executable not found\n")
                return 127
            with self._active_lock:
                self._active.add(proc.pid)
            try:
                return proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                log.write(f"\npresubmit: TIMEOUT after {timeout}s\n".encode())
                return 124
            except BaseException:
                os.killpg(proc.pid, signal.SIGKILL)
                raise
            finally:
                with self._active_lock:
                    self._active.discard(proc.pid)

    def capture(self, argv: Sequence[str], *, env: Optional[Mapping[str, str]] = None,
                timeout: float = 60) -> Tuple[int, str]:
        try:
            p = subprocess.run(list(argv), env=dict(env) if env is not None else None, capture_output=True,
                               text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        except (FileNotFoundError, PermissionError):
            return 127, ""
        except subprocess.TimeoutExpired:
            return 124, ""
        return p.returncode, p.stdout + p.stderr

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


# --------------------------------------------------------------------------
# Git helpers
# --------------------------------------------------------------------------

def git(repo: Path, *args: str, env: Optional[Mapping[str, str]] = None) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, env=dict(env) if env else None)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.decode(errors='replace').strip()}")
    return p.stdout.decode("utf-8", "surrogateescape")


def git_common_dir(repo: Path) -> Path:
    d = Path(git(repo, "rev-parse", "--git-common-dir").strip())
    return d if d.is_absolute() else (repo / d).resolve()


def resolve_rev(repo: Path, rev: str) -> Tuple[str, str]:
    sha = git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").strip()
    tree = git(repo, "rev-parse", f"{sha}^{{tree}}").strip()
    return sha, tree


def changed_paths(repo: Path, base: str, rev: Optional[str]) -> List[str]:
    """Paths changed by rev (or the working tree when rev is None) relative to merge-base(base, rev)."""
    head = rev or "HEAD"
    mb = git(repo, "merge-base", base, head).strip()
    if rev is not None:
        out = git(repo, "diff", "--name-only", "--no-renames", "-z", mb, rev)
        return sorted({p for p in out.split("\0") if p})
    out = git(repo, "diff", "--name-only", "--no-renames", "-z", mb)
    untracked = git(repo, "ls-files", "-z", "-o", "--exclude-standard")
    return sorted({p for p in (out + untracked).split("\0") if p})


def checkout_rev(repo: Path, rev: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(td) / "index"))
        git(repo, "read-tree", rev, env=env)
        git(repo, "checkout-index", "-a", f"--prefix={dest}/", env=env)


def checkout_worktree(repo: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    files = git(repo, "ls-files", "-z", "-c", "-o", "--exclude-standard").split("\0")
    for rel in files:
        if not rel:
            continue
        src = repo / rel
        if not src.exists() and not src.is_symlink():
            continue  # deleted in the working tree
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_symlink():
            os.symlink(os.readlink(src), dst)
        elif src.is_file():
            shutil.copy2(src, dst)


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

@dataclasses.dataclass
class StepResult:
    name: str
    status: str  # passed | failed | skipped | blocked | emulated
    detail: str = ""
    attempts: List[int] = dataclasses.field(default_factory=list)
    log: Optional[str] = None


@dataclasses.dataclass
class JobResult:
    key: str
    status: str  # passed | failed | incomplete | substituted | skipped
    reasons: List[str] = dataclasses.field(default_factory=list)
    python: Optional[str] = None
    servers: List[str] = dataclasses.field(default_factory=list)
    deviations: List[str] = dataclasses.field(default_factory=list)
    steps: List[StepResult] = dataclasses.field(default_factory=list)
    duration_s: float = 0.0
    log_dir: Optional[str] = None


@dataclasses.dataclass
class Options:
    tier: str = "full"
    jobs: int = 1
    repeat: int = 1
    repeat_match: str = DEFAULT_REPEAT_MATCH
    only: Sequence[str] = ()
    allow_sudo: bool = False
    substitute_python: Optional[str] = None
    keep_going: bool = False
    keep_containers: bool = False
    step_timeout: float = 3600.0
    host: str = "127.0.0.1"
    bind: str = "127.0.0.1"
    service_timeout: float = 300.0


class Executor:
    def __init__(self, *, repo: Path, run_dir: Path, rev_sha: str, opts: Options, runner: ProcessRunner,
                 out: Callable[[str], None] = print, alloc_port: Optional[Callable[[], int]] = None,
                 use_worktree: bool = False) -> None:
        self.repo = repo
        self.run_dir = run_dir
        self.rev_sha = rev_sha
        self.opts = opts
        self.runner = runner
        self._out = out
        self._lock = threading.Lock()
        self._py_cache: Dict[str, Optional[Tuple[str, str]]] = {}
        self._runtime: Optional[Tuple[Optional[List[str]], str]] = None
        self._used_ports: Set[int] = set()
        self._alloc = alloc_port or self._free_port
        self.use_worktree = use_worktree
        self.run_tag = run_dir.name.replace("_", "-").lower()[-24:]
        self.venv_cache: Optional[Path] = None
        self.cancelled = threading.Event()

    def say(self, msg: str) -> None:
        with self._lock:
            self._out(msg)

    def _free_port(self) -> int:
        with self._lock:
            while True:
                with socket.socket() as s:
                    s.bind(("127.0.0.1", 0))
                    port = s.getsockname()[1]
                if port not in self._used_ports:
                    self._used_ports.add(port)
                    return port

    # -- environment discovery ---------------------------------------------
    def find_python(self, version: str) -> Optional[Tuple[str, str]]:
        with self._lock:
            if version in self._py_cache:
                return self._py_cache[version]
        candidates = [os.environ.get(f"PRESUBMIT_PYTHON_{version.replace('.', '_')}"), f"python{version}"]
        found = None
        for c in candidates:
            if not c:
                continue
            rc, out = self.runner.capture([c, "-c", "import sys;print('%d.%d' % sys.version_info[:2]);print(sys.version.split()[0])"])
            lines = out.split()
            if rc == 0 and len(lines) >= 2 and lines[0] == version:
                found = (c, lines[1])
                break
        with self._lock:
            self._py_cache[version] = found
        return found

    def container_runtime(self) -> Tuple[Optional[List[str]], str]:
        with self._lock:
            if self._runtime is not None:
                return self._runtime
        configured = os.environ.get("PRESUBMIT_CONTAINER_RUNTIME")
        candidates = [shlex.split(configured)] if configured else [["docker"], ["podman"]]
        result: Tuple[Optional[List[str]], str] = (None, "")
        tried = []
        for c in candidates:
            if not c:
                continue
            rc, out = self.runner.capture(c + ["info"], timeout=60)
            if rc == 0:
                result = (c, shlex.join(c))
                break
            tried.append(f"'{shlex.join(c)} info' exited {rc}")
        if result[0] is None:
            src = "PRESUBMIT_CONTAINER_RUNTIME" if configured else "auto-detect (docker, podman)"
            result = (None, f"no working container runtime via {src}: {'; '.join(tried) or 'nothing configured'}")
        with self._lock:
            self._runtime = result
        return result

    def base_env(self, job: Job, job_dir: Path, workspace: Path) -> Dict[str, str]:
        env = {k: v for k, v in os.environ.items()
               if k in _ENV_ALLOWLIST or k.startswith(_ENV_ALLOWLIST_PREFIXES) or k.startswith("LC_")}
        env.update({
            "CI": "true",
            "GITHUB_ACTIONS": "true",
            "GITHUB_JOB": job.job_id,
            "GITHUB_SHA": self.rev_sha,
            "GITHUB_RUN_ID": f"local-presubmit-{self.run_dir.name}",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_WORKSPACE": str(workspace),
            "GITHUB_STEP_SUMMARY": str(job_dir / "step-summary.md"),
            "RUNNER_OS": "Linux",
            "RUNNER_TEMP": str(job_dir / "tmp"),
            "LANG": "C.UTF-8",
            "TZ": "UTC",
            "PRESUBMIT": "1",
            "PRESUBMIT_SUDO_MARKER": str(job_dir / "sudo-blocked"),
        })
        return env

    # -- jobs ----------------------------------------------------------------
    def run_job(self, job: Job, selected_steps: Optional[Set[int]] = None) -> JobResult:
        res = JobResult(job.key, "passed")
        t0 = time.monotonic()
        runtime: Optional[List[str]] = None
        if job.needs_containers:
            runtime, desc = self.container_runtime()
            if runtime is None:
                res.status = "incomplete"
                res.reasons.append(f"native engine NOT verified: {desc}")
                if job.python_version and self.find_python(job.python_version) is None:
                    res.reasons.append(f"also: python{job.python_version} not found")
                return res
        interp: Optional[Tuple[str, str]] = None
        if job.python_version:
            interp = self.find_python(job.python_version)
            if interp is None:
                if self.opts.substitute_python:
                    rc, out = self.runner.capture([self.opts.substitute_python, "-c", "import sys;print(sys.version.split()[0])"])
                    if rc != 0:
                        res.status = "incomplete"
                        res.reasons.append(f"substitute interpreter {self.opts.substitute_python!r} unusable")
                        return res
                    interp = (self.opts.substitute_python, out.strip())
                    res.deviations.append(f"python {job.python_version} unavailable; SUBSTITUTED {interp[1]}")
                else:
                    res.status = "incomplete"
                    res.reasons.append(
                        f"python{job.python_version} not found (set PRESUBMIT_PYTHON_{job.python_version.replace('.', '_')}"
                        " or pass --substitute-python, which never counts as verified)")
                    return res
        job_dir = self.run_dir / job.slug
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "tmp").mkdir(exist_ok=True)
        res.log_dir = str(job_dir)
        workspace = job_dir / "checkout"
        if self.use_worktree:
            checkout_worktree(self.repo, workspace)
        else:
            checkout_rev(self.repo, self.rev_sha, workspace)
        try:
            cplan = plan_containers(job, self.run_tag, self._alloc)
        except PlanError as exc:
            res.status = "failed"
            res.reasons.append(f"plan error: {exc}")
            return res
        if cplan.port_map:
            res.deviations.append("ports remapped " + ", ".join(f"{a}->{b}" for a, b in sorted(cplan.port_map.items())))
        env = self.base_env(job, job_dir, workspace)
        started: List[str] = []
        failed = False
        try:
            for svc in job.services:
                ok, detail = self._start_service(svc, cplan, runtime or [], job_dir)
                started.append(cplan.names[svc.name])
                if not ok:
                    res.status = "failed"
                    res.reasons.append(f"service {svc.name} ({svc.image}): {detail}")
                    failed = True
                    break
            started.extend(n for k, n in cplan.names.items() if k not in {s.name for s in job.services})
            for step in job.steps:
                if self.cancelled.is_set():
                    res.steps.append(StepResult(step.name, "skipped", "interrupted"))
                    if res.status == "passed":
                        res.status = "incomplete"
                        res.reasons.append("interrupted by user")
                    continue
                if failed and step.condition != "always" and not self.opts.keep_going:
                    res.steps.append(StepResult(step.name, "skipped", "previous step failed"))
                    continue
                if selected_steps is not None and step.index not in selected_steps:
                    res.steps.append(StepResult(step.name, "skipped", "not selected by fast tier"))
                    continue
                sr = self._run_step(job, step, env, cplan, runtime, interp, job_dir, workspace, res)
                res.steps.append(sr)
                if sr.status == "failed":
                    failed = True
                    res.status = "failed"
                    res.reasons.append(f"step failed: {step.name}")
                elif sr.status == "blocked":
                    res.status = "incomplete" if res.status != "failed" else res.status
                    res.reasons.append(f"step blocked: {step.name}: {sr.detail}")
                    failed = True
        finally:
            if runtime and started and not self.opts.keep_containers:
                for name in started:
                    if failed:
                        self.runner.run(runtime + ["logs", name], env=None, cwd=None,
                                        log_path=job_dir / f"container-{name}.log", timeout=60)
                    self.runner.run(runtime + ["rm", "-f", "-v", name], env=None, cwd=None,
                                    log_path=job_dir / "cleanup.log", timeout=120)
        if res.status == "passed" and res.deviations and any("SUBSTITUTED" in d for d in res.deviations):
            res.status = "substituted"
        res.duration_s = time.monotonic() - t0
        return res

    def _start_service(self, svc: Service, cplan: ContainerPlan, runtime: List[str], job_dir: Path) -> Tuple[bool, str]:
        log = job_dir / f"service-{svc.name}.log"
        name = cplan.names[svc.name]
        rc = self.runner.run(runtime + ["pull", svc.image], env=None, cwd=None, log_path=log, timeout=1800)
        if rc != 0:
            return False, f"image pull failed (rc={rc}); no fallback image is used"
        argv = runtime + ["run", "-d", "--name", name]
        for k, v in svc.env.items():
            argv += ["-e", f"{k}={v}"]
        for ci_host, cport in svc.ports:
            hp = cplan.port_map[ci_host]
            argv += ["-p", f"{self.opts.bind}:{hp}:{cport}" if self.opts.bind else f"{hp}:{cport}"]
        argv += svc.options + [svc.image]
        rc = self.runner.run(argv, env=None, cwd=None, log_path=log, timeout=300)
        if rc != 0:
            return False, f"container start failed (rc={rc})"
        if not svc.health_cmd:
            return True, "started (no health command)"
        deadline = time.monotonic() + self.opts.service_timeout
        attempts = 0
        while True:
            attempts += 1
            rc = self.runner.run(runtime + ["exec", name, "sh", "-c", svc.health_cmd], env=None, cwd=None,
                                 log_path=log, timeout=max(5.0, svc.health_interval * 2))
            if rc == 0:
                return True, f"healthy after {attempts} probe(s)"
            if attempts >= max(svc.health_retries, 1) and time.monotonic() >= deadline:
                return False, f"health command never succeeded ({attempts} probes)"
            if attempts >= 10_000:
                return False, "health command never succeeded"
            self.runner.sleep(svc.health_interval)

    def _run_step(self, job: Job, step: Step, env: Dict[str, str], cplan: ContainerPlan,
                  runtime: Optional[List[str]], interp: Optional[Tuple[str, str]], job_dir: Path,
                  workspace: Path, res: JobResult) -> StepResult:
        log = job_dir / f"{step.index:02d}-{re.sub(r'[^A-Za-z0-9]+', '-', step.name)[:50].strip('-')}.log"
        if step.kind == "checkout":
            return StepResult(step.name, "emulated", "workspace = checkout of tested tree", log=None)
        if step.kind == "upload-artifact":
            return StepResult(step.name, "emulated", f"artifacts kept in {workspace}")
        if step.kind == "setup-python":
            assert interp is not None
            venv = self._venv_dir(job, interp, job_dir)
            if not (venv / "bin" / "python3").exists():
                rc = self.runner.run([interp[0], "-m", "venv", str(venv)], env=None, cwd=None, log_path=log,
                                     timeout=600)
                if rc != 0:
                    return StepResult(step.name, "failed", f"venv creation failed rc={rc}", [rc], str(log))
            env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
            env["VIRTUAL_ENV"] = str(venv)
            rc, out = self.runner.capture([str(venv / "bin" / "python3"), "-c",
                                           "import sys,sqlite3;print(sys.version.split()[0], 'sqlite', sqlite3.sqlite_version)"])
            res.python = out.strip() if rc == 0 else interp[1]
            return StepResult(step.name, "emulated", f"venv from {interp[0]} ({res.python})")
        script = step.run or ""
        if self.venv_cache is not None and "pip install" in script and (Path(env.get("VIRTUAL_ENV", "/nonexistent")) / ".presubmit-ok").exists():
            return StepResult(step.name, "skipped", "fast tier: cached venv already provisioned")
        senv = dict(env)
        senv.update({k: rewrite_host_ports(v, cplan.port_map, self.opts.host) for k, v in step.env.items()})
        script = transform_script(script, cplan, runtime=runtime, host=self.opts.host, bind=self.opts.bind,
                                  allow_sudo=self.opts.allow_sudo)
        script_path = job_dir / f"{step.index:02d}.sh"
        script_path.write_text(script)
        reps = 1
        tf = step_test_file(step)
        if self.opts.repeat > 1 and tf and (workspace / tf).is_file():
            if re.search(self.opts.repeat_match, (workspace / tf).read_text(errors="replace")):
                reps = self.opts.repeat
        attempts: List[int] = []
        timeout = self.opts.step_timeout
        marker = Path(senv["PRESUBMIT_SUDO_MARKER"])
        for _ in range(reps):
            rc = self.runner.run(["bash", "-e", str(script_path)], env=senv, cwd=str(workspace), log_path=log,
                                 timeout=timeout)
            attempts.append(rc)
            if marker.exists():
                marker.unlink()
                return StepResult(step.name, "blocked",
                                  "step invoked sudo; rerun with --allow-sudo to execute it", attempts, str(log))
        venv_dir = Path(senv.get("VIRTUAL_ENV", "/nonexistent"))
        if self.venv_cache is not None and "pip install" in script and all(a == 0 for a in attempts) and venv_dir.is_dir():
            (venv_dir / ".presubmit-ok").touch()
        servers = _verified_lines(log)
        if servers:
            res.servers.extend(servers)
        failures = sum(1 for a in attempts if a != 0)
        if failures:
            detail = f"rc={attempts[-1]}" if reps == 1 else f"FLAKY/FAILED {failures}/{reps} attempts failed (rcs={attempts})"
            return StepResult(step.name, "failed", detail, attempts, str(log))
        return StepResult(step.name, "passed", f"{reps}x" if reps > 1 else "", attempts, str(log))

    def _venv_dir(self, job: Job, interp: Tuple[str, str], job_dir: Path) -> Path:
        if self.venv_cache is None:
            return job_dir / "venv"
        return self.venv_cache


def _verified_lines(log: Path) -> List[str]:
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return []
    return [ln.strip() for ln in text.splitlines() if ln.startswith("[NATIVE ENGINE VERIFIED]")]


# --------------------------------------------------------------------------
# Fast-tier selection
# --------------------------------------------------------------------------

def affected_tests(test_files: Sequence[str], changed: Sequence[str], classifier: DocsClassifier,
                   read_source: Callable[[str], Optional[str]], modules: Sequence[str] = ()) -> Tuple[Set[str], str]:
    """Return (affected test files, reason). Conservative: unknown changes affect everything.

    ``modules`` lists every top-level ``*.py`` file of the tested tree; a test is
    affected when it textually references (transitively, through any module)
    the stem of a changed top-level module.
    """
    tests = set(test_files)
    relevant = [p for p in changed if not classifier.is_docs_path(p)]
    if not relevant:
        return set(), "no non-documentation changes"
    others = [p for p in relevant if not (p.endswith(".py") and "/" not in p)]
    if others:
        return tests, f"non-top-level-python change(s) {others[:3]} -> all tests"
    affected_modules = {Path(p).stem for p in relevant if p not in tests}
    selected = {p for p in relevant if p in tests}
    sources = {}
    for f in tests | set(relevant) | set(modules):
        src = read_source(f)
        if src is not None:
            sources[f] = src
    changed_any = True
    while changed_any:
        changed_any = False
        for f, src in sources.items():
            stem = Path(f).stem
            if stem in affected_modules:
                continue
            if any(re.search(rf"\b{re.escape(m)}\b", src) for m in affected_modules):
                affected_modules.add(stem)
                changed_any = True
    selected |= {t for t in tests if Path(t).stem in affected_modules}
    return selected, f"changed: {relevant[:5]}{'...' if len(relevant) > 5 else ''}"


# --------------------------------------------------------------------------
# Stamp
# --------------------------------------------------------------------------

def stamp_path(repo: Path, tree: str) -> Path:
    return git_common_dir(repo) / "presubmit" / "stamps" / f"{tree}.json"


def write_stamp(repo: Path, *, tree: str, head: str, base: str, base_sha: str, docs_only: bool,
                required: Sequence[str], results: Sequence[JobResult]) -> Path:
    data = {
        "schema": STAMP_SCHEMA,
        "tree": tree,
        "head": head,
        "base": base,
        "base_sha": base_sha,
        "docs_only": docs_only,
        "tier": "full",
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": platform.platform(),
        "required_jobs": list(required),
        "jobs": {r.key: {"status": r.status, "python": r.python, "servers": r.servers,
                         "deviations": r.deviations} for r in results},
        "not_covered": list(KNOWN_LIMITATIONS),
    }
    p = stamp_path(repo, tree)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(p)
    return p


def required_job_keys(workflow: Mapping[str, Any], docs_only: bool) -> List[str]:
    docs_req, full_req = extract_required_jobs(workflow)
    req = docs_req if docs_only else full_req
    keys = ["classify"]
    keys += [j.key for j in build_jobs(workflow, [x for x in req if x != "classify"])]
    return keys


def check_stamp(repo: Path, rev: str, base: str) -> Tuple[bool, str]:
    try:
        head, tree = resolve_rev(repo, rev)
    except RuntimeError as exc:
        return False, str(exc)
    p = stamp_path(repo, tree)
    if not p.is_file():
        return False, f"no stamp for tree {tree[:12]} (rev {head[:12]})"
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        return False, f"unreadable stamp {p}: {exc}"
    if not isinstance(data, dict) or data.get("schema") != STAMP_SCHEMA:
        return False, "stamp schema mismatch"
    if data.get("tree") != tree:
        return False, f"stamp tree {str(data.get('tree'))[:12]} != current tree {tree[:12]}"
    if data.get("tier") != "full":
        return False, "stamp is not from a full-tier run"
    try:
        workflow = parse_workflow_text(git(repo, "show", f"{head}:{WORKFLOW_PATH}"))
        docs_only = bool(data.get("docs_only"))
        if docs_only:
            classifier = extract_docs_classifier(workflow)
            files = changed_paths(repo, base, head)
            if not classifier.docs_only(files):
                return False, f"stamp attests a docs-only change, but vs {base} the change is not docs-only"
        keys = required_job_keys(workflow, docs_only)
    except (PlanError, RuntimeError) as exc:
        return False, f"cannot re-derive required jobs: {exc}"
    jobs = data.get("jobs") or {}
    missing = [k for k in keys if k not in jobs]
    if missing:
        return False, f"stamp does not cover required job(s): {missing}"
    bad = [k for k in keys if (jobs[k] or {}).get("status") != "passed"]
    if bad:
        return False, f"stamp has non-passed required job(s): {bad}"
    return True, f"valid stamp for tree {tree[:12]} ({len(keys)} required jobs passed; created {data.get('created')})"


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

@dataclasses.dataclass
class RunOutcome:
    exit_code: int
    results: List[JobResult]
    docs_only: bool
    stamp: Optional[Path]
    run_dir: Path


def _verdict(results: Sequence[JobResult], required: Sequence[str]) -> int:
    by_key = {r.key: r for r in results}
    if any(r.status == "failed" for r in results):
        return EXIT_FAILED
    if any(by_key.get(k) is None or by_key[k].status != "passed" for k in required):
        return EXIT_INCOMPLETE
    return EXIT_GREEN


def format_report(results: Sequence[JobResult], required: Sequence[str]) -> str:
    rows = [("JOB", "STATUS", "PYTHON", "DETAIL")]
    for r in results:
        detail = "; ".join(r.reasons + r.deviations + r.servers) or ""
        rows.append((r.key, r.status.upper(), r.python or "-", detail))
    w0 = max(len(r[0]) for r in rows)
    w1 = max(len(r[1]) for r in rows)
    w2 = max(len(r[2]) for r in rows)
    lines = [f"{a:<{w0}}  {b:<{w1}}  {c:<{w2}}  {d}" for a, b, c, d in rows]
    for r in results:
        for s in r.steps:
            if s.status in ("failed", "blocked"):
                lines.append(f"  {r.key}: {s.status.upper()} step '{s.name}' {s.detail} log={s.log}")
    not_verified = [k for k in required if k not in {r.key for r in results if r.status == "passed"}]
    if not_verified:
        lines.append("NOT VERIFIED (required by CI): " + ", ".join(not_verified))
    return "\n".join(lines)


def run_presubmit(repo: Path, *, opts: Options, base: str, rev: Optional[str], runner: ProcessRunner,
                  run_dir: Optional[Path] = None, out: Callable[[str], None] = print,
                  alloc_port: Optional[Callable[[], int]] = None, force_matrix: bool = False,
                  write_stamps: bool = True) -> RunOutcome:
    use_worktree = rev is None
    head_sha, tree = resolve_rev(repo, rev or "HEAD")
    if use_worktree:
        wf_text = (repo / WORKFLOW_PATH).read_text()
    else:
        wf_text = git(repo, "show", f"{head_sha}:{WORKFLOW_PATH}")
    workflow = parse_workflow_text(wf_text)
    classifier = extract_docs_classifier(workflow)
    docs_req, full_req = extract_required_jobs(workflow)
    files = changed_paths(repo, base, None if use_worktree else head_sha)
    base_sha = git(repo, "rev-parse", "--verify", f"{base}^{{commit}}").strip()
    docs_only = classifier.docs_only(files)
    if run_dir is None:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = git_common_dir(repo) / "presubmit" / "runs" / f"{ts}-{head_sha[:10]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ex = Executor(repo=repo, run_dir=run_dir, rev_sha=head_sha, opts=opts, runner=runner, out=out,
                  alloc_port=alloc_port, use_worktree=use_worktree)
    tested = "working tree (uncommitted changes included)" if use_worktree else f"commit {head_sha[:12]} tree {tree[:12]}"
    out(f"presubmit: tier={opts.tier} testing {tested}; base {base} ({base_sha[:12]}); run dir {run_dir}")
    classify = JobResult("classify", "passed", python=None,
                         reasons=[f"docs_only={'true' if docs_only else 'false'} ({len(files)} changed path(s) vs merge-base with {base})"])
    results: List[JobResult] = [classify]
    if docs_only and not force_matrix:
        required_ids = [j for j in docs_req if j != "classify"]
    else:
        required_ids = [j for j in full_req if j != "classify"]
    jobs = build_jobs(workflow, required_ids)
    required_keys = ["classify"] + [j.key for j in jobs]

    selected: Dict[str, Optional[Set[int]]] = {j.key: None for j in jobs}
    if opts.tier == "fast":
        jobs, selected, notes = _fast_plan(ex, repo, jobs, files, classifier, docs_only and not force_matrix,
                                           head_sha if not use_worktree else None)
        for n in notes:
            out(f"presubmit: {n}")
    if opts.only:
        # Exact key/id first: keys contain '[...]', which fnmatch treats as a character class.
        keep = [j for j in jobs if any(pat in (j.key, j.job_id) or fnmatch.fnmatchcase(j.key, pat)
                                       for pat in opts.only)]
        jobs = keep
    planned = {j.key for j in jobs}
    for k in required_keys[1:]:
        if k not in planned:
            results.append(JobResult(k, "skipped", reasons=["not selected (tier/--only)"]))

    def work(job: Job) -> JobResult:
        ex.say(f"presubmit: START {job.key}")
        r = ex.run_job(job, selected.get(job.key))
        ex.say(f"presubmit: {r.status.upper():<11} {job.key} ({r.duration_s:.0f}s)")
        return r

    ordered: Dict[str, JobResult] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, opts.jobs)) as pool:
        futs = {pool.submit(work, j): j for j in jobs}
        try:
            done = list(concurrent.futures.as_completed(futs))
        except KeyboardInterrupt:
            out("presubmit: interrupted; killing running steps and removing containers")
            ex.cancelled.set()
            if hasattr(runner, "kill_all"):
                runner.kill_all()
            for f in futs:
                f.cancel()
            raise
        for f in done:
            j = futs[f]
            try:
                ordered[j.key] = f.result()
            except Exception as exc:  # pragma: no cover - defensive
                ordered[j.key] = JobResult(j.key, "failed", reasons=[f"internal error: {exc!r}"])
    job_results = [ordered[j.key] for j in jobs]
    results = [results[0]] + job_results + [r for r in results[1:]]
    order = {k: i for i, k in enumerate(required_keys)}
    results.sort(key=lambda r: order.get(r.key, 1 << 30))

    code = _verdict(results, required_keys)
    if opts.tier == "fast" and code == EXIT_INCOMPLETE and not any(r.status == "failed" for r in results):
        ran = [r for r in results if r.status in ("passed", "substituted")]
        code = EXIT_GREEN if ran else EXIT_INCOMPLETE
    stamp = None
    full_green = (opts.tier == "full" and not use_worktree and not opts.only
                  and _verdict(results, required_keys) == EXIT_GREEN)
    if full_green and write_stamps:
        stamp = write_stamp(repo, tree=tree, head=head_sha, base=base, base_sha=base_sha,
                            docs_only=docs_only and not force_matrix, required=required_keys, results=results)
    summary = {
        "exit_code": code, "tier": opts.tier, "head": head_sha, "tree": tree, "docs_only": docs_only,
        "results": [dataclasses.asdict(r) for r in results], "stamp": str(stamp) if stamp else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    out(format_report(results, required_keys))
    if opts.tier == "fast":
        verdict = {EXIT_GREEN: "FAST TIER PASSED (partial: NOT a CI-equivalence claim; no stamp written)",
                   EXIT_FAILED: "FAILED", EXIT_INCOMPLETE: "INCOMPLETE"}[code]
    else:
        verdict = {EXIT_GREEN: "GREEN: every PR-gate job/matrix row executed and passed",
                   EXIT_FAILED: "FAILED", EXIT_INCOMPLETE: "INCOMPLETE: some required jobs were NOT verified; not green"}[code]
    out(f"presubmit: {verdict}")
    if stamp:
        out(f"presubmit: stamp written {stamp}")
    if use_worktree and opts.tier == "full":
        out("presubmit: working-tree run; stamps are only written for committed revisions (--rev)")
    return RunOutcome(code, results, docs_only, stamp, run_dir)


def _fast_plan(ex: Executor, repo: Path, jobs: List[Job], files: Sequence[str], classifier: DocsClassifier,
               docs_path: bool, rev: Optional[str]) -> Tuple[List[Job], Dict[str, Optional[Set[int]]], List[str]]:
    notes: List[str] = []
    if docs_path:
        return jobs, {j.key: None for j in jobs}, ["documentation-only change: CI runs only the docs-only job"]
    sqlite_rows = [j for j in jobs if j.job_id == "sqlite"]
    if not sqlite_rows:
        raise PlanError("fast tier requires a 'sqlite' job in the workflow")
    chosen = None
    for j in sqlite_rows:
        if j.python_version and ex.find_python(j.python_version):
            chosen = j
            break
    if chosen is None:
        chosen = sqlite_rows[0]
        if not ex.opts.substitute_python:
            ex.opts.substitute_python = sys.executable
        notes.append(f"no local python for {[j.python_version for j in sqlite_rows]}; "
                     f"fast tier substitutes {ex.opts.substitute_python}")
    test_steps = {s.index: step_test_file(s) for s in chosen.steps}
    test_files = sorted({t for t in test_steps.values() if t})
    for core in FAST_CORE_TESTS:
        if core not in test_files:
            raise PlanError(f"fast-tier core test {core} is no longer a step of the sqlite job")

    def read_source(f: str) -> Optional[str]:
        try:
            if rev is None:
                return (repo / f).read_text(errors="replace")
            return git(repo, "show", f"{rev}:{f}")
        except (OSError, RuntimeError):
            return None

    if rev is None:
        listing = git(repo, "ls-files", "-z", "-c", "-o", "--exclude-standard")
    else:
        listing = git(repo, "ls-tree", "-z", "--name-only", rev)
    modules = [p for p in listing.split("\0") if p.endswith(".py") and "/" not in p]
    affected, why = affected_tests(test_files, files, classifier, read_source, modules)
    run_tests = affected | set(FAST_CORE_TESTS)
    sel = {s.index for s in chosen.steps if test_steps[s.index] is None or test_steps[s.index] in run_tests}
    # Synthetic preflight (fast tier only): byte-compile every tracked Python
    # file and import each local module that some test file references.
    all_py = [p for p in listing.split("\0") if p.endswith(".py")]
    test_src = "\n".join(read_source(t) or "" for t in test_files)
    imports = sorted(Path(m).stem for m in modules
                     if not Path(m).name.startswith("test_") and re.search(rf"\b{re.escape(Path(m).stem)}\b", test_src))
    pre_script = (
        "python3 - <<'PRESUBMIT_PY'\n"
        "import importlib, tokenize\n"
        f"for f in {all_py!r}:\n"
        "    with tokenize.open(f) as fh:\n"
        "        compile(fh.read(), f, 'exec', dont_inherit=True)\n"
        f"for m in {imports!r}:\n"
        "    importlib.import_module(m)\n"
        f"print('compiled', {len(all_py)}, 'files; imported', {imports!r})\n"
        "PRESUBMIT_PY\n"
    )
    last_install = max((i for i, s in enumerate(chosen.steps) if s.kind == "run" and "pip install" in (s.run or "")),
                       default=len(chosen.steps) - 1)
    pre_env = dict(chosen.steps[last_install].env)
    steps = list(chosen.steps)
    steps.insert(last_install + 1, Step(FAST_PREFLIGHT_INDEX, "presubmit preflight: compile + import check",
                                        "run", "success", run=pre_script, env=pre_env))
    chosen = dataclasses.replace(chosen, steps=steps)
    sel.add(FAST_PREFLIGHT_INDEX)
    notes.append(f"fast tier: {len(run_tests)}/{len(test_files)} test steps selected ({why})")
    # Cached venv per interpreter + requirements to keep the fast tier fast.
    interp = ex.find_python(chosen.python_version or "") or (ex.opts.substitute_python, "")
    req = (repo / "requirements.txt").read_bytes() if (repo / "requirements.txt").exists() else b""
    key = hashlib.sha256(str(interp).encode() + req).hexdigest()[:16]
    ex.venv_cache = git_common_dir(repo) / "presubmit" / "venv-cache" / key
    return [chosen], {chosen.key: sel}, notes


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _dry_run(repo: Path, rev: str, base: str, out: Callable[[str], None]) -> int:
    head, _ = resolve_rev(repo, rev)
    workflow = parse_workflow_text(git(repo, "show", f"{head}:{WORKFLOW_PATH}"))
    clf = extract_docs_classifier(workflow)
    docs_req, full_req = extract_required_jobs(workflow)
    files = changed_paths(repo, base, head)
    out(f"docs-only whitelist ({len(clf.sources)} patterns, files < {clf.max_files_exclusive}): {', '.join(clf.sources)}")
    out(f"changed paths vs {base}: {len(files)}; docs_only={clf.docs_only(files)}")
    out(f"required (docs path): {docs_req}; required (full path): {full_req}")
    for job in build_jobs(workflow, [j for j in full_req if j != "classify"]):
        out(f"\n== {job.key}  python={job.python_version} containers={job.needs_containers}")
        for svc in job.services:
            out(f"   service {svc.name}: {svc.image} ports={svc.ports} health={svc.health_cmd!r}")
        for s in job.steps:
            env = {k: v for k, v in s.env.items()}
            line = (s.run or "").strip().splitlines()
            first = line[0] if line else ""
            out(f"   [{s.index:02d}] {s.kind:<15} {s.condition:<7} {s.name}"
                + (f"\n        run: {first}{' ...' if len(line) > 1 else ''}" if s.kind == "run" else "")
                + (f"\n        env: {env}" if env and s.kind == "run" else ""))
    return 0


def main(argv: Optional[Sequence[str]] = None, *, runner: Optional[ProcessRunner] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=("fast", "full"), default="fast")
    ap.add_argument("--rev", help="commit to test (default: working tree for fast, HEAD for full)")
    ap.add_argument("--base", default=os.environ.get("PRESUBMIT_BASE", "origin/master"),
                    help="PR base for path classification (default: $PRESUBMIT_BASE or origin/master)")
    ap.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--repeat", type=int, default=1, help="run concurrency/barrier test steps N times")
    ap.add_argument("--repeat-match", default=DEFAULT_REPEAT_MATCH)
    ap.add_argument("--only", action="append", default=[], help="glob on job key/id (never yields GREEN)")
    ap.add_argument("--allow-sudo", action="store_true")
    ap.add_argument("--substitute-python", help="interpreter to use when a row's python is missing (row stays unverified)")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed step (non-CI semantics)")
    ap.add_argument("--keep-containers", action="store_true")
    ap.add_argument("--step-timeout", type=float, default=3600.0)
    ap.add_argument("--force-matrix", action="store_true", help="run the full matrix even for docs-only changes")
    ap.add_argument("--dry-run", action="store_true", help="print the derived plan and exit")
    ap.add_argument("--check-stamp", action="store_true", help="exit 0 iff a valid full-tier stamp covers --rev")
    ap.add_argument("--repo", default=None)
    args = ap.parse_args(argv)
    repo = Path(args.repo or git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    try:
        if args.check_stamp:
            ok, msg = check_stamp(repo, args.rev or "HEAD", args.base)
            print(f"presubmit: {'OK' if ok else 'NO VALID STAMP'}: {msg}")
            return 0 if ok else 1
        if args.dry_run:
            return _dry_run(repo, args.rev or "HEAD", args.base, print)
        rev = args.rev if (args.rev or args.tier == "fast") else "HEAD"
        if args.tier == "full" and args.rev is None:
            dirty = git(repo, "status", "--porcelain", "--untracked-files=no").strip()
            if dirty:
                print("presubmit: WARNING: uncommitted changes are NOT tested by the full tier (it tests HEAD)")
        opts = Options(tier=args.tier, jobs=args.jobs, repeat=max(1, args.repeat), repeat_match=args.repeat_match,
                       only=args.only, allow_sudo=args.allow_sudo, substitute_python=args.substitute_python,
                       keep_going=args.keep_going, keep_containers=args.keep_containers,
                       step_timeout=args.step_timeout,
                       host=os.environ.get("PRESUBMIT_CONTAINER_HOST", "127.0.0.1"),
                       bind=os.environ.get("PRESUBMIT_PORT_BIND", "127.0.0.1"),
                       service_timeout=float(os.environ.get("PRESUBMIT_SERVICE_TIMEOUT", "300")))
        outcome = run_presubmit(repo, opts=opts, base=args.base, rev=rev, runner=runner or ProcessRunner(),
                                force_matrix=args.force_matrix)
        return outcome.exit_code
    except PlanError as exc:
        print(f"presubmit: PLAN ERROR: {exc}", file=sys.stderr)
        return EXIT_PLAN_ERROR
    except RuntimeError as exc:
        print(f"presubmit: ERROR: {exc}", file=sys.stderr)
        return EXIT_PLAN_ERROR


if __name__ == "__main__":
    sys.exit(main())
