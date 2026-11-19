#!/usr/bin/env python3
"""Tests for scripts/presubmit.py, designed from its contract rather than its code paths.

Oracles used here are independent of the implementation under test:
- the real .github/workflows/ci.yml, re-read and re-expanded by small helpers in
  this file (drift tests);
- the workflow's own JavaScript classify script, executed by Node.js with a
  mocked GitHub API (docs-only parity; skipped with a message if node is absent);
- the workflow's own "CI success" aggregator Python, executed with synthetic
  NEEDS_JSON (required-job parity);
- real git repositories in temporary directories (stamp binding) and real bash
  (script transformation, sudo blocking, process-group timeouts).
No real database engine or container runtime is needed: container and engine
commands go through a fake process runner that records every call.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import presubmit as ps  # noqa: E402

CI_YML = REPO / ".github" / "workflows" / "ci.yml"
RAW_WF = yaml.safe_load(CI_YML.read_text())
NODE = shutil.which("node")


# ---------------------------------------------------------------------------
# Independent helpers (deliberately not reusing presubmit's planner)
# ---------------------------------------------------------------------------

def indep_gate_jobs(wf: Dict[str, Any]) -> List[str]:
    return [jid for jid, j in wf["jobs"].items()
            if "docs_only != 'true'" in str(j.get("if", ""))]


def indep_rows(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    m = (job.get("strategy") or {}).get("matrix") or {}
    if "include" in m:
        return list(m["include"])
    axes = {k: v for k, v in m.items()}
    if not axes:
        return [{}]
    return [dict(zip(axes, c)) for c in itertools.product(*axes.values())]


def indep_subst(s: Any, row: Dict[str, Any]) -> str:
    return re.sub(r"\$\{\{\s*matrix\.([\w-]+)\s*\}\}", lambda m: str(row.get(m.group(1), "")), str(s))


TEST_CMD = re.compile(r"python3 (test_\w+\.py|verify_engine_matrix\.py)|\bpytest\b")


def git(cwd: Path, *args: str, **kw: Any) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True, **kw).stdout


def make_repo(tmp: Path, *, extra_change: str = "matview_sync.py") -> Path:
    repo = tmp / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "master")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    (repo / ".github" / "workflows").mkdir(parents=True)
    shutil.copy(CI_YML, repo / ".github" / "workflows" / "ci.yml")
    (repo / "scripts").mkdir()
    shutil.copy(HERE / "presubmit.py", repo / "scripts" / "presubmit.py")
    (repo / "README.md").write_text("readme\n")
    (repo / "requirements.txt").write_text("pyyaml\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    git(repo, "branch", "base")
    (repo / extra_change).parent.mkdir(parents=True, exist_ok=True)
    (repo / extra_change).write_text("x = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "change")
    return repo


class FakeRunner(ps.ProcessRunner):
    """Records calls; never executes anything. Behaviour is configured per test."""

    def __init__(self, *, runtimes_ok: Sequence[str] = ("docker",), pythons: Sequence[str] = ("3.11", "3.12", "3.15"),
                 fail_when: Optional[str] = None, fail_attempts: Optional[Sequence[int]] = None,
                 verified_line: bool = True) -> None:
        super().__init__()
        self.runtimes_ok = set(runtimes_ok)
        self.pythons = set(pythons)
        self.fail_when = fail_when
        self.fail_attempts = list(fail_attempts) if fail_attempts else None
        self.verified_line = verified_line
        self.calls: List[Tuple[str, List[str], Optional[Dict[str, str]], str]] = []
        self._attempt = 0

    def capture(self, argv, *, env=None, timeout=60):
        argv = list(argv)
        self.calls.append(("capture", argv, None, ""))
        if argv[-1:] == ["info"]:
            return (0, "ok") if " ".join(argv[:-1]) in self.runtimes_ok else (1, "cannot connect")
        m = re.fullmatch(r"python(3\.\d+)", Path(argv[0]).name)
        if m:
            return (0, f"{m.group(1)}\n{m.group(1)}.9") if m.group(1) in self.pythons else (127, "")
        if argv[0].endswith("/bin/python3"):
            return 0, "3.x.9 sqlite 3.45.0"
        return 0, "3.99.0"

    def run(self, argv, *, env, cwd, log_path, timeout):
        argv = list(argv)
        script = Path(argv[2]).read_text() if argv[:2] == ["bash", "-e"] else ""
        self.calls.append(("run", argv, dict(env) if env else None, script))
        with open(log_path, "a") as f:
            f.write("$ " + " ".join(argv) + "\n")
            if "verify_engine_matrix.py" in script and self.verified_line:
                f.write("[NATIVE ENGINE VERIFIED] fake server 99.1\n")
        if self.fail_when and self.fail_when in script:
            if self.fail_attempts is None:
                return 1
            self._attempt += 1
            return 1 if self._attempt in self.fail_attempts else 0
        return 0

    def sleep(self, seconds):
        self.calls.append(("sleep", [str(seconds)], None, ""))

    def scripts(self) -> List[Tuple[str, Dict[str, str]]]:
        return [(s, e or {}) for kind, _, e, s in self.calls if kind == "run" and s]


def run_fake(repo: Path, runner: FakeRunner, tmp: Path, **opt: Any) -> Tuple[ps.RunOutcome, str]:
    lines: List[str] = []
    ports = iter(range(41000, 42000))
    rev = opt.pop("rev", "HEAD")
    base = opt.pop("base", "base")
    force = opt.pop("force_matrix", False)
    opts = ps.Options(**{"tier": "full", "jobs": 1, **opt})
    outcome = ps.run_presubmit(repo, opts=opts, base=base, rev=rev, runner=runner, run_dir=tmp / "run",
                               out=lines.append, alloc_port=lambda: next(ports), force_matrix=force)
    return outcome, "\n".join(lines)


class TmpCase(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self._env = dict(os.environ)
        for k in list(os.environ):
            if k.startswith("PRESUBMIT_"):
                del os.environ[k]

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env)
        self._td.cleanup()


# ---------------------------------------------------------------------------
# (a) Drift: the plan reproduces every CI test step with identical env
# ---------------------------------------------------------------------------

class DriftTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = indep_gate_jobs(RAW_WF)
        self.jobs = ps.build_jobs(RAW_WF, self.gate)

    def test_gate_jobs_match_ci_success(self) -> None:
        needs = set(RAW_WF["jobs"]["ci-success"]["needs"])
        self.assertEqual(set(self.gate), needs - {"classify", "docs-only"})
        _, full = ps.extract_required_jobs(RAW_WF)
        self.assertEqual(set(full), set(self.gate) | {"classify"})

    def test_every_test_step_present_with_identical_env_and_order(self) -> None:
        wf_env = RAW_WF.get("env") or {}
        checked = 0
        for jid in self.gate:
            job = RAW_WF["jobs"][jid]
            rows = indep_rows(job)
            plan_rows = [j for j in self.jobs if j.job_id == jid]
            self.assertEqual(len(plan_rows), len(rows), jid)
            for row in rows:
                matches = [j for j in plan_rows if j.matrix == row]
                self.assertEqual(len(matches), 1, f"{jid} {row}")
                pj = matches[0]
                expected = []
                for st in job["steps"]:
                    if "run" not in st or not TEST_CMD.search(st["run"]):
                        continue
                    env = {k: indep_subst(v, row) for k, v in wf_env.items()}
                    env.update({k: indep_subst(v, row) for k, v in (job.get("env") or {}).items()})
                    env.update({k: indep_subst(v, row) for k, v in (st.get("env") or {}).items()})
                    expected.append((indep_subst(st["run"], row), env))
                got = [(s.run, s.env) for s in pj.steps if s.kind == "run" and TEST_CMD.search(s.run or "")]
                self.assertEqual(got, expected, f"{pj.key}: test steps/env/order drifted")
                # No workflow step may be silently dropped.
                self.assertEqual(len(pj.steps), len(job["steps"]), pj.key)
                checked += len(expected)
        self.assertGreater(checked, 50)

    def test_services_mirror_workflow(self) -> None:
        for pj in self.jobs:
            spec = RAW_WF["jobs"][pj.job_id].get("services") or {}
            self.assertEqual([s.name for s in pj.services], list(spec))
            for svc in pj.services:
                raw = spec[svc.name]
                self.assertEqual(svc.image, indep_subst(raw["image"], pj.matrix))
                self.assertEqual(svc.env, {k: str(v) for k, v in raw["env"].items()})
                self.assertIn("pg_isready", svc.health_cmd or "")

    def test_container_jobs_are_exactly_those_using_docker_or_services(self) -> None:
        for pj in self.jobs:
            spec = RAW_WF["jobs"][pj.job_id]
            uses = bool(spec.get("services")) or any("docker " in (s.get("run") or "") for s in spec["steps"])
            self.assertEqual(pj.needs_containers, uses, pj.key)

    def test_required_jobs_agree_with_executed_ci_success_script(self) -> None:
        """Run the workflow's own aggregator: failing any presubmit-required job must fail it."""
        run = RAW_WF["jobs"]["ci-success"]["steps"][0]["run"]
        code = re.search(r"<<'PY'\n(.*?)\n\s*PY\s*$", run, re.S).group(1)
        code = textwrap.dedent(code)
        needs = RAW_WF["jobs"]["ci-success"]["needs"]
        docs_req, full_req = ps.extract_required_jobs(RAW_WF)

        def agg(docs_only: bool, results: Dict[str, str]) -> int:
            needs_json = {n: {"result": results.get(n, "success"), "outputs": {}} for n in needs}
            needs_json["classify"]["outputs"] = {"docs_only": "true" if docs_only else "false"}
            p = subprocess.run([sys.executable, "-c", code], env={**os.environ, "NEEDS_JSON": json.dumps(needs_json)},
                               capture_output=True)
            return p.returncode

        for docs_only, req in ((False, full_req), (True, docs_req)):
            for n in needs:
                rc = agg(docs_only, {n: "failure"})
                self.assertEqual(rc != 0, n in req, f"docs_only={docs_only} job={n}")

    def test_unsupported_constructs_fail_loudly(self) -> None:
        def mutated(fn) -> Dict[str, Any]:
            wf = yaml.safe_load(CI_YML.read_text())
            fn(wf)
            return wf

        cases = {
            "unknown action": lambda wf: wf["jobs"]["sqlite"]["steps"].insert(1, {"uses": "actions/cache@v4"}),
            "complex if": lambda wf: wf["jobs"]["sqlite"]["steps"][3].__setitem__("if", "${{ success() && matrix.x }}"),
            "secret expr": lambda wf: wf["jobs"]["sqlite"]["steps"][3].__setitem__("env", {"X": "${{ secrets.T }}"}),
            "exclude": lambda wf: wf["jobs"]["sqlite"]["strategy"]["matrix"].__setitem__("exclude", [{"python-version": "3.11"}]),
            "working-directory": lambda wf: wf["jobs"]["sqlite"]["steps"][3].__setitem__("working-directory", "x"),
            "continue-on-error": lambda wf: wf["jobs"]["sqlite"].__setitem__("continue-on-error", True),
            "shell pwsh": lambda wf: wf["jobs"]["sqlite"]["steps"][3].__setitem__("shell", "pwsh"),
        }
        for label, fn in cases.items():
            with self.subTest(label):
                with self.assertRaises(ps.PlanError):
                    ps.build_jobs(mutated(fn), ["sqlite"])

    def test_matrix_restricted_step_is_planned_for_exactly_the_matching_cells(self) -> None:
        restricted = [s for s in RAW_WF["jobs"]["sqlite"]["steps"] if "matrix." in str(s.get("if", ""))]
        self.assertTrue(restricted, "the sqlite job is expected to carry a matrix-restricted step (pyright)")
        jobs = ps.build_jobs(RAW_WF, ["sqlite"])
        self.assertGreater(len(jobs), 1)
        for raw in restricted:
            m = re.fullmatch(r"(?:\$\{\{\s*)?matrix\.([\w-]+)\s*==\s*'([^']*)'(?:\s*\}\})?", str(raw["if"]).strip())
            self.assertIsNotNone(m, raw["if"])
            runs = 0
            for pj in jobs:
                planned = [s for s in pj.steps if s.name == raw["name"]]
                self.assertEqual(len(planned), 1, pj.key)
                applies = str(pj.matrix[m.group(1)]).lower() == m.group(2).lower()
                self.assertEqual(planned[0].condition, "success" if applies else "never", pj.key)
                runs += applies
            self.assertEqual(runs, 1, f"{raw['name']!r} must apply to exactly one matrix cell")

    def test_matrix_condition_forms_and_failures(self) -> None:
        def sqlite_with_if(cond: str) -> Dict[str, Any]:
            wf = yaml.safe_load(CI_YML.read_text())
            wf["jobs"]["sqlite"]["steps"][3]["if"] = cond
            return wf

        def conditions(cond: str) -> Dict[str, str]:
            jobs = ps.build_jobs(sqlite_with_if(cond), ["sqlite"])
            return {j.matrix["python-version"]: j.steps[3].condition for j in jobs}

        self.assertEqual(conditions("matrix.python-version == '3.15'"), {"3.11": "never", "3.15": "success"})
        # GitHub expression braces and case-insensitive string comparison.
        self.assertEqual(conditions("${{ matrix.python-version == '3.11' }}"), {"3.11": "success", "3.15": "never"})
        self.assertEqual(conditions("matrix.python-version == '3.11 '"), {"3.11": "never", "3.15": "never"})
        for bad in ("matrix.python-version != '3.11'", "matrix.nope == 'x'",
                    "matrix.python-version == 3.11", "always() || matrix.python-version == '3.11'"):
            with self.subTest(bad):
                with self.assertRaises(ps.PlanError):
                    ps.build_jobs(sqlite_with_if(bad), ["sqlite"])

    def test_matrix_properties_render_like_github(self) -> None:
        def fuzz_targets(wf: Dict[str, Any]) -> List[str]:
            jobs = ps.build_jobs(wf, ["sqlite-compat"])
            return [
                [s for s in j.steps if s.run == "python3 test_matview_sync.py"][0].env["MATVIEW_FUZZ_TARGET"]
                for j in jobs
            ]

        wf = yaml.safe_load(CI_YML.read_text())
        rows = wf["jobs"]["sqlite-compat"]["strategy"]["matrix"]["include"]
        self.assertEqual(
            fuzz_targets(wf),
            [f"{r['label']} / Python {r['python-version']}" for r in rows],
        )
        # GitHub renders a property missing from a matrix row as the empty string.
        rows[0]["extra"] = "x"
        step = [s for s in wf["jobs"]["sqlite-compat"]["steps"] if s.get("run") == "python3 test_matview_sync.py"][0]
        step["env"]["MATVIEW_FUZZ_TARGET"] = "${{ matrix.label }}/${{ matrix.extra }}/"
        self.assertEqual(fuzz_targets(wf), [f"{r['label']}/{r.get('extra', '')}/" for r in rows])
        self.assertTrue(all(t.endswith("//") for t in fuzz_targets(wf)[1:]))

    def test_aggregator_changes_fail_loudly(self) -> None:
        wf = yaml.safe_load(CI_YML.read_text())
        wf["jobs"]["ci-success"]["steps"][0]["run"] = wf["jobs"]["ci-success"]["steps"][0]["run"].replace(
            "required_jobs = (", "required_jobs = compute(")
        with self.assertRaises(ps.PlanError):
            ps.extract_required_jobs(wf)
        wf = yaml.safe_load(CI_YML.read_text())
        wf["jobs"]["postgres"]["if"] = "always()"
        with self.assertRaises(ps.PlanError):
            ps.extract_required_jobs(wf)


# ---------------------------------------------------------------------------
# (b) Docs-only classification parity with the workflow's own JavaScript
# ---------------------------------------------------------------------------

JS_HARNESS = r"""
const fs = require('fs');
const script = fs.readFileSync(process.argv[2], 'utf8');
const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const fn = new AsyncFunction('context', 'github', 'core', script);
(async () => {
  const out = [];
  for (const files of cases) {
    let result = null;
    await fn(
      { eventName: 'pull_request', repo: { owner: 'o', repo: 'r' }, payload: { pull_request: { number: 1 } } },
      { paginate: async () => files.map(f => ({ filename: f })), rest: { pulls: { listFiles: () => null } } },
      { setOutput: (k, v) => { if (k === 'docs_only') result = v; }, info: () => {} },
    );
    out.push(result === 'true');
  }
  process.stdout.write(JSON.stringify(out));
})();
"""

ADVERSARIAL_PATHS = [
    "README.md", "readme.md", "README.MD", "README", "README.txt", "README.rst", "README.mdx", "README.markdown",
    "README.md.py", "README.md\n", "README.md\r", "\nREADME.md", "README.md ", " README.md", "README.md/x.py",
    "sub/README.md", "./README.md", "/README.md", "READMEX.md", "README.", "AGENTS.md", "agents.MD", "AGENTS.mdx",
    "TODO.md", "TODO.txt", "CHANGELOG", "changelog.rst", "LICENSE", "LICENSE.md", "LICENSE.mdx", "LICENSE.py",
    "SECURITY.md", "ſECURITY.md", "ſecurity.md", "CODE_OF_CONDUCT.md", "CONTRIBUTING.txt",
    "KLICENSE", "LıCENSE", "docs/a.md", "docs/A.MD", "docs/.md", "docs/x/y/z.png", "docs/a.jpeg",
    "docs/a.jpg", "docs/a.jpgg", "docs/a.md.py", "docs/a\nb.md", "docs/a\rb.md", "docs/a b.md",
    "docs/a b.md", "docs/a\u0085b.md", "docs/é.md", "Docs/a.md", "docs", "docs/", "docs/a.py",
    "doc/a.md", "docs\\a.md", ".github/a.md", ".github/workflows/ci.yml", ".github/workflows/README.md",
    ".GITHUB/x.MD", ".github/.md", ".github/x.md\n", "x.github/a.md", "matview_sync.py", "requirements.txt",
    "Makefile", "", ".", "README.md\u0000", "docs/a.svg", "docs/a.SVG", "docs/a.ico", "docs/a.webp", "docs/a.pdf",
]


def random_path(rng: random.Random) -> str:
    prefixes = ["", "docs/", "Docs/", "DOCS/", ".github/", ".github/workflows/", "sub/", "docs/sub/", "./", "/", "x"]
    stems = ["README", "readme", "AGENTS", "TODO", "CHANGELOG", "LICENSE", "SECURITY", "CODE_OF_CONDUCT",
             "CONTRIBUTING", "foo", "ſecurity", "agKents", "test_x", ""]
    exts = [".md", ".MD", ".mdx", ".rst", ".txt", ".py", ".yml", "", ".md.py", ".jpeg", ".jpg", ".JPG", ".svg",
            ".pdf", ".ico", ".webp", "..md", ".markdown", ".png", ".gif"]
    noise = ["", "", "", "\n", "\r", " ", " ", " ", "/", "\t"]
    parts = [rng.choice(prefixes), rng.choice(noise) if rng.random() < 0.1 else "", rng.choice(stems),
             rng.choice(noise) if rng.random() < 0.1 else "", rng.choice(exts), rng.choice(noise)]
    return "".join(parts)


class DocsParityTests(unittest.TestCase):
    SEED = int(os.environ.get("PRESUBMIT_TEST_SEED", "20261117"))

    def setUp(self) -> None:
        self.clf = ps.extract_docs_classifier(RAW_WF)

    def _cases(self) -> List[List[str]]:
        rng = random.Random(self.SEED)
        cases: List[List[str]] = [[p] for p in ADVERSARIAL_PATHS]
        cases += [[random_path(rng)] for _ in range(3000)]
        cases += [[random_path(rng) for _ in range(rng.randint(0, 6))] for _ in range(500)]
        cases += [[], ["README.md"] * 2999, ["README.md"] * 3000, ["README.md", "x.py"], ["docs/a.md", "TODO.md"]]
        return cases

    @unittest.skipUnless(NODE, "node not installed: cannot execute the workflow's own JavaScript oracle")
    def test_parity_with_workflow_javascript(self) -> None:
        script = RAW_WF["jobs"]["classify"]["steps"][0]["with"]["script"]
        cases = self._cases()
        sf, cf, hf = self.tmp_files(script, json.dumps(cases), JS_HARNESS)
        p = subprocess.run([NODE, hf, sf, cf], capture_output=True, text=True, check=True)
        js = json.loads(p.stdout)
        mismatches = [(c, j) for c, j in zip(cases, js) if self.clf.docs_only(c) != j]
        self.assertEqual(mismatches[:5], [], f"seed={self.SEED}: {len(mismatches)} mismatches")
        self.assertIn(True, js)
        self.assertIn(False, js)
        # Sensitivity: a naive translation (Python re with IGNORECASE, $ and . unchanged)
        # must disagree with the JS oracle on this case set, otherwise the set has no teeth.
        naive = [re.compile(src[1:src.rindex("/")].replace("\\/", "/"), re.I) for src in self.clf.sources]
        naive_res = [len(c) > 0 and len(c) < 3000 and all(any(r.search(f) for r in naive) for f in c) for c in cases]
        self.assertTrue(any(a != b for a, b in zip(naive_res, js)), "adversarial set does not distinguish naive regexes")

    def tmp_files(self, *contents: str) -> List[str]:
        td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, td)
        paths = []
        for i, c in enumerate(contents):
            p = Path(td) / f"f{i}.js"
            p.write_text(c)
            paths.append(str(p))
        return paths

    def test_hand_labelled_js_semantics(self) -> None:
        """Expectations written from ECMAScript semantics (no node needed)."""
        expect = {
            "README.md": False, "readme.MD": False, "TODO.md": True, "todo.MD": True, "TODO.md\n": False, "docs/a\rb.md": False,
            "docs/a b.md": False, "docs/a\u0085b.md": True, "ſECURITY.md": False,
            "sub/README.md": False, ".github/workflows/README.md": True, ".github/workflows/ci.yml": False,
            "docs/a.jpgg": False, "LICENSE.py": False,
        }
        for path, want in expect.items():
            with self.subTest(path=path):
                self.assertEqual(self.clf.docs_only([path]), want)
        self.assertFalse(self.clf.docs_only([]))
        self.assertFalse(self.clf.docs_only(["TODO.md"] * 3000))
        self.assertTrue(self.clf.docs_only(["TODO.md"] * 2999))

    def test_whitelist_extraction_fails_loudly_on_unsupported_changes(self) -> None:
        script = RAW_WF["jobs"]["classify"]["steps"][0]["with"]["script"]
        mutations = {
            "named group": script.replace("/^AGENTS\\.md$/i", "/^(?<n>AGENTS)\\.md$/i"),
            "unicode flag": script.replace("/^AGENTS\\.md$/i", "/^AGENTS\\.md$/iu"),
            "non-literal": script.replace("/^AGENTS\\.md$/i", "new RegExp('^AGENTS')"),
            "lookahead": script.replace("/^AGENTS\\.md$/i", "/^(?!x)AGENTS\\.md$/i"),
            "class escape": script.replace("/^AGENTS\\.md$/i", "/^AGENTS\\s\\.md$/i"),
            "threshold op": script.replace("files.length < 3000", "files.length <= 3000"),
            "extra condition": script.replace("files.length > 0 &&", "files.length > 0 && isTrusted &&"),
            "no event gate": script.replace('context.eventName !== "pull_request"', "false"),
            "array renamed": script.replace("documentationPaths", "docPaths"),
        }
        for label, mutated_script in mutations.items():
            with self.subTest(label):
                self.assertNotEqual(mutated_script, script, "mutation did not apply")
                wf = yaml.safe_load(CI_YML.read_text())
                wf["jobs"]["classify"]["steps"][0]["with"]["script"] = mutated_script
                with self.assertRaises(ps.PlanError):
                    ps.extract_docs_classifier(wf)


class ChangedPathTests(TmpCase):
    def test_rename_out_of_code_is_conservative(self) -> None:
        repo = make_repo(self.tmp, extra_change="src.py")
        git(repo, "branch", "-f", "base", "HEAD")
        git(repo, "mv", "src.py", "docs_src.md")
        (repo / "docs").mkdir()
        git(repo, "mv", "docs_src.md", "docs/src.md")
        git(repo, "commit", "-q", "-m", "rename")
        files = ps.changed_paths(repo, "base", "HEAD")
        self.assertEqual(files, ["docs/src.md", "src.py"])
        self.assertFalse(ps.extract_docs_classifier(RAW_WF).docs_only(files))


# ---------------------------------------------------------------------------
# (c) Missing container runtime => INCOMPLETE, never green
# ---------------------------------------------------------------------------

class ContainerRuntimeTests(TmpCase):
    def test_no_runtime_is_incomplete_and_lists_native_jobs(self) -> None:
        repo = make_repo(self.tmp)
        runner = FakeRunner(runtimes_ok=())
        outcome, text = run_fake(repo, runner, self.tmp)
        self.assertEqual(outcome.exit_code, ps.EXIT_INCOMPLETE)
        self.assertIsNone(outcome.stamp)
        by = {r.key: r for r in outcome.results}
        native = [k for k in by if k.startswith(("postgres", "mysql"))]
        self.assertEqual(len(native), 7)
        for k in native:
            self.assertEqual(by[k].status, "incomplete", k)
            self.assertIn("NOT verified", " ".join(by[k].reasons))
            self.assertIn(k, text.split("NOT VERIFIED (required by CI):")[1])
        self.assertIn("INCOMPLETE", text)
        self.assertNotIn("GREEN", text)
        self.assertFalse(any(c[1][:1] == ["docker"] and c[0] == "run" for c in runner.calls))
        self.assertEqual(ps.main(["--check-stamp", "--repo", str(repo), "--base", "base"]), 1)

    def test_configured_runtime_is_not_silently_replaced(self) -> None:
        repo = make_repo(self.tmp)
        os.environ["PRESUBMIT_CONTAINER_RUNTIME"] = "lima nerdctl"
        runner = FakeRunner(runtimes_ok=("docker",))  # docker works, configured one does not
        outcome, _ = run_fake(repo, runner, self.tmp)
        self.assertEqual(outcome.exit_code, ps.EXIT_INCOMPLETE)
        self.assertNotIn(["docker", "info"], [c[1] for c in runner.calls])

    def test_failure_dominates_incomplete(self) -> None:
        repo = make_repo(self.tmp)
        outcome, _ = run_fake(repo, FakeRunner(runtimes_ok=(), fail_when="test_cli_diff.py"), self.tmp)
        self.assertEqual(outcome.exit_code, ps.EXIT_FAILED)

    def test_missing_python_and_substitution_are_never_green(self) -> None:
        repo = make_repo(self.tmp)
        outcome, text = run_fake(repo, FakeRunner(pythons=("3.12",)), self.tmp)
        self.assertEqual(outcome.exit_code, ps.EXIT_INCOMPLETE)
        by = {r.key: r for r in outcome.results}
        self.assertEqual(by["sqlite[python-version=3.11]"].status, "incomplete")
        shutil.rmtree(self.tmp / "run")
        outcome, text = run_fake(repo, FakeRunner(pythons=("3.12",)), self.tmp, substitute_python="/usr/bin/python3")
        self.assertEqual(outcome.exit_code, ps.EXIT_INCOMPLETE)
        by = {r.key: r for r in outcome.results}
        self.assertEqual(by["sqlite[python-version=3.15]"].status, "substituted")
        self.assertIsNone(outcome.stamp)

    def test_runtime_present_provisions_isolated_services_and_cleans_up(self) -> None:
        repo = make_repo(self.tmp)
        os.environ["PRESUBMIT_CONTAINER_RUNTIME"] = "podman --remote"
        runner = FakeRunner(runtimes_ok=("podman --remote",))
        outcome, text = run_fake(repo, runner, self.tmp, only=["postgres"])
        self.assertEqual(outcome.exit_code, ps.EXIT_INCOMPLETE)  # --only never yields green
        by = {r.key: r for r in outcome.results}
        self.assertEqual(by["postgres[pg-version=15]"].status, "passed")
        self.assertEqual(by["postgres[pg-version=15]"].servers, ["[NATIVE ENGINE VERIFIED] fake server 99.1"] * 1)
        runs = [c[1] for c in runner.calls if c[0] == "run"]
        starts = [a for a in runs if a[:3] == ["podman", "--remote", "run"]]
        self.assertEqual(len(starts), 2)
        self.assertIn("postgres:15", starts[0])
        self.assertTrue(any(re.fullmatch(r"127\.0\.0\.1:41\d\d\d:5432", x) for x in starts[0]))
        self.assertIn("POSTGRES_PASSWORD=password", starts[0])
        self.assertTrue(any(a[:3] == ["podman", "--remote", "pull"] for a in runs))
        names = [a[a.index("--name") + 1] for a in starts]
        self.assertEqual(len(set(names)), 2)
        for n in names:
            self.assertIn(["podman", "--remote", "rm", "-f", "-v", n], runs)
        dsns = {e.get("POSTGRES_DSN") for _, e in runner.scripts() if e.get("POSTGRES_DSN")}
        self.assertEqual(len(dsns), 2)
        for d in dsns:
            self.assertRegex(d, r"^postgresql://postgres:password@127\.0\.0\.1:41\d\d\d/matview_test$")

    def test_docker_run_steps_are_rewritten_for_parallel_isolation(self) -> None:
        repo = make_repo(self.tmp)
        runner = FakeRunner()
        outcome, _ = run_fake(repo, runner, self.tmp, only=["mysql"])
        starts = [s for s, _ in runner.scripts() if "docker run -d" in s]
        self.assertEqual(len(starts), 2)
        for s in starts:
            self.assertIn('docker() { docker "$@"; }', s)
            self.assertNotIn("--name mysql-srv ", s)
            self.assertNotIn("exec mysql-srv ", s)
            self.assertRegex(s, r"-p 127\.0\.0\.1:41\d\d\d:3306")
        envs = [e for _, e in runner.scripts() if e.get("MYSQL_DSN")]
        self.assertTrue(all(":3306/" not in e["MYSQL_DSN"] for e in envs))
        # every renamed container is removed
        rms = [c[1][-1] for c in runner.calls if c[0] == "run" and c[1][:3] == ["docker", "rm", "-f"]]
        self.assertEqual(len(rms), 2)
        self.assertTrue(all(n.startswith("presubmit-") for n in rms))


# ---------------------------------------------------------------------------
# (d) Stamp binding and invalidation
# ---------------------------------------------------------------------------

class StampTests(TmpCase):
    def green(self, repo: Path) -> ps.RunOutcome:
        outcome, text = run_fake(repo, FakeRunner(), self.tmp)
        shutil.rmtree(self.tmp / "run", ignore_errors=True)
        self.assertEqual(outcome.exit_code, ps.EXIT_GREEN, text)
        self.assertIsNotNone(outcome.stamp)
        return outcome

    def check(self, repo: Path, rev: str = "HEAD", base: str = "base") -> Tuple[bool, str]:
        return ps.check_stamp(repo, rev, base)

    def test_stamp_bound_to_tree(self) -> None:
        repo = make_repo(self.tmp)
        outcome = self.green(repo)
        self.assertTrue(self.check(repo)[0])
        data = json.loads(outcome.stamp.read_text())
        self.assertEqual(data["tree"], git(repo, "rev-parse", "HEAD^{tree}").strip())
        self.assertEqual(data["head"], git(repo, "rev-parse", "HEAD").strip())
        self.assertEqual(len(data["required_jobs"]), 1 + 2 + 3 + 2 + 2 + 3)
        self.assertTrue(data["not_covered"])
        self.assertTrue(str(outcome.stamp).startswith(str(repo / ".git")))
        self.assertEqual(git(repo, "status", "--porcelain").strip(), "")
        # Same tree, new commit (message amend) => still valid.
        git(repo, "commit", "-q", "--amend", "-m", "reworded")
        self.assertTrue(self.check(repo)[0])
        # Any tree change => invalid.
        (repo / "matview_sync.py").write_text("x = 2\n")
        git(repo, "commit", "-q", "-am", "edit")
        ok, msg = self.check(repo)
        self.assertFalse(ok)
        self.assertIn("no stamp", msg)
        self.assertEqual(ps.main(["--check-stamp", "--repo", str(repo), "--base", "base"]), 1)
        self.assertEqual(ps.main(["--check-stamp", "--repo", str(repo), "--base", "base", "--rev", "HEAD~1"]), 0)

    def test_tampered_or_partial_stamps_rejected(self) -> None:
        repo = make_repo(self.tmp)
        path = self.green(repo).stamp
        original = path.read_text()

        def mutate(fn) -> Tuple[bool, str]:
            data = json.loads(original)
            fn(data)
            path.write_text(json.dumps(data))
            return self.check(repo)

        for label, fn in {
            "missing job": lambda d: d["jobs"].pop("mysql-compat[label=MariaDB 12.3 LTS,image=mariadb:12.3,expected-product=mariadb,expected-version=12.3]"),
            "incomplete job": lambda d: d["jobs"]["postgres[pg-version=18]"].__setitem__("status", "incomplete"),
            "substituted job": lambda d: d["jobs"]["sqlite[python-version=3.15]"].__setitem__("status", "substituted"),
            "wrong tree": lambda d: d.__setitem__("tree", "0" * 40),
            "fast tier": lambda d: d.__setitem__("tier", "fast"),
            "schema": lambda d: d.__setitem__("schema", 99),
            "docs claim": lambda d: d.__setitem__("docs_only", True),
        }.items():
            with self.subTest(label):
                ok, msg = mutate(fn)
                self.assertFalse(ok, msg)
        path.write_text("{not json")
        self.assertFalse(self.check(repo)[0])
        path.write_text(original)
        self.assertTrue(self.check(repo)[0])

    def test_non_green_runs_never_stamp(self) -> None:
        repo = make_repo(self.tmp)
        for kw in ({"runner": FakeRunner(fail_when="test_cli_diff.py")},
                   {"runner": FakeRunner(runtimes_ok=())},
                   {"runner": FakeRunner(), "only": ["sqlite"]}):
            runner = kw.pop("runner")
            outcome, _ = run_fake(repo, runner, self.tmp, **kw)
            shutil.rmtree(self.tmp / "run")
            self.assertNotEqual(outcome.exit_code, ps.EXIT_GREEN)
            self.assertIsNone(outcome.stamp)
            self.assertFalse(self.check(repo)[0])
        # Working-tree run (rev=None) never stamps either.
        outcome, _ = run_fake(repo, FakeRunner(), self.tmp, rev=None)
        self.assertIsNone(outcome.stamp)
        self.assertFalse(self.check(repo)[0])

    def test_docs_only_stamp_rechecks_classification(self) -> None:
        # History: base -> mid (code change) -> HEAD (docs change).  Stacked on
        # "mid" the PR is docs-only; against "base" it is not.
        repo = make_repo(self.tmp)  # HEAD = code change
        git(repo, "branch", "mid")
        (repo / "docs").mkdir()
        (repo / "docs" / "guide.md").write_text("g\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "docs")
        outcome, text = run_fake(repo, FakeRunner(), self.tmp, base="mid")
        self.assertEqual(outcome.exit_code, ps.EXIT_GREEN, text)
        self.assertTrue(outcome.docs_only)
        self.assertEqual([r.key for r in outcome.results], ["classify", "docs-only"])
        self.assertTrue(json.loads(outcome.stamp.read_text())["docs_only"])
        self.assertTrue(self.check(repo, base="mid")[0])
        ok, msg = self.check(repo, base="base")
        self.assertFalse(ok)
        self.assertIn("not docs-only", msg)

    def test_pre_push_hook(self) -> None:
        repo = make_repo(self.tmp)
        hook = REPO / ".githooks" / "pre-push"
        head = git(repo, "rev-parse", "HEAD").strip()
        env = {**os.environ, "PRESUBMIT_PYTHON": sys.executable, "PRESUBMIT_HOOK_RUN": "0", "PRESUBMIT_BASE": "base"}

        def push(lines: str) -> int:
            return subprocess.run(["bash", str(hook), "origin", "url"], cwd=repo, input=lines, text=True,
                                  env=env, capture_output=True).returncode

        line = f"refs/heads/master {head} refs/heads/master {'0' * 40}\n"
        self.assertEqual(push(line), 1)
        self.assertEqual(push(f"(delete) {'0' * 40} refs/heads/gone {head}\n"), 0)
        self.green(repo)
        self.assertEqual(push(line), 0)


# ---------------------------------------------------------------------------
# (e) Failing steps fail the run and are reported
# ---------------------------------------------------------------------------

class FailureTests(TmpCase):
    def test_failing_step_fails_run_with_github_skip_semantics(self) -> None:
        repo = make_repo(self.tmp)
        runner = FakeRunner(fail_when="python3 test_concurrent_backfill.py")
        outcome, text = run_fake(repo, runner, self.tmp, only=["sqlite"])
        self.assertEqual(outcome.exit_code, ps.EXIT_FAILED)
        r = {x.key: x for x in outcome.results}["sqlite[python-version=3.11]"]
        self.assertEqual(r.status, "failed")
        names = [s.name for s in r.steps]
        idx = names.index("Test independent-session backfill concurrency")
        self.assertEqual(r.steps[idx].status, "failed")
        self.assertTrue(all(s.status == "skipped" for s in r.steps[idx + 1:]))
        self.assertIn("FAILED step 'Test independent-session backfill concurrency'", text)
        self.assertRegex(text, r"presubmit: FAILED$")
        self.assertTrue(Path(r.steps[idx].log).is_file())
        summary = json.loads((self.tmp / "run" / "summary.json").read_text())
        self.assertEqual(summary["exit_code"], 1)

    def test_matrix_restricted_step_runs_once_and_is_reported_skipped_elsewhere(self) -> None:
        repo = make_repo(self.tmp)
        runner = FakeRunner()
        outcome, _ = run_fake(repo, runner, self.tmp, only=["sqlite"])
        executed = [s for s, _ in runner.scripts() if re.search(r"(?m)^pyright\b", s)]
        self.assertEqual(len(executed), 1, executed)
        results = {r.key: r for r in outcome.results}
        step = "Static type check (pyright)"
        by_cell = {k: [s for s in r.steps if s.name == step] for k, r in results.items() if k.startswith("sqlite[")}
        self.assertEqual({k: [s.status for s in v] for k, v in by_cell.items()},
                         {"sqlite[python-version=3.11]": ["passed"], "sqlite[python-version=3.15]": ["skipped"]})
        self.assertIn("matrix cell", by_cell["sqlite[python-version=3.15]"][0].detail)

    def test_failing_pyright_fails_the_gate(self) -> None:
        repo = make_repo(self.tmp)
        runner = FakeRunner(fail_when="pyright")
        outcome, text = run_fake(repo, runner, self.tmp, only=["sqlite"])
        self.assertEqual(outcome.exit_code, ps.EXIT_FAILED)
        r = {x.key: x for x in outcome.results}["sqlite[python-version=3.11]"]
        self.assertEqual({s.name: s.status for s in r.steps}["Static type check (pyright)"], "failed")
        self.assertIn("FAILED step 'Static type check (pyright)'", text)

    def test_keep_going_runs_remaining_steps(self) -> None:
        repo = make_repo(self.tmp)
        runner = FakeRunner(fail_when="python3 test_concurrent_backfill.py")
        outcome, _ = run_fake(repo, runner, self.tmp, only=["sqlite"], keep_going=True)
        r = {x.key: x for x in outcome.results}["sqlite[python-version=3.11]"]
        self.assertEqual(r.status, "failed")
        self.assertEqual(r.steps[-1].status, "passed")

    def test_repeat_detects_flaky_concurrency_step(self) -> None:
        repo = make_repo(self.tmp)
        (repo / "test_concurrent_backfill.py").write_text("import threading\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "t")
        runner = FakeRunner(fail_when="python3 test_concurrent_backfill.py", fail_attempts=[2])
        outcome, text = run_fake(repo, runner, self.tmp, only=["sqlite[python-version=3.11]"], repeat=3,
                                 keep_going=True)
        self.assertEqual(outcome.exit_code, ps.EXIT_FAILED)
        self.assertIn("FLAKY/FAILED 1/3", text)
        n = sum(1 for s, _ in runner.scripts() if "python3 test_concurrent_backfill.py" in s)
        self.assertEqual(n, 3)
        # Non-concurrency steps are not repeated.
        self.assertEqual(sum(1 for s, _ in runner.scripts() if "python3 test_cli_diff.py" in s), 1)

    def test_service_health_failure_fails_job_and_collects_logs(self) -> None:
        repo = make_repo(self.tmp)

        class Unhealthy(FakeRunner):
            def run(self, argv, **kw):
                if "exec" in argv and "pg_isready" in " ".join(argv):
                    super().run(argv, **kw)
                    return 1
                return super().run(argv, **kw)

        runner = Unhealthy()
        outcome, text = run_fake(repo, runner, self.tmp, only=["postgres[pg-version=15]"], service_timeout=0)
        self.assertEqual(outcome.exit_code, ps.EXIT_FAILED)
        self.assertIn("health command never succeeded", text)
        runs = [c[1] for c in runner.calls if c[0] == "run"]
        self.assertTrue(any(a[1:2] == ["logs"] for a in runs))
        self.assertTrue(any(a[1:3] == ["rm", "-f"] for a in runs))
        self.assertFalse(any(a[:2] == ["bash", "-e"] and "test_matview_sync" in s
                             for (k, a, _, s) in runner.calls if k == "run"))


# ---------------------------------------------------------------------------
# Real-process tests for the transformation and runner layers
# ---------------------------------------------------------------------------

class RealProcessTests(TmpCase):
    def test_sudo_is_blocked_and_reported(self) -> None:
        cplan = ps.ContainerPlan({}, {})
        script = ps.transform_script("echo before\nsudo apt-get install -y x\necho after\n", cplan,
                                     runtime=None, host="127.0.0.1", bind="127.0.0.1", allow_sudo=False)
        marker = self.tmp / "m"
        p = subprocess.run(["bash", "-e", "-c", script], capture_output=True, text=True,
                           env={**os.environ, "PRESUBMIT_SUDO_MARKER": str(marker)})
        self.assertEqual(p.returncode, 97)
        self.assertTrue(marker.exists())
        self.assertNotIn("after", p.stdout)

    def test_docker_shim_uses_configured_runtime(self) -> None:
        cplan = ps.ContainerPlan({3306: 45678}, {"mysql-srv": "presubmit-x-mysql-srv"})
        script = ps.transform_script('docker exec mysql-srv echo "127.0.0.1:3306 127.0.0.1:33060 mysql-srv-2"\n',
                                     cplan, runtime=["echo", "RT"], host="127.0.0.1", bind="", allow_sudo=True)
        p = subprocess.run(["bash", "-e", "-c", script], capture_output=True, text=True)
        self.assertEqual(p.stdout.strip(),
                         "RT exec presubmit-x-mysql-srv echo 127.0.0.1:45678 127.0.0.1:33060 mysql-srv-2")

    def test_host_port_rewrite_is_exact(self) -> None:
        m = {3306: 4001, 5432: 4002}
        self.assertEqual(ps.rewrite_host_ports("mysql://r:p@127.0.0.1:3306/db", m, "h"), "mysql://r:p@h:4001/db")
        self.assertEqual(ps.rewrite_host_ports("postgresql://localhost:5432/x", m, "h"), "postgresql://h:4002/x")
        for keep in ("127.0.0.1:33060", "10.0.0.1:3306", "127.0.0.1:3307", "-h 127.0.0.1 -P3306"):
            self.assertEqual(ps.rewrite_host_ports(keep, m, "h"), keep)

    def test_process_runner_timeout_kills_process_group(self) -> None:
        log = self.tmp / "log"
        pidfile = self.tmp / "pid"
        rc = ps.ProcessRunner().run(["bash", "-c", f"sleep 30 & echo $! > {pidfile}; wait"], env=None, cwd=None,
                                    log_path=log, timeout=1)
        self.assertEqual(rc, 124)
        self.assertIn("TIMEOUT", log.read_text())
        pid = int(pidfile.read_text())
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            import time
            time.sleep(0.05)
        else:
            self.fail("background child survived the timeout")

    def test_kill_all_stops_running_steps(self) -> None:
        import threading
        import time
        runner = ps.ProcessRunner()
        rcs: List[int] = []
        t = threading.Thread(target=lambda: rcs.append(runner.run(["bash", "-c", "sleep 30"], env=None, cwd=None,
                                                                  log_path=self.tmp / "k", timeout=60)))
        start = time.monotonic()
        t.start()
        while not runner._active and time.monotonic() - start < 5:
            time.sleep(0.01)
        runner.kill_all()
        t.join(10)
        self.assertFalse(t.is_alive())
        self.assertEqual(rcs, [-9])
        self.assertLess(time.monotonic() - start, 10)

    def test_process_runner_propagates_exit_code(self) -> None:
        rc = ps.ProcessRunner().run(["bash", "-c", "exit 7"], env=None, cwd=None, log_path=self.tmp / "l", timeout=10)
        self.assertEqual(rc, 7)
        rc = ps.ProcessRunner().run(["/nonexistent/bin"], env=None, cwd=None, log_path=self.tmp / "l", timeout=10)
        self.assertEqual(rc, 127)

    def test_developer_env_does_not_leak_into_jobs(self) -> None:
        repo = make_repo(self.tmp)
        os.environ["POSTGRES_DSN"] = "postgresql://leak"
        os.environ["MATVIEW_FUZZ_SEEDS"] = "1"
        runner = FakeRunner()
        run_fake(repo, runner, self.tmp, only=["sqlite[python-version=3.11]"])
        envs = [e for _, e in runner.scripts()]
        self.assertTrue(envs)
        for e in envs:
            self.assertNotIn("POSTGRES_DSN", e)
            self.assertNotIn("MATVIEW_FUZZ_SEEDS", e)
            self.assertEqual(e["CI"], "true")
            self.assertEqual(e["TZ"], "UTC")


class FastTierTests(TmpCase):
    def test_affected_selection(self) -> None:
        clf = ps.extract_docs_classifier(RAW_WF)
        tests = ["test_a.py", "test_b.py", "test_c.py"]
        src = {"test_a.py": "import matview_sync", "test_b.py": "import helper", "test_c.py": "x=1",
               "helper.py": "from matview_sync import y", "matview_sync.py": "", "other.py": ""}
        read = src.get
        mods = [m for m in src if m.endswith(".py")]
        self.assertEqual(ps.affected_tests(tests, ["TODO.md"], clf, read, mods)[0], set())
        # README.md is executable documentation (test_readme_examples.py), not whitelisted.
        self.assertEqual(ps.affected_tests(tests, ["README.md"], clf, read, mods)[0], set(tests))
        self.assertEqual(ps.affected_tests(tests, ["test_c.py"], clf, read, mods)[0], {"test_c.py"})
        self.assertEqual(ps.affected_tests(tests, ["matview_sync.py"], clf, read, mods)[0], {"test_a.py", "test_b.py"})
        self.assertEqual(ps.affected_tests(tests, ["helper.py"], clf, read, mods)[0], {"test_b.py"})
        self.assertEqual(ps.affected_tests(tests, ["other.py"], clf, read, mods)[0], set())
        self.assertEqual(ps.affected_tests(tests, ["requirements.txt"], clf, read, mods)[0], set(tests))
        self.assertEqual(ps.affected_tests(tests, ["scripts/x.py"], clf, read, mods)[0], set(tests))

    def test_fast_tier_runs_core_and_affected_only_and_never_stamps(self) -> None:
        repo = make_repo(self.tmp, extra_change="test_cli_diff.py")
        runner = FakeRunner(runtimes_ok=(), pythons=("3.11",))
        outcome, text = run_fake(repo, runner, self.tmp, tier="fast")
        self.assertEqual(outcome.exit_code, ps.EXIT_GREEN, text)
        self.assertIsNone(outcome.stamp)
        self.assertIn("NOT a CI-equivalence claim", text)
        ran = [s for s, _ in runner.scripts()]
        ran_tests = sorted(set(re.findall(r"(test_\w+\.py)", "\n".join(ran))))
        self.assertEqual(ran_tests, sorted({"test_cli_diff.py", *ps.FAST_CORE_TESTS}))
        self.assertFalse(any("docker" in c[1][0] for c in runner.calls if c[0] == "run"))
        self.assertIn("NOT VERIFIED (required by CI)", text)


    def test_fast_preflight_compiles_and_imports_for_real(self) -> None:
        repo = make_repo(self.tmp, extra_change="helper_mod.py")
        (repo / "test_sqlite_runtime_boundaries.py").write_text("import helper_mod\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "t")
        runner = FakeRunner(runtimes_ok=(), pythons=("3.11",))
        outcome, text = run_fake(repo, runner, self.tmp, tier="fast")
        pre = [s for s, _ in runner.scripts() if "importlib" in s]
        self.assertEqual(len(pre), 1, text)
        body = pre[0].split("\n", 2)[2] if pre[0].startswith("docker()") else pre[0]

        def run_pre() -> int:
            p = subprocess.run(["bash", "-e", "-c", body], cwd=repo, capture_output=True, text=True,
                               env={**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"})
            return p.returncode

        self.assertIn("'helper_mod'", body)
        self.assertEqual(run_pre(), 0)
        (repo / "helper_mod.py").write_text("def broken(:\n")
        self.assertNotEqual(run_pre(), 0)
        (repo / "helper_mod.py").write_text("raise ImportError('boom')\n")
        self.assertNotEqual(run_pre(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
