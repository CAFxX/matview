#!/usr/bin/env python3
"""Execute README.md's examples and check its links and CI matrix.

README fenced-code convention (documented in README.md, "Executable
documentation"):

* Every ``python``, ``sh``, ``bash`` or ``console`` block must carry either
  ``run`` or ``no-run`` in its info string. An unmarked block fails this test,
  so no example is silently unverified.
* ``python run`` blocks are executed one by one, each in a fresh interpreter
  (``sys.executable``) with the repository root on ``PYTHONPATH``, in a
  temporary working directory. They must be self-contained.
* ``sh run`` blocks are executed in document order by ``bash -euo pipefail`` in
  one shared temporary working directory (state such as ``app.db`` carries over
  between them). ``python3`` resolves to ``sys.executable``, the repository
  root is on ``PYTHONPATH`` and ``matview_sync.py`` is linked into the working
  directory, as if the commands ran from a checkout.
* A ``sql file=NAME`` block is written to ``NAME`` in that shared directory
  before the first ``sh run`` block executes.
* A ``text output`` block directly after a ``run`` block holds that block's
  exact expected standard output (trailing whitespace on each line ignored).
* ``no-run`` blocks are only syntax-checked (``compile()`` for Python,
  ``bash -n`` for shell).

The README's CI matrix tables (between ``<!-- BEGIN GENERATED: ci-matrix -->``
and ``<!-- END GENERATED: ci-matrix -->``) must equal the rendering of
.github/workflows/ci.yml, ci-extended.yml and ci-deferred.yml produced here; run
``python3 test_readme_examples.py --print-matrix`` to regenerate them.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"
TODO = ROOT / "TODO.md"
WORKFLOWS = ROOT / ".github" / "workflows"
EXEC_LANGS = {"python", "sh", "bash", "console"}
SHELL_LANGS = {"sh", "bash", "console"}
MATRIX_BEGIN = "<!-- BEGIN GENERATED: ci-matrix -->"
MATRIX_END = "<!-- END GENERATED: ci-matrix -->"
EXAMPLE_TIMEOUT_S = 300


@dataclass(frozen=True)
class Block:
    lang: str
    attrs: Tuple[str, ...]
    body: str
    line: int  # 1-based line of the opening fence

    @property
    def runs(self) -> bool:
        return "run" in self.attrs

    def attr_value(self, key: str) -> Optional[str]:
        for a in self.attrs:
            if a.startswith(key + "="):
                return a[len(key) + 1 :]
        return None


_FENCE_RE = re.compile(r"^(`{3,}|~{3,})(.*)$")


def parse_blocks(text: str) -> List[Block]:
    """Return the fenced code blocks of a Markdown document, in order."""
    blocks: List[Block] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        m = _FENCE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        fence, info = m.group(1), m.group(2).strip()
        start = i
        body: List[str] = []
        i += 1
        while i < len(lines) and not (
            lines[i].startswith(fence[0] * len(fence)) and lines[i].strip() == lines[i].strip()[0] * len(lines[i].strip())
        ):
            body.append(lines[i])
            i += 1
        if i >= len(lines):
            raise ValueError(f"README.md:{start + 1}: unterminated code fence")
        words = info.split()
        lang = words[0].lower() if words else ""
        blocks.append(Block(lang, tuple(words[1:]), "\n".join(body) + "\n", start + 1))
        i += 1
    return blocks


def _expected_output(blocks: Sequence[Block], idx: int) -> Optional[str]:
    if idx + 1 < len(blocks):
        nxt = blocks[idx + 1]
        if nxt.lang == "text" and "output" in nxt.attrs:
            return nxt.body
    return None


def _normalize(out: str) -> str:
    return "\n".join(line.rstrip() for line in out.strip("\n").split("\n"))


def _env(workdir: Path) -> Dict[str, str]:
    env = dict(os.environ)
    shim = workdir / ".bin"
    shim.mkdir(exist_ok=True)
    for name in ("python3", "python"):
        # A wrapper, not a symlink: a symlink to a virtualenv's interpreter
        # resolves to the base interpreter and loses the venv's packages.
        wrapper = shim / name
        if not wrapper.exists():
            wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
            wrapper.chmod(0o755)
    env["PATH"] = f"{shim}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("POSTGRES_DSN", None)
    env.pop("MYSQL_DSN", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run(cmd: Sequence[str], workdir: Path, stdin: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        cwd=workdir,
        env=_env(workdir),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=EXAMPLE_TIMEOUT_S,
    )


# --------------------------------------------------------------------------
# GitHub-style heading anchors and link checks
# --------------------------------------------------------------------------


def github_slug(heading: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def heading_anchors(text: str) -> List[str]:
    anchors: List[str] = []
    seen: Dict[str, int] = {}
    in_fence = False
    for line in text.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        slug = github_slug(m.group(2))
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        anchors.append(slug if n == 0 else f"{slug}-{n}")
    return anchors


def _strip_code(text: str) -> str:
    out: List[str] = []
    in_fence = False
    for line in text.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else line)
    return "\n".join(out)


_LINK_RE = re.compile(r"(?<!!)\[((?:[^\[\]]|`[^`]*`)+)\]\(([^)\s]+)\)")


def markdown_links(text: str) -> List[Tuple[str, str]]:
    return _LINK_RE.findall(_strip_code(text))


def todo_titles(todo_text: str) -> List[str]:
    return re.findall(r"^- \[[ x]\] \*\*(.+?)\*\*", todo_text, flags=re.MULTILINE)


# --------------------------------------------------------------------------
# CI matrix rendering
# --------------------------------------------------------------------------


def load_workflow(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        from test_strength_reporting import _stdlib_yaml_parse

        return _stdlib_yaml_parse(text)


_EXPR_RE = re.compile(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}")


def _subst(value: Any, row: Dict[str, Any]) -> str:
    # GitHub Actions evaluates an undefined matrix property to ''.
    def repl(m: "re.Match[str]") -> str:
        return str(row.get(m.group(1), ""))

    return _EXPR_RE.sub(repl, str(value))


def _matrix_rows(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    if not matrix:
        return [{}]
    if "include" in matrix and len(matrix) == 1:
        return [dict(r) for r in matrix["include"]]
    if "include" in matrix or "exclude" in matrix:
        raise ValueError("matrix with both axes and include/exclude is not supported by the README renderer")
    rows: List[Dict[str, Any]] = [{}]
    for key, values in matrix.items():
        rows = [dict(r, **{key: v}) for r in rows for v in values]
    return rows


def _job_runs_tests(job: Dict[str, Any]) -> bool:
    return any("test_matview_sync.py" in str(s.get("run", "")) for s in job.get("steps") or [])


def _job_python(job: Dict[str, Any], row: Dict[str, Any]) -> str:
    for step in job.get("steps") or []:
        if "setup-python" in str(step.get("uses", "")):
            return _subst((step.get("with") or {}).get("python-version", "?"), row)
    return "runner default"


def _job_database(job: Dict[str, Any], row: Dict[str, Any]) -> str:
    images: List[str] = []
    for svc in (job.get("services") or {}).values():
        images.append(_subst(svc.get("image", ""), row))
    for key in ("image", "pg-image", "mysql-image"):
        if row.get(key):
            images.append(str(row[key]))
    for step in job.get("steps") or []:
        env = step.get("env") or {}
        if "MYSQL_IMAGE" in env and "docker run" in str(step.get("run", "")):
            img = _subst(env["MYSQL_IMAGE"], row)
            if img:
                images.append(img)
    images = list(dict.fromkeys(i for i in images if i))
    if images:
        return ", ".join(f"`{i}`" for i in images)
    driver = row.get("pip-pkg") or row.get("sqlite-engine") or row.get("engine")
    if not driver and row.get("backend") not in (None, "", "sqlite"):
        driver = row["backend"]
    if driver:
        return f"SQLite via `{driver}`"
    return "SQLite bundled with CPython (`sqlite3`)"


def render_matrix_rows(workflow: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    for job in (workflow.get("jobs") or {}).values():
        if not _job_runs_tests(job):
            continue
        for row in _matrix_rows(job):
            rows.append((_subst(job.get("name", ""), row), _job_python(job, row), _job_database(job, row)))
    return rows


def render_matrix() -> str:
    parts: List[str] = []
    for title, fname in (
        ("PR gate ([`ci.yml`](.github/workflows/ci.yml); required, every pull request; fixed fuzz seeds only)", "ci.yml"),
        ("Extended random-seed fuzz ([`ci-extended.yml`](.github/workflows/ci-extended.yml); every push to master, nightly and on demand; not a merge gate, failures open an issue)", "ci-extended.yml"),
        ("Deferred compatibility sweep ([`ci-deferred.yml`](.github/workflows/ci-deferred.yml); weekly, not a merge gate)", "ci-deferred.yml"),
    ):
        parts.append(f"{title}:\n")
        parts.append("| CI job | Python | Database |")
        parts.append("| --- | --- | --- |")
        for name, py, db in render_matrix_rows(load_workflow(WORKFLOWS / fname)):
            parts.append(f"| {name} | {py} | {db} |")
        parts.append("")
    return "\n".join(parts).rstrip("\n")


def readme_matrix_section(text: str) -> str:
    start = text.index(MATRIX_BEGIN) + len(MATRIX_BEGIN)
    end = text.index(MATRIX_END)
    return text[start:end].strip("\n")


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


class ReadmeBlocks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = README.read_text(encoding="utf-8")
        cls.blocks = parse_blocks(cls.text)

    def test_every_executable_block_is_marked(self) -> None:
        for b in self.blocks:
            if b.lang in EXEC_LANGS:
                with self.subTest(line=b.line):
                    marks = {"run", "no-run"} & set(b.attrs)
                    self.assertEqual(len(marks), 1, f"README.md:{b.line}: ```{b.lang} block needs exactly one of run/no-run")

    def test_output_blocks_follow_a_run_block(self) -> None:
        for i, b in enumerate(self.blocks):
            if b.lang == "text" and "output" in b.attrs:
                with self.subTest(line=b.line):
                    self.assertTrue(i > 0 and self.blocks[i - 1].runs, f"README.md:{b.line}: output block without a preceding run block")

    def test_there_are_runnable_examples(self) -> None:
        self.assertGreaterEqual(sum(1 for b in self.blocks if b.lang == "python" and b.runs), 3)
        self.assertGreaterEqual(sum(1 for b in self.blocks if b.lang in SHELL_LANGS and b.runs), 1)

    def test_python_examples(self) -> None:
        for i, b in enumerate(self.blocks):
            if b.lang != "python" or not b.runs:
                continue
            with self.subTest(line=b.line), tempfile.TemporaryDirectory() as tmp:
                res = _run([sys.executable, "-"], Path(tmp), stdin=b.body)
                self.assertEqual(res.returncode, 0, f"README.md:{b.line} failed:\n{res.stdout}\n{res.stderr}")
                expected = _expected_output(self.blocks, i)
                if expected is not None:
                    self.assertEqual(_normalize(res.stdout), _normalize(expected), f"README.md:{b.line}: output differs")

    def test_shell_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "matview_sync.py").symlink_to(ROOT / "matview_sync.py")
            for b in self.blocks:
                name = b.attr_value("file")
                if name is not None:
                    self.assertNotIn("/", name)
                    (work / name).write_text(b.body, encoding="utf-8")
            for i, b in enumerate(self.blocks):
                if b.lang not in SHELL_LANGS or not b.runs:
                    continue
                with self.subTest(line=b.line):
                    res = _run(["bash", "-euo", "pipefail", "-c", b.body], work)
                    self.assertEqual(res.returncode, 0, f"README.md:{b.line} failed:\n{res.stdout}\n{res.stderr}")
                    expected = _expected_output(self.blocks, i)
                    if expected is not None:
                        self.assertEqual(_normalize(res.stdout), _normalize(expected), f"README.md:{b.line}: output differs")

    def test_no_run_blocks_are_syntactically_valid(self) -> None:
        for b in self.blocks:
            if b.runs or b.lang not in EXEC_LANGS:
                continue
            with self.subTest(line=b.line):
                if b.lang == "python":
                    compile(b.body, f"README.md:{b.line}", "exec")
                else:
                    res = subprocess.run(["bash", "-n"], input=b.body, capture_output=True, text=True)
                    self.assertEqual(res.returncode, 0, f"README.md:{b.line}: {res.stderr}")


class ReadmeLinks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = README.read_text(encoding="utf-8")
        cls.links = markdown_links(cls.text)

    def test_internal_anchors_exist(self) -> None:
        anchors = set(heading_anchors(self.text))
        for _, target in self.links:
            if target.startswith("#"):
                with self.subTest(target=target):
                    self.assertIn(target[1:], anchors)

    def test_relative_file_links_exist(self) -> None:
        for _, target in self.links:
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            path = target.split("#", 1)[0]
            with self.subTest(target=target):
                self.assertTrue((ROOT / path).exists(), f"README links to missing {path}")

    def test_todo_links_name_existing_items(self) -> None:
        titles = todo_titles(TODO.read_text(encoding="utf-8"))
        named = [label for label, target in self.links if target == "TODO.md" and label != "TODO.md" and not label.startswith("`")]
        self.assertGreater(len(named), 0)
        for label in named:
            with self.subTest(label=label):
                self.assertIn(label, titles, "README names a TODO.md item that does not exist (by exact title)")


class ReadmeMatrix(unittest.TestCase):
    def test_matrix_matches_workflows(self) -> None:
        text = README.read_text(encoding="utf-8")
        self.assertEqual(
            readme_matrix_section(text),
            render_matrix(),
            "README CI matrix is stale; regenerate with: python3 test_readme_examples.py --print-matrix",
        )

    def test_renderer_reflects_workflow_changes(self) -> None:
        wf = load_workflow(WORKFLOWS / "ci.yml")
        before = render_matrix_rows(wf)
        wf["jobs"]["postgres"]["strategy"]["matrix"]["pg-version"] = ["15", "99"]
        after = render_matrix_rows(wf)
        self.assertIn(("PostgreSQL 99", "3.12", "`postgres:99`"), after)
        self.assertNotEqual(before, after)

    def test_stdlib_parser_renders_the_same_matrix(self) -> None:
        try:
            import yaml  # type: ignore  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed; the stdlib parser is already the one in use")
        from test_strength_reporting import _stdlib_yaml_parse

        for fname in ("ci.yml", "ci-extended.yml", "ci-deferred.yml"):
            with self.subTest(workflow=fname):
                path = WORKFLOWS / fname
                self.assertEqual(
                    render_matrix_rows(_stdlib_yaml_parse(path.read_text(encoding="utf-8"))),
                    render_matrix_rows(load_workflow(path)),
                )

    def test_docker_compose_images_are_the_pr_gate_servers(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        compose_images = set(re.findall(r"^\s+image:\s*([^\s#]+)", compose, flags=re.MULTILINE))
        gate_images = set()
        for _, _, db in render_matrix_rows(load_workflow(WORKFLOWS / "ci.yml")):
            gate_images.update(re.findall(r"`([^`]+:[^`]+)`", db))
        self.assertEqual(compose_images - {i for i in compose_images if not i.startswith(("postgres", "mysql", "mariadb"))}, gate_images)


class ConventionParserSelfTest(unittest.TestCase):
    """The parser and link checks must detect the faults they exist for."""

    def test_unmarked_block_is_reported(self) -> None:
        blocks = parse_blocks("```python\nprint(1)\n```\n")
        self.assertEqual(blocks[0].attrs, ())
        self.assertFalse({"run", "no-run"} & set(blocks[0].attrs))

    def test_attributes_and_output_pairing(self) -> None:
        blocks = parse_blocks("```python run\nprint(1)\n```\n\n```text output\n1\n```\n```sql file=a.sql\nSELECT 1;\n```\n")
        self.assertTrue(blocks[0].runs)
        self.assertEqual(_expected_output(blocks, 0), "1\n")
        self.assertEqual(blocks[2].attr_value("file"), "a.sql")

    def test_slugs(self) -> None:
        self.assertEqual(github_slug("Transactions and drivers"), "transactions-and-drivers")
        self.assertEqual(github_slug("`bulk_changes()`"), "bulk_changes")
        self.assertEqual(github_slug("Replacement, schema changes & repair"), "replacement-schema-changes--repair")
        self.assertEqual(heading_anchors("# A\n## A\n```\n# B\n```\n"), ["a", "a-1"])

    def test_links_inside_code_are_ignored(self) -> None:
        self.assertEqual(markdown_links("```\n[x](#nope)\n```\n[y](#yes)\n"), [("y", "#yes")])


def main(argv: Sequence[str]) -> int:
    if "--print-matrix" in argv:
        print(MATRIX_BEGIN)
        print(render_matrix())
        print(MATRIX_END)
        return 0
    return 0 if unittest.main(argv=[argv[0], "-v"], exit=False).result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
