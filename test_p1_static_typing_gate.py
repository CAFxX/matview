#!/usr/bin/env python3
"""Contract tests for the pyright gate: configuration, mixin host contract, fault detection.

What is tested, from the documented behavior of the gate rather than from how
pyright is wired:

* The pyright configuration cannot be weakened silently: it targets the oldest
  supported Python, uses at least the ``standard`` mode, and disables no rule.
  Every in-source suppression names its rule and states a reason.
* The CI step runs once (the oldest-Python matrix cell) and the tool version is
  pinned with an upper bound.
* ``matview/_host.py`` declares the cross-mixin contract that lets pyright check
  the mixins. It is type-checking-only, so a stale declaration would silently
  hide a real error: every declaration must match the live implementation.
* Fault detection (native pyright run, oldest-Python cell only): reintroducing
  the defects this gate found, or making a mixin call a misspelled sibling or
  pass a wrong argument type, must produce exactly the expected diagnostics.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

import matview
import matview.compiler as compiler_mod
from matview._host import MixinHost
from matview.compiler import IncrementalViewCompiler

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "pyrightconfig.json").read_text())
CI = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
FLOOR = min(
    (str(v) for v in CI["jobs"]["sqlite"]["strategy"]["matrix"]["python-version"]),
    key=lambda v: tuple(int(p) for p in v.split(".")),
)


def _production_python_files() -> List[Path]:
    files = [ROOT / "matview_sync.py"]
    files += sorted((ROOT / "matview").rglob("*.py"))
    files += sorted(p for p in (ROOT / "scripts").glob("*.py") if not p.name.startswith("test_"))
    return files


class TestPyrightConfiguration(unittest.TestCase):
    def test_targets_the_oldest_supported_python(self) -> None:
        self.assertEqual(CONFIG["pythonVersion"], FLOOR)

    def test_mode_is_at_least_standard_and_no_rule_is_disabled(self) -> None:
        self.assertIn(CONFIG["typeCheckingMode"], {"standard", "strict"})
        weakened = {
            k: v for k, v in CONFIG.items()
            if k.startswith("report") and v in (False, "none", "off", "information")
        }
        self.assertEqual(weakened, {})
        for key in ("ignore", "reportMissingImports", "reportMissingModuleSource"):
            self.assertNotIn(key, CONFIG)

    def test_production_code_is_in_scope_and_tests_are_the_only_exclusion(self) -> None:
        self.assertIn("matview", CONFIG["include"])
        self.assertIn("matview_sync.py", CONFIG["include"])
        self.assertIn("scripts", CONFIG["include"])
        self.assertEqual(
            [e for e in CONFIG["exclude"] if "__pycache__" not in e], ["scripts/test_*.py"]
        )

    def test_every_in_source_suppression_names_its_rule_and_reason(self) -> None:
        pattern = re.compile(r"#\s*(pyright|type):\s*(ignore|basic|off|standard|strict)\b(.*)")
        seen = 0
        for path in _production_python_files():
            for n, line in enumerate(path.read_text().splitlines(), 1):
                m = pattern.search(line)
                if not m:
                    continue
                seen += 1
                where = f"{path.relative_to(ROOT)}:{n}"
                self.assertEqual(m.group(1), "pyright", f"{where}: use '# pyright: ignore[rule]', not '# type:'")
                self.assertEqual(m.group(2), "ignore", f"{where}: file-level mode comments are not allowed")
                self.assertRegex(
                    m.group(3), r"^\[[A-Za-z]+(?:, ?[A-Za-z]+)*\]\s+# \S.*", f"{where}: needs [rule] and a reason"
                )
        self.assertGreater(seen, 0, "scan found no suppressions at all: the scan itself is broken")

    def test_ci_runs_pyright_exactly_once_on_the_floor_cell(self) -> None:
        steps = [
            s for job in CI["jobs"].values() for s in job.get("steps", [])
            if re.search(r"(?m)^\s*(python3? -m )?pyright\b", str(s.get("run", "")))
        ]
        self.assertEqual(len(steps), 1, steps)
        self.assertIn(steps[0], CI["jobs"]["sqlite"]["steps"])
        self.assertEqual(steps[0]["if"].replace("${{", "").replace("}}", "").strip(),
                         f"matrix.python-version == '{FLOOR}'")
        self.assertIn("sqlite", CI["jobs"]["ci-success"]["needs"])

    def test_tool_version_is_pinned_with_an_upper_bound(self) -> None:
        reqs = [
            ln.strip() for ln in (ROOT / "requirements-dev.txt").read_text().splitlines()
            if ln.strip().startswith("pyright")
        ]
        self.assertEqual(len(reqs), 1, reqs)
        self.assertRegex(reqs[0], r"^pyright\s*(==\s*\d[\w.]*|>=\s*\d[\w.]*\s*,\s*<\s*\d[\w.]*)$")


class TestMixinHostContract(unittest.TestCase):
    """``MixinHost`` declarations (type-checking only) must equal the live implementations."""

    @classmethod
    def setUpClass(cls) -> None:
        tree = ast.parse((ROOT / "matview" / "_host.py").read_text())
        host = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MixinHost")
        guard = next(n for n in host.body if isinstance(n, ast.If))
        assert ast.unparse(guard.test) == "TYPE_CHECKING"
        cls.declared: Dict[str, ast.FunctionDef] = {
            n.name: n for n in guard.body if isinstance(n, ast.FunctionDef)
        }

    @staticmethod
    def _live_function_ast(name: str) -> Tuple[ast.FunctionDef, Any]:
        raw = inspect.getattr_static(IncrementalViewCompiler, name)
        fn = raw.fget if isinstance(raw, property) else getattr(raw, "__func__", raw)
        src = textwrap.dedent(inspect.getsource(fn))
        node = ast.parse(src).body[0]
        assert isinstance(node, ast.FunctionDef)
        return node, raw

    def test_declarations_exist_and_are_substantial(self) -> None:
        # The contract is generated from every cross-mixin access pyright reported;
        # a near-empty block would mean it stopped describing anything.
        self.assertGreater(len(self.declared), 80)
        self.assertIn("__init__", self.declared)

    def test_each_declaration_matches_the_live_implementation(self) -> None:
        for name, stub in sorted(self.declared.items()):
            with self.subTest(member=name):
                self.assertTrue(hasattr(IncrementalViewCompiler, name), f"{name} no longer exists")
                live, raw = self._live_function_ast(name)
                self.assertEqual(
                    [ast.dump(d) for d in stub.decorator_list],
                    [ast.dump(d) for d in live.decorator_list],
                    f"{name}: decorators differ",
                )
                self.assertEqual(ast.dump(stub.args), ast.dump(live.args), f"{name}: parameters differ")
                self.assertEqual(
                    ast.dump(stub.returns) if stub.returns else None,
                    ast.dump(live.returns) if live.returns else None,
                    f"{name}: return annotation differs",
                )
                is_property = any(ast.unparse(d) == "property" for d in stub.decorator_list)
                self.assertEqual(isinstance(raw, property), is_property, name)

    def test_host_is_empty_at_runtime_so_real_implementations_are_the_only_ones(self) -> None:
        runtime_members = {k for k in vars(MixinHost) if not (k.startswith("__") and k.endswith("__"))}
        self.assertEqual(runtime_members, set())
        for name in self.declared:
            self.assertNotIn(name, vars(MixinHost), name)

    def test_every_mixin_derives_from_the_host(self) -> None:
        import importlib
        import pkgutil

        mixins: Set[type] = set()
        for info in pkgutil.walk_packages(matview.__path__, "matview."):
            mod = importlib.import_module(info.name)
            for obj in vars(mod).values():
                if isinstance(obj, type) and obj.__name__.endswith("Mixin") and obj.__module__ == mod.__name__:
                    mixins.add(obj)
        self.assertGreaterEqual(len(mixins), 14)
        for mixin in sorted(mixins, key=lambda c: c.__name__):
            self.assertTrue(issubclass(mixin, MixinHost), f"{mixin.__name__} does not derive from MixinHost")
        self.assertIs(compiler_mod.IncrementalViewCompiler.__mro__[-2], MixinHost)


class TestPyrightDetectsInjectedFaults(unittest.TestCase):
    """Native pyright against a copy of the production tree with deliberate faults."""

    @classmethod
    def setUpClass(cls) -> None:
        if tuple(map(int, FLOOR.split("."))) != sys.version_info[:2]:
            raise unittest.SkipTest(f"pyright gate is defined for Python {FLOOR}; running {sys.version_info[:2]}")
        exe = shutil.which("pyright")
        if exe is None:
            raise unittest.SkipTest("pyright is not installed (pip install -r requirements-dev.txt)")
        cls.exe = exe

    def _run(self, work: Path) -> List[Dict[str, Any]]:
        cfg = dict(CONFIG)
        cfg["include"] = ["matview", "matview_sync.py"]
        (work / "pyrightconfig.json").write_text(json.dumps(cfg))
        proc = subprocess.run(
            [self.exe, "--outputjson", "--pythonpath", sys.executable],
            cwd=work, capture_output=True, text=True, timeout=600,
        )
        data = json.loads(proc.stdout)
        return [d for d in data["generalDiagnostics"] if d["severity"] == "error"]

    @staticmethod
    def _copy_tree(dst: Path) -> None:
        shutil.copytree(ROOT / "matview", dst / "matview", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(ROOT / "matview_sync.py", dst / "matview_sync.py")

    @staticmethod
    def _edit(path: Path, old: str, new: str) -> None:
        text = path.read_text()
        assert text.count(old) == 1, (path, old, text.count(old))
        path.write_text(text.replace(old, new))

    def test_clean_tree_has_no_errors_and_each_fault_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            self._copy_tree(work)
            self.assertEqual(self._run(work), [], "the unmodified production tree must be clean")

            planner = work / "matview" / "planner.py"
            frontend = work / "matview" / "sql_frontend.py"
            engines = work / "matview" / "engines.py"
            # 1. Defect found by this gate: a staticmethod that uses `self`.
            self._edit(
                planner,
                "    def _extract_index_col_base_ident(self, col_spec: str) -> str:",
                "    @staticmethod\n    def _extract_index_col_base_ident(col_spec: str) -> str:",
            )
            # 2. Defect found by this gate: attribute that does not exist on the sqlglot node.
            self._edit(
                frontend,
                'for c in (params_node.args.get("columns") or [])',
                "for c in params_node.columns",
            )
            # 3. A mixin calls a misspelled sibling method (the host contract must not hide typos).
            # 4. A mixin passes a wrong argument type to a sibling method (signatures are live).
            self._edit(
                planner,
                "    @staticmethod\n    def _extract_index_col_base_ident(col_spec: str) -> str:",
                "    def _fault_misspelled(self) -> None:\n"
                "        self._no_such_sibling_method()\n\n"
                "    def _fault_wrong_type(self) -> str:\n"
                "        return self._q(12345)\n\n"
                "    @staticmethod\n    def _extract_index_col_base_ident(col_spec: str) -> str:",
            )
            # 5. Possibly unbound variable.
            with engines.open("a") as fh:
                fh.write(
                    "\n\ndef _fault_unbound(flag: bool) -> int:\n"
                    "    if flag:\n        value = 1\n    return value\n"
                )

            got = {(d["file"].replace(str(work) + "/", ""), d["rule"]) for d in self._run(work)}
            expected = {
                ("matview/planner.py", "reportUndefinedVariable"),
                ("matview/sql_frontend.py", "reportAttributeAccessIssue"),
                ("matview/planner.py", "reportAttributeAccessIssue"),
                ("matview/planner.py", "reportArgumentType"),
                ("matview/engines.py", "reportPossiblyUnboundVariable"),
            }
            self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
