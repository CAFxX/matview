"""Focused tests for schema-graph counterexample shrinking."""
import unittest

from schema_graph_minimizer import minimize_schema_graph


class SchemaGraphMinimizerTests(unittest.TestCase):
    def scenario(self):
        schema = """
CREATE TABLE root (id INTEGER PRIMARY KEY, value INTEGER NOT NULL);
CREATE INDEX ix_root_value ON root (value);
CREATE TABLE middle (
    id INTEGER PRIMARY KEY,
    root_id INTEGER REFERENCES root(id),
    value INTEGER NOT NULL
);
CREATE TABLE leaf (
    id INTEGER PRIMARY KEY,
    middle_id INTEGER REFERENCES middle(id),
    value INTEGER NOT NULL
);
CREATE INDEX ix_leaf_value ON leaf (value);
"""
        views = [
            ("root_view", "CREATE VIEW root_view AS SELECT id, value FROM root;"),
            ("middle_view", "CREATE VIEW middle_view AS SELECT id, value FROM middle;"),
            ("leaf_view", "CREATE VIEW leaf_view AS SELECT id, value FROM leaf;"),
            (
                "root_middle",
                "CREATE VIEW root_middle AS SELECT r.id, m.id FROM root r JOIN middle m ON m.root_id = r.id;",
            ),
            (
                "middle_leaf",
                "CREATE VIEW middle_leaf AS SELECT m.id, l.id FROM middle m JOIN leaf l ON l.middle_id = m.id;",
            ),
        ]
        seed = [
            "INSERT INTO root VALUES (1, 10);",
            "INSERT INTO middle VALUES (1, 1, 20);",
            "INSERT INTO leaf VALUES (1, 1, 30);",
        ]
        mutations = [
            ("root_update", "UPDATE root SET value = value + 1 WHERE id = 1;"),
            ("middle_update", "UPDATE middle SET value = value + 1 WHERE id = 1;"),
            ("leaf_delete", "DELETE FROM leaf WHERE id = 1;"),
        ]
        shape = {
            "topology": "chain",
            "table_count": 3,
            "column_counts": (2, 3, 3),
            "row_counts": (1, 1, 1),
            "foreign_key_count": 2,
            "index_count": 2,
        }
        return schema, views, seed, mutations, shape

    def test_removes_leaf_tables_and_all_dependent_artifacts(self):
        original = self.scenario()

        def signature(candidate):
            schema, views, seed, mutations, _ = candidate
            if (
                "CREATE TABLE root" in schema
                and any(name == "root_view" for name, _ in views)
                and any("UPDATE root " in sql for _, sql in mutations)
            ):
                return ("same-defect", "root-update")
            return None

        minimized = minimize_schema_graph(original, signature)
        schema, views, seed, mutations, shape = minimized
        self.assertIn("CREATE TABLE root", schema)
        self.assertNotIn("CREATE TABLE middle", schema)
        self.assertNotIn("CREATE TABLE leaf", schema)
        self.assertNotIn("ix_leaf", schema)
        self.assertEqual([name for name, _ in views], ["root_view"])
        self.assertEqual(seed, ["INSERT INTO root VALUES (1, 10);"])
        self.assertEqual([name for name, _ in mutations], ["root_update"])
        self.assertEqual(shape["table_count"], 1)
        self.assertEqual(shape["foreign_key_count"], 0)
        self.assertEqual(shape["row_counts"], (1,))
        self.assertEqual(shape["index_count"], 1)
        self.assertEqual(shape["topology"], "shrunk")
        self.assertFalse(shape["coverage_metadata_valid"])

    def test_non_reproducing_scenario_is_returned_unchanged(self):
        original = self.scenario()
        self.assertIs(minimize_schema_graph(original, lambda _: None), original)

    def test_evaluation_budget_is_deterministic(self):
        original = self.scenario()
        minimized = minimize_schema_graph(
            original,
            lambda candidate: "failure" if "CREATE TABLE root" in candidate[0] else None,
            max_evaluations=1,
        )
        self.assertIn("CREATE TABLE middle", minimized[0])
        self.assertNotIn("CREATE TABLE leaf", minimized[0])

    def test_rejects_invalid_evaluation_budget(self):
        with self.assertRaisesRegex(ValueError, "max_evaluations"):
            minimize_schema_graph(self.scenario(), lambda _: "failure", max_evaluations=0)


if __name__ == "__main__":
    unittest.main()
