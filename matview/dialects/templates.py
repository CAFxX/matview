"""Jinja2 procedural SQL templates for triggers, functions, and chunked backfill."""

from __future__ import annotations

from typing import Any, Dict

from jinja2 import BaseLoader, DictLoader, Environment, StrictUndefined

PROCEDURAL_SQL_TEMPLATES: Dict[str, str] = {
    "ro_trigger_sqlite": """CREATE TRIGGER {{ trg_name }}
INSTEAD OF {{ op }} ON {{ db_view }}
BEGIN
    SELECT RAISE(ABORT, 'Materialized view {{ raw_db_view }} is read-only and maintained automatically');
END;""",
    "ro_function_pg": """CREATE OR REPLACE FUNCTION {{ fn_name }}()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Materialized view % is read-only and maintained automatically', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;""",
    "ro_trigger_pg": """CREATE TRIGGER {{ trg_name }}
INSTEAD OF INSERT OR UPDATE OR DELETE ON {{ db_view }}
FOR EACH ROW EXECUTE FUNCTION {{ fn_name }}();""",
    "backfill_pk_sqlite": """WITH RECURSIVE __mv_bounds AS (
    SELECT MIN({{ q_pk_col }}) AS min_id, MAX({{ q_pk_col }}) AS max_id FROM {{ q_base_t }}
),
__mv_batches(lo, hi, max_id) AS (
    SELECT min_id - 1, MIN(max_id, min_id - 1 + {{ batch_size }}), max_id
    FROM __mv_bounds
    WHERE min_id IS NOT NULL AND max_id IS NOT NULL
    UNION ALL
    SELECT hi, MIN(max_id, hi + {{ batch_size }}), max_id
    FROM __mv_batches
    WHERE hi < max_id
)
{{ joined_tmpl }};""",
    "backfill_pk_pg": """DO $$
DECLARE
    __mv_min_pk BIGINT;
    __mv_max_pk BIGINT;
    __mv_cur_pk BIGINT;
    __mv_top_pk BIGINT;
    __mv_batch_size BIGINT := {{ batch_size }};
BEGIN
    SELECT MIN({{ q_pk_col }}), MAX({{ q_pk_col }}) INTO __mv_min_pk, __mv_max_pk FROM {{ q_base_t }};
    IF __mv_min_pk IS NOT NULL AND __mv_max_pk IS NOT NULL THEN
        __mv_cur_pk := __mv_min_pk - 1;
        WHILE __mv_cur_pk < __mv_max_pk LOOP
            __mv_top_pk := LEAST(__mv_max_pk, __mv_cur_pk + __mv_batch_size);
            {{ pg_chunk }};
            __mv_cur_pk := __mv_top_pk;
        END LOOP;
    END IF;
{% for pop_sql in fallback_populate_sqls %}
    IF NOT EXISTS (SELECT 1 FROM {{ q_backing_table }}) THEN
        {{ pop_sql }};
    END IF;
{% endfor %}
END $$;""",
    "drop_procedure_mysql": "DROP PROCEDURE IF EXISTS {{ proc_name }};",
    "call_procedure_mysql": "CALL {{ proc_name }}();",
    "backfill_pk_mysql_proc": """CREATE PROCEDURE {{ proc_name }}()
BEGIN
    DECLARE __mv_min_pk BIGINT;
    DECLARE __mv_max_pk BIGINT;
    DECLARE __mv_cur_pk BIGINT;
    DECLARE __mv_top_pk BIGINT;
    DECLARE __mv_batch_size BIGINT DEFAULT {{ batch_size }};
    SELECT MIN({{ q_pk_col }}), MAX({{ q_pk_col }}) INTO __mv_min_pk, __mv_max_pk FROM {{ q_base_t }};
    IF __mv_min_pk IS NOT NULL AND __mv_max_pk IS NOT NULL THEN
        SET __mv_cur_pk = __mv_min_pk - 1;
        WHILE __mv_cur_pk < __mv_max_pk DO
            SET __mv_top_pk = LEAST(__mv_max_pk, __mv_cur_pk + __mv_batch_size);
            {{ my_chunk }};
            SET __mv_cur_pk = __mv_top_pk;
        END WHILE;
    END IF;
{% for pop_sql in fallback_populate_sqls %}
    IF NOT EXISTS (SELECT 1 FROM {{ q_backing_table }}) THEN
        {{ pop_sql }};
    END IF;
{% endfor %}
END;""",
    "backfill_offset_sqlite": """WITH __mv_batched_src AS (
    SELECT
        ((ROW_NUMBER() OVER () - 1) / {{ batch_size }}) AS __mv_batch_id,
        __mv_raw_src.*
    FROM (
        {{ aliased_inner }}
    ) AS __mv_raw_src
    ORDER BY 1
)
INSERT INTO {{ q_target_tbl }} ({{ q_cols_csv }})
SELECT {{ q_cols_csv }}
FROM __mv_batched_src
WHERE TRUE{{ upsert_tail }};""",
    "backfill_offset_pg": """DO $$
DECLARE
    __mv_offset BIGINT := 0;
    __mv_batch_size BIGINT := {{ batch_size }};
    __mv_rows BIGINT := 0;
BEGIN
    LOOP
{% if use_merge %}
        MERGE INTO {{ q_target_tbl }} AS __mv_dst
        USING (
            SELECT {{ q_cols_csv }}
            FROM (
                {{ aliased_inner }}
            ) AS __mv_init{{ ord_tail }}
            LIMIT __mv_batch_size OFFSET __mv_offset
        ) AS __mv_src
        ON ({{ merge_on_csv }})
        {{ matched_clause }}
        WHEN NOT MATCHED THEN
            INSERT ({{ q_cols_csv }})
            VALUES ({{ merge_vals_csv }});
{% else %}
        INSERT INTO {{ q_target_tbl }} ({{ q_cols_csv }})
        SELECT {{ q_cols_csv }}
        FROM (
            {{ aliased_inner }}
        ) AS __mv_init{{ ord_tail }}
        LIMIT __mv_batch_size OFFSET __mv_offset{{ upsert_tail }};
{% endif %}
        GET DIAGNOSTICS __mv_rows = ROW_COUNT;
        EXIT WHEN __mv_rows < __mv_batch_size;
        __mv_offset := __mv_offset + __mv_rows;
    END LOOP;
END $$;""",
    "backfill_offset_mysql_proc": """CREATE PROCEDURE {{ proc_name }}()
BEGIN
    DECLARE __mv_offset BIGINT DEFAULT 0;
    DECLARE __mv_batch_size BIGINT DEFAULT {{ batch_size }};
    DECLARE __mv_rows BIGINT DEFAULT 0;
    REPEAT
        INSERT INTO {{ q_target_tbl }} ({{ q_cols_csv }})
        SELECT {{ q_cols_csv }}
        FROM (
            {{ aliased_inner }}
        ) AS __mv_init{{ ord_tail }}
        LIMIT __mv_batch_size OFFSET __mv_offset{{ upsert_tail }};
        SET __mv_rows = ROW_COUNT();
        SET __mv_offset = __mv_offset + __mv_batch_size;
    UNTIL __mv_rows <= 0 END REPEAT;
END;""",
    "trigger_sqlite_after_insert": """CREATE TRIGGER {{ trg_name }}
AFTER INSERT ON {{ q_base_t }}
FOR EACH ROW{% if when_clause %}

WHEN {{ when_clause }}{% endif %}

BEGIN
    {{ body }}
END;""",
    "trigger_sqlite_after_delete": """CREATE TRIGGER {{ trg_name }}
AFTER DELETE ON {{ q_base_t }}
FOR EACH ROW{% if when_clause %}

WHEN {{ when_clause }}{% endif %}

BEGIN
    {{ body }}
END;""",
    "trigger_sqlite_after_update": """CREATE TRIGGER {{ trg_name }}
AFTER UPDATE{{ upd_of_clause }} ON {{ q_base_t }}
FOR EACH ROW{% if when_clause %}

WHEN {{ when_clause }}{% endif %}

BEGIN
    {{ body }}
END;""",
    "pg_truncate_function": """CREATE OR REPLACE FUNCTION {{ fn_name }}()
RETURNS TRIGGER AS $$
BEGIN
    {{ trunc_block }}
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;""",
    "pg_truncate_trigger": """CREATE TRIGGER {{ trg_name }}
AFTER TRUNCATE ON {{ q_base_t }}
FOR EACH STATEMENT EXECUTE FUNCTION {{ fn_name }}();""",
    "pg_transition_simple_function": """CREATE OR REPLACE FUNCTION {{ fn_name }}()
RETURNS TRIGGER AS $$
BEGIN
    {{ body_sql }}
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;""",
    "pg_transition_loop_function": """CREATE OR REPLACE FUNCTION {{ fn_name }}()
RETURNS TRIGGER AS $$
DECLARE
    {{ row_var }} RECORD;
BEGIN
    FOR {{ row_var }} IN SELECT * FROM {{ transition_table }} LOOP
        {{ section_sql }}
    END LOOP;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;""",
    "pg_transition_au_loop_function": """CREATE OR REPLACE FUNCTION {{ fn_name }}()
RETURNS TRIGGER AS $$
DECLARE
    OLD RECORD;
    NEW RECORD;
BEGIN
{% for loop_spec in loops %}
    FOR {{ loop_spec.row_var }} IN SELECT * FROM {{ loop_spec.transition_table }} LOOP
        {{ loop_spec.section_sql }}
    END LOOP;
{% endfor %}
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;""",
    "pg_transition_trigger": """CREATE TRIGGER {{ trg_name }}
AFTER {{ event }} ON {{ q_base_t }}
REFERENCING {{ referencing }}
FOR EACH STATEMENT EXECUTE FUNCTION {{ fn_name }}();""",
    "pg_row_sync_function": """CREATE OR REPLACE FUNCTION {{ fn_name }}()
RETURNS TRIGGER AS $$
BEGIN
    IF COALESCE(current_setting('matview.skip_triggers', true), '') = 'on' THEN
        RETURN NULL;
    END IF;
    IF (TG_OP = 'DELETE') THEN
        {{ del_section }}
    ELSIF (TG_OP = 'INSERT') THEN
        {{ ins_section }}
    ELSIF (TG_OP = 'UPDATE') THEN
        {{ upd_body }}
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;""",
    "pg_row_sync_trigger": """CREATE TRIGGER {{ trg_name }}
AFTER INSERT OR UPDATE{{ upd_of_clause }} OR DELETE ON {{ q_base_t }}
FOR EACH ROW EXECUTE FUNCTION {{ fn_name }}();""",
    "mysql_row_trigger": """CREATE TRIGGER {{ trg_name }}
AFTER {{ event }} ON {{ q_base_t }}
FOR EACH ROW
BEGIN
    IF COALESCE(@__matview_skip_triggers, 0) = 0 THEN
    {{ body }}
    END IF;
END;""",
}

_PROCEDURAL_TEMPLATE_ENV = Environment(
    loader=DictLoader(PROCEDURAL_SQL_TEMPLATES),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    undefined=StrictUndefined,
)


def render_procedural_sql(template_name: str, **context: Any) -> str:
    """Render a dialect procedural trigger or backfill SQL statement via Jinja2."""
    return _PROCEDURAL_TEMPLATE_ENV.get_template(template_name).render(**context)

