"""Best-effort record of recent diffs in the plugin's own SQLite DB (<hermes home>/plugin-data/weightscope/)."""

from __future__ import annotations

import datetime as _dt
import logging

logger = logging.getLogger(__name__)
PLUGIN_NAME = "weightscope"
KEEP = 200

_SCHEMA = """CREATE TABLE IF NOT EXISTS diffs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  surface TEXT,
  path_a TEXT, path_b TEXT,
  compared_tensors INTEGER, changed_tensors INTEGER,
  global_rel_change REAL, top_tensor TEXT, top_rel_change REAL
)"""


def _db():
    from plugins.plugin_storage import plugin_db
    conn = plugin_db(PLUGIN_NAME)
    conn.execute(_SCHEMA)
    return conn


def record(result: dict, surface: str) -> bool:
    """Store a diff's metadata (paths + headline numbers, never tensor data). Never raises."""
    try:
        top = (result.get("top_tensors") or [{}])[0]
        totals = result.get("totals", {})
        conn = _db()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO diffs (created_at, surface, path_a, path_b, compared_tensors, changed_tensors,"
                    " global_rel_change, top_tensor, top_rel_change) VALUES (?,?,?,?,?,?,?,?,?)",
                    (_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), surface,
                     result["a"]["path"], result["b"]["path"], totals.get("compared_tensors"),
                     totals.get("changed_tensors"), totals.get("global_rel_change"), top.get("name"),
                     top.get("rel_change")))
                conn.execute("DELETE FROM diffs WHERE id NOT IN (SELECT id FROM diffs ORDER BY id DESC LIMIT ?)",
                             (KEEP,))
        finally:
            conn.close()
        return True
    except Exception as exc:  # history must never break a diff
        logger.debug("weightscope history record failed: %s", exc)
        return False


def recent(limit: int = 10) -> list[dict]:
    try:
        conn = _db()
        try:
            rows = conn.execute(
                "SELECT created_at, surface, path_a, path_b, compared_tensors, changed_tensors, global_rel_change,"
                " top_tensor, top_rel_change FROM diffs ORDER BY id DESC LIMIT ?", (max(1, min(int(limit), KEEP)),)
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        logger.debug("weightscope history read failed: %s", exc)
        return []
    cols = ("created_at_utc", "surface", "a", "b", "compared_tensors", "changed_tensors", "global_rel_change",
            "top_tensor", "top_rel_change")
    return [dict(zip(cols, r)) for r in rows]
