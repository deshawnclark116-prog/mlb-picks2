#!/usr/bin/env python3
"""
HITS_CHAMPION_RECONSTRUCTION_A

Purpose
-------
Reconstruct the exact CURRENT batter-hits production champion before any new
research branch is opened.

This script is read-only. It does not modify code, database tables, predictions,
or production state.

It inventories:
1. Source files containing batter-hits logic.
2. Relevant constants, thresholds, gates, MC settings, and status logic.
3. Functions/classes that appear to drive batter-hits projections or filtering.
4. Relevant database tables/columns.
5. Recent batter-hits prediction/candidate artifacts from /data.
6. A compact reconstruction report for manual review.

Run
---
python -u hits_champion_reconstruction_a.py 2>&1 | tee /data/hr_model/hits_champion_reconstruction_a.log

Output
------
/data/hr_model/hits_champion_reconstruction_a_report.json
/data/hr_model/hits_champion_reconstruction_a_report.txt
"""

import ast
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


ROOT = Path(".").resolve()
DATA_ROOT = Path("/data")
OUT_DIR = Path("/data/hr_model")
OUT_JSON = OUT_DIR / "hits_champion_reconstruction_a_report.json"
OUT_TXT = OUT_DIR / "hits_champion_reconstruction_a_report.txt"

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    "dist",
    "build",
}

SOURCE_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".md", ".txt"}

PRIMARY_PATTERNS = [
    r"\bbatter_hits\b",
    r"\bhitter_hits\b",
    r"\bhit_mc\b",
    r"\bhits_mc\b",
    r"\braw_mc_prob\b",
    r"\bmc_prob\b",
    r"\bconfirmed_lineup\b",
    r"\blineup_confirm",
    r"\bofficial_prediction\b",
    r"\bwatchlist_prediction\b",
    r"\bhit_lean\b",
    r"\bhit.*threshold\b",
    r"\bthreshold.*hit\b",
    r"\bover\s*0\.5\b",
]

SECONDARY_PATTERNS = [
    r"\bprobability\b",
    r"\bprojection\b",
    r"\bmonte.?carlo\b",
    r"\b250000\b",
    r"\b250_000\b",
    r"\bpoisson\b",
    r"\bbvp\b",
    r"\bstatus\b",
    r"\breject_reason\b",
]

NAME_PATTERNS = [
    "batter_hits",
    "hitter_hits",
    "hit_mc",
    "hits_mc",
    "raw_mc_prob",
    "mc_prob",
    "confirmed_lineup",
    "official_prediction",
    "watchlist_prediction",
    "hit_lean",
    "batter_hit",
    "hitter_hit",
]

ARTIFACT_HINTS = [
    "prediction",
    "candidate",
    "record",
    "log",
    "pick",
]

MAX_TEXT_BYTES = 2_000_000
MAX_ARTIFACT_FILES = 60
MAX_ARTIFACT_MATCHES_PER_FILE = 20


def is_excluded(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & EXCLUDE_DIRS)


def iter_source_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if is_excluded(path):
            continue
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_TEXT_BYTES:
                continue
        except OSError:
            continue
        yield path


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def line_matches(text: str, patterns):
    compiled = [re.compile(p, re.I) for p in patterns]
    rows = []
    for i, line in enumerate(text.splitlines(), start=1):
        if any(rx.search(line) for rx in compiled):
            rows.append((i, line.rstrip()))
    return rows


def ast_inventory(path: Path, text: str):
    out = {
        "functions": [],
        "classes": [],
        "assignments": [],
    }
    if path.suffix.lower() != ".py":
        return out

    try:
        tree = ast.parse(text, filename=str(path))
    except Exception:
        return out

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            lname = name.lower()
            if any(k in lname for k in NAME_PATTERNS):
                out["functions"].append({
                    "name": name,
                    "line": getattr(node, "lineno", None),
                })

        elif isinstance(node, ast.ClassDef):
            name = node.name
            lname = name.lower()
            if any(k in lname for k in NAME_PATTERNS):
                out["classes"].append({
                    "name": name,
                    "line": getattr(node, "lineno", None),
                })

        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            else:
                targets = [node.target]

            names = []
            for t in targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
                elif isinstance(t, ast.Attribute):
                    names.append(t.attr)

            for name in names:
                lname = name.lower()
                if any(k in lname for k in NAME_PATTERNS):
                    value_repr = None
                    try:
                        value_node = node.value
                        value_repr = ast.unparse(value_node)
                    except Exception:
                        pass

                    out["assignments"].append({
                        "name": name,
                        "line": getattr(node, "lineno", None),
                        "value": value_repr,
                    })

    return out


def inspect_sources():
    source_hits = []

    for path in iter_source_files(ROOT):
        text = read_text(path)
        primary = line_matches(text, PRIMARY_PATTERNS)

        if not primary:
            continue

        secondary = line_matches(text, SECONDARY_PATTERNS)
        ast_info = ast_inventory(path, text)

        source_hits.append({
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "primary_matches": [
                {"line": n, "text": line}
                for n, line in primary[:200]
            ],
            "secondary_matches": [
                {"line": n, "text": line}
                for n, line in secondary[:100]
            ],
            "ast": ast_info,
        })

    source_hits.sort(
        key=lambda x: (
            -len(x["primary_matches"]),
            x["path"],
        )
    )
    return source_hits


def discover_sqlite_files():
    candidates = []
    for base in [Path("/data"), ROOT]:
        if not base.exists():
            continue
        for path in base.rglob("*.sqlite"):
            if not path.is_file():
                continue
            if is_excluded(path):
                continue
            try:
                candidates.append({
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                })
            except OSError:
                pass
    return candidates


def inspect_sqlite(path: str):
    result = {
        "path": path,
        "tables": [],
        "hits_relevant_tables": [],
    }

    try:
        conn = sqlite3.connect(path)
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table','view') "
                "ORDER BY name"
            ).fetchall()
        ]
        result["tables"] = tables

        for table in tables:
            try:
                cols = [
                    r[1]
                    for r in conn.execute(
                        f'PRAGMA table_info("{table}")'
                    ).fetchall()
                ]
            except Exception:
                cols = []

            joined = " ".join([table] + cols).lower()
            if any(k in joined for k in [
                "batter_hit",
                "hitter_hit",
                "hits",
                "prediction",
                "candidate",
                "pick",
                "mc_prob",
            ]):
                result["hits_relevant_tables"].append({
                    "table": table,
                    "columns": cols,
                })

        conn.close()

    except Exception as e:
        result["error"] = repr(e)

    return result


def likely_artifact_file(path: Path):
    lname = path.name.lower()
    if not any(h in lname for h in ARTIFACT_HINTS):
        return False
    return path.suffix.lower() in {
        ".json",
        ".jsonl",
        ".log",
        ".txt",
        ".csv",
    }


def inspect_artifacts():
    out = []

    if not DATA_ROOT.exists():
        return out

    files = []
    for path in DATA_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if not likely_artifact_file(path):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue

        files.append((stat.st_mtime, stat.st_size, path))

    files.sort(reverse=True)
    files = files[:MAX_ARTIFACT_FILES]

    hit_rx = re.compile(
        r"batter_hits|hitter_hits|\"market\"\s*:\s*\"hits\"|"
        r"\"market\"\s*:\s*\"batter_hits\"|raw_mc_prob|"
        r"official_prediction|watchlist_prediction",
        re.I,
    )

    for mtime, size_bytes, path in files:
        if size_bytes > 20_000_000:
            continue

        text = read_text(path)
        matches = []

        for i, line in enumerate(text.splitlines(), start=1):
            if hit_rx.search(line):
                matches.append({
                    "line": i,
                    "text": line[:2000],
                })
                if len(matches) >= MAX_ARTIFACT_MATCHES_PER_FILE:
                    break

        if matches:
            out.append({
                "path": str(path),
                "size_bytes": size_bytes,
                "mtime": mtime,
                "matches": matches,
            })

    return out


def extract_candidate_summary(source_hits):
    summary = {
        "top_source_files": [],
        "possible_threshold_lines": [],
        "possible_mc_lines": [],
        "possible_status_lines": [],
        "possible_functions": [],
        "possible_assignments": [],
    }

    for item in source_hits[:20]:
        summary["top_source_files"].append({
            "path": item["path"],
            "primary_match_count": len(item["primary_matches"]),
            "secondary_match_count": len(item["secondary_matches"]),
        })

        for row in item["primary_matches"] + item["secondary_matches"]:
            text = row["text"]
            lower = text.lower()

            entry = {
                "path": item["path"],
                "line": row["line"],
                "text": text,
            }

            if "threshold" in lower or re.search(r"\b0\.\d+\b", lower):
                summary["possible_threshold_lines"].append(entry)

            if "mc" in lower or "monte" in lower or "250000" in lower or "250_000" in lower:
                summary["possible_mc_lines"].append(entry)

            if (
                "official_prediction" in lower
                or "watchlist_prediction" in lower
                or "reject" in lower
                or "status" in lower
            ):
                summary["possible_status_lines"].append(entry)

        for f in item["ast"]["functions"]:
            summary["possible_functions"].append({
                "path": item["path"],
                **f,
            })

        for a in item["ast"]["assignments"]:
            summary["possible_assignments"].append({
                "path": item["path"],
                **a,
            })

    # Trim to keep report manageable.
    for key in [
        "possible_threshold_lines",
        "possible_mc_lines",
        "possible_status_lines",
    ]:
        summary[key] = summary[key][:120]

    return summary


def build_text_report(report):
    lines = []

    lines.append("HITS_CHAMPION_RECONSTRUCTION_A")
    lines.append("=" * 32)
    lines.append("READ-ONLY RECONSTRUCTION REPORT")
    lines.append("")

    lines.append("TOP SOURCE FILES")
    lines.append("----------------")
    for row in report["candidate_summary"]["top_source_files"]:
        lines.append(
            f"{row['path']} | "
            f"primary={row['primary_match_count']} "
            f"secondary={row['secondary_match_count']}"
        )

    lines.append("")
    lines.append("POSSIBLE BATTER-HITS FUNCTIONS")
    lines.append("------------------------------")
    for row in report["candidate_summary"]["possible_functions"][:80]:
        lines.append(
            f"{row['path']}:{row['line']} {row['name']}"
        )

    lines.append("")
    lines.append("POSSIBLE BATTER-HITS ASSIGNMENTS / CONSTANTS")
    lines.append("--------------------------------------------")
    for row in report["candidate_summary"]["possible_assignments"][:80]:
        lines.append(
            f"{row['path']}:{row['line']} "
            f"{row['name']} = {row.get('value')}"
        )

    lines.append("")
    lines.append("POSSIBLE THRESHOLDS")
    lines.append("-------------------")
    for row in report["candidate_summary"]["possible_threshold_lines"][:100]:
        lines.append(
            f"{row['path']}:{row['line']} {row['text']}"
        )

    lines.append("")
    lines.append("POSSIBLE MC LOGIC")
    lines.append("-----------------")
    for row in report["candidate_summary"]["possible_mc_lines"][:100]:
        lines.append(
            f"{row['path']}:{row['line']} {row['text']}"
        )

    lines.append("")
    lines.append("POSSIBLE STATUS / OFFICIAL / WATCHLIST LOGIC")
    lines.append("--------------------------------------------")
    for row in report["candidate_summary"]["possible_status_lines"][:100]:
        lines.append(
            f"{row['path']}:{row['line']} {row['text']}"
        )

    lines.append("")
    lines.append("SQLITE DATABASES")
    lines.append("----------------")
    for db in report["sqlite"]:
        lines.append(
            f"{db['path']} | relevant_tables="
            f"{len(db.get('hits_relevant_tables', []))}"
        )
        for table in db.get("hits_relevant_tables", [])[:30]:
            lines.append(
                f"  {table['table']}: "
                + ", ".join(table["columns"])
            )

    lines.append("")
    lines.append("RECENT ARTIFACT MATCHES")
    lines.append("-----------------------")
    for artifact in report["artifacts"][:40]:
        lines.append(
            f"{artifact['path']} | size={artifact['size_bytes']}"
        )
        for match in artifact["matches"][:10]:
            lines.append(
                f"  line {match['line']}: {match['text']}"
            )

    lines.append("")
    lines.append("NEXT REVIEW TARGET")
    lines.append("------------------")
    lines.append(
        "Use this report to lock the exact current batter-hits champion:"
    )
    lines.append("1. exact projection formula/model")
    lines.append("2. exact feature set")
    lines.append("3. MC overlay and raw-vs-final probability")
    lines.append("4. official/watchlist/reject thresholds")
    lines.append("5. confirmed-lineup gates")
    lines.append("6. candidate logs and production behavior")
    lines.append("7. existing known weaknesses")
    lines.append("8. historical/live performance available today")

    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("HITS_CHAMPION_RECONSTRUCTION_A", flush=True)
    print("==============================", flush=True)
    print(f"project_root: {ROOT}", flush=True)
    print("mode: READ ONLY", flush=True)

    print("\nscanning source files...", flush=True)
    source_hits = inspect_sources()
    print(
        f"source files with batter-hits relevance: {len(source_hits)}",
        flush=True,
    )

    print("\ninspecting SQLite databases...", flush=True)
    sqlite_files = discover_sqlite_files()
    sqlite_results = []

    for row in sqlite_files:
        print(f"checking {row['path']}...", flush=True)
        sqlite_results.append(inspect_sqlite(row["path"]))

    print("\nscanning recent /data artifacts...", flush=True)
    artifacts = inspect_artifacts()
    print(
        f"artifact files with batter-hits matches: {len(artifacts)}",
        flush=True,
    )

    candidate_summary = extract_candidate_summary(source_hits)

    report = {
        "script": "HITS_CHAMPION_RECONSTRUCTION_A",
        "mode": "READ_ONLY",
        "project_root": str(ROOT),
        "source_file_count": len(source_hits),
        "source_hits": source_hits,
        "candidate_summary": candidate_summary,
        "sqlite": sqlite_results,
        "artifacts": artifacts,
    }

    OUT_JSON.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    text_report = build_text_report(report)
    OUT_TXT.write_text(text_report, encoding="utf-8")

    print("\nRECONSTRUCTION SUMMARY", flush=True)
    print("----------------------", flush=True)
    print(
        f"top_source_files={len(candidate_summary['top_source_files'])}",
        flush=True,
    )
    print(
        f"possible_functions={len(candidate_summary['possible_functions'])}",
        flush=True,
    )
    print(
        f"possible_assignments={len(candidate_summary['possible_assignments'])}",
        flush=True,
    )
    print(
        f"possible_threshold_lines="
        f"{len(candidate_summary['possible_threshold_lines'])}",
        flush=True,
    )
    print(
        f"possible_mc_lines="
        f"{len(candidate_summary['possible_mc_lines'])}",
        flush=True,
    )
    print(
        f"possible_status_lines="
        f"{len(candidate_summary['possible_status_lines'])}",
        flush=True,
    )
    print(
        f"artifact_files_with_matches={len(artifacts)}",
        flush=True,
    )

    print("\nTOP SOURCE FILES", flush=True)
    print("----------------", flush=True)
    for row in candidate_summary["top_source_files"][:20]:
        print(
            f"{row['path']} | "
            f"primary={row['primary_match_count']} "
            f"secondary={row['secondary_match_count']}",
            flush=True,
        )

    print("\nOUTPUTS", flush=True)
    print("-------", flush=True)
    print(str(OUT_JSON), flush=True)
    print(str(OUT_TXT), flush=True)

    print("\nNEXT STEP", flush=True)
    print("---------", flush=True)
    print(
        "Paste the TOP SOURCE FILES section first. "
        "Then we will reconstruct the current batter-hits champion exactly "
        "before testing any new feature.",
        flush=True,
    )


if __name__ == "__main__":
    main()
