"""
utils/progress_tracker.py
──────────────────────────────────────────────────────────────────────────────
SQLite-backed student progress tracker for StudyPal.

Tracks:
  • quiz_results   — every completed quiz (subject, score, total, timestamp)
  • questions      — every question asked (subject, timestamp)
  • weak_topics    — subjects where average quiz score < 60%

Public API
──────────
  init_db()
  save_quiz_result(subject, score, total)
  save_question(subject, question_text)
  get_quiz_history()          → list of dicts
  get_question_history()      → list of dicts
  get_subject_stats()         → dict  {subject: {attempts, avg_score, best, worst}}
  get_weak_topics(threshold)  → list of subject strings
  get_streak()                → int  (consecutive days with activity)
  get_summary_stats()         → dict  (totals for the overview cards)
  clear_all_progress()
"""

from __future__ import annotations

import sqlite3
import os
from datetime import datetime, date, timedelta
from typing import List, Dict, Any

DB_PATH = "./studypal_progress.db"


# ── Init ──────────────────────────────────────────────────────────────────────
def init_db() -> None:
    """Create tables if they don't exist."""
    with _conn() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS quiz_results (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            subject   TEXT    NOT NULL,
            score     INTEGER NOT NULL,
            total     INTEGER NOT NULL,
            pct       REAL    NOT NULL,
            ts        TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS questions (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            subject  TEXT NOT NULL,
            text     TEXT NOT NULL,
            ts       TEXT NOT NULL
        );
        """)


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


# ── Writers ───────────────────────────────────────────────────────────────────
def save_quiz_result(subject: str, score: int, total: int) -> None:
    pct = round((score / total) * 100, 1) if total else 0
    ts  = datetime.now().isoformat(timespec="seconds")
    with _conn() as con:
        con.execute(
            "INSERT INTO quiz_results (subject, score, total, pct, ts) VALUES (?,?,?,?,?)",
            (subject, score, total, pct, ts),
        )


def save_question(subject: str, question_text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with _conn() as con:
        con.execute(
            "INSERT INTO questions (subject, text, ts) VALUES (?,?,?)",
            (subject, question_text[:500], ts),
        )


# ── Readers ───────────────────────────────────────────────────────────────────
def get_quiz_history(limit: int = 50) -> List[Dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM quiz_results ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_question_history(limit: int = 50) -> List[Dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM questions ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_subject_stats() -> Dict[str, Dict]:
    """Return per-subject stats: attempts, avg_score, best, worst, last_ts."""
    with _conn() as con:
        rows = con.execute("""
            SELECT subject,
                   COUNT(*)       AS attempts,
                   AVG(pct)       AS avg_pct,
                   MAX(pct)       AS best,
                   MIN(pct)       AS worst,
                   MAX(ts)        AS last_ts
            FROM quiz_results
            GROUP BY subject
        """).fetchall()
    return {
        r["subject"]: {
            "attempts":  r["attempts"],
            "avg_score": round(r["avg_pct"], 1),
            "best":      round(r["best"], 1),
            "worst":     round(r["worst"], 1),
            "last_ts":   r["last_ts"],
        }
        for r in rows
    }


def get_weak_topics(threshold: float = 60.0) -> List[str]:
    """Subjects where average quiz score is below threshold %."""
    stats = get_subject_stats()
    return [s for s, v in stats.items() if v["avg_score"] < threshold]


def get_streak() -> int:
    """Count consecutive calendar days (ending today) with any activity."""
    with _conn() as con:
        quiz_dates = {
            r[0][:10]
            for r in con.execute("SELECT ts FROM quiz_results").fetchall()
        }
        q_dates = {
            r[0][:10]
            for r in con.execute("SELECT ts FROM questions").fetchall()
        }
    active_days = quiz_dates | q_dates
    streak, day = 0, date.today()
    while str(day) in active_days:
        streak += 1
        day -= timedelta(days=1)
    return streak


def get_summary_stats() -> Dict[str, Any]:
    """High-level numbers for the dashboard overview cards."""
    with _conn() as con:
        total_quizzes   = con.execute("SELECT COUNT(*) FROM quiz_results").fetchone()[0]
        total_questions = con.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        avg_score_row   = con.execute("SELECT AVG(pct) FROM quiz_results").fetchone()[0]
        subjects_row    = con.execute(
            "SELECT COUNT(DISTINCT subject) FROM quiz_results"
        ).fetchone()[0]
    return {
        "total_quizzes":   total_quizzes,
        "total_questions": total_questions,
        "avg_score":       round(avg_score_row or 0, 1),
        "subjects_studied": subjects_row,
        "streak":          get_streak(),
        "weak_topics":     get_weak_topics(),
    }


def get_score_trend(subject: str | None = None, limit: int = 20) -> List[Dict]:
    """Return recent quiz results for a trend line chart."""
    with _conn() as con:
        if subject:
            rows = con.execute(
                "SELECT ts, pct, subject FROM quiz_results WHERE subject=? "
                "ORDER BY ts DESC LIMIT ?", (subject, limit)
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT ts, pct, subject FROM quiz_results "
                "ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in reversed(rows)]  # oldest → newest for chart


def clear_all_progress() -> None:
    with _conn() as con:
        con.executescript("DELETE FROM quiz_results; DELETE FROM questions;")