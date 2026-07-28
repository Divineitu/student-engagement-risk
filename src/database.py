"""Small sqlite wrapper. Nothing fancy, just enough to log uploads, retrain
runs and request latency so the UI has something real to show on the
Status and Upload & Retrain tabs."""
import os
import sqlite3
import time

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "app.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            label TEXT NOT NULL,
            uploaded_at REAL NOT NULL,
            used_in_retrain INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS retrain_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ran_at REAL NOT NULL,
            n_new_images INTEGER NOT NULL,
            old_macro_f1 REAL,
            new_macro_f1 REAL,
            model_version TEXT
        );

        CREATE TABLE IF NOT EXISTS request_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            endpoint TEXT NOT NULL,
            latency_ms REAL NOT NULL
        );
    """)
    conn.commit()
    conn.close()


def log_upload(filename, label):
    conn = get_connection()
    conn.execute(
        "INSERT INTO uploads (filename, label, uploaded_at) VALUES (?, ?, ?)",
        (filename, label, time.time()),
    )
    conn.commit()
    conn.close()


def get_unused_uploads():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM uploads WHERE used_in_retrain = 0").fetchall()
    conn.close()
    return rows


def mark_uploads_used(ids):
    if not ids:
        return
    conn = get_connection()
    q_marks = ",".join("?" * len(ids))
    conn.execute(f"UPDATE uploads SET used_in_retrain = 1 WHERE id IN ({q_marks})", ids)
    conn.commit()
    conn.close()


def log_retrain_run(n_new_images, old_macro_f1, new_macro_f1, model_version):
    conn = get_connection()
    conn.execute(
        "INSERT INTO retrain_runs (ran_at, n_new_images, old_macro_f1, new_macro_f1, model_version) "
        "VALUES (?, ?, ?, ?, ?)",
        (time.time(), n_new_images, old_macro_f1, new_macro_f1, model_version),
    )
    conn.commit()
    conn.close()


def get_last_retrain():
    conn = get_connection()
    row = conn.execute("SELECT * FROM retrain_runs ORDER BY ran_at DESC LIMIT 1").fetchone()
    conn.close()
    return row


def log_request(endpoint, latency_ms):
    conn = get_connection()
    conn.execute(
        "INSERT INTO request_log (ts, endpoint, latency_ms) VALUES (?, ?, ?)",
        (time.time(), endpoint, latency_ms),
    )
    conn.commit()
    conn.close()


def get_recent_requests(limit=200):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM request_log ORDER BY ts DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return rows


def get_upload_counts():
    conn = get_connection()
    rows = conn.execute("SELECT label, COUNT(*) as n FROM uploads GROUP BY label").fetchall()
    conn.close()
    return {row["label"]: row["n"] for row in rows}
