"""SQLite database layer (plain sqlite3 – no ORM needed, easy to explain in viva)."""
import os
import sqlite3

from flask import g

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "instance", "healthcare.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name     TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    age           INTEGER,
    gender        TEXT,
    phone         TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS login_logs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   INTEGER REFERENCES users(id) ON DELETE CASCADE,
    email     TEXT,
    success   INTEGER NOT NULL,
    ip        TEXT,
    logged_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS reports (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    original_name  TEXT NOT NULL,
    stored_name    TEXT NOT NULL,
    scan_method    TEXT,
    ocr_text       TEXT,
    extracted_json TEXT,
    uploaded_at    TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    report_id     INTEGER REFERENCES reports(id) ON DELETE SET NULL,
    disease       TEXT NOT NULL,
    inputs_json   TEXT NOT NULL,
    probability   REAL NOT NULL,
    risk_level    TEXT NOT NULL,
    result_json   TEXT NOT NULL,
    model_name    TEXT,
    source        TEXT NOT NULL DEFAULT 'manual',
    created_at    TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_pred_user ON predictions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rep_user ON reports(user_id);
"""


def get_db():
    if "db" not in g:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    con.commit()
    con.close()
