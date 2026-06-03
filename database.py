# -*- coding: utf-8 -*-
import sqlite3, os
from werkzeug.security import generate_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'kdrama.db')

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn(); c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS dramas (
        drama_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        title        TEXT NOT NULL UNIQUE,
        genre        TEXT, rating_avg REAL, synopsis TEXT,
        director     TEXT, screenwriter TEXT, cast TEXT,
        tags TEXT, year INTEGER, episodes INTEGER,
        poster_url   TEXT, is_active INTEGER DEFAULT 1,
        source_id    TEXT DEFAULT ''
    )""")
    # Tambah kolom source_id jika belum ada (upgrade existing DB)
    try:
        c.execute("ALTER TABLE dramas ADD COLUMN source_id TEXT DEFAULT ''")
    except Exception:
        pass

    c.execute("""CREATE TABLE IF NOT EXISTS auth_users (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        username     TEXT NOT NULL UNIQUE,
        email        TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        display_name TEXT,
        role         TEXT DEFAULT 'user',
        fav_genres   TEXT DEFAULT '',
        bio          TEXT DEFAULT '',
        avatar_color TEXT DEFAULT '#e8506a',
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login   TIMESTAMP,
        is_active    INTEGER DEFAULT 1
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        total_ratings INTEGER DEFAULT 0,
        auth_user_id INTEGER REFERENCES auth_users(id)
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS ratings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL, drama_id INTEGER NOT NULL,
        rating REAL NOT NULL, review TEXT DEFAULT '',
        split TEXT DEFAULT 'train',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id),
        FOREIGN KEY (drama_id) REFERENCES dramas(drama_id),
        UNIQUE(user_id, drama_id)
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS watchlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        auth_user_id INTEGER NOT NULL,
        drama_id INTEGER NOT NULL,
        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (auth_user_id) REFERENCES auth_users(id),
        FOREIGN KEY (drama_id) REFERENCES dramas(drama_id),
        UNIQUE(auth_user_id, drama_id)
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS activity_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        auth_user_id INTEGER, username TEXT,
        action TEXT NOT NULL, detail TEXT, ip_addr TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS cbf_similarity (
        drama_id_a INTEGER NOT NULL, drama_id_b INTEGER NOT NULL,
        similarity REAL NOT NULL, PRIMARY KEY (drama_id_a, drama_id_b)
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS cf_similarity (
        drama_id_a INTEGER NOT NULL, drama_id_b INTEGER NOT NULL,
        similarity REAL NOT NULL, PRIMARY KEY (drama_id_a, drama_id_b)
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS cbf_model_meta (
        key TEXT PRIMARY KEY, value TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS evaluation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        method TEXT NOT NULL, k INTEGER NOT NULL,
        precision_k REAL, recall_k REAL, f1_k REAL,
        rmse REAL, mae REAL, coverage REAL, ndcg REAL,
        train_size INTEGER, test_size INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS training_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        method TEXT NOT NULL, status TEXT NOT NULL,
        message TEXT, duration_s REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    conn.commit(); conn.close()
    _seed_admin()

def _seed_admin():
    conn = get_conn()
    exists = conn.execute("SELECT id FROM auth_users WHERE role='admin'").fetchone()
    if not exists:
        conn.execute("""INSERT OR IGNORE INTO auth_users
            (username, email, password_hash, display_name, role)
            VALUES (?,?,?,?,?)""",
            ('admin', 'admin@kdrama.local',
             generate_password_hash('admin123'),
             'Administrator', 'admin'))
        conn.commit()
    conn.close()

def query(sql, params=()):
    conn = get_conn()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def query_one(sql, params=()):
    conn = get_conn()
    row = conn.execute(sql, params).fetchone()
    conn.close()
    return dict(row) if row else None

def execute(sql, params=()):
    conn = get_conn()
    cur = conn.execute(sql, params)
    conn.commit(); lid = cur.lastrowid; conn.close()
    return lid

def executemany(sql, data):
    conn = get_conn()
    conn.executemany(sql, data)
    conn.commit(); conn.close()

def log_activity(auth_user_id, username, action, detail='', ip=''):
    execute("INSERT INTO activity_log (auth_user_id,username,action,detail,ip_addr) VALUES (?,?,?,?,?)",
            (auth_user_id, username, action, detail, ip))
