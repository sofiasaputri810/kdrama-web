# -*- coding: utf-8 -*-
"""
data/import_data.py
Import dataset gabungan (single CSV) ke SQLite.

Format kolom dataset:
    user_id       : ID pengguna (mis. user_00001)
    drama_id      : ID drama    (mis. kdrama_0001)
    rating_x      : Rating pengguna (1-10) → masuk tabel ratings
    title         : Judul drama
    genre         : Genre drama
    rating_y      : Rata-rata rating drama → masuk kolom rating_avg di tabel dramas
    synopsis      : Sinopsis
    director      : Sutradara
    screenwriter  : Penulis skenario
    cast          : Pemeran
    tags          : Tag konten

Alur:
    1. Baca satu CSV gabungan
    2. Ekstrak data drama unik   → tabel dramas
    3. Ekstrak data rating       → tabel ratings
    4. Split 80:20 per user      → kolom split = 'train' / 'test'
"""
import sys, os, re
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database import get_conn, init_db

DATA_DIR    = os.path.dirname(__file__)

# Nama file dataset gabungan — ubah sesuai nama file Anda
DATASET_CSV = os.path.join(DATA_DIR, 'dataset_kdrama.csv')

# Fallback ke nama alternatif yang umum
FALLBACK_NAMES = [
    'dataset_kdrama.csv',
    'kdrama_dataset.csv',
    'dataset.csv',
    'data_kdrama.csv',
    'dataset_gabungan.csv',
]


def _find_dataset():
    """Cari file dataset di folder data/."""
    # Cek nama file utama
    if os.path.exists(DATASET_CSV):
        return DATASET_CSV

    # Cek nama-nama alternatif
    for name in FALLBACK_NAMES:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return path

    # Cari file CSV pertama yang ada di folder data/
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if len(csv_files) == 1:
        return os.path.join(DATA_DIR, csv_files[0])
    elif len(csv_files) > 1:
        print(f"[IMPORT] Ditemukan beberapa CSV: {csv_files}")
        print(f"[IMPORT] Menggunakan: {csv_files[0]}")
        return os.path.join(DATA_DIR, csv_files[0])

    return None


def _clean(s):
    if pd.isna(s):
        return ''
    return re.sub(r'\s+', ' ', str(s).strip().strip('"').strip("'"))


def import_merged(test_ratio=0.2, random_state=42):
    """
    Import dataset gabungan dengan kolom:
        user_id, drama_id, rating_x, title, genre, rating_y,
        synopsis, director, screenwriter, cast, tags
    """
    path = _find_dataset()
    if not path:
        print("[IMPORT] ERROR: File dataset tidak ditemukan di folder data/")
        print("[IMPORT] Letakkan file CSV di folder data/ lalu jalankan ulang.")
        print("[IMPORT] Nama file yang dicari:", ', '.join(FALLBACK_NAMES))
        return False

    print(f"[IMPORT] Membaca dataset: {os.path.basename(path)}")
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()

    print(f"[IMPORT] {len(df)} baris, {len(df.columns)} kolom")
    print(f"[IMPORT] Kolom: {list(df.columns)}")

    # ── Mapping kolom otomatis ─────────────────────────────────
    col = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == 'user_id'      or 'user' in cl:              col['user_id']      = c
        if cl == 'drama_id'     or cl in ('dramaid','id'):     col['drama_id']     = c
        if cl == 'rating_x'     or cl == 'rating_user':       col['rating_x']     = c
        if cl == 'title'        or 'judul' in cl:             col['title']        = c
        if cl == 'genre':                                       col['genre']        = c
        if cl == 'rating_y'     or cl == 'rating_avg':        col['rating_y']     = c
        if cl == 'synopsis'     or 'deskripsi' in cl:         col['synopsis']     = c
        if cl == 'director'     or 'sutradara' in cl:         col['director']     = c
        if cl == 'screenwriter' or 'penulis' in cl:           col['screenwriter'] = c
        if cl == 'cast'         or 'pemain' in cl:            col['cast']         = c
        if cl == 'tags'         or 'tag' in cl:               col['tags']         = c

    # Jika rating_x belum ketemu, coba kolom 'rating' biasa
    if 'rating_x' not in col:
        for c in df.columns:
            if 'rating' in c.lower():
                col['rating_x'] = c
                break

    print(f"[IMPORT] Mapping kolom: {col}")

    # Validasi kolom wajib
    required = ['user_id', 'drama_id', 'rating_x', 'title']
    missing  = [r for r in required if r not in col]
    if missing:
        print(f"[IMPORT] ERROR: Kolom wajib tidak ditemukan: {missing}")
        print(f"[IMPORT] Kolom tersedia: {list(df.columns)}")
        return False

    # ── Rename kolom sesuai mapping ───────────────────────────
    rename = {v: k for k, v in col.items()}
    df = df.rename(columns=rename)

    # ── Tambah kolom yang tidak ada ───────────────────────────
    for c in ['genre','rating_y','synopsis','director','screenwriter','cast','tags']:
        if c not in df.columns:
            df[c] = ''

    # ── Bersihkan data ─────────────────────────────────────────
    df['user_id']  = df['user_id'].astype(str).str.strip()
    df['drama_id'] = df['drama_id'].astype(str).str.strip()
    df['rating_x'] = pd.to_numeric(df['rating_x'], errors='coerce')
    df['rating_y'] = pd.to_numeric(df['rating_y'], errors='coerce')

    for c in ['title','genre','synopsis','director','screenwriter','cast','tags']:
        df[c] = df[c].apply(_clean)

    # Hapus baris dengan rating atau user/drama kosong
    before = len(df)
    df.dropna(subset=['user_id','drama_id','rating_x'], inplace=True)
    df = df[df['rating_x'].between(1, 10)]
    df = df[df['user_id'] != ''] 
    df = df[df['drama_id'] != '']
    print(f"[IMPORT] {len(df)}/{before} baris valid setelah pembersihan")

    # ================================================================
    # STEP 1: Import Drama Unik
    # ================================================================
    print("[IMPORT] Step 1/3: Mengekstrak data drama unik...")

    # Ambil satu baris per drama (berdasarkan drama_id)
    drama_df = (
        df.drop_duplicates(subset=['drama_id'])
        [['drama_id','title','genre','rating_y','synopsis',
          'director','screenwriter','cast','tags']]
        .copy()
    )
    drama_df = drama_df[drama_df['title'] != '']
    drama_df.drop_duplicates(subset=['title'], inplace=True)

    conn = get_conn()
    drama_records = [
        (row['title'], row['genre'], row['rating_y'] if pd.notna(row['rating_y']) else None,
         row['synopsis'], row['director'], row['screenwriter'],
         row['cast'], row['tags'], row['drama_id'])
        for _, row in drama_df.iterrows()
    ]
    conn.executemany("""
        INSERT OR IGNORE INTO dramas
            (title, genre, rating_avg, synopsis, director, screenwriter,
             "cast", tags, source_id)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, drama_records)
    conn.commit()

    n_dramas = conn.execute("SELECT COUNT(*) FROM dramas").fetchone()[0]
    print(f"[IMPORT] OK {n_dramas} drama di database")

    # Buat mapping source_id (kdrama_0001) → drama_id (integer PK)
    drama_rows   = conn.execute("SELECT drama_id, title, source_id FROM dramas").fetchall()
    source_to_id = {str(r[2]).strip(): r[0] for r in drama_rows if r[2]}
    title_to_id  = {r[1].lower().strip(): r[0] for r in drama_rows}

    def resolve_drama_id(ref):
        ref_s = str(ref).strip()
        if ref_s in source_to_id:
            return source_to_id[ref_s]
        if ref_s.lower() in title_to_id:
            return title_to_id[ref_s.lower()]
        for t, did in title_to_id.items():
            if ref_s.lower() in t:
                return did
        return None

    # ================================================================
    # STEP 2: Resolusi drama_id & Import User
    # ================================================================
    print("[IMPORT] Step 2/3: Memetakan drama_id dan user...")

    df['drama_pk'] = df['drama_id'].apply(resolve_drama_id)
    matched = df['drama_pk'].notna().sum()
    print(f"[IMPORT] {matched}/{len(df)} rating berhasil dipetakan ke drama")

    df.dropna(subset=['drama_pk'], inplace=True)
    df['drama_pk'] = df['drama_pk'].astype(int)

    # Insert users unik
    users = df['user_id'].unique().tolist()
    conn.executemany(
        "INSERT OR IGNORE INTO users (user_id) VALUES (?)",
        [(u,) for u in users]
    )
    conn.commit()
    print(f"[IMPORT] {len(users)} user ditemukan")

    # ================================================================
    # STEP 3: Split 80:20 per User → Train / Test
    # ================================================================
    print("[IMPORT] Step 3/3: Membagi data train/test (80:20 per user)...")

    np.random.seed(random_state)
    df = df.sort_values(['user_id', 'drama_pk'])
    df['split'] = 'train'

    for uid, grp in df.groupby('user_id'):
        n = len(grp)
        if n >= 5:   # minimal 5 rating agar bisa di-split
            n_test = max(1, int(n * test_ratio))
            test_idx = grp.index[-n_test:]
            df.loc[test_idx, 'split'] = 'test'
        # jika < 5 rating, semua masuk train

    # Insert ratings
    conn.execute("DELETE FROM ratings")
    rating_records = [
        (row['user_id'], row['drama_pk'],
         float(row['rating_x']), row['split'])
        for _, row in df.iterrows()
    ]
    conn.executemany("""
        INSERT OR REPLACE INTO ratings (user_id, drama_id, rating, split)
        VALUES (?,?,?,?)
    """, rating_records)

    # Update total_ratings per user
    conn.execute("""
        UPDATE users SET total_ratings = (
            SELECT COUNT(*) FROM ratings r WHERE r.user_id = users.user_id
        )
    """)
    conn.commit()

    n_train = conn.execute("SELECT COUNT(*) FROM ratings WHERE split='train'").fetchone()[0]
    n_test  = conn.execute("SELECT COUNT(*) FROM ratings WHERE split='test'").fetchone()[0]
    n_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()

    print(f"[IMPORT] Split: {n_train} train | {n_test} test | {n_users} users")
    return {
        'n_dramas': n_dramas,
        'n_train':  n_train,
        'n_test':   n_test,
        'n_users':  n_users,
    }


def run_import():
    print("=" * 55)
    print("  KDrama — Import Dataset Gabungan")
    print("=" * 55)
    init_db()

    result = import_merged()
    if not result:
        return False

    print("=" * 55)
    print(f"  Selesai!")
    print(f"  Drama  : {result['n_dramas']}")
    print(f"  Users  : {result['n_users']}")
    print(f"  Train  : {result['n_train']} rating (80%)")
    print(f"  Test   : {result['n_test']}  rating (20%)")
    print("=" * 55)
    return True


if __name__ == '__main__':
    run_import()
