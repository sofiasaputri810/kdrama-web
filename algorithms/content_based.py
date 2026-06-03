# -*- coding: utf-8 -*-
"""
algorithms/content_based.py
Content-Based Filtering dengan Item Rating Augmentation

Sesuai revisi dosen:
- CBF MURNI — tidak ada unsur collaborative (tidak menyentuh tabel ratings)
- Feature Augmentation menggunakan rating_y (rating rata-rata DRAMA dari dataset)
  bukan rating_x dari pengguna → rating_y adalah atribut ITEM, bukan sinyal user
- Batasan "tidak hybrid" tetap terjaga

Arsitektur:
    1. Content Vector = TF-IDF(genre, cast, synopsis, tags, director)
    2. Item Rating    = rating_y (kolom metadata drama, ternormalisasi)
    3. Augmented      = hstack([content × (1-alpha), item_rating × alpha])
    4. Similarity     = Cosine Similarity antar Augmented Vector
    5. Prediksi       = Weighted Sum (pakai rating user hanya saat prediksi,
                        BUKAN saat training similarity — tetap pure CBF)

Perbedaan dengan FA-CBF (hybrid):
    FA-CBF     : rating diambil dari AVG(tabel ratings) → sinyal kolaboratif
    CBF ini    : rating diambil dari kolom rating_y tiap drama → atribut item

Referensi:
    Pazzani & Billsus (2007) - Content-Based Recommendation Systems
    Lops et al. (2011) - Content-based Recommender Systems: State of the Art
"""
import numpy as np, time, re, sys, os
from scipy.sparse import hstack, csr_matrix
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import database as db

# ================================================================
# KONSTANTA
# ================================================================
ALPHA_ITEM_RATING = 0.3  # Bobot item rating (0.3 = 30% rating_y, 70% konten)
MAX_SIMILARITY    = 50   # Top-N similarity tersimpan per drama
TOP_K_NEIGHBORS   = 20   # K tetangga untuk prediksi rating
MIN_RATINGS_FOR_RECS = 5 # Minimum rating user untuk rekomendasi personal


# ================================================================
# FEATURE ENGINEERING
# ================================================================
def _clean(s):
    return re.sub(r'["\']', ' ', re.sub(r'\s+', ' ', str(s or '').strip()))


def _build_content_text(row, features='all'):
    """
    Teks gabungan dari atribut konten drama.
    Bobot via pengulangan: genre×3, cast×2, tags×2, synopsis×1, director×1
    """
    genre    = ' '.join(re.split(r'[,/|]+', _clean(row.get('genre', '')))).strip()
    synopsis = _clean(row.get('synopsis', ''))
    cast     = _clean(row.get('cast', ''))
    tags     = _clean(row.get('tags', ''))
    director = _clean(row.get('director', ''))

    parts = []
    if features in ('all', 'genre_cast', 'genre_plot', 'genre_only'):
        parts += [genre] * 3
    if features in ('all', 'genre_cast'):
        parts += [cast] * 2
    if features in ('all', 'genre_plot'):
        parts += [synopsis]
    if features == 'all':
        parts += [tags] * 2
        parts += [director]

    return ' '.join(parts).lower()


def _build_item_rating_vector(dramas):

    mat = np.zeros((len(dramas), 1))

    for i, d in enumerate(dramas):
        mat[i, 0] = float(d.get('rating_avg') or 0)

    scaler = MinMaxScaler()
    mat_norm = scaler.fit_transform(mat)

    return csr_matrix(mat_norm)



# ================================================================
# TRAINING — CBF dengan Item Rating Augmentation
# ================================================================
def train_cbf(features='all', min_df=1, max_features=5000,
              alpha=ALPHA_ITEM_RATING):
    """
    Training CBF murni dengan augmentasi fitur rating item.

    Langkah:
    1. TF-IDF dari atribut teks drama (genre, cast, synopsis, tags, director)
    2. Vektor item rating dari metadata drama 
       → rating_avg (average_rating dataset)
    3. Gabung: [content × (1-alpha)] + [item_rating × alpha]
    4. Cosine Similarity → simpan Top-50 ke database

    Args:
        features : 'all' | 'genre_cast' | 'genre_plot' | 'genre_only'
        alpha    : Bobot item rating metadata (default 0.3)
                   0   = CBF murni konten saja
                   0.3 = 70% konten + 30% item rating (recommended)
                   1   = item rating saja
    """
    t0 = time.time()
    print(f"[CBF] Training CBF + Item Rating Augmentation...")
    print(f"[CBF] features='{features}', alpha={alpha} "
          f"({int((1-alpha)*100)}% konten teks + {int(alpha*100)}% item rating metadata)")
    print(f"[CBF] Sumber rating: kolom rating_avg/rating_y (atribut item) "
          f"— BUKAN dari tabel ratings pengguna")

    dramas = db.query("""
        SELECT drama_id, title, genre, synopsis, "cast", tags,
               director, rating_avg, year, episodes
        FROM dramas WHERE is_active = 1 ORDER BY drama_id
    """)
    if not dramas:
        raise ValueError("Database drama kosong. Import data terlebih dahulu.")

    n   = len(dramas)
    ids = [d['drama_id'] for d in dramas]
    print(f"[CBF] {n} drama dimuat dari database")

    # ── STEP 1: TF-IDF Content Vector ────────────────────────────
    print("[CBF] Step 1/3: TF-IDF vectorization dari konten teks...")
    corpus = [_build_content_text(d, features) for d in dramas]
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents='unicode',
        token_pattern=r'[a-zA-Z]{2,}'
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    vocab_size   = len(vectorizer.vocabulary_)
    print(f"[CBF]   Matriks TF-IDF: {tfidf_matrix.shape}, vocab={vocab_size}")

    # ── STEP 2: Item Rating Metadata Vector ───────────────────────
    # Menggunakan rating_avg (= rating_y dari CSV) — ATRIBUT ITEM
    print("[CBF] Step 2/3: Membangun vektor item rating dari metadata drama...")
    item_rating_matrix = _build_item_rating_vector(dramas)

    has_rating = any(d.get('rating_avg') for d in dramas)
    if has_rating:
        print(f"[CBF]   Item rating vector shape: {item_rating_matrix.shape}")
        print(f"[CBF]   Fitur: [rating_avg] — dari metadata drama")
    else:
        print("[CBF]   PERINGATAN: rating_avg kosong, hanya menggunakan konten teks")

    # ── STEP 3: Augmented Vector ──────────────────────────────────
    print(f"[CBF] Step 3/3: Menggabungkan vektor (alpha={alpha})...")

    if has_rating and alpha > 0:
        content_weighted      = tfidf_matrix      * (1.0 - alpha)
        item_rating_weighted  = item_rating_matrix * alpha
        augmented = hstack([content_weighted, item_rating_weighted])
        method_used = (f'CBF + Item Rating Augmentation '
                       f'({int((1-alpha)*100)}% konten + {int(alpha*100)}% item rating)')
    else:
        augmented   = tfidf_matrix
        method_used = 'CBF murni (konten teks saja)'
        print("[CBF]   Fallback ke CBF murni")

    print(f"[CBF]   Augmented vector shape: {augmented.shape}")

    # ── STEP 4: Cosine Similarity ─────────────────────────────────
    sim_matrix = cosine_similarity(augmented)   # (n × n)

    # ── STEP 5: Simpan Top-50 per drama ──────────────────────────
    records = []
    for i, drama_i in enumerate(ids):
        sorted_idx = np.argsort(sim_matrix[i])[::-1]
        count = 0
        for j in sorted_idx:
            if j == i:
                continue
            if sim_matrix[i][j] <= 0:
                break
            if count >= MAX_SIMILARITY:
                break
            records.append((int(drama_i), int(ids[j]), float(sim_matrix[i][j])))
            count += 1

    db.execute("DELETE FROM cbf_similarity")
    db.executemany("INSERT OR REPLACE INTO cbf_similarity VALUES (?,?,?)", records)

    # Simpan metadata model
    db.execute("DELETE FROM cbf_model_meta")
    meta = {
        'method':       'CBF + Item Rating Augmentation',
        'features':     features,
        'alpha':        str(alpha),
        'n_dramas':     str(n),
        'vocab':        str(vocab_size),
        'has_rating':   str(has_rating),
        'rating_source':'rating_avg (item metadata — bukan dari tabel ratings)',
        'method_used':  method_used,
        'trained_at':   time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    for k, v in meta.items():
        db.execute("INSERT INTO cbf_model_meta VALUES (?,?)", (k, v))

    dur = round(time.time() - t0, 2)
    print(f"[CBF] OK Selesai {dur}s | {len(records)} pasang similarity tersimpan")
    print(f"[CBF] Metode: {method_used}")

    db.execute(
        "INSERT INTO training_log (method,status,message,duration_s) VALUES (?,?,?,?)",
        ('CBF', 'done',
         f'{method_used}: features={features}, alpha={alpha}, '
         f'vocab={vocab_size}, pairs={len(records)}', dur)
    )

    return {
        'status':      'done',
        'method':      'CBF + Item Rating Augmentation',
        'n_dramas':    n,
        'vocab':       vocab_size,
        'alpha':       alpha,
        'has_rating':  has_rating,
        'n_pairs':     len(records),
        'duration':    dur,
        'features':    features,
    }


# ================================================================
# PREDIKSI RATING — Weighted Sum
# ================================================================
def predict_rating_cbf(user_id, drama_id, top_k=TOP_K_NEIGHBORS):
    """
    Prediksi rating user terhadap drama menggunakan CBF similarity.

    Formula Weighted Sum (sama dengan CF — apple-to-apple):
        Prediksi(u, i) = Σ[sim_cbf(i,j) × r(u,j)] / Σ|sim_cbf(i,j)|

    Catatan:
    - sim_cbf dibangun dari konten+item_rating (TANPA data user) → pure CBF
    - r(u,j) adalah rating user terhadap drama j yang sudah ditonton
    - Penggunaan r(u,j) di sini hanya untuk PREDIKSI, bukan training
    - Ini adalah cara standar CBF menghasilkan prediksi numerik
    """
    user_ratings = db.query(
        "SELECT drama_id, rating FROM ratings WHERE user_id=? AND split='train'",
        (user_id,)
    )
    if not user_ratings:
        return None

    rated     = {r['drama_id']: r['rating'] for r in user_ratings}
    rated_ids = list(rated.keys())
    if not rated_ids:
        return None

    placeholders = ','.join(['?'] * len(rated_ids))
    neighbors = db.query(
        f"""SELECT drama_id_b, similarity
            FROM cbf_similarity
            WHERE drama_id_a = ?
              AND drama_id_b IN ({placeholders})
            ORDER BY similarity DESC
            LIMIT ?""",
        [drama_id] + rated_ids + [top_k]
    )
    if not neighbors:
        return None

    num = sum(n['similarity'] * rated[n['drama_id_b']] for n in neighbors)
    den = sum(abs(n['similarity']) for n in neighbors)

    return round(num / den, 4) if den > 0 else None


# ================================================================
# GET RECOMMENDATIONS
# ================================================================
def get_cbf_recommendations(drama_id, top_n=10, genre_filter=None):
    """
    Rekomendasi CBF berdasarkan satu drama referensi.
    Interface tidak berubah — website tidak perlu dimodifikasi.
    """
    rows = db.query("""
        SELECT cs.drama_id_b AS drama_id,
               cs.similarity,
               d.title, d.genre, d.synopsis,
               d.rating_avg, d."cast", d.tags
        FROM cbf_similarity cs
        JOIN dramas d ON d.drama_id = cs.drama_id_b
        WHERE cs.drama_id_a = ?
        ORDER BY cs.similarity DESC
        LIMIT ?
    """, (drama_id, top_n * 3 if genre_filter else top_n))

    if genre_filter:
        rows = [r for r in rows
                if genre_filter.lower() in (r.get('genre') or '').lower()]

    return rows[:top_n]


def get_cbf_user_recommendations(user_id, top_n=10):
    """
    Rekomendasi personal CBF untuk user tertentu.
    Minimal MIN_RATINGS_FOR_RECS rating diperlukan.
    """
    user_ratings = db.query(
        "SELECT drama_id, rating FROM ratings WHERE user_id=? AND split='train'",
        (user_id,)
    )
    n_rated = len(user_ratings)

    if n_rated < MIN_RATINGS_FOR_RECS:
        return {
            'error':         'insufficient_ratings',
            'message':       (f'Anda baru merating {n_rated} drama. '
                              f'Minimal {MIN_RATINGS_FOR_RECS} diperlukan.'),
            'min_required':  MIN_RATINGS_FOR_RECS,
            'current_count': n_rated,
            'needed':        MIN_RATINGS_FOR_RECS - n_rated,
        }

    rated    = {r['drama_id']: r['rating'] for r in user_ratings}
    watched  = set(rated.keys())
    scores   = {}   # drama_id → [sum_weighted, sum_abs_sim]

    for rated_id, user_rating in rated.items():
        neighbors = db.query("""
            SELECT drama_id_b, similarity FROM cbf_similarity
            WHERE drama_id_a = ? ORDER BY similarity DESC LIMIT ?
        """, (rated_id, TOP_K_NEIGHBORS * 3))

        for nb in neighbors:
            cid = nb['drama_id_b']
            sim = nb['similarity']
            if cid in watched:
                continue
            if cid not in scores:
                scores[cid] = [0.0, 0.0]
            scores[cid][0] += sim * user_rating
            scores[cid][1] += abs(sim)

    preds = [
        (cid, round(num / den, 4))
        for cid, (num, den) in scores.items() if den > 0
    ]
    preds.sort(key=lambda x: x[1], reverse=True)

    results = []
    for cid, pred_score in preds[:top_n]:
        info = db.query_one(
            'SELECT drama_id, title, genre, synopsis, rating_avg, "cast", year '
            'FROM dramas WHERE drama_id=? AND is_active=1',
            (cid,)
        )
        if info:
            row = dict(info)
            row['predicted_rating'] = pred_score
            row['similarity']       = round(min(pred_score / 10.0, 1.0), 4)
            results.append(row)

    return results


def get_cbf_model_info():
    """Info metadata model yang sedang aktif."""
    rows = db.query("SELECT key, value FROM cbf_model_meta")
    meta = {r['key']: r['value'] for r in rows}
    n    = db.query_one("SELECT COUNT(*) as n FROM cbf_similarity")
    meta['n_similarity_pairs'] = n['n'] if n else 0
    return meta
