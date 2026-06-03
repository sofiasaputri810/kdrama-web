# -*- coding: utf-8 -*-
"""
algorithms/collaborative.py
Item-Based Collaborative Filtering (IBCF)

Sesuai proposal:
- Minimum rating untuk mendapat rekomendasi: 5 drama
- Similarity: Cosine Similarity dengan mean-centering
- Prediksi: Weighted Sum (Sarwar et al., 2001)
  Prediksi(u,i) = sum(sim(i,j) * r(u,j)) / sum(|sim(i,j)|)
"""
import numpy as np, pandas as pd, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import database as db

# ================================================================
# KONSTANTA
# ================================================================
MIN_RATINGS_FOR_RECS = 5    # Minimum rating sebelum dapat rekomendasi CF
MIN_COMMON_USERS     = 2    # Minimum user bersama untuk similarity
TOP_K_NEIGHBORS      = 20   # K tetangga untuk prediksi
MAX_SIMILARITY_STORE = 50   # Top-N similarity tersimpan per drama


# ================================================================
# TRAINING
# ================================================================
def build_rating_matrix(split='train'):
    rows = db.query(
        "SELECT r.user_id, r.drama_id, r.rating FROM ratings r WHERE r.split=?",
        (split,)
    )
    if not rows: return None, [], []
    df = pd.DataFrame(rows)
    matrix = df.pivot_table(index='user_id', columns='drama_id', values='rating')
    return matrix, list(matrix.index), list(matrix.columns)


def train_cf(method='cosine'):
    """
    Training IBCF:
    1. Bangun matriks User-Item
    2. Mean-centering (hilangkan bias rating per user)
    3. Cosine Similarity antar item (kolom matriks)
    4. Simpan top-50 pasang similarity ke DB
    """
    t0 = time.time()
    print(f"[CF] Training Item-Based CF, method='{method}'...")
    print(f"[CF] Min rating untuk rekomendasi: {MIN_RATINGS_FOR_RECS}")

    matrix, users, dramas = build_rating_matrix('train')
    if matrix is None: raise ValueError("Data rating kosong.")
    n_users, n_items = matrix.shape
    print(f"[CF] Matriks: {n_users} users x {n_items} items")

    # Mean-centering
    if method in ('cosine', 'adjusted_cosine'):
        user_means = matrix.mean(axis=1)
        mat_adj = matrix.sub(user_means, axis=0)
    else:
        mat_adj = matrix.copy()

    mat_filled = mat_adj.fillna(0).values

    # Hitung similarity
    if method == 'pearson':
        n = mat_filled.shape[1]
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                mask = (mat_filled[:, i] != 0) & (mat_filled[:, j] != 0)
                if mask.sum() < MIN_COMMON_USERS: continue
                r, _ = pearsonr(mat_filled[mask, i], mat_filled[mask, j])
                sim[i, j] = sim[j, i] = 0.0 if np.isnan(r) else r
    else:
        sim = cosine_similarity(mat_filled.T)

    # Simpan top-50 per drama
    records = []
    for i, drama_i in enumerate(dramas):
        count = 0
        for j in np.argsort(sim[i])[::-1]:
            if j == i: continue
            if sim[i][j] <= 0: break
            if count >= MAX_SIMILARITY_STORE: break
            records.append((int(drama_i), int(dramas[j]), float(sim[i][j])))
            count += 1

    db.execute("DELETE FROM cf_similarity")
    db.executemany("INSERT OR REPLACE INTO cf_similarity VALUES (?,?,?)", records)
    dur = round(time.time() - t0, 2)
    print(f"[CF] OK Selesai {dur}s | {len(records)} pasang similarity")
    db.execute(
        "INSERT INTO training_log (method,status,message,duration_s) VALUES (?,?,?,?)",
        ('CF', 'done', f'method={method},items={n_items},pairs={len(records)}', dur)
    )
    return {'status':'done','method':method,'n_users':n_users,
            'n_items':n_items,'n_pairs':len(records),'duration':dur}


# ================================================================
# PREDIKSI RATING — Weighted Sum (Sarwar et al., 2001)
# ================================================================
def predict_rating_cf(user_id, drama_id, top_k=TOP_K_NEIGHBORS):
    """
    Prediksi(u,i) = sum(sim(i,j) * r(u,j)) / sum(|sim(i,j)|)
    j = drama yang sudah dirating user u
    """
    user_ratings = db.query(
        "SELECT drama_id, rating FROM ratings WHERE user_id=? AND split='train'",
        (user_id,)
    )
    if not user_ratings: return None
    rated = {r['drama_id']: r['rating'] for r in user_ratings}
    rated_ids = list(rated.keys())
    placeholders = ','.join('?' * len(rated_ids))
    neighbors = db.query(
        f"""SELECT drama_id_b, similarity FROM cf_similarity
            WHERE drama_id_a=? AND drama_id_b IN ({placeholders})
            ORDER BY similarity DESC LIMIT ?""",
        [drama_id] + rated_ids + [top_k]
    )
    if not neighbors: return None
    num = sum(n['similarity'] * rated[n['drama_id_b']] for n in neighbors)
    den = sum(abs(n['similarity']) for n in neighbors)
    return round(num / den, 4) if den > 0 else None


# ================================================================
# GET RECOMMENDATIONS
# ================================================================
def get_cf_recommendations(user_id=None, drama_id=None, top_n=10):
    """
    Dua mode:
    - drama_id : ambil drama serupa dari cf_similarity (item-based)
    - user_id  : rekomendasi personal (butuh >= MIN_RATINGS_FOR_RECS rating)

    Jika user belum cukup rating, kembalikan dict error bukan list.
    """

    # ── Mode 1: Item similarity ──────────────────────────────────
    if drama_id is not None:
        return db.query("""
            SELECT cs.drama_id_b AS drama_id, cs.similarity,
                   d.title, d.genre, d.synopsis, d.rating_avg, d."cast", d.year
            FROM cf_similarity cs JOIN dramas d ON d.drama_id=cs.drama_id_b
            WHERE cs.drama_id_a=? ORDER BY cs.similarity DESC LIMIT ?
        """, (drama_id, top_n))

    # ── Mode 2: Rekomendasi personal user ────────────────────────
    if user_id is not None:
        user_ratings = db.query(
            "SELECT drama_id, rating FROM ratings WHERE user_id=? AND split='train'",
            (user_id,)
        )
        n_rated = len(user_ratings)

        # Cek minimum 5 rating
        if n_rated < MIN_RATINGS_FOR_RECS:
            return {
                'error': 'insufficient_ratings',
                'message': (
                    f'Anda baru merating {n_rated} drama. '
                    f'Minimal {MIN_RATINGS_FOR_RECS} drama diperlukan '
                    f'untuk rekomendasi Collaborative Filtering yang akurat.'
                ),
                'min_required': MIN_RATINGS_FOR_RECS,
                'current_count': n_rated,
                'needed': MIN_RATINGS_FOR_RECS - n_rated
            }

        rated       = {r['drama_id']: r['rating'] for r in user_ratings}
        watched_ids = set(rated.keys())

        # Kumpulkan kandidat dari similarity table
        # (efisien — tidak perlu loop semua drama)
        candidate_scores = {}  # drama_id -> [sum_weighted, sum_abs_sim]

        for rated_drama_id, user_rating in rated.items():
            neighbors = db.query("""
                SELECT drama_id_b, similarity FROM cf_similarity
                WHERE drama_id_a=? ORDER BY similarity DESC LIMIT ?
            """, (rated_drama_id, TOP_K_NEIGHBORS * 3))

            for nb in neighbors:
                cand_id = nb['drama_id_b']
                sim_val = nb['similarity']
                if cand_id in watched_ids: continue
                if cand_id not in candidate_scores:
                    candidate_scores[cand_id] = [0.0, 0.0]
                # Weighted Sum: sim * rating_user
                candidate_scores[cand_id][0] += sim_val * user_rating
                candidate_scores[cand_id][1] += abs(sim_val)

        if not candidate_scores:
            return []

        # Hitung prediksi dan urutkan
        predictions = [
            (cand_id, round(num / den, 4))
            for cand_id, (num, den) in candidate_scores.items()
            if den > 0
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Ambil info drama untuk top-N
        results = []
        for cand_id, pred_score in predictions[:top_n]:
            info = db.query_one(
                "SELECT drama_id, title, genre, synopsis, rating_avg, \"cast\", year "
                "FROM dramas WHERE drama_id=? AND is_active=1",
                (cand_id,)
            )
            if info:
                row = dict(info)
                row['predicted_rating'] = pred_score
                row['similarity'] = round(min(pred_score / 10.0, 1.0), 4)
                results.append(row)
        return results

    return []


# ================================================================
# HELPER
# ================================================================
def check_user_eligibility(user_id):
    """Cek apakah user memenuhi syarat minimum rating untuk CF."""
    row = db.query_one(
        "SELECT COUNT(*) as n FROM ratings WHERE user_id=? AND split='train'",
        (user_id,)
    )
    n_rated = row['n'] if row else 0
    eligible = n_rated >= MIN_RATINGS_FOR_RECS
    return {
        'eligible': eligible,
        'n_rated': n_rated,
        'min_required': MIN_RATINGS_FOR_RECS,
        'needed': max(0, MIN_RATINGS_FOR_RECS - n_rated),
        'progress_pct': min(100, int(n_rated / MIN_RATINGS_FOR_RECS * 100)),
        'message': (
            f'Siap! {n_rated} drama dirating.' if eligible
            else f'Butuh {MIN_RATINGS_FOR_RECS - n_rated} rating lagi ({n_rated}/{MIN_RATINGS_FOR_RECS})'
        )
    }
