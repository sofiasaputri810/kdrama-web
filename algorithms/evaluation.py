# -*- coding: utf-8 -*-
"""
algorithms/evaluation.py
Metrik Evaluasi Utama: RMSE dan MAE (sesuai proposal skripsi)
Metrik Tambahan: Precision@K, Recall@K, F1@K, NDCG@K, Coverage

Referensi proposal:
- Rumusan Masalah 2: "kinerja... berdasarkan metrik evaluasi RMSE dan MAE"
- Tujuan 2: "mengukur kinerja... menggunakan metrik evaluasi RMSE dan MAE"
"""
import numpy as np, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import database as db

# ================================================================
# HELPER FUNCTIONS
# ================================================================
def dcg_at_k(relevances, k):
    r = np.array(relevances[:k], dtype=float)
    if not r.size: return 0.0
    return float(np.sum(r / np.log2(np.arange(2, r.size+2))))

def ndcg_at_k(relevances, k):
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg/ideal if ideal > 0 else 0.0

# ================================================================
# CBF EVALUATION
# ================================================================
def evaluate_cbf(k=10, relevant_threshold=7.0):
    """
    Evaluasi CBF.
    Metrik Utama: RMSE dan MAE (sesuai proposal)
    Metrik Tambahan: Precision@K, Recall@K, F1@K, NDCG@K, Coverage
    """
    t0 = time.time()
    print(f"[EVAL-CBF] Menghitung RMSE dan MAE...")

    # ── RMSE & MAE (METRIK UTAMA) ───────────────────────────────
    rmse_mae = _cbf_rmse_mae_primary()

    # ── PRECISION / RECALL / F1 / NDCG (METRIK TAMBAHAN) ────────
    test_users = db.query("SELECT DISTINCT user_id FROM ratings WHERE split='test'")
    if not test_users:
        raise ValueError("Tidak ada data test.")

    precisions, recalls, f1s, ndcgs = [], [], [], []
    all_rec_ids = set()
    total_dramas = db.query_one("SELECT COUNT(*) as n FROM dramas")['n']

    for tu in test_users:
        uid = tu['user_id']
        test_ratings = db.query("SELECT drama_id, rating FROM ratings WHERE user_id=? AND split='test'", (uid,))
        relevant = {r['drama_id'] for r in test_ratings if r['rating'] >= relevant_threshold}
        if not relevant: continue
        train_dramas = db.query("SELECT drama_id FROM ratings WHERE user_id=? AND split='train'", (uid,))
        if not train_dramas: continue
        rec_scores = {}
        for td in train_dramas:
            sims = db.query("SELECT drama_id_b, similarity FROM cbf_similarity WHERE drama_id_a=? ORDER BY similarity DESC LIMIT ?",
                            (td['drama_id'], k*3))
            for s in sims:
                did = s['drama_id_b']
                if any(t['drama_id']==did for t in train_dramas): continue
                rec_scores[did] = max(rec_scores.get(did,0), s['similarity'])
        rec_ids = [x[0] for x in sorted(rec_scores.items(), key=lambda x:x[1], reverse=True)[:k]]
        all_rec_ids.update(rec_ids)
        hits = [1 if rid in relevant else 0 for rid in rec_ids]
        n_hits = sum(hits)
        prec = n_hits/k; rec = n_hits/len(relevant) if relevant else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
        ndcgs.append(ndcg_at_k(hits, k))

    if not precisions:
        raise ValueError("Tidak cukup data untuk evaluasi CBF")

    result = {
        'method': 'CBF', 'k': k,
        # METRIK UTAMA (proposal)
        'rmse': rmse_mae['rmse'],
        'mae':  rmse_mae['mae'],
        # METRIK TAMBAHAN
        'precision_k': round(float(np.mean(precisions)), 4),
        'recall_k':    round(float(np.mean(recalls)), 4),
        'f1_k':        round(float(np.mean(f1s)), 4),
        'ndcg':        round(float(np.mean(ndcgs)), 4),
        'coverage':    round(len(all_rec_ids)/total_dramas, 4),
        'n_users_eval': len(precisions),
        'duration':    round(time.time()-t0, 2),
    }
    _save_eval(result)
    print(f"[EVAL-CBF] RMSE={result['rmse']} MAE={result['mae']} | "
          f"P@{k}={result['precision_k']} R@{k}={result['recall_k']}")
    return result

def _cbf_rmse_mae_primary():
    """
    RMSE & MAE untuk CBF menggunakan Weighted Sum.

    Penjelasan apple-to-apple:
        CBF  : sim(i,j) dari konten + item rating metadata
               → Prediksi(u,i) = Σ[sim_cbf × r(u,j)] / Σ|sim_cbf|
        CF   : sim(i,j) dari pola rating pengguna
               → Prediksi(u,i) = Σ[sim_cf  × r(u,j)] / Σ|sim_cf|

    Formula prediksi IDENTIK. Perbedaan HANYA pada asal similarity:
        CBF  → dari atribut item (konten + rating_y metadata)  ← pure CBF
        CF   → dari perilaku user (rating matrix)
    """
    from algorithms.content_based import predict_rating_cbf

    test_data = db.query("""
        SELECT DISTINCT r.user_id, r.drama_id, r.rating
        FROM ratings r WHERE r.split = 'test'
        LIMIT 2000
    """)

    sq_errors, abs_errors = [], []
    for td in test_data:
        pred = predict_rating_cbf(td['user_id'], td['drama_id'])
        if pred is None:
            continue
        sq_errors.append((pred - td['rating']) ** 2)
        abs_errors.append(abs(pred - td['rating']))

    if not sq_errors:
        return {'rmse': None, 'mae': None}

    return {
        'rmse': round(float(np.sqrt(np.mean(sq_errors))), 4),
        'mae':  round(float(np.mean(abs_errors)), 4),
    }

# ================================================================
# CF EVALUATION
# ================================================================
def evaluate_cf(k=10, relevant_threshold=7.0):
    """
    Evaluasi CF.
    Metrik Utama: RMSE dan MAE (sesuai proposal)
    Metrik Tambahan: Precision@K, Recall@K, F1@K, NDCG@K, Coverage
    """
    from algorithms.collaborative import predict_rating_cf
    t0 = time.time()
    print(f"[EVAL-CF] Menghitung RMSE dan MAE...")

    test_users = db.query("SELECT DISTINCT user_id FROM ratings WHERE split='test'")
    if not test_users:
        raise ValueError("Tidak ada data test.")

    precisions, recalls, f1s, ndcgs = [], [], [], []
    sq_errors, abs_errors = [], []
    all_rec_ids = set()
    total_dramas = db.query_one("SELECT COUNT(*) as n FROM dramas")['n']

    for tu in test_users:
        uid = tu['user_id']
        test_items = db.query("SELECT drama_id, rating FROM ratings WHERE user_id=? AND split='test'", (uid,))
        relevant = {r['drama_id'] for r in test_items if r['rating'] >= relevant_threshold}
        train_items = db.query("SELECT drama_id FROM ratings WHERE user_id=? AND split='train'", (uid,))
        if not train_items: continue
        train_set = {t['drama_id'] for t in train_items}

        # RMSE/MAE: prediksi setiap item test
        for ti in test_items:
            pred = predict_rating_cf(uid, ti['drama_id'])
            if pred is not None:
                sq_errors.append((pred - ti['rating'])**2)
                abs_errors.append(abs(pred - ti['rating']))

        if not relevant: continue
        top_train = db.query("SELECT drama_id FROM ratings WHERE user_id=? AND split='train' ORDER BY rating DESC LIMIT 3", (uid,))
        rec_scores = {}
        for td in top_train:
            sims = db.query("SELECT drama_id_b, similarity FROM cf_similarity WHERE drama_id_a=? ORDER BY similarity DESC LIMIT ?",
                            (td['drama_id'], k*3))
            for s in sims:
                did = s['drama_id_b']
                if did in train_set: continue
                rec_scores[did] = max(rec_scores.get(did,0), s['similarity'])
        rec_ids = [x[0] for x in sorted(rec_scores.items(), key=lambda x:x[1], reverse=True)[:k]]
        all_rec_ids.update(rec_ids)
        hits = [1 if rid in relevant else 0 for rid in rec_ids]
        n_hits = sum(hits)
        prec = n_hits/k; rec = n_hits/len(relevant) if relevant else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
        ndcgs.append(ndcg_at_k(hits, k))

    if not precisions:
        raise ValueError("Tidak cukup data untuk evaluasi CF")

    result = {
        'method': 'CF', 'k': k,
        # METRIK UTAMA (proposal)
        'rmse': round(float(np.sqrt(np.mean(sq_errors))), 4) if sq_errors else None,
        'mae':  round(float(np.mean(abs_errors)), 4) if abs_errors else None,
        # METRIK TAMBAHAN
        'precision_k': round(float(np.mean(precisions)), 4),
        'recall_k':    round(float(np.mean(recalls)), 4),
        'f1_k':        round(float(np.mean(f1s)), 4),
        'ndcg':        round(float(np.mean(ndcgs)), 4),
        'coverage':    round(len(all_rec_ids)/total_dramas, 4),
        'n_users_eval': len(precisions),
        'duration':    round(time.time()-t0, 2),
    }
    _save_eval(result)
    print(f"[EVAL-CF] RMSE={result['rmse']} MAE={result['mae']} | "
          f"P@{k}={result['precision_k']} R@{k}={result['recall_k']}")
    return result

# ================================================================
# SHARED HELPERS
# ================================================================
def _save_eval(result):
    train_n = db.query_one("SELECT COUNT(*) as n FROM ratings WHERE split='train'")['n']
    test_n  = db.query_one("SELECT COUNT(*) as n FROM ratings WHERE split='test'")['n']
    db.execute("""INSERT INTO evaluation_results
        (method,k,precision_k,recall_k,f1_k,rmse,mae,coverage,ndcg,train_size,test_size)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (result['method'], result['k'],
         result.get('precision_k'), result.get('recall_k'), result.get('f1_k'),
         result.get('rmse'), result.get('mae'),
         result.get('coverage'), result.get('ndcg'),
         train_n, test_n))

def get_latest_evaluation():
    return db.query("""SELECT * FROM evaluation_results
        WHERE id IN (SELECT MAX(id) FROM evaluation_results GROUP BY method)
        ORDER BY method""")

def get_evaluation_history():
    return db.query("SELECT * FROM evaluation_results ORDER BY created_at DESC LIMIT 50")

def run_full_evaluation(k=10, threshold=7.0):
    print(f"\n{'='*50}\n  EVALUASI LENGKAP @ K={k}, threshold={threshold}\n"
          f"  Metrik Utama: RMSE & MAE (sesuai proposal)\n{'='*50}")
    results = {}
    n_cbf = db.query_one("SELECT COUNT(*) as n FROM cbf_similarity")['n']
    n_cf  = db.query_one("SELECT COUNT(*) as n FROM cf_similarity")['n']
    if n_cbf == 0:
        results['CBF'] = {'error': 'Model CBF belum di-train.'}
    else:
        try: results['CBF'] = evaluate_cbf(k, threshold)
        except Exception as e: results['CBF'] = {'error': str(e)}
    if n_cf == 0:
        results['CF'] = {'error': 'Model CF belum di-train.'}
    else:
        try: results['CF'] = evaluate_cf(k, threshold)
        except Exception as e: results['CF'] = {'error': str(e)}
    return results
