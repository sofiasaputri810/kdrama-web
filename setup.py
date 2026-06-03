# -*- coding: utf-8 -*-
"""
setup.py — Jalankan sekali setelah meletakkan dataset di folder data/

Dataset yang dibutuhkan:
    data/dataset_kdrama.csv   (satu file gabungan)

    Kolom: user_id, drama_id, rating_x, title, genre, rating_y,
           synopsis, director, screenwriter, cast, tags
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database as db
from data.import_data import run_import
from algorithms.content_based import train_cbf
from algorithms.collaborative import train_cf
from algorithms.evaluation import run_full_evaluation

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  KDrama Recommend — Setup")
    print("  Dataset : data/dataset_kdrama.csv")
    print("  Algoritma: FA-CBF vs Item-Based CF")
    print("  Metrik   : RMSE dan MAE (utama)")
    print("="*55)

    # Hapus database lama sebelum setup ulang
    DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'kdrama.db')
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("\n[SETUP] Database lama dihapus: data/kdrama.db")
    else:
        print("\n[SETUP] Tidak ada database lama.")

    db.init_db()

    result = run_import()
    if result:
        print("\nTraining model...")
        # FA-CBF: 70% konten + 30% rating (alpha=0.3)
        train_cbf(features='all', alpha=0.3)
        # Item-Based CF: Cosine Similarity
        train_cf(method='cosine')

        print("\nEvaluasi (RMSE dan MAE)...")
        run_full_evaluation(k=10, threshold=7.0)

        print("\n" + "="*55)
        print("  [OK] Setup selesai!")
        print("  Jalankan : python run.py")
        print("  Buka     : http://localhost:5000")
        print("  Admin    : admin / admin123")
        print("  Evaluasi : http://localhost:5000/about")
        print("="*55 + "\n")
    else:
        print("\n" + "="*55)
        print("  [ERROR] Import gagal!")
        print("  Pastikan file ada di: data/dataset_kdrama.csv")
        print("  Kolom wajib: user_id, drama_id, rating_x, title")
        print("="*55 + "\n")
