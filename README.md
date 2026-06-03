# KDrama Recommend — Sistem Rekomendasi Drama Korea

**Judul Penelitian:**
Analisis Perbandingan Kinerja Metode Collaborative Filtering dan
Feature Augmented Content-Based Filtering untuk Sistem Rekomendasi Drama Korea

---

## Instalasi

```bash
pip install -r requirements.txt
```

---

## Struktur Dataset (SATU FILE GABUNGAN)

Letakkan file CSV di folder `data/` dengan nama `dataset_kdrama.csv`

| Kolom | Keterangan |
|-------|------------|
| user_id | ID pengguna (mis. user_00001) |
| drama_id | ID drama (mis. kdrama_0001) |
| rating_x | Rating dari pengguna (1–10) → data latih/uji |
| title | Judul drama |
| genre | Genre drama |
| rating_y | Rata-rata rating drama (metadata) |
| synopsis | Sinopsis |
| director | Sutradara |
| screenwriter | Penulis skenario |
| cast | Pemeran |
| tags | Tag konten |

---

## Menjalankan (Pertama Kali)

```bash
# 1. Letakkan dataset di:
#    data/dataset_kdrama.csv

# 2. Setup: import + train + evaluasi
python setup.py

# 3. Jalankan server
python run.py
```

Buka: http://localhost:5000  
Admin: admin / admin123

---

## Pembagian Data Latih / Uji

Dilakukan otomatis oleh sistem saat import:
- **80% Data Latih** — digunakan training FA-CBF dan CF
- **20% Data Uji** — digunakan evaluasi RMSE, MAE, Precision@K
- Split per user: pengguna dengan ≥5 rating → 20% terakhir jadi data uji

---

## Algoritma

### Feature Augmented CBF (FA-CBF)
- Vektor konten: TF-IDF (genre×3, cast×2, tags×2, synopsis, director)
- Vektor rating: avg_rating, n_ratings, range_rating (MinMaxScaler)
- Augmented: 70% konten + 30% rating (alpha=0.3)
- Prediksi: Weighted Sum — sama dengan CF (apple-to-apple)

### Item-Based Collaborative Filtering (CF)
- Matriks User-Item dari data rating
- Mean-centering per user
- Cosine Similarity antar item
- Prediksi: Weighted Sum (Sarwar et al., 2001)

### Metrik Evaluasi Utama (Sesuai Proposal)
- **RMSE** — Root Mean Square Error
- **MAE** — Mean Absolute Error
- Lower is Better

### Metrik Tambahan
- Precision@K, Recall@K, F1@K, NDCG@K, Coverage
