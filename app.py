# -*- coding: utf-8 -*-
"""
KDrama Recommendation System - Full Web Application
Routes: Public, User (auth), Admin
"""
import os, sys, time, threading, math
import os
import gdown

# Download database dari Google Drive
DB_PATH = 'data/kdrama.db'
os.makedirs('data', exist_ok=True)

if not os.path.exists(DB_PATH):
    print("📥 Downloading database from Google Drive...")
    FILE_ID = '19riM_z-4MaSVTocyShiuAY88iE3ii4Hb'
    url = f'https://drive.google.com/uc?id={19riM_z-4MaSVTocyShiuAY88iE3ii4Hb}'
    gdown.download(url, DB_PATH, quiet=False)
    print("✅ Database downloaded!")
from functools import wraps
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify, g)
from werkzeug.security import generate_password_hash, check_password_hash

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import database as db
from algorithms.content_based import train_cbf, get_cbf_recommendations
from algorithms.collaborative  import train_cf,  get_cf_recommendations, check_user_eligibility
from algorithms.evaluation     import run_full_evaluation, get_latest_evaluation

app = Flask(__name__)
# Jinja2 custom filters
app.jinja_env.filters['enumerate'] = enumerate

app.secret_key = os.environ.get('SECRET_KEY', 'kdrama-secret-2025-change-in-production')

# ================================================================
# CONTEXT PROCESSORS & HELPERS
# ================================================================
@app.context_processor
def inject_globals():
    user = None
    if 'user_id' in session:
        user = db.query_one("SELECT * FROM auth_users WHERE id=?", (session['user_id'],))
    n_dramas = db.query_one("SELECT COUNT(*) as n FROM dramas")
    return dict(current_user=user, total_dramas=n_dramas['n'] if n_dramas else 0)

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Silakan login terlebih dahulu.', 'warning')
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = db.query_one("SELECT role FROM auth_users WHERE id=?", (session['user_id'],))
        if not user or user['role'] != 'admin':
            flash('Akses ditolak. Halaman ini hanya untuk admin.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

def get_user_uid(auth_user_id):
    """Get or create CF user_id for auth user."""
    uid = f"U{auth_user_id:05d}"
    db.execute("INSERT OR IGNORE INTO users (user_id, auth_user_id) VALUES (?,?)",
               (uid, auth_user_id))
    return uid

def paginate(items, page, per_page=20):
    total = len(items)
    pages = math.ceil(total / per_page)
    start = (page - 1) * per_page
    return items[start:start+per_page], pages, total

# ================================================================
# PUBLIC ROUTES
# ================================================================
@app.route('/')
def index():
    # Top rated dramas
    popular = db.query("""
        SELECT d.*, COUNT(r.id) as n_ratings, ROUND(AVG(r.rating),1) as avg_r
        FROM dramas d LEFT JOIN ratings r ON r.drama_id=d.drama_id
        WHERE d.is_active=1
        GROUP BY d.drama_id ORDER BY avg_r DESC, n_ratings DESC LIMIT 12
    """)
    # Latest dramas
    latest = db.query("SELECT * FROM dramas WHERE is_active=1 ORDER BY year DESC, drama_id DESC LIMIT 8")
    # Genre list
    genres = _get_genres()
    # Stats
    stats = {
        'n_dramas': db.query_one("SELECT COUNT(*) as n FROM dramas WHERE is_active=1")['n'],
        'n_users': db.query_one("SELECT COUNT(*) as n FROM auth_users WHERE role='user'")['n'],
        'n_ratings': db.query_one("SELECT COUNT(*) as n FROM ratings")['n'],
    }
    return render_template('index.html', popular=popular, latest=latest,
                           genres=genres, stats=stats)

@app.route('/dramas')
def dramas_list():
    page     = request.args.get('page', 1, type=int)
    genre    = request.args.get('genre', '')
    sort     = request.args.get('sort', 'popular')
    per_page = 20

    where = "WHERE d.is_active=1"
    params = []
    if genre:
        where += " AND LOWER(d.genre) LIKE ?"
        params.append(f'%{genre.lower()}%')

    order = {
        'popular': 'n_ratings DESC, avg_r DESC',
        'rating':  'avg_r DESC, n_ratings DESC',
        'newest':  'd.year DESC, d.drama_id DESC',
        'alpha':   'd.title ASC',
    }.get(sort, 'n_ratings DESC')

    total_row = db.query_one(f"SELECT COUNT(*) as n FROM dramas d {where}", params)
    total = total_row['n'] if total_row else 0
    pages = math.ceil(total / per_page)
    offset = (page - 1) * per_page

    dramas = db.query(f"""
        SELECT d.*, COUNT(r.id) as n_ratings, ROUND(AVG(r.rating),1) as avg_r
        FROM dramas d LEFT JOIN ratings r ON r.drama_id=d.drama_id
        {where} GROUP BY d.drama_id ORDER BY {order} LIMIT ? OFFSET ?
    """, params + [per_page, offset])

    return render_template('dramas.html', dramas=dramas, page=page, pages=pages,
                           total=total, genre=genre, sort=sort, genres=_get_genres())

@app.route('/dramas/<int:drama_id>')
def drama_detail(drama_id):
    drama = db.query_one("SELECT * FROM dramas WHERE drama_id=? AND is_active=1", (drama_id,))
    if not drama:
        flash('Drama tidak ditemukan.', 'error')
        return redirect(url_for('dramas_list'))

    stats = db.query_one("""
        SELECT COUNT(*) as n_ratings, ROUND(AVG(rating),2) as avg_rating,
               MIN(rating) as min_r, MAX(rating) as max_r
        FROM ratings WHERE drama_id=?
    """, (drama_id,))
    dist = db.query("SELECT CAST(rating AS INT) as r, COUNT(*) as cnt FROM ratings WHERE drama_id=? GROUP BY r ORDER BY r", (drama_id,))
    reviews = db.query("""
        SELECT r.rating, r.review, r.created_at, u.username, u.display_name, u.avatar_color
        FROM ratings r JOIN users us ON us.user_id=r.user_id
        JOIN auth_users u ON u.id=us.auth_user_id
        WHERE r.drama_id=? AND r.review != '' ORDER BY r.created_at DESC LIMIT 10
    """, (drama_id,))

    # CBF recommendations
    recs_cbf = []
    if db.query_one("SELECT COUNT(*) as n FROM cbf_similarity")['n'] > 0:
        recs_cbf = get_cbf_recommendations(drama_id, top_n=6)

    # User's rating & watchlist
    user_rating = None
    in_watchlist = False
    if 'user_id' in session:
        uid = get_user_uid(session['user_id'])
        ur = db.query_one("SELECT rating, review FROM ratings WHERE user_id=? AND drama_id=?", (uid, drama_id))
        user_rating = ur
        wl = db.query_one("SELECT id FROM watchlist WHERE auth_user_id=? AND drama_id=?", (session['user_id'], drama_id))
        in_watchlist = bool(wl)

    return render_template('drama_detail.html', drama=drama, stats=stats,
                           dist=dist, reviews=reviews, recs_cbf=recs_cbf,
                           user_rating=user_rating, in_watchlist=in_watchlist)

@app.route('/search')
def search():
    q     = request.args.get('q', '').strip()
    genre = request.args.get('genre', '')
    cast  = request.args.get('cast', '')
    page  = request.args.get('page', 1, type=int)
    per_page = 15
    results = []
    total = 0

    if q or genre or cast:
        where = "WHERE d.is_active=1"
        params = []
        if q:
            where += " AND (LOWER(d.title) LIKE ? OR LOWER(d.synopsis) LIKE ?)"
            params += [f'%{q.lower()}%', f'%{q.lower()}%']
        if genre:
            where += " AND LOWER(d.genre) LIKE ?"
            params.append(f'%{genre.lower()}%')
        if cast:
            where += " AND LOWER(d.\"cast\") LIKE ?"
            params.append(f'%{cast.lower()}%')
        tr = db.query_one(f"SELECT COUNT(*) as n FROM dramas d {where}", params)
        total = tr['n'] if tr else 0
        pages = math.ceil(total / per_page)
        offset = (page - 1) * per_page
        results = db.query(f"""
            SELECT d.*, COUNT(r.id) as n_ratings, ROUND(AVG(r.rating),1) as avg_r
            FROM dramas d LEFT JOIN ratings r ON r.drama_id=d.drama_id
            {where} GROUP BY d.drama_id ORDER BY avg_r DESC LIMIT ? OFFSET ?
        """, params + [per_page, offset])
    else:
        pages = 0

    return render_template('search.html', results=results, q=q, genre=genre,
                           cast=cast, page=page, pages=pages, total=total,
                           genres=_get_genres())

@app.route('/about')
def about():
    eval_results = get_latest_evaluation()
    return render_template('about.html', eval_results=eval_results)

# ================================================================
# AUTH ROUTES
# ================================================================
@app.route('/login', methods=['GET','POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','')
        user = db.query_one("SELECT * FROM auth_users WHERE (username=? OR email=?) AND is_active=1",
                            (username, username))
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            db.execute("UPDATE auth_users SET last_login=CURRENT_TIMESTAMP WHERE id=?", (user['id'],))
            db.log_activity(user['id'], user['username'], 'LOGIN', '', request.remote_addr)
            flash(f'Selamat datang kembali, {user["display_name"] or user["username"]}!', 'success')
            nxt = request.args.get('next')
            return redirect(nxt if nxt else (url_for('admin_dashboard') if user['role']=='admin' else url_for('dashboard')))
        flash('Username/email atau password salah.', 'error')
    return render_template('auth/login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        email    = request.form.get('email','').strip()
        password = request.form.get('password','')
        confirm  = request.form.get('confirm','')
        display  = request.form.get('display_name','').strip() or username

        if len(username) < 3:
            flash('Username minimal 3 karakter.', 'error')
        elif len(password) < 6:
            flash('Password minimal 6 karakter.', 'error')
        elif password != confirm:
            flash('Konfirmasi password tidak cocok.', 'error')
        elif db.query_one("SELECT id FROM auth_users WHERE username=?", (username,)):
            flash('Username sudah digunakan.', 'error')
        elif db.query_one("SELECT id FROM auth_users WHERE email=?", (email,)):
            flash('Email sudah terdaftar.', 'error')
        else:
            uid = db.execute("""INSERT INTO auth_users (username,email,password_hash,display_name)
                                VALUES (?,?,?,?)""",
                             (username, email, generate_password_hash(password), display))
            db.log_activity(uid, username, 'REGISTER', '', request.remote_addr)
            flash('Registrasi berhasil! Silakan login.', 'success')
            return redirect(url_for('login'))
    return render_template('auth/register.html')

@app.route('/logout')
def logout():
    if 'user_id' in session:
        db.log_activity(session['user_id'], session.get('username',''), 'LOGOUT')
    session.clear()
    flash('Anda telah logout.', 'info')
    return redirect(url_for('index'))

# ================================================================
# USER ROUTES (Login Required)
# ================================================================
@app.route('/dashboard')
@login_required
def dashboard():
    uid = get_user_uid(session['user_id'])
    user = db.query_one("SELECT * FROM auth_users WHERE id=?", (session['user_id'],))
    recent_ratings = db.query("""
        SELECT r.*, d.title, d.genre, d.year FROM ratings r
        JOIN dramas d ON d.drama_id=r.drama_id
        WHERE r.user_id=? ORDER BY r.created_at DESC LIMIT 6
    """, (uid,))
    wl_count = db.query_one("SELECT COUNT(*) as n FROM watchlist WHERE auth_user_id=?",
                             (session['user_id'],))
    rating_count = db.query_one("SELECT COUNT(*) as n FROM ratings WHERE user_id=?", (uid,))
    avg_rating = db.query_one("SELECT ROUND(AVG(rating),1) as a FROM ratings WHERE user_id=?", (uid,))
    has_model = db.query_one("SELECT COUNT(*) as n FROM cbf_similarity")['n'] > 0
    quick_recs = []
    if has_model and recent_ratings:
        ref_id = recent_ratings[0]['drama_id']
        quick_recs = get_cbf_recommendations(ref_id, top_n=4)
    eligibility = check_user_eligibility(uid)
    return render_template('user/dashboard.html', user=user,
                           recent_ratings=recent_ratings,
                           wl_count=wl_count['n'] if wl_count else 0,
                           rating_count=rating_count['n'] if rating_count else 0,
                           avg_rating=avg_rating['a'] if avg_rating else 0,
                           quick_recs=quick_recs,
                           eligibility=eligibility)

@app.route('/recommendations')
@login_required
def recommendations():
    uid = get_user_uid(session['user_id'])
    method = request.args.get('method', 'cbf')
    top_n  = request.args.get('n', 10, type=int)

    rated     = db.query("SELECT drama_id FROM ratings WHERE user_id=?", (uid,))
    rated_ids = {r['drama_id'] for r in rated}

    recs_cbf, recs_cf = [], []
    cf_error   = None   # pesan error jika belum cukup rating
    cbf_ready  = db.query_one("SELECT COUNT(*) as n FROM cbf_similarity")['n'] > 0
    cf_ready   = db.query_one("SELECT COUNT(*) as n FROM cf_similarity")['n'] > 0
    eligibility = check_user_eligibility(uid)

    # ── CBF: tidak ada minimum rating, bisa langsung ─────────────
    if cbf_ready and rated_ids:
        top_rated = db.query("""SELECT drama_id FROM ratings WHERE user_id=?
                                ORDER BY rating DESC LIMIT 3""", (uid,))
        seen = set()
        for tr in top_rated:
            for r in get_cbf_recommendations(tr['drama_id'], top_n=top_n*2):
                if r['drama_id'] not in rated_ids and r['drama_id'] not in seen:
                    recs_cbf.append(r); seen.add(r['drama_id'])
        recs_cbf = recs_cbf[:top_n]

    # ── CF: butuh minimal 5 rating ───────────────────────────────
    if cf_ready:
        cf_result = get_cf_recommendations(user_id=uid, top_n=top_n)
        if isinstance(cf_result, dict) and cf_result.get('error'):
            cf_error = cf_result   # simpan info error untuk template
        elif isinstance(cf_result, list):
            recs_cf = cf_result

    return render_template('user/recommendations.html',
                           recs_cbf=recs_cbf, recs_cf=recs_cf,
                           cbf_ready=cbf_ready, cf_ready=cf_ready,
                           rated_count=len(rated_ids), method=method,
                           eligibility=eligibility, cf_error=cf_error)

@app.route('/rate/<int:drama_id>', methods=['GET','POST'])
@login_required
def rate_drama(drama_id):
    drama = db.query_one("SELECT * FROM dramas WHERE drama_id=? AND is_active=1", (drama_id,))
    if not drama:
        flash('Drama tidak ditemukan.', 'error')
        return redirect(url_for('dramas_list'))
    uid = get_user_uid(session['user_id'])
    if request.method == 'POST':
        rating = request.form.get('rating', type=float)
        review = request.form.get('review','').strip()
        if not rating or not (1 <= rating <= 10):
            flash('Rating harus antara 1-10.', 'error')
        else:
            db.execute("""INSERT INTO ratings (user_id,drama_id,rating,review,split)
                          VALUES (?,?,?,?,'train')
                          ON CONFLICT(user_id,drama_id) DO UPDATE
                          SET rating=excluded.rating, review=excluded.review""",
                       (uid, drama_id, rating, review))
            db.execute("UPDATE users SET total_ratings=(SELECT COUNT(*) FROM ratings WHERE user_id=?) WHERE user_id=?", (uid,uid))
            db.log_activity(session['user_id'], session['username'], 'RATE',
                            f'drama_id={drama_id} rating={rating}')
            flash(f'Rating untuk "{drama["title"]}" berhasil disimpan!', 'success')
            return redirect(url_for('drama_detail', drama_id=drama_id))
    existing = db.query_one("SELECT * FROM ratings WHERE user_id=? AND drama_id=?", (uid, drama_id))
    return render_template('user/rate.html', drama=drama, existing=existing)

@app.route('/watchlist')
@login_required
def watchlist():
    items = db.query("""
        SELECT w.*, d.title, d.genre, d.year, d.synopsis,
               d.rating_avg, d."cast"
        FROM watchlist w JOIN dramas d ON d.drama_id=w.drama_id
        WHERE w.auth_user_id=? ORDER BY w.added_at DESC
    """, (session['user_id'],))
    return render_template('user/watchlist.html', items=items)

@app.route('/watchlist/toggle/<int:drama_id>', methods=['POST'])
@login_required
def toggle_watchlist(drama_id):
    exists = db.query_one("SELECT id FROM watchlist WHERE auth_user_id=? AND drama_id=?",
                          (session['user_id'], drama_id))
    if exists:
        db.execute("DELETE FROM watchlist WHERE auth_user_id=? AND drama_id=?",
                   (session['user_id'], drama_id))
        result = 'removed'
    else:
        db.execute("INSERT OR IGNORE INTO watchlist (auth_user_id,drama_id) VALUES (?,?)",
                   (session['user_id'], drama_id))
        result = 'added'
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'status': result})
    return redirect(request.referrer or url_for('watchlist'))

@app.route('/history')
@login_required
def history():
    uid = get_user_uid(session['user_id'])
    page = request.args.get('page', 1, type=int)
    per_page = 15
    total_row = db.query_one("SELECT COUNT(*) as n FROM ratings WHERE user_id=?", (uid,))
    total = total_row['n'] if total_row else 0
    pages = math.ceil(total / per_page)
    offset = (page - 1) * per_page
    items = db.query("""
        SELECT r.*, d.title, d.genre, d.year, d."cast"
        FROM ratings r JOIN dramas d ON d.drama_id=r.drama_id
        WHERE r.user_id=? ORDER BY r.created_at DESC LIMIT ? OFFSET ?
    """, (uid, per_page, offset))
    stats = db.query_one("""
        SELECT COUNT(*) as n, ROUND(AVG(rating),2) as avg,
               MAX(rating) as max_r, MIN(rating) as min_r
        FROM ratings WHERE user_id=?
    """, (uid,))
    return render_template('user/history.html', items=items, page=page,
                           pages=pages, total=total, stats=stats)

@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    user = db.query_one("SELECT * FROM auth_users WHERE id=?", (session['user_id'],))
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'update_profile':
            display = request.form.get('display_name','').strip()
            bio     = request.form.get('bio','').strip()
            genres  = ','.join(request.form.getlist('fav_genres'))
            color   = request.form.get('avatar_color','#e8506a')
            db.execute("UPDATE auth_users SET display_name=?,bio=?,fav_genres=?,avatar_color=? WHERE id=?",
                       (display, bio, genres, color, session['user_id']))
            flash('Profil berhasil diperbarui!', 'success')
        elif action == 'change_password':
            old_pw  = request.form.get('old_password','')
            new_pw  = request.form.get('new_password','')
            confirm = request.form.get('confirm_password','')
            if not check_password_hash(user['password_hash'], old_pw):
                flash('Password lama salah.', 'error')
            elif len(new_pw) < 6:
                flash('Password baru minimal 6 karakter.', 'error')
            elif new_pw != confirm:
                flash('Konfirmasi password tidak cocok.', 'error')
            else:
                db.execute("UPDATE auth_users SET password_hash=? WHERE id=?",
                           (generate_password_hash(new_pw), session['user_id']))
                flash('Password berhasil diubah!', 'success')
        return redirect(url_for('profile'))
    all_genres = ['Romance','Fantasy','Thriller','Action','Historical',
                  'Crime','Comedy','Horror','Slice of Life','Drama',
                  'Sci-Fi','Legal','Business','Psychological','Youth']
    fav = user['fav_genres'].split(',') if user['fav_genres'] else []
    return render_template('user/profile.html', user=user,
                           all_genres=all_genres, fav_genres=fav)

@app.route('/compare')
@login_required
def compare():
    uid = get_user_uid(session['user_id'])

    # ambil input manual dari textbox
    drama_title = request.args.get('drama_title', '').strip()

    top_n = request.args.get('n', 10, type=int)

    cbf_ready = db.query_one(
        "SELECT COUNT(*) as n FROM cbf_similarity"
    )['n'] > 0

    cf_ready = db.query_one(
        "SELECT COUNT(*) as n FROM cf_similarity"
    )['n'] > 0

    drama = None
    recs_cbf = []
    recs_cf = []
    eval_data = []

    # cari drama berdasarkan judul yang diketik
    if drama_title:
        drama = db.query_one("""
            SELECT *
            FROM dramas
            WHERE LOWER(title) LIKE LOWER(?)
            LIMIT 1
        """, (f'%{drama_title}%',))

        # jika ditemukan
        if drama:
            drama_id = drama['drama_id']

            if cbf_ready:
                recs_cbf = get_cbf_recommendations(
                    drama_id,
                    top_n=top_n
                )

            if cf_ready:
                recs_cf = get_cf_recommendations(
                    drama_id=drama_id,
                    top_n=top_n
                )

        else:
            flash('Drama tidak ditemukan.', 'warning')

        eval_data = get_latest_evaluation()

    return render_template(
        'user/compare.html',
        drama=drama,
        recs_cbf=recs_cbf,
        recs_cf=recs_cf,
        eval_data=eval_data,
        cbf_ready=cbf_ready,
        cf_ready=cf_ready,
        drama_title=drama_title,
        top_n=top_n
    )
# ================================================================
# ADMIN ROUTES
# ================================================================
@app.route('/admin')
@admin_required
def admin_dashboard():
    stats = {
        'n_dramas':  db.query_one("SELECT COUNT(*) as n FROM dramas WHERE is_active=1")['n'],
        'n_users':   db.query_one("SELECT COUNT(*) as n FROM auth_users WHERE role='user'")['n'],
        'n_ratings': db.query_one("SELECT COUNT(*) as n FROM ratings")['n'],
        'n_wl':      db.query_one("SELECT COUNT(*) as n FROM watchlist")['n'],
        'cbf_ready': db.query_one("SELECT COUNT(*) as n FROM cbf_similarity")['n'] > 0,
        'cf_ready':  db.query_one("SELECT COUNT(*) as n FROM cf_similarity")['n'] > 0,
        'n_train':   db.query_one("SELECT COUNT(*) as n FROM ratings WHERE split='train'")['n'],
        'n_test':    db.query_one("SELECT COUNT(*) as n FROM ratings WHERE split='test'")['n'],
    }
    recent_users = db.query("""SELECT * FROM auth_users ORDER BY created_at DESC LIMIT 5""")
    recent_logs  = db.query("""SELECT * FROM activity_log ORDER BY created_at DESC LIMIT 10""")
    eval_data    = get_latest_evaluation()
    top_dramas   = db.query("""
        SELECT d.title, COUNT(r.id) as n_r, ROUND(AVG(r.rating),1) as avg_r
        FROM dramas d LEFT JOIN ratings r ON r.drama_id=d.drama_id
        GROUP BY d.drama_id ORDER BY n_r DESC LIMIT 5
    """)
    return render_template('admin/dashboard.html', stats=stats,
                           recent_users=recent_users, recent_logs=recent_logs,
                           eval_data=eval_data, top_dramas=top_dramas)

@app.route('/admin/dramas', methods=['GET','POST'])
@admin_required
def admin_dramas():
    page = request.args.get('page', 1, type=int)
    q    = request.args.get('q','').strip()
    per_page = 20
    where = "WHERE 1=1"
    params = []
    if q:
        where += " AND LOWER(title) LIKE ?"
        params.append(f'%{q.lower()}%')
    total = db.query_one(f"SELECT COUNT(*) as n FROM dramas {where}", params)['n']
    pages = math.ceil(total / per_page)
    dramas = db.query(f"SELECT * FROM dramas {where} ORDER BY drama_id DESC LIMIT ? OFFSET ?",
                      params + [per_page, (page-1)*per_page])
    return render_template('admin/dramas.html', dramas=dramas, page=page,
                           pages=pages, total=total, q=q)

@app.route('/admin/dramas/add', methods=['GET','POST'])
@admin_required
def admin_drama_add():
    if request.method == 'POST':
        f = request.form
        db.execute("""INSERT INTO dramas (title,genre,rating_avg,synopsis,director,
                      screenwriter,"cast",tags,year,episodes)
                      VALUES (?,?,?,?,?,?,?,?,?,?)""",
                   (f.get('title','').strip(), f.get('genre',''),
                    f.get('rating_avg', type=float),
                    f.get('synopsis',''), f.get('director',''),
                    f.get('screenwriter',''), f.get('cast',''),
                    f.get('tags',''), f.get('year', type=int),
                    f.get('episodes', type=int)))
        db.log_activity(session['user_id'], session['username'], 'ADMIN_ADD_DRAMA',
                        f'title={f.get("title")}')
        flash('Drama berhasil ditambahkan!', 'success')
        return redirect(url_for('admin_dramas'))
    return render_template('admin/drama_form.html', drama=None, action='Tambah')

@app.route('/admin/dramas/edit/<int:drama_id>', methods=['GET','POST'])
@admin_required
def admin_drama_edit(drama_id):
    drama = db.query_one("SELECT * FROM dramas WHERE drama_id=?", (drama_id,))
    if not drama:
        flash('Drama tidak ditemukan.', 'error')
        return redirect(url_for('admin_dramas'))
    if request.method == 'POST':
        f = request.form
        db.execute("""UPDATE dramas SET title=?,genre=?,rating_avg=?,synopsis=?,
                      director=?,screenwriter=?,"cast"=?,tags=?,year=?,episodes=?
                      WHERE drama_id=?""",
                   (f.get('title','').strip(), f.get('genre',''),
                    f.get('rating_avg', type=float),
                    f.get('synopsis',''), f.get('director',''),
                    f.get('screenwriter',''), f.get('cast',''),
                    f.get('tags',''), f.get('year', type=int),
                    f.get('episodes', type=int), drama_id))
        db.log_activity(session['user_id'], session['username'], 'ADMIN_EDIT_DRAMA',
                        f'drama_id={drama_id}')
        flash('Drama berhasil diperbarui!', 'success')
        return redirect(url_for('admin_dramas'))
    return render_template('admin/drama_form.html', drama=drama, action='Edit')

@app.route('/admin/dramas/delete/<int:drama_id>', methods=['POST'])
@admin_required
def admin_drama_delete(drama_id):
    db.execute("UPDATE dramas SET is_active=0 WHERE drama_id=?", (drama_id,))
    db.log_activity(session['user_id'], session['username'], 'ADMIN_DELETE_DRAMA',
                    f'drama_id={drama_id}')
    flash('Drama berhasil dihapus.', 'success')
    return redirect(url_for('admin_dramas'))

@app.route('/admin/users')
@admin_required
def admin_users():
    page = request.args.get('page', 1, type=int)
    q    = request.args.get('q','').strip()
    per_page = 20
    where = "WHERE 1=1"
    params = []
    if q:
        where += " AND (LOWER(username) LIKE ? OR LOWER(email) LIKE ?)"
        params += [f'%{q.lower()}%', f'%{q.lower()}%']
    total = db.query_one(f"SELECT COUNT(*) as n FROM auth_users {where}", params)['n']
    pages = math.ceil(total / per_page)
    users = db.query(f"""
        SELECT u.*, COUNT(r.id) as n_ratings
        FROM auth_users u
        LEFT JOIN users us ON us.auth_user_id=u.id
        LEFT JOIN ratings r ON r.user_id=us.user_id
        {where} GROUP BY u.id ORDER BY u.created_at DESC LIMIT ? OFFSET ?
    """, params + [per_page, (page-1)*per_page])
    return render_template('admin/users.html', users=users, page=page,
                           pages=pages, total=total, q=q)

@app.route('/admin/users/toggle/<int:user_id>', methods=['POST'])
@admin_required
def admin_toggle_user(user_id):
    user = db.query_one("SELECT is_active, username FROM auth_users WHERE id=?", (user_id,))
    if user:
        new_state = 0 if user['is_active'] else 1
        db.execute("UPDATE auth_users SET is_active=? WHERE id=?", (new_state, user_id))
        db.log_activity(session['user_id'], session['username'],
                        'ADMIN_TOGGLE_USER', f'user={user["username"]} active={new_state}')
    return redirect(url_for('admin_users'))

@app.route('/admin/import', methods=['GET','POST'])
@admin_required
def admin_import():
    result = None
    if request.method == 'POST':
        from data.import_data import run_import
        try:
            db.init_db()
            import_result = run_import()
            if import_result:
                result = {
                    'success': True,
                    'n_dramas': import_result.get('n_dramas', 0) if isinstance(import_result, dict) else db.query_one("SELECT COUNT(*) as n FROM dramas")['n'],
                    'n_users':  import_result.get('n_users',  0) if isinstance(import_result, dict) else 0,
                    'n_train':  import_result.get('n_train',  0) if isinstance(import_result, dict) else db.query_one("SELECT COUNT(*) as n FROM ratings WHERE split='train'")['n'],
                    'n_test':   import_result.get('n_test',   0) if isinstance(import_result, dict) else db.query_one("SELECT COUNT(*) as n FROM ratings WHERE split='test'")['n'],
                }
                db.log_activity(session['user_id'], session['username'], 'ADMIN_IMPORT')
                flash('Import berhasil!', 'success')
            else:
                result = {'success': False, 'error': 'File dataset_kdrama.csv tidak ditemukan di folder data/'}
                flash('Import gagal: File CSV tidak ditemukan.', 'error')
        except Exception as e:
            result = {'success': False, 'error': str(e)}
            flash(f'Import gagal: {e}', 'error')
    return render_template('admin/import.html', result=result)

_training_status = {}

@app.route('/admin/retrain', methods=['GET','POST'])
@admin_required
def admin_retrain():
    global _training_status
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'train_cbf':
            features = request.form.get('features','all')
            alpha = float(request.form.get('alpha', 0.3))
            _training_status['CBF'] = {'status':'running','started':time.time()}
            def _do():
                try:
                    r = train_cbf(features=features, alpha=alpha)
                    _training_status['CBF'] = {'status':'done','result':r}
                except Exception as e:
                    _training_status['CBF'] = {'status':'error','error':str(e)}
            threading.Thread(target=_do, daemon=True).start()
            db.log_activity(session['user_id'], session['username'], 'ADMIN_TRAIN_CBF')
            flash('Training CBF dimulai...', 'info')
        elif action == 'train_cf':
            method = request.form.get('method','cosine')
            _training_status['CF'] = {'status':'running','started':time.time()}
            def _do2():
                try:
                    r = train_cf(method=method)
                    _training_status['CF'] = {'status':'done','result':r}
                except Exception as e:
                    _training_status['CF'] = {'status':'error','error':str(e)}
            threading.Thread(target=_do2, daemon=True).start()
            db.log_activity(session['user_id'], session['username'], 'ADMIN_TRAIN_CF')
            flash('Training CF dimulai...', 'info')
        elif action == 'evaluate':
            k = int(request.form.get('k', 10))
            threshold = float(request.form.get('threshold', 7.0))
            try:
                run_full_evaluation(k=k, threshold=threshold)
                flash('Evaluasi selesai!', 'success')
            except Exception as e:
                flash(f'Evaluasi gagal: {e}', 'error')
        return redirect(url_for('admin_retrain'))
    train_log = db.query("SELECT * FROM training_log ORDER BY created_at DESC LIMIT 10")
    eval_results = get_latest_evaluation()
    return render_template('admin/retrain.html', training_status=_training_status,
                           train_log=train_log, eval_results=eval_results)

@app.route('/admin/retrain/status')
@admin_required
def retrain_status():
    return jsonify(_training_status)

@app.route('/admin/logs')
@admin_required
def admin_logs():
    page = request.args.get('page', 1, type=int)
    action_filter = request.args.get('action','')
    per_page = 30
    where = "WHERE 1=1"
    params = []
    if action_filter:
        where += " AND action=?"; params.append(action_filter)
    total = db.query_one(f"SELECT COUNT(*) as n FROM activity_log {where}", params)['n']
    pages = math.ceil(total / per_page)
    logs = db.query(f"SELECT * FROM activity_log {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    params + [per_page, (page-1)*per_page])
    actions = db.query("SELECT DISTINCT action FROM activity_log ORDER BY action")
    return render_template('admin/logs.html', logs=logs, page=page, pages=pages,
                           total=total, actions=actions, action_filter=action_filter)

# ================================================================
# API ENDPOINTS (for AJAX)
# ================================================================
@app.route('/api/status')
def api_status():
    return jsonify({'status':'ok','n_dramas':db.query_one("SELECT COUNT(*) as n FROM dramas")['n']})

@app.route('/api/watchlist/check/<int:drama_id>')
@login_required
def api_watchlist_check(drama_id):
    exists = db.query_one("SELECT id FROM watchlist WHERE auth_user_id=? AND drama_id=?",
                          (session['user_id'], drama_id))
    return jsonify({'in_watchlist': bool(exists)})

# ================================================================
# HELPERS
# ================================================================
def _get_genres():
    dramas = db.query("SELECT genre FROM dramas WHERE is_active=1 AND genre IS NOT NULL")
    genres = set()
    for d in dramas:
        for g in (d['genre'] or '').split(','):
            g = g.strip()
            if g: genres.add(g)
    return sorted(genres)

if __name__ == '__main__':
    db.init_db()
    print("\n" + "="*55)
    print("  KDrama Web App - Full System")
    print("  URL: http://localhost:5000")
    print("  Admin: admin / admin123")
    print("="*55 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
