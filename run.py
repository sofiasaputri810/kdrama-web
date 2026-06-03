# -*- coding: utf-8 -*-
"""run.py — Jalankan web app setelah setup data"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import database as db
db.init_db()
from app import app
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  KDrama Recommend - Full Web App")
    print("  URL: http://localhost:5000")
    print("  Admin: admin / admin123")
    print("="*55 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
