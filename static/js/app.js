// KDrama Recommend — Main JS

// Auto-dismiss flash messages
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => { el.style.opacity='0'; el.style.transform='translateY(-6px)';
      el.style.transition='all .3s'; setTimeout(()=>el.remove(), 300); }, 4500);
  });

  // Sidebar close on outside click (mobile)
  document.addEventListener('click', e => {
    const sb = document.getElementById('sidebar');
    const toggle = document.querySelector('.sidebar-toggle');
    if (sb && sb.classList.contains('open') && !sb.contains(e.target) && !toggle.contains(e.target)) {
      sb.classList.remove('open');
    }
  });
});

// Watchlist toggle (AJAX)
function toggleWatchlist(dramaId, btn) {
  fetch(`/watchlist/toggle/${dramaId}`, {
    method: 'POST',
    headers: { 'X-Requested-With': 'XMLHttpRequest',
                'Content-Type': 'application/json' }
  }).then(r => r.json()).then(data => {
    if (data.status === 'added') {
      btn.textContent = '♥'; btn.classList.add('active');
      btn.title = 'Hapus dari Watchlist';
      showToast('Ditambahkan ke watchlist!', 'success');
    } else {
      btn.textContent = '♡'; btn.classList.remove('active');
      btn.title = 'Tambah ke Watchlist';
      showToast('Dihapus dari watchlist.', 'info');
    }
  }).catch(() => showToast('Gagal. Coba lagi.', 'error'));
}

// Toast notification
function showToast(msg, type='info') {
  const t = document.createElement('div');
  t.className = `flash flash-${type}`;
  t.style.cssText = 'position:fixed;bottom:20px;right:20px;z-index:9999;min-width:240px;animation:slideDown .25s ease';
  const icons = {success:'✓', error:'✕', info:'ℹ', warning:'⚠'};
  t.innerHTML = `<span>${icons[type]||'ℹ'}</span> ${msg}`;
  document.body.appendChild(t);
  setTimeout(() => { t.style.opacity='0'; t.style.transition='opacity .3s';
    setTimeout(()=>t.remove(), 300); }, 3000);
}

// Star rating interactive
function initStarRating() {
  const input = document.getElementById('rating-value');
  const labels = document.querySelectorAll('.star-interactive');
  if (!labels.length || !input) return;
  labels.forEach((lbl, i) => {
    lbl.addEventListener('click', () => {
      input.value = labels.length - i;
      labels.forEach((l, j) => { l.style.color = j >= i ? 'var(--gold)' : 'var(--border)'; });
    });
  });
}
document.addEventListener('DOMContentLoaded', initStarRating);

// Training status polling
let _pollTimer = null;
function startTrainingPoll() {
  _pollTimer = setInterval(() => {
    fetch('/admin/retrain/status').then(r=>r.json()).then(data => {
      ['CBF','CF'].forEach(m => {
        if (!data[m]) return;
        const el = document.getElementById(`train-status-${m}`);
        if (!el) return;
        if (data[m].status === 'running') {
          el.innerHTML = '<span class="spinner spinner-dark"></span> Training...';
        } else if (data[m].status === 'done') {
          el.innerHTML = '<span style="color:var(--green-dark)">✓ Selesai</span>';
          clearInterval(_pollTimer);
        } else if (data[m].status === 'error') {
          el.innerHTML = `<span style="color:var(--cherry)">✕ Error: ${data[m].error||''}</span>`;
          clearInterval(_pollTimer);
        }
      });
    });
  }, 1500);
}

// Bar chart animation
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.bar-fill').forEach(el => {
    const w = el.getAttribute('data-w') || '0';
    el.style.width = '0%';
    setTimeout(() => { el.style.width = w + '%'; }, 100);
  });
});

// Confirm delete
function confirmDelete(msg, form) {
  if (confirm(msg || 'Yakin ingin menghapus?')) form.submit();
}

// Genre checkbox limit & select all
function toggleGenreAll(cb) {
  document.querySelectorAll('.genre-cb').forEach(el => el.checked = cb.checked);
}
