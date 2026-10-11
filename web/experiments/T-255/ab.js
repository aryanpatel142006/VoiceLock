/* T-255 blind A/B listening harness. No dependencies, no server.
   Manifest: { trials: [ { trial_id, prompt, input_ref, clips: [ {id, file, condition, run_id?} ] } ] }
   Conditions stay blind: page shows only randomized X/Y labels. */
'use strict';

const TAGS = ['Wrong person', 'Other voice leaks', 'Unclear/robotic', 'Too slow', 'Good'];
const STORE_KEY = 't255_ab_session_v1';

const $ = (id) => document.getElementById(id);
const state = {
  trials: [], idx: 0, results: [], played: new Set(), order: [],
  listener: '', startedAt: null,
};

function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function validateManifest(m) {
  if (!m || !Array.isArray(m.trials) || m.trials.length === 0)
    return 'manifest needs a non-empty trials array';
  for (const t of m.trials) {
    if (!t.trial_id || !Array.isArray(t.clips) || t.clips.length < 2)
      return `trial ${t.trial_id || '?'} needs trial_id and >=2 clips`;
    for (const c of t.clips) {
      if (!c.id || !c.file || !c.condition)
        return `trial ${t.trial_id}: every clip needs id, file, condition`;
    }
  }
  return null;
}

function save() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({
      idx: state.idx, results: state.results, listener: state.listener,
      startedAt: state.startedAt,
    }));
  } catch (e) { /* private mode: session simply won't resume */ }
}

function loadSaved() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) { return null; }
}

function startSession(manifest, listener) {
  const err = validateManifest(manifest);
  if (err) { alert('Invalid manifest: ' + err); return; }
  state.trials = manifest.trials;
  state.listener = listener || 'anon';
  state.startedAt = new Date().toISOString();
  const saved = loadSaved();
  if (saved && saved.results && saved.results.length > 0 &&
      confirm(`Resume previous session (${saved.results.length} trials done)?`)) {
    state.idx = saved.idx; state.results = saved.results;
  } else {
    state.idx = 0; state.results = [];
    localStorage.removeItem(STORE_KEY);
  }
  $('loader').hidden = true;
  $('done').hidden = true;
  renderTrial();
}

function renderTrial() {
  const t = state.trials[state.idx];
  if (!t) return finish();
  state.played = new Set();
  // Randomize: shuffle clip order, label X/Y in shuffled positions.
  state.order = shuffle(t.clips.map((c) => c.id));
  $('prompt').textContent = `Trial ${state.idx + 1}/${state.trials.length}: ${t.prompt || 'Which clip is better?'}`;
  $('inputref').textContent = t.input_ref ? `Input: ${t.input_ref}` : '';
  $('session-line').textContent = `Listener: ${state.listener}`;
  $('progress').hidden = false;
  updateProgress();

  const clipsEl = $('clips');
  clipsEl.innerHTML = '';
  const labels = ['X', 'Y', 'Z', 'W'];
  state.order.forEach((clipId, i) => {
    const clip = t.clips.find((c) => c.id === clipId);
    const label = labels[i] || `C${i + 1}`;
    const div = document.createElement('div');
    div.className = 'clip';
    div.id = `clip-${label}`;
    div.innerHTML = `<h2>${label}</h2><audio controls preload="auto" src="${clip.file}"></audio>`;
    const audio = div.querySelector('audio');
    audio.addEventListener('play', () => {
      state.played.add(label);
      div.classList.add('played');
      checkGate();
    });
    clipsEl.appendChild(div);
  });

  // Build tag checkboxes per label
  document.querySelectorAll('.tagrow').forEach((row) => {
    const label = row.dataset.clip;
    const list = row.querySelector('.taglist');
    list.innerHTML = '';
    TAGS.forEach((tag) => {
      const lab = document.createElement('label');
      lab.innerHTML = `<input type="checkbox" name="tag_${label}" value="${tag}"><span>${tag}</span>`;
      list.appendChild(lab);
    });
    row.style.display = ['X', 'Y'].includes(label) ? '' : 'none';
  });

  $('rateform').reset();
  $('comment').value = '';
  $('nextbtn').disabled = true;
  $('gate-note').hidden = false;
  $('trial').hidden = false;
}

function checkGate() {
  const need = Math.min(2, state.order.length);
  const ok = state.played.size >= need;
  $('nextbtn').disabled = !ok;
  $('gate-note').hidden = ok;
}

function updateProgress() {
  const pct = Math.round((state.idx / state.trials.length) * 100);
  $('progressbar').style.width = pct + '%';
  $('progresstext').textContent = `${state.idx}/${state.trials.length}`;
}

function labelToClip(t, label) {
  const idx = ['X', 'Y', 'Z', 'W'].indexOf(label);
  const clipId = state.order[idx];
  return t.clips.find((c) => c.id === clipId);
}

function submitTrial(ev) {
  ev.preventDefault();
  const t = state.trials[state.idx];
  const fd = new FormData($('rateform'));
  const choice = fd.get('choice');
  if (!choice) return;

  const unblind = {};
  ['X', 'Y'].forEach((label) => { unblind[label] = labelToClip(t, label).condition; });

  const rec = {
    trial_id: t.trial_id,
    input_ref: t.input_ref || null,
    presentation_order: state.order.slice(),
    // what the listener saw:
    choice_label: choice,
    // truth (added at export, never shown during listening):
    choice_condition: choice === 'tie' ? 'tie' : unblind[choice],
    unblind_map: unblind,
    dimensions: {
      X: { clarity: fd.get('x_clarity') || null, leak: fd.get('x_leak') || null, natural: fd.get('x_natural') || null },
      Y: { clarity: fd.get('y_clarity') || null, leak: fd.get('y_leak') || null, natural: fd.get('y_natural') || null },
    },
    tags: {
      X: fd.getAll('tag_X'), Y: fd.getAll('tag_Y'),
    },
    comment: $('comment').value || null,
  };
  // Attach true conditions to dimensions/tags for analysis convenience
  rec.dimensions_by_condition = {};
  rec.tags_by_condition = {};
  ['X', 'Y'].forEach((label) => {
    const cond = unblind[label];
    rec.dimensions_by_condition[cond] = rec.dimensions[label];
    rec.tags_by_condition[cond] = rec.tags[label];
  });

  state.results.push(rec);
  state.idx += 1;
  save();
  updateProgress();
  renderTrial();
}

function finish() {
  $('trial').hidden = true;
  $('progress').hidden = true;
  const wins = {};
  state.results.forEach((r) => {
    const c = r.choice_condition;
    wins[c] = (wins[c] || 0) + 1;
  });
  $('summary').textContent =
    `${state.results.length} trials rated. Wins: ` +
    Object.entries(wins).map(([k, v]) => `${k}: ${v}`).join(', ');
  $('done').hidden = false;
}

function exportResults() {
  const payload = {
    tool: 't255-ab-harness',
    listener: state.listener,
    started_at: state.startedAt,
    exported_at: new Date().toISOString(),
    trial_count: state.trials.length,
    results: state.results,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  a.href = URL.createObjectURL(blob);
  a.download = `ab_results_${state.listener}_${stamp}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function loadExample() {
  fetch('trials.example.json')
    .then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then((m) => {
      const name = prompt('Listener name (for the results file):', 'joseph') || 'anon';
      startSession(m, name.trim() || 'anon');
    })
    .catch((e) => alert('Could not load trials.example.json (serve over http, not file://): ' + e.message));
}

// Manifest validation is also exposed for the Python test harness.
if (typeof module !== 'undefined') module.exports = { validateManifest, shuffle };

$('manifest').addEventListener('change', (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    try {
      const m = JSON.parse(rd.result);
      const name = prompt('Listener name (for the results file):', 'joseph') || 'anon';
      startSession(m, name.trim() || 'anon');
    } catch (e) { alert('Could not parse JSON: ' + e.message); }
  };
  rd.readAsText(f);
});
$('demobtn').addEventListener('click', loadExample);
$('rateform').addEventListener('submit', submitTrial);
$('exportbtn').addEventListener('click', exportResults);
$('restartbtn').addEventListener('click', () => {
  localStorage.removeItem(STORE_KEY);
  location.reload();
});
