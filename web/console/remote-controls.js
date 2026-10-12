// Input-only presenter remote probe. No product, audio, or speaker actions.
export function describeInput(e) {
  return { type: e.type, code: e.code || (e.button == null ? 'Unidentified' : `Mouse${e.button}`),
    key: e.key || '', repeat: !!e.repeat, button: e.button ?? null };
}
export function mountRemoteTest(doc = document, win = window) {
  const rows = doc.getElementById('events');
  const signal = doc.getElementById('signal');
  let presses = 0;
  function record(e) {
    const item = describeInput(e);
    const press = (e.type === 'keydown' && !e.repeat) || e.type === 'pointerdown';
    if (press) presses++;
    signal.textContent = `TEST ${presses} · ${item.code} · ${item.type}${item.repeat ? ' · REPEAT' : ''}`;
    const row = doc.createElement('li');
    row.textContent = `${new Date().toLocaleTimeString()} · ${JSON.stringify(item)}`;
    rows.prepend(row);
    while (rows.children.length > 60) rows.lastElementChild.remove();
    if (e.type.startsWith('key') && !e.ctrlKey && !e.metaKey && !e.altKey) e.preventDefault();
  }
  doc.addEventListener('keydown', record);
  doc.addEventListener('keyup', record);
  const pad = doc.getElementById('pad');
  pad.addEventListener('pointerdown', record);
  pad.addEventListener('pointerup', record);
  pad.addEventListener('contextmenu', e => e.preventDefault());
  pad.addEventListener('auxclick', e => e.preventDefault());
  doc.getElementById('clear').addEventListener('click', () => {
    presses = 0; rows.replaceChildren(); signal.textContent = 'READY · press a remote button'; pad.focus();
  });
  function focusStatus() {
    doc.getElementById('focus').textContent = doc.hasFocus() && !doc.hidden
      ? 'PAGE FOCUSED · listening for input' : 'INPUT PAUSED · click this page';
  }
  win.addEventListener('focus', focusStatus);
  win.addEventListener('blur', focusStatus);
  doc.addEventListener('visibilitychange', focusStatus);
  focusStatus(); pad.focus();
}
