// Shared replay controls: the server feeds this socket's real Session, never a second viewer.
import { KIND_AUDIO_IN, KIND_VIDEO_IN, pcm16ToFloat } from './protocol.js';

export function replayControls(state, { startAudio, stopMedia, error }) {
  const button = document.getElementById('replay');
  const badge = document.getElementById('replay-badge');
  const preview = document.getElementById('replay-preview');
  const picker = document.getElementById('replay-clip');
  let imageUrl;
  let pending = false;

  // T-173: replay audio is never shown without the REPLAY badge. While replaying the badge is forced visible
  // (CSS `[data-replaying]` beats any `hidden`), re-shown if anything hides it, and re-asserted on every
  // replayed audio frame, so a prerecorded mix can never pass for live.
  function showBadge() {
    badge.dataset.replaying = 'true';
    if (badge.hidden) badge.hidden = false;
  }
  new MutationObserver(() => { if (state.replaying && badge.hidden) showBadge(); })
    .observe(badge, { attributes: true, attributeFilter: ['hidden'] });

  function clear() {
    state.replaying = false;
    delete badge.dataset.replaying;
    badge.hidden = true;
    button.textContent = 'Replay clip';
    document.getElementById('start').disabled = false;
    preview.hidden = true;
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    imageUrl = null;
    stopMedia();
  }

  function endpoint(path) {
    const url = new URL(state.url);
    url.protocol = url.protocol === 'wss:' ? 'https:' : 'http:';
    url.pathname = path;
    url.search = '';
    return url;
  }

  // Every clip in eval/fixtures/demo: <name>.mp4 and ingested <name>/clip.mp4 (T-090).
  async function listClips() {
    if (!picker) return;
    try {
      const response = await fetch(endpoint('/replay/clips'));
      if (!response.ok) return;
      const { clips, default: fallback } = await response.json();
      const keep = picker.value;
      picker.replaceChildren(...(clips.length ? clips : [{ clip: fallback, label: fallback }]).map((c) => {
        const option = document.createElement('option');
        option.value = c.clip;
        option.textContent = c.label + (c.duration_s ? ` (${Math.round(c.duration_s)} s${c.target ? ', target ' + c.target : ''})` : '');
        return option;
      }));
      if ([...picker.options].some((o) => o.value === keep)) picker.value = keep;
      else if ([...picker.options].some((o) => o.value === fallback)) picker.value = fallback;
    } catch {
      // older server without /replay/clips: keep the default entry
    }
  }

  async function request(action) {
    const url = endpoint('/replay');
    const clip = picker?.value || 'demo-mf-easy/clip.mp4';
    url.search = new URLSearchParams({ session_id: state.sessionId, action, loop: 'true', clip });
    const response = await fetch(url, { method: 'POST' });
    if (!response.ok) {
      const body = await response.json();
      throw new Error(body.detail || 'Replay failed');
    }
  }

  button.addEventListener('click', async () => {
    if (pending) return;
    pending = true;
    button.disabled = true;
    try {
      if (!state.sessionId || state.ws?.readyState !== WebSocket.OPEN) throw new Error('Connect to the server first');
      if (state.replaying) {
        await request('stop');
      } else {
        stopMedia();
        state.replaying = true;
        await startAudio(true); // playback only: never asks for microphone permission
        // Keep the existing overlay/tier rendering over the replay JPEGs.
        state.video = { stream: new MediaStream(), timer: null, clock: { stop() {} } }; // clock: /judge's stopMedia() stops it (T-173)
        document.getElementById('view').classList.add('live');
        document.getElementById('start').disabled = true;
        await request('start');
      }
    } catch (e) {
      clear();
      error(e.message);
    } finally {
      pending = false;
      button.disabled = false;
    }
  });

  return {
    text(msg) {
      if (msg.type === 'session') {
        state.sessionId = msg.session_id;
        listClips();
      }
      if (msg.type === 'replay') {
        if (!msg.active) clear();
        else {
          state.replaying = true;
          showBadge();
          button.textContent = 'Stop replay';
          preview.hidden = false;
        }
      }
      if (msg.type === 'error' && msg.stage === 'replay') error(msg.message);
    },
    binary(frame) {
      if (!state.replaying) return false;
      if (frame.kind === KIND_VIDEO_IN) {
        const previous = imageUrl;
        imageUrl = URL.createObjectURL(new Blob([frame.payload], { type: 'image/jpeg' }));
        preview.src = imageUrl;
        if (previous) URL.revokeObjectURL(previous);
        return true;
      }
      if (frame.kind === KIND_AUDIO_IN) {
        showBadge();
        const samples = pcm16ToFloat(frame.payload);
        state.audio?.node.port.postMessage({ type: 'replay-raw', samples }, [samples.buffer]);
        return true;
      }
      return false;
    },
    disconnect() {
      state.sessionId = null;
      if (state.replaying) clear();
    },
  };
}
