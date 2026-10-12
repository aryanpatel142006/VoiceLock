// Play My Sounds: a small player of calm sounds the person picks themself, outside the SOS frame.
//
// Every sound is generated on this device with WebAudio buffers: no file, no network, no licence to track. Brown and
// white noise and the soft pad reuse the SOS calm loops (sos.js noiseSamples / musicSamples); gentle rain is
// generated here (rainSamples). The repo has no licensed lo-fi or piano recording, so there is none here: the pad
// is labelled "Soft pad" (its note says it is made here).
//
// Rules: mounting plays nothing. The one autoplay is the SOS press itself (comfort.js calls autoplay() inside that
// gesture): the remembered favourite, else gentle rain, at the remembered volume. Every start fades in over 1.5 s
// from silence at a low default volume; every volume change, duck and stop is a ramp, never a jump. Output is this
// page's own AudioContext -> destination only: it never touches the mic, the engine input or session.js's graph.
//
// The favourite sound and the comfortable volume are saved to PUT /comfort/profiles/{id} (merged into the profile
// the server already has, fields favorite_sound / sound_volume) and always to localStorage `hearing.sounds.<id>`.
// After the server echoes the favourite back, window gets `comfort:profile-updated`
// {detail: {profile_id, change: "favorite_sound" | "volume"}}.
//
// Exports: mountSounds(el, {profileId, onState}) -> {stop(), autoplay(), duck(on), offerRemember(id), destroy()};
// stopAllSounds(); offerRemember(id).
// The controller (createSoundsController) holds all the audio and saving logic without a DOM, for node --test.
import { RAIN_LOOP_S, cachedSamples, hasWorker, knownRate, noteRate, prewarmSamples, rainSamples, renderSamples, samplesFor, samplesSync } from "./synth.js";

export { RAIN_LOOP_S, rainSamples, samplesFor }; // the synth moved to synth.js (a Worker runs it); the names stay here

export const FADE_IN_MS = 1500; // every start
export const RAMP_MS = 250; // every volume change
export const STOP_MS = 400; // Stop / Pause / swap
export const SILENCE_MS = 2500; // Return to silence: a slower fade
export const DUCK = 0.25; // while Talk to Me speaks, the sound sits at a quarter of its gain
export const DUCK_MS = 300; // the duck and its release are ramps too
export const DEFAULT_VOLUME = 30; // 0..100, low on purpose
export const MAX_GAIN = 2; // the loops are -30 dBFS RMS; 100 % = 2x = about -24 dBFS RMS
export const STORAGE_PREFIX = "hearing.sounds.";

export const SOUNDS = [
  { id: "brown", name: "Brown noise", note: "Deep, soft rumble.", source: "Generated on this device (sos.js brown noise). No licence needed." },
  { id: "rain", name: "Gentle rain", note: "Filtered hiss with soft drops.", source: "Generated on this device (filtered noise). No licence needed." },
  { id: "white", name: "Soft white noise", note: "Even hush, like a fan.", source: "Generated on this device (sos.js white noise, low-passed). No licence needed." },
  { id: "pad", name: "Soft pad", note: "Slow soft chords, made here.", source: "Generated on this device (sos.js chill music synth). No recording, no licence needed." },
];
const BY_ID = new Map(SOUNDS.map((s) => [s.id, s]));
export const soundName = (id) => BY_ID.get(id)?.name || "Nothing";

/** The profile's avoid list holds Comfort strategy ids or labels (hearing/server/comfort.py records the feedback;
 * comfort_catalog.json maps each sound strategy to its favourite sound). These are the entries that mean each sound. */
const AVOID_NAMES = {
  brown: ["brown", "brown-noise", "brown noise"],
  rain: ["rain", "rain sound"],
  white: ["white", "white-noise", "white noise"],
  pad: ["pad", "lofi", "soft music"],
};
/** The sound ids the profile avoids. */
export function avoidedSounds(avoid) {
  const said = new Set((Array.isArray(avoid) ? avoid : []).map((x) => String(x).toLowerCase()));
  return new Set(Object.keys(AVOID_NAMES).filter((id) => AVOID_NAMES[id].some((n) => said.has(n))));
}
/** SOUNDS in their usual order, with the ones the profile avoids moved last (stable). */
export function orderSounds(avoid) {
  const out = avoidedSounds(avoid);
  return [...SOUNDS.filter((s) => !out.has(s.id)), ...SOUNDS.filter((s) => out.has(s.id))];
}

/** 0..100 slider -> linear gain. Squared so the low half of the slider is fine-grained. */
export function volumeGain(v) {
  const x = Math.min(100, Math.max(0, Number(v) || 0)) / 100;
  return MAX_GAIN * x * x;
}

// ---- local fallback ----
export function loadLocal(storage, profileId) {
  try {
    const raw = storage?.getItem(STORAGE_PREFIX + (profileId || "default"));
    const v = raw ? JSON.parse(raw) : {};
    return {
      favorite: BY_ID.has(v.favorite) ? v.favorite : null,
      volume: Number.isFinite(v.volume) ? Math.min(100, Math.max(0, v.volume)) : null,
    };
  } catch { return { favorite: null, volume: null }; }
}
export function saveLocal(storage, profileId, { favorite, volume }) {
  try { storage?.setItem(STORAGE_PREFIX + (profileId || "default"), JSON.stringify({ favorite, volume })); return true; }
  catch { return false; }
}

const live = new Set(); // every controller on the page, for stopAllSounds()

/**
 * The player without a DOM. `AudioContext` is only constructed on the first play() (a user gesture).
 * onChange(state) runs after every change; state = {current, playing, volume, favorite, saved, avoid}.
 */
export function createSoundsController({ profileId = null, AudioContext: Ctx = globalThis.AudioContext || globalThis.webkitAudioContext,
  fetchImpl = globalThis.fetch, storage = null, win = globalThis, base = "", onChange = () => {} } = {}) {
  const state = { current: null, playing: false, volume: DEFAULT_VOLUME, favorite: null, saved: null, avoid: [] };
  let audio = null; // {ctx, master, buffers}
  let voice = null; // {id, src, gain}
  let waiting = null; // a play() whose loop the worker is still rendering
  let profile = null; // the server's copy, merged into on save
  let volTimer = null;
  let ducked = false; // Talk to Me is speaking: the master sits at DUCK x the volume
  let auto = null; // {id} while an autoplay (the SOS press) waits for load() to bring the server's favourite
  const level = () => volumeGain(state.volume) * (ducked ? DUCK : 1);
  const emit = () => onChange({ ...state });

  function makeGraph() {
    if (audio) return audio;
    if (typeof Ctx !== "function") return null;
    const ctx = new Ctx();
    noteRate(ctx.sampleRate);
    const master = ctx.createGain();
    master.gain.value = 0;
    master.connect(ctx.destination); // this page's output only
    audio = { ctx, master, buffers: new Map() };
    return audio;
  }
  function graph() {
    if (!makeGraph()) return null;
    audio.ctx.resume?.()?.catch?.(() => {});
    return audio;
  }
  function ramp(param, to, ms) {
    const t = audio.ctx.currentTime;
    param.cancelScheduledValues?.(t);
    param.setValueAtTime?.(param.value, t);
    param.linearRampToValueAtTime(to, t + ms / 1000);
  }
  /** The AudioBuffer for `id`, or null while the worker is still rendering it (synth.js: never synthesised on this
   * thread when a worker exists; the pad alone was a 160 ms stall on a click). */
  function buffer(id) {
    if (audio.buffers.has(id)) return audio.buffers.get(id);
    const rate = audio.ctx.sampleRate;
    const x = cachedSamples(id, rate) || (hasWorker() ? null : samplesSync(id, rate));
    if (!x) return null;
    const b = audio.ctx.createBuffer(1, x.length, rate);
    b.copyToChannel ? b.copyToChannel(x, 0) : b.getChannelData(0).set(x);
    audio.buffers.set(id, b);
    return b;
  }
  /** The loop is rendered: start it (or swap to it), fading in. */
  function startVoice(a, id, buf) {
    const g = a.ctx.createGain();
    g.gain.value = 0;
    g.connect(a.master);
    const src = a.ctx.createBufferSource();
    src.buffer = buf;
    src.loop = true;
    src.connect(g);
    src.start();
    const old = voice;
    voice = { id, src, gain: g };
    if (old) endVoice(old, STOP_MS);
    ramp(g.gain, 1, old ? STOP_MS : 1); // a swap cross-fades; a start is faded by the master below
    if (!old) ramp(a.master.gain, level(), FADE_IN_MS); // from silence (master is 0 when stopped)
  }
  function endVoice(v, ms) {
    if (!v) return;
    ramp(v.gain.gain, 0, ms);
    try { v.src.stop(audio.ctx.currentTime + ms / 1000 + 0.05); } catch { /* already stopped */ }
  }

  const api = {
    get state() { return { ...state }; },
    /** Read the remembered favourite and volume (server first, then localStorage). Never starts a sound; while one
     * plays it keeps playing (an autoplay swaps to the server's favourite, nothing else changes the selection). */
    async load() {
      const volBefore = state.volume;
      const local = loadLocal(storage, profileId);
      let fav = local.favorite, vol = local.volume;
      if (profileId && typeof fetchImpl === "function") {
        try {
          const r = await fetchImpl(`${base}/comfort/profiles`, { headers: { accept: "application/json" } });
          if (r.ok) {
            const list = await r.json();
            profile = Array.isArray(list) ? list.find((p) => p?.id === profileId) || null : null;
            if (profile && BY_ID.has(profile.favorite_sound)) fav = profile.favorite_sound;
            if (profile && Number.isFinite(profile.sound_volume)) vol = profile.sound_volume;
            if (profile && Array.isArray(profile.avoid)) state.avoid = [...profile.avoid];
          }
        } catch { /* offline: localStorage */ }
      }
      state.favorite = fav;
      if (state.volume === volBefore) { // the person did not move the slider meanwhile
        state.volume = vol ?? DEFAULT_VOLUME;
        if (audio && voice && state.playing) ramp(audio.master.gain, level(), RAMP_MS);
      }
      const a = auto;
      auto = null;
      if (!state.playing) state.current = fav; // preselected, not playing
      else if (a && fav && state.current === a.id && fav !== a.id) api.play(fav); // the server's favourite: a cross-fade
      emit();
      return { ...state };
    },
    /** Start (or swap to) `id`, fading in over 1.5 s. */
    play(id = state.current || SOUNDS[0].id) {
      if (!BY_ID.has(id)) return false;
      const a = graph();
      if (!a) return false;
      if (voice && voice.id === id && state.playing) return true;
      const buf = buffer(id);
      if (buf) { waiting = null; startVoice(a, id, buf); }
      else { // a cold cache: the worker renders the loop and it starts, still fading in from silence, when it is here
        const token = { id };
        waiting = token;
        renderSamples(id, a.ctx.sampleRate).then(() => {
          if (waiting !== token) return;
          waiting = null;
          if (!audio || !state.playing || state.current !== id || voice?.id === id) return;
          const b = buffer(id);
          if (b) startVoice(audio, id, b);
        }, () => { if (waiting === token) waiting = null; });
      }
      state.current = id;
      state.playing = true;
      emit();
      return true;
    },
    /**
     * The SOS press: start the remembered favourite (else `fallback`, gentle rain) at the remembered volume, fading in.
     * Call it inside the press (pointerdown / keydown) so the browser lets the context run. load() may still swap to
     * the server's favourite. Nothing happens when a sound already plays.
     */
    autoplay(fallback = "rain") {
      if (state.playing) return true;
      const local = loadLocal(storage, profileId);
      const id = local.favorite || (BY_ID.has(fallback) ? fallback : "rain");
      if (local.volume != null) state.volume = local.volume;
      auto = { id };
      return api.play(id);
    },
    /** Talk to Me is speaking (on) or done (off): the sound ramps down to a quarter, then back. */
    duck(on = true) {
      on = Boolean(on);
      if (ducked === on) return;
      ducked = on;
      if (audio && voice && state.playing) ramp(audio.master.gain, level(), DUCK_MS);
    },
    get ducked() { return ducked; },
    /** Perf: open the context (the slow part of a first play) inside an earlier gesture, suspended; play() resumes it. */
    prime() {
      if (audio || !makeGraph()) return false;
      audio.ctx.suspend?.()?.catch?.(() => {});
      return true;
    },
    pause() { api.stop(STOP_MS); },
    toggle(id) {
      if (state.playing && (!id || id === state.current)) api.stop(STOP_MS);
      else api.play(id || state.current || SOUNDS[0].id);
    },
    /** Fade out and end the loop; the selection stays. */
    stop(ms = STOP_MS) {
      waiting = null;
      auto = null;
      if (audio) {
        ramp(audio.master.gain, 0, ms);
        endVoice(voice, ms);
      }
      voice = null;
      state.playing = false;
      emit();
    },
    /** A slower fade, and no sound selected. */
    returnToSilence() {
      api.stop(SILENCE_MS);
      state.current = null;
      emit();
    },
    setVolume(v) {
      state.volume = Math.round(Math.min(100, Math.max(0, Number(v) || 0)));
      if (audio && voice && state.playing) ramp(audio.master.gain, level(), RAMP_MS);
      saveLocal(storage, profileId, { favorite: state.favorite, volume: state.volume });
      emit();
      clearTimeout(volTimer);
      volTimer = setTimeout(() => { api.save("volume"); }, 800);
    },
    async setFavorite(id) {
      state.favorite = state.favorite === id ? null : (BY_ID.has(id) ? id : null);
      emit();
      return api.save("favorite_sound");
    },
    /**
     * Save favourite + volume. Always to localStorage; to the server when there is a profile. `saved` = "server"
     * only when the PUT answered with the same values back, then the window event fires.
     */
    async save(change = "favorite_sound") {
      saveLocal(storage, profileId, { favorite: state.favorite, volume: state.volume });
      let where = "device";
      if (profileId && typeof fetchImpl === "function") {
        try {
          if (!profile) {
            const r = await fetchImpl(`${base}/comfort/profiles`, { headers: { accept: "application/json" } });
            if (r.ok) { const list = await r.json(); profile = Array.isArray(list) ? list.find((p) => p?.id === profileId) || null : null; }
          }
          if (profile) {
            const body = { ...profile, favorite_sound: state.favorite, sound_volume: state.volume };
            const r = await fetchImpl(`${base}/comfort/profiles/${encodeURIComponent(profileId)}`, {
              method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
            });
            if (r.ok) {
              const back = await r.json();
              profile = back;
              if (back && back.favorite_sound === state.favorite && back.sound_volume === state.volume) {
                where = "server";
                try {
                  win?.dispatchEvent?.(new win.CustomEvent("comfort:profile-updated", { detail: { profile_id: profileId, change } }));
                } catch { /* no CustomEvent */ }
              }
            }
          }
        } catch { /* offline: kept on this device */ }
      }
      state.saved = where;
      emit();
      return where;
    },
    destroy() {
      api.stop(STOP_MS);
      clearTimeout(volTimer);
      live.delete(api);
      const a = audio;
      if (a) setTimeout(() => { a.ctx.close?.()?.catch?.(() => {}); }, STOP_MS + 100);
    },
  };
  live.add(api);
  return api;
}

/** Stop every player on the page (SOS, a route change, the integrator's panic button). */
export function stopAllSounds() {
  for (const c of live) c.stop(STOP_MS);
}

const mounted = []; // UI instances, newest last

/** Ask the newest mounted player to offer "Remember <sound> as your favourite?". false when nothing is mounted. */
export function offerRemember(soundId) {
  const m = mounted[mounted.length - 1];
  return m ? m.offerRemember(soundId) : false;
}

/**
 * Mount the player into `el`. Nothing plays on mount: the remembered favourite is preselected at the remembered
 * volume and waits for a press. Returns {stop(), offerRemember(id), destroy(), controller}.
 */
export function mountSounds(el, { profileId = null, doc = el?.ownerDocument || globalThis.document, win = doc?.defaultView || globalThis,
  fetchImpl, AudioContext: Ctx, base = "", onState = null } = {}) {
  if (doc?.head && !doc.querySelector?.('link[data-sounds-css]')) {
    const link = doc.createElement("link");
    link.rel = "stylesheet";
    link.setAttribute("data-sounds-css", "");
    link.href = new URL("./sounds.css", import.meta.url).href;
    doc.head.append(link);
  }
  const storage = (() => { try { return win.localStorage; } catch { return null; } })();
  const h = (tag, attrs = {}, ...kids) => {
    const n = doc.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") n.textContent = v;
      else n.setAttribute(k, v);
    }
    n.append(...kids.filter(Boolean));
    return n;
  };

  const title = h("h2", { class: "snd-title", id: "snd-title", text: "Play my sounds" });
  const intro = h("p", { class: "snd-intro", text: "Calm sounds made on this device. Nothing plays until you press Play." });
  const nowName = h("span", { class: "snd-now-name", text: "Nothing playing" });
  const nowState = h("span", { class: "snd-now-state", role: "status", "aria-live": "polite", text: "" });
  const glow = h("span", { class: "snd-glow", "aria-hidden": "true" });
  const playBtn = h("button", { type: "button", class: "snd-play", "aria-label": "Play", text: "Play" });
  const stopBtn = h("button", { type: "button", class: "snd-stop", text: "Stop" });
  const favNow = h("button", { type: "button", class: "snd-fav-now", "aria-pressed": "false", text: "☆ Favorite" });
  const volOut = h("output", { class: "snd-vol-out", for: "snd-vol" });
  const vol = h("input", { type: "range", id: "snd-vol", class: "snd-vol", min: "0", max: "100", step: "1", "aria-label": "Volume" });
  const volRow = h("div", { class: "snd-vol-row" },
    h("label", { class: "snd-vol-label", for: "snd-vol", text: "Volume" }), vol, volOut);
  const now = h("section", { class: "snd-now", "aria-label": "Now playing" },
    glow, h("div", { class: "snd-now-text" }, nowName, nowState),
    h("div", { class: "snd-controls" }, playBtn, stopBtn, favNow), volRow);
  const rows = SOUNDS.map((s) => {
    const pick = h("button", { type: "button", class: "snd-pick", "data-sound": s.id, "aria-pressed": "false" },
      h("span", { class: "snd-pick-name", text: s.name }), h("span", { class: "snd-pick-note", text: s.note }));
    const fav = h("button", { type: "button", class: "snd-fav", "data-sound": s.id, "aria-pressed": "false", "aria-label": `Favorite: ${s.name}`, text: "☆" });
    return { id: s.id, pick, fav, li: h("li", { class: "snd-row", "data-sound": s.id }, pick, fav) };
  });
  const list = h("ul", { class: "snd-list", "aria-label": "Sounds" }, ...rows.map((r) => r.li));
  const silence = h("button", { type: "button", class: "snd-silence", text: "Return to silence" });
  const offerText = h("p", { class: "snd-offer-text" });
  const offerYes = h("button", { type: "button", class: "snd-offer-yes", text: "Remember it" });
  const offerNo = h("button", { type: "button", class: "snd-offer-no", text: "Not now" });
  const offer = h("div", { class: "snd-offer", role: "group", "aria-label": "Remember this sound", hidden: "" }, offerText,
    h("div", { class: "snd-offer-buttons" }, offerYes, offerNo));
  const saveNote = h("p", { class: "snd-save", role: "status", "aria-live": "polite" });
  const root = h("div", { class: "snd", role: "region", "aria-labelledby": "snd-title" }, title, intro, now, offer, list, silence, saveNote);
  el.append(root);

  let offered = null;
  let order = rows.map((r) => r.id).join(",");
  function render(s) {
    // the sounds this profile avoids sit last; the list is only re-ordered when that changes
    const want = orderSounds(s.avoid).map((x) => x.id);
    if (want.join(",") !== order) {
      order = want.join(",");
      list.append(...want.map((id) => rows.find((r) => r.id === id).li));
    }
    const playing = s.playing;
    root.classList?.toggle("is-playing", playing);
    nowName.textContent = s.current ? soundName(s.current) : "Nothing playing";
    nowState.textContent = playing ? "Playing" : s.current ? "Ready" : "";
    playBtn.textContent = playing ? "Pause" : "Play";
    playBtn.setAttribute("aria-label", playing ? `Pause ${soundName(s.current)}` : `Play ${s.current ? soundName(s.current) : soundName(SOUNDS[0].id)}`);
    const isFav = Boolean(s.current) && s.favorite === s.current;
    favNow.setAttribute("aria-pressed", String(isFav));
    favNow.textContent = isFav ? "★ Favorite" : "☆ Favorite";
    favNow.disabled = !s.current;
    vol.value = String(s.volume);
    volOut.textContent = `${s.volume}%`;
    for (const r of rows) {
      r.pick.setAttribute("aria-pressed", String(playing && s.current === r.id));
      r.li.classList?.toggle("is-current", s.current === r.id);
      const f = s.favorite === r.id;
      r.fav.setAttribute("aria-pressed", String(f));
      r.fav.textContent = f ? "★" : "☆";
    }
    saveNote.textContent = s.saved === "server" ? "Saved to your profile." : s.saved === "device" ? "Saved on this device." : "";
    if (typeof onState === "function") { try { onState(s); } catch { /* the host's own problem */ } }
  }

  const c = createSoundsController({ profileId, storage, win, base,
    ...(fetchImpl ? { fetchImpl } : {}), ...(Ctx ? { AudioContext: Ctx } : {}), onChange: render });
  render(c.state);
  c.load();
  // the loops render in the worker while the page is idle (no AudioContext yet: nothing plays), so Play is instant
  prewarmSamples(SOUNDS.map((s) => s.id), knownRate(), win);

  // the first gesture after the card is up opens the context (suspended); Play then only resumes it
  const prime = () => { win.removeEventListener?.("pointerdown", prime, true); win.removeEventListener?.("keydown", prime, true); c.prime(); };
  win.addEventListener?.("pointerdown", prime, true);
  win.addEventListener?.("keydown", prime, true);
  playBtn.addEventListener("click", () => c.toggle());
  stopBtn.addEventListener("click", () => c.stop());
  silence.addEventListener("click", () => c.returnToSilence());
  favNow.addEventListener("click", () => { if (c.state.current) c.setFavorite(c.state.current); });
  vol.addEventListener("input", () => c.setVolume(vol.value));
  for (const r of rows) {
    r.pick.addEventListener("click", () => c.toggle(r.id));
    r.fav.addEventListener("click", () => c.setFavorite(r.id));
  }
  offerYes.addEventListener("click", () => {
    if (offered) { if (c.state.favorite !== offered) c.setFavorite(offered); }
    offer.hidden = true; offered = null;
  });
  offerNo.addEventListener("click", () => { offer.hidden = true; offered = null; });

  const inst = {
    controller: c,
    stop: () => c.stop(),
    autoplay(fallback) {
      // the SOS press started the sound, so "nothing plays until you press Play" would be false here
      intro.textContent = "Your sound is playing. Pick another, or tap Stay Quiet for silence.";
      return c.autoplay(fallback);
    },
    duck: (on) => c.duck(on),
    offerRemember(soundId) {
      if (!BY_ID.has(soundId) || c.state.favorite === soundId) return false;
      offered = soundId;
      offerText.textContent = `${soundName(soundId)} seemed to help. Remember it as your favorite?`;
      offer.hidden = false;
      return true;
    },
    destroy() {
      win.removeEventListener?.("pointerdown", prime, true);
      win.removeEventListener?.("keydown", prime, true);
      c.destroy();
      const i = mounted.indexOf(inst);
      if (i >= 0) mounted.splice(i, 1);
      root.remove();
    },
  };
  mounted.push(inst);
  return inst;
}
