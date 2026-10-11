# Contracts

Draft written by the orchestrator before M0. The M0 "Contracts and core types" task owns this file, makes the code match it, and may refine it (say which consumers change).

## http
| Endpoint | Method | Response |
|---|---|---|
| `/health` | GET | `{"ok": true, "device": "mps", "stages": {"dfn": "ready", "tse": "unavailable"}, "warm": "ready", "tiers_config": {"tiers.yaml": "9f983fa9f13e", "tiers-remote.yaml": "a575916270c5"}, "version": "0.1.0"}`; `tiers_config` (T-118): short sha256 of each `config/tiers*.yaml` at start, compared by `demo_regress` with the local file; `warm` (T-088): `off` (`--no-warm`) / `warming` / `ready`, the start-up model preload incl. one enrolled inference (`hearing.server.warm`); `demo.sh --judge` waits for it; `client_devices` (T-133): `null` or `{"client":"judge","input":"RØDE VideoMic GO II","output":"External Headphones","missing":[],"at":1760000000.0}` from the latest hello with a device label, asserted by `scripts/judge_check.py`; `profiles` (T-172): the voice profile names under `$HEARING_PROFILES_DIR` (`[]` when off), printed by `scripts/enroll_profiles.sh`; `tse_window` (T-419): `{"window_s": 1.0, "hop_s": 0.5, "delay_ms": 1500, "env": false}`, the balanced enrolled tier's window, hop and expected delay (window + hop, threaded), moved by `HEARING_TSE_WINDOW_S` / `HEARING_TSE_HOP_S` on the server (only an enrolled `tse` at exactly 1.0 / 0.5 is rewritten; the speed/accuracy presets and `remote.enrolled` keep theirs; the window alone halves into the hop); pages that print a fixed "about 1.5 s behind" should read this; `profile_versions` (T-172, Q-467): `{"aryan": "v2", "joseph": "v1"}`, the version each loads (`<name>/current` names a `v<N>/` dir holding the wavs and `manifest.json`; `null` for a flat pre-T-172 `<name>/*.wav` profile, still loaded) |
| `/` | GET | the debug console (`web/console/index.html`) |
| `/ws/session` | WebSocket | one live session (below) |
| `/judge` | GET | the judge demo page (`web/console/judge.html`, T-088) |
| `/connect` | GET | QR code + links for phones (T-087); the URLs `hearing serve` printed, else the page's own origin |
| `/hearing-ca.crt` | GET | the served TLS certificate (PEM, `application/x-x509-ca-cert`) for the phone trust step; 404 without `--tls` |
| `/sdk/hearing-client.js`, `/console/*` | GET | browser SDK (`web/sdk/`) and the console files it imports |

With `hearing serve --tls` (or `--cert/--key`) every route above is https and the socket is `wss://`;
see `docs/DEMO.md#phones`.

## events
One WebSocket per session. **Binary** messages carry media, **text** messages carry JSON.

Binary frame layout (little endian): `u8 kind` (1 = audio in, 2 = video in, 3 = audio out) · `u32 seq` · `f64 capture_ts` (seconds, client clock) · payload.
- audio in/out payload: PCM16 mono at 16 kHz (configurable via `hello`), 10-40 ms per message.
- video in payload: JPEG, ≤ 640×360, ≤ 25 fps.

JSON messages (`type` field):
| type | direction | shape |
|---|---|---|
| `hello` | client → server | `{"type":"hello","sample_rate":16000,"frame_ms":20,"video":true,"client":"console"}`; `/judge` and `/simple` add `"input_device"` / `"output_device"` (labels of the mic and output in use, `null` after Stop, T-121); the recording keeps them and `session_report` prints `mic: ...`; `"devices_missing":["mic"|"output"]` (T-133) lists saved devices the page could not open (it fell back and shows a banner) |
| `control` | client → server | `{"type":"control","mode":"natural|focus_hq|synthetic|text|quiet|auto|name_call","name":{"text":"Aryan","aliases":[]},"name_call":"release","suppression":0-100,"target":{"lock_track_id":17}|null,"voice_override":"female_adult"|null,"hq_backend":"local|remote|auto","latency_preset":"speed|balanced|accuracy","synth_feed":"auto|mixture|natural|focus","ab":"raw|processed","attention":{"dwell_s":0.5,"hold_s":2.0,"switch_margin":0.04,"tie_margin":0.15,"hold_lock_s":10.0,"preset":"normal"},"agc":true|false}` (`agc`: judge-profile output AGC on/off, T-169) |
| `telemetry` | server → client, ~10 Hz | `{"type":"telemetry","seq":123,"tier":"gate","latency_ms":{"enhance":11.2,"gate":0.4,"total":63.0},"faces":[{"track_id":17,"box":[x,y,w,h],"p_speaking":0.83}],"target":{"track_id":17,"confidence":0.91,"locked":false},"voice":{"bucket":"male_adult","source":"face+voice","override":null},"acquisition":{"state":"isolated","since_s":4.2,"detail":""},"mode":"focus_hq","effective_mode":"focus_hq","pipeline":"tiers","device":"mps","tiers":{"active":"enrolled","reason":"enrolled ready","switches":1,"fading_from":null,"fade":null,"fade_kind":null,"delay_ms":1500,"same_bucket":false,"tiers":[{"name":"enrolled","eligible":true,"reason":"ready","delay_ms":1500,"health":[]}]},"transition":null,"ab":"processed","attention":{"dwell_s":0.5,"hold_s":2.0,"switch_margin":0.04,"tie_margin":0.15},"hq_backend":"auto","latency_preset":"balanced","synth_feed":"auto","synth_feed_request":"auto","hq_active":"local","hq_reason":"...","remote_rtt_ms":null}` (abridged: also `vision`) |
| (telemetry, Focus HQ backend) | server → client | the `telemetry` event also carries `"hq_backend":"auto","hq_active":"remote","hq_reason":"remote healthy, RTT 12 ms < 150 ms","remote_rtt_ms":12.3` (see below) |
| `caption` | server → client | `{"type":"caption","text":"see you at noon","final":true,"t0":12.31,"t1":13.02}` |
| `context` | server → client, 1 Hz | `{"type":"context","noise_db":-28.5,"snr_db":6.1,"speakers":2,"overlap":0.35,"target_conf":0.9,"mode":"natural"}` |
| `recommendation` | server → client | `{"type":"recommendation","mode":"synthetic","confidence":0.78,"reason":"noisy, 3+ speakers"}` |
| `error` | server → client | `{"type":"error","stage":"tse","message":"...","degraded_to":"gate"}` |

### Name Call mode (`mode: "name_call"`, T-120)
Camera-free: whoever says the wearer's name becomes the target. `{"type":"control","mode":"name_call","suppression":100,"name":{"text":"Aryan","aliases":["Ary"]}}`;
`name` (text 1-64 chars, up to 8 aliases) is a partial update and may be sent alone; `{"type":"control","name_call":"release"}`
drops the target (back to listening). Pipeline: `tiers(only="enrolled,audio-only", upgrade_after_s=0.5, min_switch_s=0)`
(`hearing.server.namecall.NAME_CALL_SPEC`, latency preset and isolation applied like Focus HQ): `dfn_ll` until a clip exists,
then the Enrolled tier. A background `NameCaller` cleans the mic with its own `dfn_ll`, transcribes 2.5 s windows every
1 s (1.5 s overlap, faster-whisper `base.en`, word timestamps) and fuzzy-matches the name (case/punctuation-free,
edit distance <= 1, or the same Soundex-like key at distance <= 2). False-trigger guards (T-123):
`name.match` = `strict` (the name or an alias exactly, possessive allowed) | `normal` (default: fuzzy, but never a common
other name the fuzzy rule would take, `namecall.CONFUSABLE`, e.g. "Ryan"/"Brian"/"Orion" for "Aryan", unless it is itself an
alias) | `loose` (the T-120 matcher: "Ariane", "Arian", "Ryan" match "Aryan"); `name.exclude` (up to 16 words, never the
name or an alias) is checked before the fuzzy match at every level. Both are optional partial updates kept by the session.
Whisper gets no name prompt by default (`NameCaller(prompt=True)` passes `initial_prompt` = `hotwords` = "Aryan, Ary.";
measured: it hallucinates the name, 7 false triggers / 5 min, `docs/BENCHMARKS.md` T-123). The prompt mode (`HEARING_NAMECALL_PROMPT=1`
on the server, T-131) always runs with a hit guard: a hit counts only if the window has >= 2 words besides the name, every word of the
name has Whisper probability >= 0.1 (measured, T-131: the hallucinated "Aryan." scores 0.00-0.01, real prompted names 0.05-0.86, so the first-planned 0.5 kept only 3/16 names), its segment `avg_logprob` > -1.0 and `no_speech_prob` < 0.6, and, when voice profiles
(`$HEARING_PROFILES_DIR`, T-125) exist, the caller's ECAPA cosine to one of them is >= 0.45 (`namecall.GUARD_*`). The raw voiced
audio from up to 2 s before the name to the end of its window becomes a calibrated enrollment
(`AutoEnroller.enroll_clip`, the T-085 path); the 6 s before it (if a different voice) the negative. The name from a voice
with ECAPA cosine < 0.45 to the target re-targets (the old clip becomes the negative; the enrolled stage keeps the old clip
playing until the new one is prepared); same voice: the clips merge. The clip grows with matching voiced audio (cosine
>= 0.55) to 5 s. Target not heard (mixture cosine >= 0.4, or the playing enrolled window chose a stream at cosine >= 0.25) for 6 s: `quiet`; for 60 s: released. The camera's enroller/auto-calibration
is not fed in this mode. `hello.video` may be false. Every `telemetry` carries `"name_call"`: `null` until the mode is first
used, else `{"state":"listening|locking|isolated|quiet|off","name":"Aryan","aliases":[],"engine":"whisper|none|custom","match":"normal","exclude":["Ryan"],"prompt":false,"guard":false,"guarded":0,"profiles":0,
"target":true,"pending":false,"clip_s":3.1,"since_heard_s":0.4,"quiet":false,"detections":1,"last":{"word":"Ariane",
"text":"...","latency_ms":420.0,"new_voice":true,"similarity":null},"events":[{"event":"heard|retarget|released",...}]}`.
Consumers: `/judge` and `/simple` (toggle, name field, pill), `scripts/demo_regress.py --modes name_call`, `eval/namecall_eval.py`.

### Judge profile (`control.profile`, T-088)
`{"type":"control","profile":"judge"}` applies, in one message, `mode: focus_hq`, `suppression: 100`,
`isolation: strict`, `synth_feed: auto`, `attention` = the sticky preset, `hq_backend: auto` (only when a
GPU url is configured) and the latency preset for the backend playing (`balanced` on both until T-086
picks the GPU's), and turns on automatic calibration (`hearing.server.judge.AutoCalibrator`): once a
target is selected (attention dwell or a tap) and ASD shows it speaking alone (its `p_speaking` >= 0.5,
every other face < 0.5, gaps < 0.3 s tolerated) for 1 s, the session opens the T-085 calibration window
for 2 s (`AUTO_CALIBRATE_S`; the manual C/`control.enroll` window stays 3 s), at most 3 attempts per
target. The judge session's `tiers` ladder moves up after 0.5 s of eligibility (`upgrade_after_s=0.5`,
the enrolled hop; `config/tiers.yaml` keeps 1 s for everyone else). Measured on the replay clip
(demo-mf-easy, laptop-a): lock -> Isolated **5.0-5.2 s** (3 runs). Other fields in the same message override the profile. Under the profile a `mode: natural` sent
without `suppression` (the page's camera-lost fallback) plays at **80** (`JUDGE_NATURAL_SUPPRESSION`, T-103; plain `dfn_ll`). Every
`telemetry` gains `"profile": "judge"|null` and `"auto_calibrate": {"state": "off|waiting|calibrating|done|gave_up",
"solo_s": 0.4, "attempts": 1}`. Unknown profiles are a `protocol` error. Consumer: `/judge` (`web/console/judge.*`).
T-135: under the profile the session's output goes through `hearing.audio.agc` (not in a session whose `hello.client` is `remote-stage`, the GPU end of Focus HQ: the laptop levels once, T-129) (`agc` stage: -20 dBFS, at most +12 / -20 dB,
envelope 50 ms attack / 800 ms release, held below -50 dBFS or 12 dB under the envelope (pauses), soft limiter above 0.8) and every `telemetry` carries
`"agc": {"gain_db": 6.2, "pause_db": 0.0, "level_db": -26.1, "target_db": -20.0}` (`null` without the profile or with `control.agc: false`);
T-169: blocks within 10 dB of the noise floor (10th percentile of the last 1.5 s of 10 ms block levels) hold the envelope too, and after
300 ms of held blocks the applied gain relaxes to at most 0 dB (`pause_db`, 150 ms; back in 20 ms after 30 ms of speech), so room noise
and pauses are never raised. T-179 floor rule, opt-in (`HEARING_AGC_FLOOR_RULE=1` on the server; off by default pending review, Q-485): a block under the
effective floor `max(-50 dBFS, noise floor + 10 dB)` never raises the applied gain (it holds or moves back toward 0 dB) and the gain is capped at +9 dB;
telemetry `agc` gains `applied_db` (`gain_db + pause_db`, after that clamp when on; what plays) and `floor_rule` (bool; `/judge` then runs the same rule
on its RAW side); `session_score` reports
`pause_lift_db` (out.wav minus out_pre_agc.wav over target-silent 20 ms frames, bar <= +3 dB) and `speech_level_db` (out.wav over target-active frames, energy mean), plus `si_sdri_pre_agc_same_crop_db` (out_pre_agc.wav on out.wav's own delay-aligned crop), `speech_level_pre_agc_db` and `speech_level_p90_db` (T-179, Q-485). `{"type":"control","agc":false}` turns the output AGC off for the session (`true` back on): the `remote`
stage sends it with every control it forwards, so on the Focus HQ remote path only the laptop levels (the remote AGC had raised quiet
room windows +12 dB before the laptop's, live session 20261009T203531Z-2); a remote older than T-169 ignores the field (restart it on new code); `/judge` levels its raw A/B side
with the same rule (worklet message `{"type":"agc","on":true}`); with `HEARING_SESSION_LOG` the tap also writes `out_pre_agc.wav`
(16 kHz sessions), and `session_score`/`demo_regress` report `si_sdri_pre_agc_db` ("separation SI-SDRi, before output leveling",
the value demo_regress judges regressions on) next to `si_sdri_db` and every other number measured on `out.wav`. The `tse` stages
keep the adaptive hop until the acquisition first reads `locked` (isolated), then the session pins each at its current hop
(`FocusHQ.pin_hop()`, health `"hop_pinned": true`; Q-341); `HEARING_JUDGE_ADAPT_HOP=0` pins it from Start instead
(`tiers(adapt_hop=false)` passes it down; no `HOP_STEPS` widening, so `hop_changes` stays 0). The profile's
ladders also promote at most every **3 s** (`min_switch_s=3`, `JUDGE_MIN_SWITCH_S`; `config/tiers.yaml` keeps 10 s): with 10 s
the audio-only -> gate step ~1 s after Start held the calibrated Enrolled tier back until ~11 s. After the calibration
window, until the ladder isolates, the pill reads "Calibrated — locking on…" (state `calibrating`, not back to "speak").
`/judge` keeps Focus HQ when the camera ends while the Enrolled tier is healthy (pill "Camera lost — voice profile still
isolating"; otherwise `mode: natural` as before), plays Focus HQ through a 120 ms jitter buffer (Natural 40 ms), and
takes C/R/N only with Shift or the drawer open.
**Team preset (T-130):** with `HEARING_DEMO_PRESET=team` (`scripts/demo.sh --judge --team`) every `profile: "judge"`
also applies `config/demo-team.yaml` (`hearing.server.team`); `profile: "team"` asks for it without the env. The session
still reports `"profile": "judge"`; every `telemetry` gains `"preset": null` (off) or `{"name": "team", "pins": {...}}`
(the HQ backend: `local` by default, never probed, `pins.hq_reason` = `"Mac (measured: beats the A100 with profiles, T-129)"`
(T-181: `hq_backend: local` in the YAML); only under `HEARING_HQ_BACKEND=remote` (`demo.sh --judge --team --gpu`, the env
wins over the YAML key) pinned per server from 5 `/health` probes, a failed one retried once: >= 3/5 answered and their RTT p95
< 60 ms -> `remote` else `local`; under it a local pin caused by the GPU is re-probed every 10 s and `pins.hq_backend` moves to
`remote` at the next safe point, before Isolated or after a 0.3 s pause, with `pins.hq_reason` saying so (T-154); same-bucket
guard; `isolation` standard over the judge profile's strict (T-181, T-129 Q-470; `control.isolation` or the drawer's
Isolation picker still selects strict); lock on Start + `hold_lock_s` 60; Name Call name/aliases/`strict`/quiet 10 s/release 180 s; `voice_override`;
ladder `min_switch_s` 3; `synth_feed` focus; caption ASR env `HEARING_ASR`/`HEARING_ASR_TEMPERATURE`; the preset sets no `caption_prompt` since T-139, so no
`HEARING_ASR_PROMPT`/`HEARING_ASR_VOCAB`: a preset that sets `caption_prompt` also gets `HEARING_ASR_VOCAB` = the prompt's words for T-136's guard).
Consumers: `/judge` drawer (`TEAM PRESET` row), `scripts/judge_check.py --team`.

### Auto mode and `recommendation` (T-076)
`mode: "auto"` plays one of `natural` / `focus_hq` / `synthetic` (telemetry `effective_mode`),
re-decided once per `context` tick (1 s, `hearing.context.prefs.Advisor`):
- **Model** in charge when its pick has confidence >= `min_conf` (0.6): it switches after the
  same pick for `stable_s` (3) ticks and at most every `auto_cooldown_s` (10 s).
- **Rule** otherwise (the `synth_feed` rule): `speakers >= 2` -> `focus_hq` at once; after 5
  ticks in a row below 2 -> `natural` (any other mode is held until then). Auto is never dumber
  than the rule.
- Every Auto switch, and the first tick after entering Auto, sends a `recommendation` with
  `"auto": true` and `"source": "rule"|"model"`; rule events carry `confidence: 1.0` and a
  reason like `"rule: 2 talkers"` / `"rule: fewer than 2 talkers"`. Toasts outside Auto are
  unchanged (`"auto": false`, `"source"` absent). Consumers: `web/console` (shows auto toasts as
  "OK" only, unchanged), phone app (may show `source`), `scripts/session_report.py`.

### Focus HQ backend (`hq_backend`)
Where Focus HQ runs. `control` messages are partial updates: send `hq_backend` alone to change it.
- `local`: the session's configured `focus_hq` pipeline (default `tiers`) on this machine.
- `remote`: the same mode on the GPU session server at `HEARING_REMOTE_URL`, through the `remote`
  stage, whose fallback is the local `focus_hq` pipeline. A remote that is unreachable,
  disconnected or silent for 500 ms is replaced by the local pipeline inside the stage (one-hop
  crossfade) within 1 s; it returns on its own when the remote answers again.
  While remote audio plays, telemetry `tier` stays `focus_hq` and `tiers` is the **remote**
  session's ladder (T-108): `active` is the remote's playing tier (e.g. `enrolled`), `delay_ms` is
  what the stream carries (200 ms playout buffer + the remote's own delay; the RTT sits inside the
  buffer), plus `"remote": true`, `remote_delay_ms` and `buffer_ms`. The session forwards its
  `latency_preset` to the remote as is (T-086; was `balanced` as `speed`): the GPU ladder `config/tiers-remote.yaml` runs its balanced enrolled tier at 0.5 s / 0.125 s (625 ms).
- `auto`: `remote` while the remote's `GET /health` is green (`200`, `"ok": true`, checked every
  1 s) **and** the RTT is below **150 ms**, else `local`. RTT = the audio stream's p50 round trip
  while the remote plays, else the median `/health` round trip. Local is chosen at the first bad
  check; the remote is chosen once it has been good for 2 s (`ServerConfig.hq_auto_upgrade_s`).

The default is `HEARING_HQ_BACKEND` or `auto`. With no `HEARING_REMOTE_URL` (or a `focus_hq`
pipeline that is itself `remote(...)`) every value plays local, and asking for `remote`/`auto`
answers an `error` with `"stage":"hq_backend","degraded_to":"local"`. Unknown values are a
`protocol` error. A backend change is an ordinary make-before-break pipeline switch: the old
backend plays until the new one has filled its delay, then a 100 ms fade, so audio never
drops out. Every pipeline switch fades like the tier ladder (T-078, `hearing.fusion.tiers.fade_kind`
on the jump in delay): equal delays crossfade, a jump up to 500 ms is an equal-power splice, a
larger one dips (old out over 50 ms, then new in) so two different moments never overlap. A new
pipeline holding a `tiers` ladder also keeps the old one playing, up to 6 s
(`SWITCH_MATCH_S`), until it plays an isolating tier (`enrolled`/`focus_hq`) whenever the old one
did. `telemetry.transition` gains `fade_kind` (`null` while priming, then
`aligned`/`splice`/`dip`). While a `remote` stage plays, `latency_ms.algorithmic` (and `total`)
is the playout buffer `delay_ms` **plus** the remote session's own `latency_ms.algorithmic`
(e.g. 200 + 750 ms for the GPU enrolled spec); `extras.remote.remote_algorithmic_ms` shows the
second part.

### Quiet (`mode: "quiet"`, T-149)
Quiet = **no audio out, captions only from the selected person** (Focus = hear that person). The session
plays the very pipeline Focus HQ plays (same spec, same instance: judge/team preset, latency preset,
isolation and the remote backend included), so the target, voice profile, enrollment and the ladder's
isolating tier survive Focus <-> Quiet toggles without a reset or re-enrollment. The output is ramped to
digital zeros over 20 ms (`QUIET_FADE_MS`; zeros from the first sample when Quiet is the first mode), so a
crossfade or a priming pipeline never leaks into Quiet. The ladder's output feeds a session-owned
captions-only `synth(speak=false)` (the Text ASR, `HEARING_ASR`, T-139 guards, no caption prompt) **only
while the playing tier isolates the target** (`enrolled` / `focus_hq`); otherwise the ASR hears silence
(never the room) and the captions pause. `synth_feed` reads `focus` in Quiet whatever was requested.
Every `telemetry` carries `"quiet"`: `null` outside Quiet, else
`{"feed":"focus|paused","tier":"enrolled","reason":"captions from the enrolled tier"}` (`reason` e.g.
`"not isolating the selected person yet (gate tier): captions paused"`, `"Focus chain failed (playing 'dfn_ll'): captions paused"`).
A `focus -> paused` change also sends `{"type":"error","stage":"quiet","message":<reason>,"degraded_to":"paused"}`
in the same reply, so the UI shows a degrade within one audio frame (telemetry within 0.1 s). Entering or
leaving Quiet resets its synth (a phrase from before never captions). `pipelines={"quiet": ...}` /
`HEARING_PIPELINES="quiet=silence"` keeps the old behaviour (that spec plays, no captions, `quiet: null`).
Measured: `docs/BENCHMARKS.md` T-149 (`eval/quiet_toggle.py`). Consumers: `/judge`, `/simple`, console (`mode` button), phone app.

### Synthetic/Text ASR feed (`synth_feed`, T-074, T-079)
What the ASR of Synthetic and Text hears; a partial `control` update independent of `mode`.
| value | chain before `synth` | use |
|---|---|---|
| `mixture` | `passthrough`: the raw room, also when a natural chain is configured (T-079) | one talker |
| `natural` | the `natural` pipeline (`dfn_ll`, T-103; or the configured one, e.g. `natural_gate`) | noise, one talker (measured slower, no WER gain) |
| `focus` | the `focus_hq` ladder (`tiers`, with the `latency_preset`): whichever tier plays, gate / enrolled / AV | two or more talkers |
| `auto` (default) | `mixture`; `focus` once `context.speakers >= 2` for 2 consecutive seconds (a 1 s blip does not flip), back to `mixture` after 5 s below 2; `mixture` when the ladder failed to build | |
Default `auto`, or `HEARING_SYNTH_FEED` / `ServerConfig.synth_feed`. A feed change (asked or by
`auto`) swaps only the chain in front of `synth`: the synth stage instance moves to the new chain
with its ASR state, queued phrases and pending captions, nothing is dropped; the new upstream
starts from fresh state. A **mode** change (Synthetic <-> Text, or leaving them) still resets synth.
An explicit `pipelines` override for `synthetic`/`text` (e.g. `--pipelines "synthetic=..."`) is
used as given and **disables** `synth_feed` for that mode (`ServerConfig.synth_explicit`); the
demo serve line therefore names only `natural=`. Unknown values answer a `protocol` error.
Every `telemetry` carries `"synth_feed"` (the feed in effect, never `auto`) and
`"synth_feed_request"` (what was asked, may be `auto`). `latency_ms.synth_phrase_p50` is the
median of ASR phrase end -> first TTS sample. The local Whisper ASR now also ends a phrase at an
energy dip after 1.8 s and hard at 3 s (`HEARING_ASR_MAX_PHRASE_S`, was 8 s), and Kokoro speaks
sentence by sentence. Consumers: `web/console` (no selector yet), phone app.

**Judge pin (T-136).** Under `control.profile: "judge"` a requested `auto` feed is pinned by the
server's device instead of the speaker count: `focus` on a CUDA session server (the A100), `mixture`
everywhere else (the Mac). An explicit feed still wins.

**Caption guards (T-136, `hearing.synth.asr`).** The local Whisper engine gets no sentence prompt; the
only priming is `$HEARING_ASR_VOCAB` (comma-separated names/domain words, entries over 3 words refused),
passed as `hotwords`. A Whisper segment with `no_speech_prob >= 0.6` and `avg_logprob < -1.0` is dropped;
a repetition loop (an n-gram, n <= 6, 3+ times in a row) is kept once; a final made only of vocabulary
words or a stock phrase ("thank you", "thanks for watching", "you", ...) while the phrase's mean RMS is
under 2x the room floor (10th percentile of the last 5 s of 10 ms frame RMS) is dropped. The synth
stage drops a final repeating the previous one within 3 s (any engine). Consumer: captions in every UI.

### Focus HQ latency preset (`latency_preset`)
`speed` | `balanced` (default) | `accuracy`, independent of `mode` and `hq_backend`; a partial
`control` update. It sets the window/hop of the **enrolled** tier of the `tiers` ladder
(`hearing/latency_preset.py`; `tiers(preset=...)`): speed 0.5 s / 0.25 s, accuracy 1.5 s / 0.75 s,
and **balanced = the configured windows, untouched** (T-078): 1.0 s / 0.5 s on the Mac ladder
(`config/tiers.yaml`), 0.5 s / 0.125 s for the GPU box's `remote.enrolled` (T-086). The AV tier, gate and
audio-only are unchanged. The enrolled tier's algorithmic playout delay is window + hop
(0.75 / 1.5 / 2.25 s on a threaded stage); this excludes capture/output buffers, transport,
and overruns. See `docs/BENCHMARKS.md` for SI-SDRi and compute per window. A change is an
ordinary make-before-break pipeline switch (above: the old ladder plays until the new one's
enrolled tier is eligible, then a dip). **Enrollment survives it**: the session owns one
`AutoEnroller` (`ctx.telemetry.extras["enroller"]`), fed every hop in every mode, and every
enrolled `tse` stage uses it, so a preset click or a Focus HQ ↔ Synthetic/Text round trip never
goes back to "Acquiring voice". Unknown values answer an `error`
through control validation. Every `telemetry` carries `latency_preset`. It applies to the
local ladder and direct enrolled TSE pipeline. Remote stages forward the preset as a partial
control update, so the GPU session applies it to its ladder or direct enrolled pipeline too.
The remote client's configured jitter/playout buffer remains fixed; preset changes do not
shorten that buffer. Other remote pipelines (e.g. AV-only) remain unchanged.

### Enrolled tier under load: adaptive hop and compute headroom (T-089)
Every threaded `tse` stage (the Mac ladder's enrolled tier says `threaded=true` explicitly: a slow
window is late or skipped, never blocks the 10 ms loop) widens its hop when the machine is too slow:
**2 of the last 4 windows overrun or late** -> hop x1.5 -> x2, capped at the window (0.5 -> 0.75 ->
1.0 s for the 1 s window), and steps back one rung after **10 clean windows** whose compute p95
fits in 80 % of the smaller hop. The playout delay is window + hop, so it moves with the hop
(1.5 -> 1.75 -> 2.0 s): the output dips over 10 ms, then stalls or skips the difference. One
overrun among the last 4 is tolerated (the neighbouring windows cover it); at the widest hop 3 of 4
make the stage unhealthy. Any other skip (`enrolling`, `no lips`, a model error) makes it unhealthy
at once, and an unhealthy stage is healthy again after **N = 4 consecutive clean windows**
(`hearing.tse.stage.RECOVER_WINDOWS`). The first computed window after a reset is cold (MPS
kernels, enrollment embeddings): windows it blocks read `warming up` and it is left out of the
compute stats. `tse(adapt_hop=false)` turns the adaptation off.

T-178: each `tse` health report also carries `contrast` (the enrolled pick floor, `null` for other candidates), `pair_cos`
(cos(target, negative enrollment) of the last window, `null` without a negative) and `negative_active` (`false` when
`pair_cos` > `SAME_VOICE` 0.7 dropped the negative and the LDA); `/judge` shows them in the drawer next to the confidence.
Each `tse` health report (`telemetry.tiers.tiers[].health[]`, `extras.tse`) gains `hop_s` (current),
`hop_base_s` (configured), `hop_changes`, `utilisation` (= compute p50 / current hop, over the
last 16 windows); `compute_p50_ms` / `compute_p95_ms` are now over those 16 windows and `delay_ms`
follows the hop. `telemetry.tiers.compute_headroom` = 1 - `utilisation` of the slowest running
tier (T-116: idle tiers are suspended, see `suspended` below), `null` until one has timed a window. The ladder's
name lists the tiers built on this host (e.g. `tiers(enrolled,gate,audio-only)` on the Mac), and
its health, eligibility and `telemetry.tiers` status are recomputed at the telemetry rate (every
100 ms of audio, and on every switch or fade), not every 10 ms hop. `scripts/demo.sh --check`
(`python -m hearing.tse.headroom`) times the enrolled tier on this machine and exits 1 below 30 %
headroom. Consumers: `web/console` (diagnostics row "Compute headroom"), phone app (may show it),
`scripts/session_report.py` (unchanged).

### Telemetry `tiers` and `ab`
`tiers` is `null` unless the active pipeline holds a `tiers` ladder (`hearing.fusion.tiers.status()`); otherwise:
| field | meaning |
|---|---|
| `active` | name of the tier playing now |
| `reason` | why it was picked, for humans |
| `switches` | number of tier switches since the ladder was built |
| `fading_from` | tier being crossfaded out, else `null` |
| `fade` | crossfade progress 0-1, `null` when not fading |
| `fade_kind` | kind of the running crossfade (`hearing.fusion.tiers.fade_kind`), `null` when not fading |
| `delay_ms` | the playing tier's own delay |
| `same_bucket` | `true` when the target and another recently seen face share a voice group (Enrolled is skipped) |
| `compute_headroom` | 1 - `utilisation` of the slowest running tier, `null` until a window is timed (T-089, above) |
| `hop_ms` | `{"p50","p95"}` the ladder's own compute per 10 ms hop (every running tier, priming included) over the last 500 chunks, `null` before the first (T-116) |
| `suspended` | names of the built tiers not running now (T-116): a tier runs while it plays, fades out, is better than the playing tier, is the fallback below it (first eligible lower tier, else the last), needs `healthy`, holds a `remote` stage or says `idle: run` (config or `HEARING_TIERS_IDLE=run`); a suspended tier replays the input it missed (at most `prime_s`, 0.5 s) before it runs again, so every switch stays sample aligned |
| `tiers[]` | one row per tier in order, then unavailable ones: `name`, `eligible` (bool), `reason` (`unavailable: ...` when not built on this host), `delay_ms` (`null` when unavailable), `health` (list of per-stage health dicts, non-finite floats as `null`); built tiers also carry `cooling_s` (seconds before an overrun tier may be promoted again, 0 = not cooling) (T-095), `running` (bool) and `primes` (times it was re-primed after a suspension) (T-116) |

`ab` (`control.ab`, `raw` or `processed`, default `processed`) is what the listener hears in the client-side
Raw | Selective hearing toggle. The server only stores it and echoes it in every `telemetry`; it does not
change the audio it sends. Consumers: `/simple`, `web/console`.

Telemetry fields (every `telemetry` event):
| field | meaning |
|---|---|
| `hq_backend` | the requested backend, `local` / `remote` / `auto` |
| `hq_active` | what Focus HQ plays now (or would play if Focus HQ were on): `local` / `remote` |
| `hq_reason` | why, for humans (e.g. `remote down, local fallback (...)`) |
| `remote_rtt_ms` | RTT in ms as above, `null` when not measured or the remote is down |
| `remote` | T-156: while the remote Focus HQ pipeline is the session's pipeline, its `remote` stage's stats, else `null`: `source` (`remote` / `delayed` / `local`: what played on the last hop), `remote_tier` (the remote ladder's playing tier), `connected`, `rtt_ms` `{p50,p95,max,n}`, `remote_hops`, `misses`, `sent`/`received`/`dropped`/`connects`, `delay_ms` (playout buffer), `remote_algorithmic_ms`, `latency_ms` (what the stage adds), `remote_latency_ms` + `remote_device` (from the remote's telemetry), `last_error`, `url`, `session`: the remote session's own `target` / `acquisition` / `tse` / `enrollment`, and (T-170) `git_rev` (the GPU's `/health` `git_rev`, last read by the health monitor or a probe; `null` before one) and `rev_warning` (`"A100 rev <x> is behind local <y>: rerun gpu_up"` when this checkout is a git repo and its HEAD is not an ancestor of the GPU's rev, `git merge-base --is-ancestor`; else `null`). The laptop's `/health` carries the same as `remote_git_rev` / `remote_rev_warning`, the team preset's pins as `remote_git_rev` / `rev_warning` (drawer), and `demo.sh --judge`'s JUDGE line appends the warning |

`latency_ms` then also carries `remote_stage` (the stage's delay: buffer + remote algorithmic while the
GPU plays, the fallback's while local) and `remote_rtt` (p50). While the GPU's audio plays
(`remote.source != "local"`) `target.locked`/`target.confidence`, `acquisition` and `tse.confidence` are
the **remote** session's (each marked `"source": "remote"`; `target.track_id` stays the local id), so the
/judge pill shows what the A100 isolates (`… · iLab GPU · enrolled`); the local values return with the
fallback. `hq_reason` says `remote selected; no microphone audio from the client yet` until the first
audio frame arrives (`seq` is `null` until then).

While the remote plays, the laptop forwards every camera JPEG (`video in`, kind 2) to the remote
session (its `hello` says `"video": true`) so the GPU runs vision and the audio-visual model on the
same frames. Track ids are numbered per server: a locked target is sent as the remote track whose
box overlaps the local target's box most (IoU >= 0.3, from the remote's `telemetry.faces`), held
back until one does; with no local camera the local id is sent as is.

Consumers: `web/console` (selector next to the Focus HQ toggle), the phone app (send
`hq_backend`, show `hq_active` + `remote_rtt_ms`), `hearing.server.protocol.Control`.

### Attention (`control.attention`)
Sticky attention: how readily the target switches or is dropped. Partial update, applied live to the
session's `TargetSelector`; every `telemetry` event echoes the current values as `"attention":{"dwell_s","hold_s","switch_margin","tie_margin","hold_lock_s"}`.
`tie_margin` (default 0.15, 0-0.5, T-101): at acquisition only, when the best candidates are within this centre-cost of each other
(both off-centre), the face with the higher mean `p_speaking` over the last 1.5 s (lead >= 0.2) wins, and acquisition then waits 1.5 s
for that evidence; a lock and switching (dwell/hold/`switch_margin`) are untouched.
| field | range | default | meaning |
|---|---|---|---|
| `dwell_s` | 0.1-3.0 | 0.5 | a challenger must beat the target for this long before the target switches |
| `hold_s` | 0.2-10 | **2.0** | a target whose face is missing (glance away, looking down, occlusion) is kept this long before it is dropped. Raised from 1.0 because 1-1.5 s lapses are normal and handed the target to the other talker |
| `switch_margin` | 0-0.5 | 0.04 | how much closer to centre the challenger must be (higher = stickier) |
| `hold_lock_s` | 1-60 | 10.0 | T-107: an explicit lock (`control.target.lock_track_id`) whose face is missing stays `reacquiring` (telemetry keeps its `track_id`, `locked: true`) this long; see below |
| `preset` | `glance` / `normal` / `sticky` | none | label only, written to the prefs log with the values |

**Explicit lock (T-107).** A lock never falls back to free selection and is released only by
`control.target: null` or another lock: no challenger, no `hold_s` drop. While its face is missing the
lock is re-attached to a new track whose box centre is within 0.15 (normalised, `REATTACH_DIST`) of the
face's last box (a re-detection with a new id, or the tracker restarting ids after a vision stall); a track
that was visible next to the locked face never inherits it, and the re-attach keeps `switch_seq` (the
enrolled voice is kept). After `hold_lock_s` the lock is lost: `target` is `{"track_id":null,"locked":true}`,
`acquisition` is `searching` with detail `lost the person you locked — look at them again`, and the same
face coming back at that position re-attaches. A vision stall (no frame applied for 1 s) keeps the lock
with `track_id: null` until the first frame re-attaches it. Measured cause (session
20261006T223844Z-1): vision went dark under load at 2.2 s, the selector was reset, and free selection
then took the other, more central face for the rest of the run.

Presets: Glance 0.3 / 1.0 / 0.02, Normal 0.5 / 2.0 / 0.04, Sticky 1.0 / 4.0 / 0.10 (all `hold_lock_s` 10). Out-of-range values are a `protocol` error. Each change is appended to the prefs log
(`{"kind":"attention",...}` lines, skipped by the mode recommender). Consumers: `web/console` (Glance / Normal / Sticky buttons), phone app.

### Local prerecorded replay (T-066)

`POST /replay?session_id=<id>&clip=grid-two-face.mp4&action=start&loop=false`
starts a local feeder into the **existing** WebSocket Session. `action=stop`
cancels it; a second start replaces it. `clip` must be a single `.mp4` filename
under `eval/fixtures/demo/` (`ServerConfig.replay_dir` can override that directory).
Returns `{ "active": true, "clip": "grid-two-face.mp4" }` on start,
`{ "active": false }` on stop. Invalid names/actions/undecodable clips return 400;
missing clips or disconnected sessions return 404. The endpoint never downloads.

Opt in with `/ws/session?replay=1` to receive the initial
`{ "type": "session", "session_id": "<opaque id>" }` event. Ordinary sockets
keep their existing initial-message behaviour. Both bundled pages opt in.
The session ID expires when its socket closes; each page controls only its own
session. Replay sends hello (16 kHz, 20 ms, video true) and an empty control
update, preserving the page's mode, suppression, target lock and voice choice.
Live binary input is ignored while replay is active; text controls remain active.
Frames pass through the same `Session.handle_binary` and processing stages.

Server → page: `{ "type": "replay", "active": true|false, "clip": "..." }`
shows/hides the REPLAY badge. During replay only, the server also echoes input
binary kinds **1** (raw PCM for A/B) and **2** (JPEG preview) on the same socket.
Kind **3** remains the processed output, with ordinary telemetry/caption/error
events. Capture timestamps share an epoch clock, advance by clip time, and remain
monotonic when looping; sequence numbers count **per media kind** (audio 0, 1, 2, ...; video
0, 1, 2, ...), like the browser clients, and keep increasing across loops (T-072: one shared
counter made every video frame look like a lost audio frame in `scripts/session_report.py`).
PCM frames are 320 samples (20 ms), final partial frames zero-padded; JPEGs are
25 fps and fit inside 640×360. Deadlines use a monotonic clock, avoiding cumulative
processing drift. The last common A/V duration bounds replay.

`HEARING_SESSION_LOG=/path/taps` enables main's per-session wire recorder:
each session gets a directory with `events.jsonl`, `in.wav`, `out.wav`, and sampled
JPEGs under `frames/`. Replay inputs are recorded once before processing; their
preview echoes are excluded from recording so the input WAV is not doubled.
Outgoing telemetry omits the voice field. Consumers changed: session server,
both console pages, shared audio worklet, and `scripts/replay.py`.

### Manual voice calibration (`control.enroll`) and isolation (`control.isolation`) (T-085)

`{"type":"control","enroll":{"action":"start","seconds":3}}`: the session's shared enroller
(`AutoEnroller.calibrate`) keeps **every voiced 10 ms chunk** (RMS > 0.01) of the next `seconds`
(1-10, default 3) of room audio as the target's enrollment, whatever ASD says: the wearer asked the
target to speak alone. At the end of the window the clip (all the voiced audio, up to `seconds`) is
ready at once if it holds >= 1 s of speech (`calibrated`), else the window is dropped (`failed`) and
automatic collection carries on. A calibrated clip is bound to the current target (or the first one
seen), follows a re-detected face (new track id, no deliberate switch, the calibrated face no longer
in view), and is dropped only by
`{"enroll":{"action":"clear"}}` or a target switch (lock / challenger win, or a new target while the
calibrated face is still in view; it then becomes the negative enrollment). While calibrating, the previous clip keeps playing. Ready <= 3.5 s from the
press (window + one 20 ms frame; `tests/test_calibration.py`). Unknown actions or `seconds` out of
range are a `protocol` error.

`{"type":"control","isolation":"standard"|"strict"}` (default `standard`): `strict` adds
`pause_floor=0.25, pause_p=0.5, pause_action="duck"` (`hearing.latency_preset.STRICT_KWARGS`) to every
enrolled `tse` stage and `isolation="strict"` to the `tiers` ladder (`apply_isolation`). A window where
no separated stream reaches cos 0.25 to the target while the target's ASD is below 0.5 plays silence
(the mixer plays only the attenuated room): fewer wrong-person seconds, slightly more dropouts
(`docs/decisions/association.md`, T-085 addendum). A change is a make-before-break pipeline switch;
the enrollment is kept (shared enroller). The embedding margin (`contrast`) is deliberately **not**
part of strict: on the scenes measured it cost 0.01-0.04 accuracy and cut no wrong-person audio. The
same-gender guard (T-070, `avoid_when: [same_bucket]`) still skips the Enrolled tier, calibrated or
not, because AV Focus HQ leaks less than calibrated + strict Enrolled on same-gender pairs.

Telemetry: every event carries `"isolation"` and
`"enrollment":{"state":"calibrating|calibrated|ready|collecting|failed","collected_s","calibrate_elapsed_s","calibrate_s"}` plus (T-125) `"source":"profile|calibrated|auto"|null` (null while no clip plays) and `"profile":"<name>"|null`.
`acquisition` says `acquiring` / `calibrating 1.2/3.0 s` while the window is open (before every other
rule) and `locked` / `calibrated` once the enrolled tier plays a calibrated clip. Consumers:
`web/console` (both pages, `calibrate.js`), the phone app, `hearing.server.protocol.Control`.

### Voice profiles (`control.enroll` `action: "profile"`, T-125)
Off unless `HEARING_PROFILES_DIR` names a directory. A profile is `$HEARING_PROFILES_DIR/<name>/*.wav`
(gitignored voices, written by `scripts/make_profile.py <name> <wav...>`, which trims silence).
`{"type":"control","enroll":{"action":"profile","name":"aryan","other":"joseph"}}` (names
`[A-Za-z0-9_-]{1,64}`; `other` optional, default **every other profile** in the directory) puts the
profile on the calibrated path (`AutoEnroller.enroll_profile` -> `enroll_clip`, the T-085 path Name
Call uses): ready at once, bound to the first selected face, `other` is the known negative from t=0
and is not replaced by the camera's automatic negative. The Enrolled candidate embeds a profile clip
as the normalised mean of the unit ECAPA embeddings of its 2-3 s pieces (`PROFILE_PIECE_S` 2.5 s; a
tail under 1.5 s joins the last piece), **without** the separator pass a room clip gets. An unknown
name, or profiles off, answers an `error` with `"stage":"enroll"` and keeps the current clip;
`{"enroll":{"action":"clear"}}` drops it. A failed calibration window over a profile keeps the profile;
a successful one (or a deliberate target switch) replaces it. The judge profile (`control.profile`)
loads `$HEARING_VOICE_PROFILE` (default: the first profile by name) when profiles exist and none is
loaded. With `hq_backend` remote the `remote` stage forwards `{"enroll":{"action":"profile",...}}`
(and later `{"action":"clear"}`) **in its own control message**, so a remote without profiles (or older
code) rejects only that message; the GPU host needs the same profile directory under its own
`HEARING_PROFILES_DIR`. `acquisition.detail` is `known voice: <name>` once the Enrolled tier plays a
profile; `/judge`'s pill reads "Isolated — known voice: <name>". Consumers: `/judge`
(`judge-pill.js`), `scripts/session_report.py` (unchanged, reads `enrollment`), the remote session.

**Saved-profile fast paths (T-136).** At `hello` (and at the first profile load) the session hands every
profile's `(target, negative)` pair, the same clips `enroll_profile` builds, to the shared enroller
(`AutoEnroller.preload`, `hearing.tse.enroll.profile_pairs`); the threaded Enrolled stage queues their
`prepare()` behind the current clip, and the candidate keeps up to 16 enrollment embeddings
(`enrolled.CACHE_ENTRIES`), so a later swap is a cache hit: the old clip plays until the next stage
call sees the new one prepared (one window, no `enrolling` gap). `telemetry.enrollment.profiles` lists
the saved profiles; `/judge` keys **1** / **2** send `{"enroll":{"action":"profile","name":profiles[0|1]}}`.
Name Call: when profiles exist the `NameCaller` gets them (`saved=`) and, from 0.5 s of voiced audio
around the name (`SNAP_MIN_S`, retried every 0.25 s until the normal clip is due), compares the
utterance's ECAPA embedding with each profile's (mean of unit piece embeddings); cosine >= 0.45
(`SNAP_SIM`) and >= 0.1 ahead of the runner-up (`SNAP_MARGIN`) enrolls **that profile**
(`enroll_profile`, every other profile the negative) instead of the mixed utterance, never grown with
room audio; the same profile saying the name again is ignored, another profile re-targets at once
(no `RETARGET_WAIT_S`). Otherwise the clip path above is unchanged. `telemetry.name_call` gains
`saved_profiles` (names), `target_profile` (name or null), `last.profile` / `last.profile_sim`, and its
`events` carry `profile`; the pill reads "Isolated · saved profile <name>". Consumers: `/judge`, `/simple`
(`name-call.js`).

### Pick confidence and the residual floor (`telemetry.tse`, T-134)
The Enrolled candidate gives every window a pick confidence in [0, 1]: the lead of the chosen
stream's score over the other's divided by `CONF_MARGIN` (0.54, cosine-minus-negative scores) or, when
target **and** `other` are saved profiles, `CONF_MARGIN_LDA` (0.58) on a closed-set 2-class LDA
over both profiles' piece embeddings (centroids at +1/-1, pooled covariance shrunk 0.5). Both margins
were fitted by `eval/confidence_fit.py` on a male-male GRID stand-in, not yet on the teammates' takes.
An F0 cue (autocorrelation median F0 of each separated stream, >= 50 % voiced) vetoes the embedding
pick when the other stream sits in the negative profile's p10-p90 range and the two ranges do not
overlap (confidence capped at 0.5), or adds 0.25 when it agrees. A gated window has confidence 0. The
`tse` stage crossfades it like the audio and passes it to the mixer as `extras["tse_confidence"]`;
the mixer then sets the residual floor to `6 + 34 * c` dB (unsure: about the mix, sure: -40 dB),
never quieter than the slider allows, ramped over 30 ms. Without it (other candidates) the mixer keeps
`max_duck_db`; `HEARING_TSE_CONFIDENCE=0` ignores it (A/B arm). Telemetry: `"tse":{"confidence":0.74}`
(`null` when no confidence-giving tier plays), `extras.mixer.floor_db`. With both voices saved as
profiles `same_bucket` is forced false (the closed set separates them). Consumers: `/judge` drawer
meter + "profile: <name> (saved)" label, `eval/pipeline_eval.py --profile` (median confidence per run).

### Acquisition status

Every `telemetry` event includes `acquisition: {state, since_s, detail}`. `state`
is one of `searching`, `candidate`, `acquiring`, `locked-visual`, `locked`,
`reacquiring`. `since_s` is nonnegative elapsed server-monotonic seconds since
this state/target/candidate was first observed by telemetry; it resets when any
of those change. `detail` is a display-safe explanation, not a machine enum.

- `searching`: no target or dwelling challenger; `looking for a speaking face`; or an explicit lock lost
  for `hold_lock_s` (T-107): `lost the person you locked — look at them again` (`/judge` pill shows it).
- `candidate`: selector challenger is dwelling, including a possible target switch;
  `waiting for face dwell`.
- `acquiring`: target selected, enrollment collecting; `enrolling 1.2/2.0 s`.
  The denominator is the actual configured enrollment duration. Silence/overlap
  can pause collected speech time.
- `locked-visual`: face selected but no isolation tier playing; `face selected;
  waiting for isolation`. An inferred `tier=av` for live video alone does not
  establish isolation.
- `locked`: the pipeline reports an AV/Focus HQ or enrolled isolation tier playing;
  `target isolation active`. AV can play while voice enrollment is still collecting.
- `reacquiring`: selected target missing during the selector's hold interval (`hold_lock_s` for a lock)
  (`target lost 0.4 s`, on the video capture clock), or enrollment restarted after
  prior progress (the enrollment detail). Enrollment restart stays in this state
  until ready; after hold expires and no challenger exists the state is `searching`.

Precedence: held target loss, dwelling challenger, no target, playing AV isolation,
collecting/restarted enrollment, playing enrolled isolation, visual selection.
Consumers: `/simple` and the debug console share the pill labels Searching,
Candidate, Acquiring voice, Target locked, Isolated, Reacquiring respectively.
Clients connecting to older servers show Waiting for engine if the field is absent.
The existing `target.locked` still means an explicit user pin, independent of
acquisition's visual or audio lock.

## schemas
The in-process types are defined in `hearing/types.py`. Dataclasses retain the
provided arrays without copying, coercing, clipping or validating them; media
adapters and stage implementations must honour these formats.

```python
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable
import numpy as np

Mode = Literal["natural", "focus_hq", "synthetic", "text", "quiet", "auto"]

@dataclass
class AudioChunk:
    samples: np.ndarray  # float32 mono [-1, 1]
    sample_rate: int
    seq: int
    capture_ts: float

@dataclass
class VideoFrame:
    image: np.ndarray  # HxWx3 uint8 RGB
    seq: int
    capture_ts: float

@dataclass
class FaceTrack:
    track_id: int
    box: tuple[float, float, float, float]  # normalised x, y, width, height
    landmarks: np.ndarray | None
    p_speaking: float
    last_seen_ts: float

@dataclass
class TargetState:
    track_id: int | None
    confidence: float
    locked: bool
    since_ts: float

@dataclass
class Telemetry:
    seq: int = 0
    tier: str = ""
    latency_ms: dict[str, float] = field(default_factory=dict)
    device: str = ""
    extras: dict = field(default_factory=dict)

@dataclass
class StageContext:
    mode: Mode
    suppression: float  # 0-1
    target: TargetState
    tracks: list[FaceTrack]
    telemetry: Telemetry

    @classmethod
    def default(cls) -> "StageContext":
        return cls(
            mode="natural",
            suppression=0.5,
            target=TargetState(None, 0.0, False, 0.0),
            tracks=[],
            telemetry=Telemetry(),
        )

@runtime_checkable
class Stage(Protocol):
    name: str
    sample_rate: int  # external input AND output rate
    latency_ms: float  # algorithmic lookahead/windowing, not compute time
    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk: ...
    def reset(self) -> None: ...
```

Suppression 0 = untouched input, 1 = maximum suppression. Each call to
`StageContext.default()` creates independent target, tracks, telemetry and maps.
An empty telemetry tier/device means unset, not a detected CPU or selected tier;
the session/runner sets those fields and updates `seq` for each audio chunk.
The server combines this internal telemetry with tracks/target/voice when forming
the existing wire event above; `extras` is internal extension data, not an
instruction to serialize arbitrary contents to clients. Never store apparent
age/gender estimates in telemetry or extras.

### Stage registry and pipeline grammar
`hearing.registry` exposes:

- `register(name)`: decorator for a stage class or factory accepting keyword
  arguments. Returns the original callable. Names match
  `[A-Za-z_][A-Za-z0-9_-]*`; duplicate or invalid names raise `ValueError`.
- `get(name, **kwargs) -> Stage`: constructs a fresh stage via its registered
  factory. Unknown names raise `KeyError` listing known names. Factories must
  return independent streaming state for each invocation.
- `names() -> list[str]`: sorted registered names. Import the stage's module
  before lookup so its decorator runs; there is no automatic discovery.
- `build_pipeline(spec: str) -> Pipeline`: parses the complete spec, constructs
  its stages, and checks consecutive external sample rates at build time.

```text
pipeline := stage (">" stage)*
stage    := name ["(" [key "=" value ("," key "=" value)*] ")"]
key      := [A-Za-z_][A-Za-z0-9_]*
value    := signed_integer | decimal_or_exponent_float | boolean | string
example  := a>b(x=2)>c
example  := dfn(strength=0.5)>gate(enabled=true,label="front > left")
```

Whitespace around stages, keys, values and separators is ignored. Integers use
base 10; floats include `0.5`, `.5`, `1.` and `1e-3`; booleans are case-insensitive
`true`/`false`. Quoted strings use Python string-literal escapes (single or double
quotes); quotes preserve strings such as `"2"` and `"true"` and allow delimiters
such as `>`, `,`, `=`, and parentheses. Bare strings match
`[A-Za-z_][A-Za-z0-9_./:-]*`. Expressions, collections, positional arguments,
duplicate keys, trailing commas and empty stages/specs raise `ValueError`.
Parsing never executes expressions. Factory keyword errors propagate to callers.

`Pipeline` is itself a runtime-checkable `Stage`. Its `name` is the original
spec, `stages` is an ordered tuple, `sample_rate` is the first stage's rate, and
`latency_ms` is the sum of stage algorithmic latencies, read at call time (a `tiers` ladder
reports its playing tier's delay, a `remote` stage its active path's). `process` passes each
output to the next stage with the same context; `reset` resets every stage in
order. Pipelines do not add buffering, resampling, clipping, or compute timing.

The pipeline format is mono float32 at **16 kHz**, normally **10 ms / 160 samples**.
A stage's advertised `sample_rate` describes both its input and output rate.
Stages needing another internal rate (for example 48 kHz DFN) resample on their
own entry and exit edges, retain the external 16 kHz rate, and include the added
algorithmic delay in `latency_ms`. Consecutive unequal advertised sample rates
raise `ValueError` at pipeline construction; there is no implicit resampler.
Stages must preserve chunk metadata and maintain streaming state so chunked and
whole-file processing agree within tolerance. Runtime format enforcement belongs
to media adapters and stages, not the generic pipeline.

### Device and model helpers
`hearing.device.pick()` honours `HEARING_DEVICE` (`cuda`, `mps`, `cpu`) verbatim,
without probing availability for an explicit override; other values raise
`ValueError`. With no override it lazily checks CUDA, then MPS, then CPU; absent
PyTorch means CPU. Importing the module does not import optional ML runtimes.
`describe()` returns a JSON-safe dict with `device`, `torch` (version or null),
`onnxruntime_providers` (list or null), `platform`, `machine` (architecture),
`hostname`, `python` (version), and `cpu_count`. `python -m hearing.device` prints
that object as JSON.

`hearing.models.register_loader(name)` decorates a zero-argument loader without
calling it; duplicate names raise `ValueError`. `load(name)` caches one instance
per name, including concurrent callers and `None` results. Failed loads can be
retried; unknown names raise `KeyError` listing registered names. `clear()` drops
cached references but retains loaders and does not invalidate instances still
held by callers. Loads/clear are serialized under a reentrant lock.
`cache_dir() -> pathlib.Path` expands `~` and resolves `HEARING_MODELS_DIR`, or
`models/` relative to the current working directory. It creates no directories
and downloads no weights; registered adapters own those operations.

Consumers clarified by M0: all enhancement/vision/fusion/TSE/synthesis stages,
media adapters, the session server, CLI and benchmark runner. The HTTP and wire
event contracts are unchanged.

## Companion session mirror (T-350)

Consumers: `/app/` session discovery and the phone companion (`web/app/watch.js`,
`web/app/screens/phone.js`, `web/app/connect.js`). Existing `/judge`, `/simple`,
and `/ws/session` initial-message behavior stay unchanged.

- `GET /sessions` returns an array of open engine sessions, e.g.
  `[{"id":"<opaque id>","client":"app","started":"2026-10-10T18:00:00+00:00"}]`.
  `client` is the latest accepted `hello.client` (the protocol default before hello);
  `started` is the session registration time in UTC ISO 8601. Watchers are not sessions
  and do not increase `/health.sessions_active`. IDs expire when the owner disconnects.
  A stage page may obtain its own ID using the existing `/ws/session?replay=1` handshake.
- `WS /ws/watch?session=<id>` attaches to that existing session without creating an
  engine, microphone, or output. It sends the session's **telemetry, caption, error,
  and replay JSON only**, unchanged; no binary media or other JSON types are mirrored.
  New watchers first receive the latest telemetry and replay state, if available,
  then future events. Captions/errors are not replayed as history.
- Watchers may send existing `control` JSON messages. They go through the owner's
  `SessionTap.incoming` (when recording is enabled) exactly once, then the same
  `Session.handle_text` validation/processing as the page's own controls. Resulting
  errors go to the owner and all companions; new state appears in telemetry.
  Other message types, malformed JSON, and binary inputs receive a watcher-local
  `{"type":"error","stage":"watch","message":"only control JSON is accepted"}`
  and never reach the engine or its recording.
- Unknown/missing IDs receive a `watch` error with `message: "session disconnected"`,
  then WebSocket close **1008**. Owner closure closes companions with **1000**;
  companion closure never closes or changes the owner. Each watcher has a 128-event
  pending queue; overflow disconnects that watcher with **1013**, rather than slowing
  the stage or silently dropping captions. Reconnect to receive current state.
- `GET /connect?page=<URL-encoded origin-relative path>` generates the QR and links
  for that path on the server's advertised origins; for example encode
  `/app/?watch=<id>` as the `page` query value. Query strings/fragments are preserved.
  Absolute URLs, protocol-relative URLs, backslashes and control characters are
  rejected with HTTP 400. Without `page`, the existing connect links are unchanged.

These endpoints use the server's existing trusted-LAN access model, without added
authentication: anyone with access to this server can list and control its sessions.
The registry is process-local, like `/replay`; use the same single server process for
stage and companion connections.
