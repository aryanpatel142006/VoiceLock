# Selective Hearing — test everything yourself

Written Oct 6, 2026 (main at `85ec7e5`) so you can test the whole system without an agent.
Every command is copy-paste from a terminal. Run them from the repo:

```bash
cd ~/CODE/selective-hearing
```

If a command prints something different from "Expect", jump to **Troubleshooting** at the end.

---

## 0. What exists (one minute)

| Piece | What it does | How you reach it |
|---|---|---|
| Engine (`hearing serve`) | Mic + camera in, processed audio out, over a WebSocket on port 8000 | started by `scripts/demo.sh` |
| `/judge` page | THE demo: big RAW / SELECTIVE HEARING switch, status pill, captions, hidden drawer | `http://localhost:8000/judge` |
| `/simple` page | Phone-style test page: 6 mode buttons (Natural, Focus HQ, Synthetic, Text, Quiet, Auto), slider, Runs on Local/GPU/Auto, Delay Speed/Balanced/Accuracy | `http://localhost:8000/simple` |
| `/` console | Full debug console: face boxes, tiers, latency, every control | `http://localhost:8000/` |
| Modes | **Natural** (noise suppression, 13 ms), **Focus HQ** (only the person you look at, ~1.5 s Mac / ~0.8 s GPU), **Synthetic** (re-spoken voice), **Text** (captions), **Quiet** | buttons on any page |
| GPU backend | Same engine on a Rutgers iLab A100; laptop streams to it | `scripts/gpu_up.sh ilab` |
| Replay clips | Built-in two-person scenes so you can test with no camera / no second person | `R` on `/judge`, or `scripts/replay.py` |
| Scoring | Recorder + scorer that turn any session into numbers | `scripts/demo_regress.sh`, `scripts/session_score.py` |
| Swarm harness | The agents that built this (Notion board + runners) | `swarm status` (section 11) |

**Before measuring anything, stop the agents** (section 11) — they load the Mac and make numbers worse.

---

## 1. Start the demo (the one command)

**Audio path (T-143, final):** mic = the laptop's **MacBook Pro Microphone**, output = **AirPods Pro 3** (Bluetooth A2DP,
about 150-250 ms), picked on `/judge` under **D** > Output. RØDE mic and wired headphones: not used. AirPods: Noise
Cancellation on; Transparency, Adaptive, Conversation Awareness, Spatial Audio and Automatic Ear Detection off; Mic Mode
Standard. Never pick the AirPods mic (it switches them to HFP: phone-call audio, no noise cancelling). Then:

```bash
bash scripts/demo.sh --judge
```

Expect one line like:

```
JUDGE · backend Mac ladder · expected delay ~1.5 s (...) · rev 85ec7e5 · http://localhost:8000/judge
```

(or `backend iLab A100 ... ~0.83 s` if the GPU is up — section 7). Chrome opens `/judge`. Allow mic + camera.

**Pick the devices (T-121).** Press **D** on `/judge` (the Diagnostics drawer) or scroll to the top of the controls on `/simple`: **Microphone** and **Output** pickers list what Chrome sees (pick **MacBook Pro Microphone** and the **AirPods**). The choice is remembered in the browser (`localStorage`), so it survives a reload; "System default" follows macOS. Changing the mic while running restarts capture on it; the output switches at once (`AudioContext.setSinkId`, Chrome/Edge only: the picker is greyed out elsewhere, use the OS output). The **level meter** next to them is the mic's peak in dBFS: amber **too quiet** (below -48 dBFS since T-145, was -40: move closer or raise the input gain), green **good**, red **clipping** (above -2 dBFS: back off or lower gain). Talk at normal volume 1 m from the laptop: the bar should sit in the green. An info note **Bluetooth output (the demo AirPods): about 150-250 ms of delay** shows when the output's name contains AirPods / Bluetooth / BT (expected on the demo path). A red banner shows when the **mic** is Bluetooth (the AirPods mic = HFP: pick MacBook Pro Microphone) and, with the team preset (T-130), when the mic or output is not the preset's (`config/demo-team.yaml` `input_device` / `output_device`), with an output picker in the banner. The mic label is sent in `hello.input_device` and printed by `session_report.py` (`mic: ...` under `## client`), so a recording says which mic it used.

**Built-in MacBook mic levels (T-145).** The demo input is **MacBook Pro Microphone** (no external mic; the AirPods mic is never the input, it would drop the AirPods to HFP). Speech reaches it at -50..-35 dBFS over a -58 dBFS room (T-119), so the meter's **too quiet** line is **-48 dBFS**: the 10th percentile of what the meter shows while someone talks at -45 dBFS (-45.2 dBFS, measured on Aryan's and Joseph's real takes) minus 3 dB. When the target talks at arm's length the bar should sit at -45..-35 dBFS (green); amber while they talk means they are farther than ~1 m or facing away: have them step closer. The enrollment gate (`VoicedGate`, `hearing/tse/enroll.py`) follows the room floor and, since T-145, stays open 6 dB lower right after a voiced chunk (word tails), so at -45 dBFS the voice print is ready after **1.95-2.47 s** of the target talking alone (both voices, 3/3 runs, 71-90 % of their talking time kept); at -50 dBFS it takes up to 4.1 s. Re-measure on new takes (offline, nothing leaves the machine): `.venv/bin/python -m eval.enroll_eval --team "$HEARING_FIXTURES_DIR/team"` (`--level-db -20,-35,-45,-50 --floor-db -58 --runs 3`; `--no-pick` skips the ECAPA closed-set pick).

**Never a silent fallback (T-133).** The saved mic is opened **exactly** (`deviceId: {exact}`), not as a preference. If it is unplugged or renumbered, the page opens the system default anyway but shows a red banner across the top of `/judge` and `/simple`: **Demo mic not found (MacBook Pro Microphone): using ...**, with a **Retry** button (plug the mic back in, tap Retry). A saved output that is gone (`setSinkId` fails, or it is not listed) raises the same banner: **Demo output not found (...)**. The camera has no picker (it asks for `facingMode: user`), so it has no saved device to lose. The page sends what it is really using (`hello.input_device`, `output_device`, `devices_missing`), and `GET /health` shows the latest one as `client_devices`. Before the judges arrive, with the page started: `.venv/bin/python scripts/judge_check.py --no-hw` (or `bash scripts/demo.sh --judge --check`) prints **Demo mic in use (MacBook Pro Microphone)** (PASS only if the mic label contains it, case/accent-free; `HEARING_EXPECT_MIC` overrides; FAIL when the saved mic was not found, and in a full check when no page is running), **Mic is not a Bluetooth (HFP) mic** (FAIL on an AirPods / Bluetooth input), **Output is not the laptop speakers**, **Output is the demo output (AirPods)** (`HEARING_EXPECT_OUTPUT` overrides), and two reminders no script can read: **macOS Mic Mode** (Control Centre > Mic Mode while the page records: **Standard**, not Voice Isolation) and the **AirPods settings** (Noise Cancellation on, Conversation Awareness off, ear detection off). `--server URL` points it at a server on another port (default `http://127.0.0.1:$PORT`, else 8000). Tests: `tests/test_device_exact.py` (Playwright, fake devices: missing id -> banner + `/health` `missing: ["mic"]`; present id -> one exact request, no banner), `tests/test_judge_check.py` (the rows).

Other ways to start:

```bash
bash scripts/demo.sh --simple          # opens /simple instead
bash scripts/demo.sh --restart         # replace whatever runs on :8000
bash scripts/demo.sh --stop            # stop it
bash scripts/demo.sh --judge --no-open # start without opening a browser
PORT=8123 bash scripts/demo.sh --judge --no-open   # any script above on another port (the examples below keep :8000)
```

`--simple` on a server that is already running does not restart it: use `--restart --simple` (or `--judge`, which always restarts).

---

### 1a. Team preset (T-130): one flag for the teammates' demo

```bash
bash scripts/demo.sh --judge --team                 # the engine with config/demo-team.yaml pinned
bash scripts/demo.sh --judge --check --team --no-hw # dry run: + "Team preset pins" row, replays demo-team-easy
bash scripts/demo_regress.sh --clip demo-team-easy --modes natural,focus_hq,text,synthetic,name_call --profile team --no-free
```

`--profile team` is the judge profile plus the preset; `--profile judge` is the same run with the preset off (the
comparison). Each row's JSON has `tier_switches_after_isolated` (the preset's bar: 0).

Open the drawer (D): a `TEAM PRESET` row lists every pin (HQ backend picked once from the GPU probe, same-bucket guard,
lock on Start + hold 60 s, Name Call name/aliases/strict/10 s/180 s, voice, min switch 3 s, feed `focus`, ASR local at
temperature 0 with a vocabulary-only prompt, noise level). Edit `config/demo-team.yaml` to change a pin (restart the
engine). Without `--team` the judge page behaves exactly as before.

## 2. Test the judge demo with NO camera and NO second person (replay)

On `/judge`, press **R**. A 30 s two-person clip plays through the real engine; a **REPLAY** badge shows.

Keys on `/judge`:

| Key | Does |
|---|---|
| **Space** | RAW ↔ SELECTIVE HEARING (same moment, delay-matched — the judge A/B) |
| **C** | Calibrate: grabs 3 s of the target's voice now (ask them to say one sentence) |
| **R** | Replay clip on/off |
| **D** | Diagnostics drawer: tier, delay, backend, RTT, numbers |

Expect: pill goes **Look at the person → Ask them to say one sentence → Calibrating → Isolated**, captions appear, and pressing Space flips between the mixed room and the isolated voice. On the replay clip, Isolated comes after ~5-7 s on the GPU and ~13-15 s on the Mac ladder (`/judge`'s drawer shows which backend).

Same thing from a terminal (the server from section 1 must be running; it is recorded only if the server was started with `HEARING_SESSION_LOG`, section 4):

```bash
HEARING_MODELS_DIR=models .venv/bin/python scripts/replay.py eval/fixtures/demo/demo-mf-easy --lock-target --mode focus_hq --suppression 100
```

If `eval/fixtures/demo/demo-mf-easy/clip.mp4` is missing, build the clips once (offline, about 3 minutes, writes the four demo scenes):

```bash
HEARING_MODELS_DIR=models .venv/bin/python -m eval.make_fixtures --demo
```

---

## 2a. Known voices: record and load a voice profile (T-125)

Off unless `HEARING_PROFILES_DIR` is set; without it nothing changes.

1. Record each teammate alone, on the demo mic, in a quiet room: 10-20 s of normal speech
   (3-6 sentences), saved as 16 kHz+ WAV (QuickTime -> export audio, or `sox -d rec.wav`). Do not
   reuse the sentences you will say in the demo test clip when measuring.
2. Write the profiles (silence trimmed, -3 dBFS peak):
   `export HEARING_PROFILES_DIR=~/hearing-profiles; mkdir -p $HEARING_PROFILES_DIR`
   `.venv/bin/python scripts/make_profile.py aryan aryan1.wav aryan2.wav`
   `.venv/bin/python scripts/make_profile.py joseph joseph1.wav`
   It prints the seconds of speech kept per file; aim for >= 8 s per person.
3. Start the demo with the variable set (`HEARING_PROFILES_DIR=... bash scripts/demo.sh --judge`).
   The judge profile loads `HEARING_VOICE_PROFILE` (default the first name alphabetically) as the
   target and every other profile as the known negative. Console / other target:
   `{"type":"control","enroll":{"action":"profile","name":"joseph"}}`; forget it with
   `{"type":"control","enroll":{"action":"clear"}}`.
4. Expect the pill to read **Isolated — known voice: aryan** as soon as the face is selected, and
   telemetry `enrollment.source = "profile"`.
5. GPU: copy the same directory to the iLab host and start the remote server with the same
   `HEARING_PROFILES_DIR` (a new server step, `ILAB_REUSE=<job> ... gpu_up.sh ilab`); the Mac forwards
   the profile by name. A remote without it answers an error and keeps calibrating from the camera.

## 2b. Name Call: no camera, "say my name" (T-120)

The camera-free fallback. Whoever says the wearer's name becomes the target; the Enrolled tier isolates their voice.

1. Start the demo (section 1). On `/judge` tick **Name Call** (or press **N**) and type your name in the field
   (commas add aliases: `Aryan, Ary`). On `/simple` click the **Name Call** mode and fill **Your name**.
   The camera may stay unplugged: Name Call never uses it.
2. Pill: **Listening for “Aryan”**. Have someone say "Aryan, can you hear me?" in a normal voice.
3. Pill: **Heard “Aryan” from a new voice — locking** (detected about 0.3-0.5 s after the name), then
   **Isolated** about 3-4 s after the name. Press Space on `/judge` for the RAW / SELECTIVE A/B.
4. A second person says "Aryan": the target switches to them (the pill goes back to locking, then Isolated).
   The same person saying it again only strengthens their voice print.
5. **Release** (button next to the name field, or `{"type":"control","name_call":"release"}`) goes back to
   listening; so does 60 s without hearing the target. **Target quiet** = isolated, but the target has not
   been heard for 6 s.

Expected on laptop-a (M4 Pro, `eval/namecall_eval.py`, camera-free scene: babble 10 dB SNR, a GRID talker
throughout, a Kokoro voice saying "Aryan, hey, can you hear me over here?" at 7 s then overlapping):
name detected **0.40-0.42 s** after it ends, **Isolated 2.3-2.8 s** after the name, SI-SDRi **+10.1 dB** from Isolated on,
other talker **-19 dB**, wrong target **0.81-0.83 s per 10 s**; **0 false triggers** in 5.06 min of speech without the name
(`docs/BENCHMARKS.md`, T-120). Whisper `base.en` hears
"Aryan" as "Ariane"/"Arian"/"Arion"; the matcher accepts those. A name Whisper cannot spell close enough
(check with the drawer's diagnostics: `telemetry.name_call.last.word`) needs an alias with its spelling.

False-trigger guards (T-123), next to the name field on `/judge` and `/simple`: the **match** select and the
**not:** field (words that never count as the name, commas: `Ryan, Brian`), both remembered by the browser.
- `strict`: only the name or an alias spelled exactly. Whisper spells "Aryan" right only 7/16 times: add the spellings you see as aliases.
- `normal` (default): close spellings, but never a common other name the fuzzy rule would take ("Ryan", "Brian", "Orion" for "Aryan").
- `loose`: the T-120 matcher; "Ryan" triggers it unless it is in the **not:** list.
Measured (`docs/BENCHMARKS.md` T-123): `normal` hears 14/16 "Aryan" sentences, 0/16 "Ryan" sentences, 0 false triggers in 5 min;
`loose` triggers on 14/16 "Ryan" sentences. Whisper is not given the name as a prompt: that made it invent "Aryan" (7 false triggers / 5 min).
Check: with match `normal`, someone says "Ryan, can you hear me?": the pill stays on **Listening**; switch to `loose`: it locks.

Measure it (replays the scene through a Session at real time; then 5 min of speech without the name):

```bash
HEARING_MODELS_DIR=models swarm-lock --exclusive -- .venv/bin/python -m eval.namecall_eval scene
HEARING_MODELS_DIR=models .venv/bin/python -m eval.namecall_eval false-triggers
HEARING_MODELS_DIR=models .venv/bin/python -m eval.namecall_eval matrix   # match x prompt, incl. a "Ryan" set (T-123)
bash scripts/demo_regress.sh --modes name_call --no-baseline   # the demo clip, camera-free, name = a word only the target says ("green")
```

Limits: two people saying the name at the same time enroll the louder one; a confusable word (for "Aryan":
"Ryan", "Arian") triggers like the name; Whisper needs ~0.35 s per 2.5 s window of one CPU core.

---

## 3. Test every mode by hand (`/simple`)

Open `http://localhost:8000/simple`, pick the microphone and output at the top of the controls (MacBook Pro Microphone, the AirPods; level meter in the green, no red banner; section 1), press **Start mic and camera**, AirPods in.

| Step | Do | Expect |
|---|---|---|
| Natural | Click Natural, talk, drag the slider 0 → 100 | Background noise drops smoothly; your voice stays; ~35 ms total delay |
| Focus HQ | Click Focus HQ, face the camera, someone talks beside you | Pill reaches Isolated; the side talker is pushed down; ~1.5 s delay on the Mac |
| Synthetic | Click Synthetic, speak a sentence | Your words come back in a synthetic voice matched to the face (Kokoro offline; ElevenLabs if a key is set) |
| Text | Click Text, speak | Captions appear |
| Quiet | Click Quiet | Silence |
| Auto | Click Auto | Natural until two people talk, then Focus HQ (a toast says why), back to Natural 5 s after |
| Runs on | Flip Local / GPU / Auto | GPU only works if section 7 is up; otherwise it stays Local; Auto uses the GPU whenever it answers |

Two-person live test (the most important one an agent could not do):
1. You sit facing the laptop camera with the AirPods in; a friend sits off to the side.
2. Focus HQ. Look at the friend — wait for the pill to say **Isolated** (or press **C** on `/judge` while they say one sentence alone).
3. Have a third voice (a person or a phone video) talk at the same time. The friend should stay clear; the other voice should drop.
4. Press Space on `/judge` to A/B.
5. Try: the friend pauses; you look down for 2 s; the friend covers their face briefly. The lock should hold (it survives camera stalls since T-107).
6. Same-gender pair: on the Mac it falls back to the gate (less isolation, pill says so); on the GPU it uses the lip-based AV tier (works).

---

## 4. Record and score a session (numbers, not impressions)

Turn on the recorder by starting the server with `HEARING_SESSION_LOG` (demo.sh does this when the variable is set):

```bash
HEARING_SESSION_LOG=results/sessions bash scripts/demo.sh --judge
```

Every browser session is saved under `results/sessions/<time>/` (mic in, engine out, telemetry, sample frames).

Read what happened in the newest session:

```bash
.venv/bin/python scripts/session_report.py
```

Score it against the clip's clean voices (replay sessions only):

```bash
.venv/bin/python scripts/session_score.py results/sessions/<that-folder>
```

One-command regression (starts its own server on a free port, replays, scores, compares to the last runs):

```bash
bash scripts/demo_regress.sh --modes natural,focus_hq            # 60 s each + the free-selection row, ~3.5 min
bash scripts/demo_regress.sh --modes focus_hq --seconds 60       # Focus HQ only (60 s is the default), ~2.5 min; --seconds 120 is steadier
```

Read the "demo score" lines. `target agreement 100%` must hold; `WRONG TARGET` on the **free** row is expected (the demo has the wearer face the target).

---

## 5. Turn a phone video into a test clip

Record 30 s in landscape, both faces visible, ~1 m away: the target talks alone 10 s, then the other person 10 s, then both at once. Then:

```bash
HEARING_MODELS_DIR=models .venv/bin/python scripts/ingest_clip.py ~/Downloads/myclip.mp4 --target left --name myclip
HEARING_MODELS_DIR=models .venv/bin/python scripts/replay.py eval/fixtures/demo/myclip --lock-target --mode focus_hq --suppression 100
.venv/bin/python scripts/session_score_nr.py "$(ls -td results/sessions/*/ | head -1)" --clip eval/fixtures/demo/myclip
```

(`--target left|right|largest` = which face is the person you want to hear.) The replay is recorded only if the server was started with `HEARING_SESSION_LOG=results/sessions` (section 4). The clip also shows up under the REPLAY button.

### 5b. Record the teammates' own takes (demo-team clips, T-126)

The judge demo is tuned on **your two voices**. Each teammate records alone, at the laptop, in a quiet room
(the noise is added later), facing the webcam, centred, about an arm's length away, normal speaking voice.
Use the mic you will demo with (`--list-devices`, then `--mic <index>`: the MacBook Pro Microphone). One person, one take at a time:

```bash
.venv/bin/python scripts/record_take.py --list-devices
.venv/bin/python scripts/record_take.py --name aryan  --kind profile --bucket male_adult --mic <index>   # 30 s
.venv/bin/python scripts/record_take.py --name aryan  --kind fixture --mic <index>                      # 60 s
.venv/bin/python scripts/record_take.py --name joseph --kind profile --bucket <bucket> --mic <index>
.venv/bin/python scripts/record_take.py --name joseph --kind fixture --mic <index>
.venv/bin/python scripts/record_take.py --venue-noise --seconds 60 --mic <index>   # at HackRU, nobody near the mic talks
```

What each step does: a 3 s **level check** (talk normally; exit 3 = too quiet below -40 dBFS or clipping, fix the
gain or distance and run again), a 3-2-1 countdown, then the take. Read the sentences it prints, **one per line with a
short pause between them** (the pauses are where the take is cut into turns). At the end it prints the speech level and
seconds of speech; it asks for another take below 10 s (profile) / 20 s (fixture) of speech. `--bucket` (one of
`male_adult`, `female_adult`, `male_child`, `female_child`, `neutral`, your choice; only used to decide whether the
`demo-team-same` clip is built) is stored in `team/<name>/meta.json`.

Files: `$HEARING_FIXTURES_DIR/team/<name>/{profile,fixture}-NN.{wav,mp4}` (QuickTime files: below) and `team/venue_noise.wav` (main checkout,
gitignored, never commit them). **Profile** takes are for enrollment and are never cut into a clip; **fixture** takes
are what the clip is made of; they must be different recordings (the builder refuses an identical file). More
fixture takes = more varied turns. Then build and test:

```bash
.venv/bin/python -m eval.make_fixtures --demo-team          # demo-team-easy (SNR 10), -hard (SNR 5), -same if one bucket
bash scripts/demo_regress.sh --clip demo-team-easy --modes natural,focus_hq --no-free --no-baseline
HEARING_HQ_BACKEND=remote HEARING_REMOTE_URL=ws://127.0.0.1:18765/ws/session \
  bash scripts/demo_regress.sh --clip demo-team-easy --modes focus_hq --no-free --no-baseline   # the A100
```

**QuickTime / phone recordings (RECORDING_SCRIPT.md).** No conversion needed: drop them under
`$HEARING_FIXTURES_DIR/team/<name>/` (e.g. `~/CODE/selective-hearing/eval/fixtures/generated/team/aryan/`) with a name
that starts with `profile` or `fixture`: `profile.wav` / `profile.m4a` / `profile_venue.wav` (Takes 1 and the venue
profile, enrollment only), `fixture.mov` / `fixture.mp4` / `fixture-02.m4a` (Take 2 and any more solo takes). Accepted
formats: `.wav`, `.m4a`, `.aac` (audio only: the teammate shows as a grey card in the clip), `.mp4`, `.mov` (video plus
its own sound; a `.wav`/`.m4a` with the **same name** next to it is used as the sound instead). A long take is fine: it
is cut at its pauses and only its first 40 s of speech (`TAKE_BUDGET_S`) goes into the clips. Not used by
`--demo-team`: `noisy.wav` (Take 3) and `team/conversation/` (Take 4, `room.mp4` + `<name>_close.m4a` synced by
claps); the conversation is two people in one recording, so it cannot be cut into solo turns. It is listed in the
manifest (`source.conversation`, `"used": false`), and scoring it against the close-talk refs is the T-132 eval's job.

The first teammate by name is the target (left face); `--team-target joseph` swaps. The noise is the venue recording
when `team/venue_noise.wav` exists, else the built-in babble. Until two teammates have fixture takes, `--demo-team`
builds the same clips from two GRID talkers (s2 male, s7 female) and says "stand-in" in the manifest's `note`.
The clips appear under REPLAY on `/judge` like the other demo clips.

---

## 6. Rehearsal self-check (do this the morning of Oct 10)

```bash
bash scripts/demo.sh --judge --check
```

First it checks the Enrolled tier's compute headroom (one line; below 30 % it prints FAIL and stops: close apps / stop the agents). Then it prints PASS / FAIL / SKIP per success criterion with the measured numbers (judge flow on the replay clip, Natural and Focus HQ scores, mic-to-ear latency, camera lock, GPU probe), about 5 minutes. Sit in front of the camera while it runs. Things to know:

- With the GPU up, add `HEARING_REMOTE_URL=ws://127.0.0.1:18765/ws/session` in front, otherwise the GPU row says "no HEARING_REMOTE_URL: Mac ladder" and Focus HQ is judged on the Mac bar.
- The mic-to-ear row runs `hearing latency-test` on the demo devices (`--in "MacBook Pro Microphone" --out <the AirPods> --sr 48000 --clicks 10`, from `config/demo-team.yaml`) and is labelled **Bluetooth path**: PASS at median <= 300 ms and p95 <= 350 ms. Hold one AirPod at the laptop's mic (left of the keyboard) while the clicks play (ear detection off keeps it playing). FAIL when the AirPods or the mic are not connected. The camera row opens the webcam (allow the terminal in System Settings → Privacy → Camera).
- Two rows are always SKIP (target selection, bench table): each names the command that measures it. Synthetic and Text are measured (T-117, about 2 minutes): `demo-mf-easy` is replayed 30 s in Synthetic and then in Text with the manifest's target locked. **Synthetic first-word latency** = per final caption, its `t1` (end of the phrase, client clock) -> the first non-silent sample of that session's `out.wav` that no earlier phrase used; the row prints p50, p95, the phrase count and which TTS ran, and PASSes at p50 <= 1.5 s with ElevenLabs or <= 2.5 s with Kokoro (a mock TTS is not a latency). It is not the app's `synth_phrase_p50` telemetry, which also counts the queue behind earlier phrases. A hygiene row FAILs on an empty or `. . .` caption or more than 2 voice-bucket changes. **Text caption WER** scores the final captions against the target's words (manifest `transcript`); informational, no bar.
- Agents and dry runs: `bash scripts/demo.sh --judge --check --no-hw` never opens the camera, mic or speakers and skips the GPU probe (Synthetic/Text still run: they use no device). The hardware rows print `SKIP … skipped (no hardware)`, Focus HQ is judged on the Mac bar, and its demo_regress run is written with `--no-baseline`.
- A demo_regress run made for docs or a dry run should not become the baseline the next `verify_full` compares against: pass `--no-baseline` or set `HEARING_REGRESS_NO_BASELINE=1` (the JSON gets `"no_baseline": true` and later runs skip it).
- On the Mac ladder (no GPU) the **judge flow** (lock → Isolated ≤ 6 s) and **Focus HQ** (Isolated ≤ 8 s, leak ≤ −12 dB) rows can FAIL: Oct 6 on a loaded Mac they measured Isolated 10.6-13.5 s and leak −7.5 dB. That is the Mac's real behaviour, the GPU meets both; the exit code is 1 while any row FAILs.

Also measure real mic-to-ear latency on the demo devices by hand (one AirPod held at the laptop's mic):

```bash
HEARING_MODELS_DIR=models .venv/bin/python -m hearing latency-test --pipeline dfn_ll --in "MacBook Pro Microphone" --out AirPods --sr 48000 --clicks 10
```

Expect median <= 300 ms, p95 <= 350 ms on the AirPods (Bluetooth adds 150-250 ms on top of the engine path).

---

## 7. GPU backend (Rutgers iLab A100)

1. Open the SSH tunnel (asks for your NetID password; keep this terminal open):

```bash
ssh -o ControlMaster=yes -o ControlPath=~/.ssh/cm-ilab -o ControlPersist=12h -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -fN ap2772@ilab.cs.rutgers.edu
```

2. Start the GPU server (4 h job on an A100):

```bash
ILAB_GRES=gpu:a100:1 ILAB_TIME=04:00:00 bash scripts/gpu_up.sh ilab
```

3. Check it, start the judge on it, stop it:

```bash
bash scripts/gpu_up.sh status
bash scripts/gpu_up.sh probe
HEARING_REMOTE_URL=ws://127.0.0.1:18765/ws/session bash scripts/demo.sh --judge
bash scripts/gpu_up.sh down
```

Expect the judge line to say `backend iLab A100 ... ~0.83 s`. If the GPU drops mid-demo the page falls back to the Mac by itself and switches back when it returns.

If the tunnel died: `ssh -O exit -o ControlPath=~/.ssh/cm-ilab ap2772@ilab.cs.rutgers.edu` then redo step 1.
CPUs (T-115): `--cpus-per-task`, `--cpus-per-gpu` and `--ntasks-per-node` are all rejected (`gpu_up.sh` passes none; `ILAB_CPUS` is a no-op that warns, T-118): `srun.real: error: Job rejected: Please do not specify the number of CPUs. There is actually no limit.` The job's `squeue` says 2 CPUs, but a step inside it sees all 96 (`nproc` 96, affinity 0-95), so more CPUs are not the lever for enrolled overruns.
Other GPU types: `ILAB_GRES=gpu:4500_ada:1` or `gpu:5000_Blackwell:1`. Never add `--cpus-per-task` or `--mem` (the cluster rejects them).

**Restart the GPU server on new code without losing the A100 (`ILAB_REUSE`).** `gpu_up.sh ilab` normally cancels the running `hearing-hq` job, and a new A100 job may queue. Instead, start the new server as an extra `srun --jobid --overlap` step inside the job you already hold, on a free port:

```bash
ssh -o ControlPath=~/.ssh/cm-ilab ap2772@ilab.cs.rutgers.edu squeue -u ap2772 -n hearing-hq -h -o '%i %N %L'   # job id, node, time left
ILAB_REUSE=<job id> REMOTE_PORT=18766 ILAB_GRES=gpu:a100:1 bash scripts/gpu_up.sh ilab
HEARING_REMOTE_URL=ws://127.0.0.1:18766/ws/session bash scripts/demo.sh --judge
```

It re-syncs the tree (`/health` then reports the new `git_rev`), runs on the job's own node, logs to `~/hearing-remote/serve-18766.log`, never runs `scancel` and points the tunnel at the new port. The old server keeps its port until the job ends. The step shares the job's CPUs. For an A/B in a separate job, `scripts/gpu_up.sh ilab --side` keeps its own state, job name and ports so the live job's tunnel survives. Add `REMOTE_PIPELINES='focus_hq=…'` to run a second config side by side with the first (two ports, one GPU) and compare them with `gpu_up.sh probe ws://127.0.0.1:<port>/ws/session`. `gpu_up.sh down` cancels the whole job, so it stops both servers.

---

## 8. Phones on the same Wi-Fi

```bash
bash scripts/demo.sh --lan
```

Prints an `https://<laptop-ip>:8000/simple` URL and a QR code. On the phone: open it, accept the self-signed certificate once (iPhone: Settings → General → About → Certificate Trust Settings if asked), press Start, allow the mic. Phones need https for the mic — plain http will not work.

---

## 9. Expected numbers (what "working" looks like)

Realistic two-person demo clip (`demo-mf-easy`, 30-60 s), wearer facing the target (rows from `docs/BENCHMARKS.md`: T-084/T-086, and a `demo_regress.sh` run on laptop-a, Oct 6, load ~5):

| Path | Isolation (SI-SDRi) | Other voice leak | Dropouts | Quality (DNSMOS, input ≈ 2.7) | Delay |
|---|---|---|---|---|---|
| Mac, Focus HQ | +3 to +7 dB (+5.3 / +6.6 measured Oct 6) | −8 to −10 dB | < 1 s / 30 s (0.2 s measured) | ≈ 2.7 | ~1.5 s |
| iLab A100, Focus HQ | +2.6 to +4.6 dB (T-115, 3 locked 60 s runs; one first replay −3.9 dB, its re-replay +2.6) | −12.7 to −18.4 dB (−3.8 on that replay) | 0-0.48 s | ≈ 2.4-2.6 (the first ~7 s play the room or the plain gate at 1.6-1.95) | ~0.95 s while Enrolled plays (0.5 s window + 0.25 s hop live + 200 ms buffer); 3.2 s while AV plays |
| Natural | ≈ 0 dB (it only removes noise) | ≈ −8 dB | 0 | ≈ 2.6-2.7 | 13 ms |

Synthetic and Text on the same clip (`judge_check`, T-117, laptop-a, Kokoro, no ElevenLabs key, mixture feed, load 4): first-word latency (caption `t1` -> first TTS sample) **p50 1.70 s, p95 3.07 s** over 13 phrases of 14 (bar 2.5 s with Kokoro, 1.5 s with ElevenLabs); the one phrase over 3 s queued behind the previous one. Text caption WER **92 %** (71 caption words vs 78 target words): local Whisper mishears GRID letters and digits ("bin blue by f six now" -> "I think she's embarrassing now"), so read it as an ASR-on-synthetic-speech floor, not a product claim.

Same-gender pairs on the A100 (lip-based tier): 0.97 accuracy, < 0.5 s of the wrong voice per 10 s.
On the Mac, Isolated appears 13-15 s into the clip; on the A100 7-9 s (T-113: the GPU ladder now promotes at most every 3 s, not 10 s). A100 DNSMOS stays below 2.9: the first ~7 s play the room or the plain gate (DNSMOS 1.6-1.95; the GPU box runs no DeepFilterNet: a dfn_ll bridge there scored 2.2-2.5 in those 7 s but every tier runs every chunk, so it ran all along, put the box 1.5-2 s behind and cut SI-SDRi to -4.4..+2.2 dB, T-115 in docs/BENCHMARKS.md), and the live enrolled windows overrun now and then (hop 0.125 s widens to 0.25 s; not the CPU allocation: the job sees all 96 cores, section 7). The `free` row of `demo_regress.sh` (no lock) is allowed to say WRONG TARGET.

---

## 10. ElevenLabs (better Synthetic voice)

```bash
echo 'ELEVENLABS_API_KEY=your-key' >> .env
bash scripts/demo.sh --restart
```

`python -m hearing` reads `.env` (current folder or the repo root) at start; a variable already set in your shell wins. Then Synthetic mode uses ElevenLabs Flash (phrase latency target ≤ 1.5 s). Without a key it uses Kokoro (offline). Details: `docs/DEMO.md` → "ElevenLabs drop-in checklist".

---

## 12b. Demo noise kit (speaker beside the target)
`scripts/demo_noise.py` plays looping café babble (1), a competing talker (2) or both (3) on a chosen output device, with `+`/`-` level, Space pause, q quit. Sources come from WAVs under `HEARING_FIXTURES_DIR`; with none, Kokoro TTS and a synthetic murmur. Steps: `--list`; `--device NAME --calibrate` (pink noise, aim for ~65 dB(A) at the listener with a phone SPL app); then run without `--calibrate` using the printed `--level`. It refuses the engine's output (`--engine-output`, default the system output): check by passing the headphones name and seeing "refusing". Unit tests (`tests/test_demo_noise.py`) use a fake device list; the real speaker is a human check (never from verify).

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| Page says it can't reach the engine | `bash scripts/demo.sh --restart`; check `curl -s localhost:8000/health` says `"ok":true` |
| Port 8000 busy / old code still running | `bash scripts/demo.sh --restart` (finds the old server by port). Check the rev in `/health` (`git_rev`) matches `git log -1` |
| Python crashes when the camera starts | Face detection must use the CPU on macOS (default). Do not set `HEARING_FACES_DELEGATE=gpu` |
| No face boxes / never "Isolated" | Light the face, face the camera, press **C** while the target talks alone (`/judge` and `/`; on `/simple` the **Calibrate voice** button), or tap the face to lock it |
| Echo / doubled voice | AirPods: Noise Cancellation on, Transparency/Adaptive and Conversation Awareness off; mic = MacBook Pro Microphone, never the AirPods mic |
| Focus HQ too slow | Mac is loaded: stop the agents (section 11). Or use the GPU (section 7) |
| GPU line says "Mac ladder" | Tunnel down or job ended: `bash scripts/gpu_up.sh status`; redo section 7 |
| Synthetic silent / robotic | No ElevenLabs key → Kokoro; check `.env`; restart |
| Anything else | `tail -50 "${TMPDIR:-/tmp}/hearing-serve-8000.log"` (server log of the server `demo.sh` started on :8000) and `docs/DEMO.md` sections 6 and C |

Deeper docs: `docs/DEMO.md` (full runbook + judge script), `docs/BENCHMARKS.md` (all numbers), `docs/CONTRACTS.md` (protocol for the app team).

**Browser checks (T-124).** They are Python pytest tests marked `slow` that use the Python Playwright in `.venv` (see `tests/test_phone_check.py`). Do not write or run node Playwright `PLAYWRIGHT=… node …-e2e.mjs` scripts: no node Playwright is installed.
