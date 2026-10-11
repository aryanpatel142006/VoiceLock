# Final test sheet (Oct 10)

The last full rehearsal before the demo. Two people, one MacBook (laptop-a), laptop-b for the noise, about 15 minutes.
`bash scripts/rehearse.sh` prints the 12 steps one at a time and waits for Enter. It runs the parts it can and then writes
`results/rehearsal_<stamp>.md`. Print that file, check it, and both sign it.

## Demo audio path (user decision Oct 9, final)
- **Input:** `MacBook Pro Microphone` (exact label). No external mic. The AirPods mic is never the input: picking it
  moves the buds to HFP (16 kHz SCO), which adds delay and lowers quality.
- **Output:** AirPods Pro 3 over A2DP, picked explicitly in the /judge drawer (the page calls `setSinkId`). Never
  "system default".
- **AirPods settings:** ANC on. Transparency, Adaptive, Conversation Awareness, ear detection and spatial audio off.
  macOS Mic Mode Standard (not Voice Isolation).

## Run it
```
caffeinate -dims &                                           # the Mac must not sleep
bash scripts/rehearse.sh --caption-ref captions.txt          # Mac ladder: Isolated bar 8 s
bash scripts/rehearse.sh --gpu --caption-ref captions.txt    # A100 instead of the Mac (GPU up via scripts/gpu_up.sh): bar 6 s
```
- `captions.txt` holds the words Joseph reads in step 10. Without it, the caption WER row prints NOT MEASURED.
- **Voice profiles:** step 3 restarts the server with `HEARING_PROFILES_DIR` set (yours if exported, else
  `eval/fixtures/generated/profiles`) and prints the profiles it found. Without profiles, keys 1/2 do nothing and the
  re-learn row prints NOT MEASURED. The wrong-speaker row reads the same profiles.
- **`demo.sh --judge --team` without `HEARING_PROFILES_DIR`** (T-328): the launcher defaults it to
  `$HEARING_FIXTURES_DIR/profiles` (the main checkout's `eval/fixtures/generated/profiles`) when that holds the `aryan`
  and `joseph` profiles and says so on the JUDGE line (`· profiles <dir>`); otherwise the JUDGE line ends in
  `WARN no voice profiles (...)` and the audio-only session never enrolls: it plays the room. An exported
  `HEARING_PROFILES_DIR` always wins (the Sat mic gate may re-enroll into another dir). `--check` prints the same note.
- **Judge flow row under the team preset** (T-328): `audio_only` (T-168) shows no face, so `judge_check --team`
  counts the lock from the first tick whose `enrollment.source` is `profile` (or whose acquisition says `known voice:`)
  and Isolated from the first tick after it on an isolating tier (the session's acquisition stays `searching` without a
  face); it reports `lock = saved voice <name>, audio_only` next to Start->lock. Measured on
  `t141/aryan/demo-team-easy` with the default profiles: lock->Isolated 3.8 / 3.9 s, Start->lock 0.1 / 0.0 s.
- If no audio frame ever reached the engine (Start not pressed: `seq` is null in every tick), the sheet prints one line
  **NO AUDIO REACHED THE ENGINE**, and every row read from the session audio is NOT MEASURED instead of FAIL.
- Server sessions are recorded under `$HEARING_SESSION_LOG` (default `results/sessions`). The sheet reads only the
  sessions that started after the script did.
- Each step's Enter time is saved, and the step's numbers are read from that part of the recording. **Press Enter first,
  then do the step.**
- Dry run (agents, tests, or to read the steps the night before). It opens no device and does not wait for Enter:
  `bash scripts/rehearse.sh --dry-run --no-hw --session results/sessions/<dir>`.

## The 12 steps
| # | Step | The script runs | Read from the recording |
|---|---|---|---|
| 1 | Devices: AirPods as output, ANC on, the settings above. Bluetooth **off** on both phones and on laptop-b. Zoom/Teams closed | `hearing devices` (labels); checks that `caffeinate` is running | - |
| 2 | Latency: Noise Control **off** for this step only. One bud ~2 cm from the mic grille | `scripts/bt_latency.sh` (3 x 5 clicks) | round trip median / p95 from `bt_latency_<stamp>.json` |
| 3 | Server up. The team preset plays Focus HQ on the Mac (T-181, T-129: the Mac beats the A100 with profiles); the JUDGE line says `backend Mac (measured: beats the A100 with profiles, T-129)`. Only to rehearse the A100: start it first (`scripts/gpu_up.sh`) and pass `--gpu` (the server then gets `HEARING_HQ_BACKEND=remote`, probes and pins the A100). ANC back on | `scripts/demo.sh --judge --team`, then waits until `/health` is no longer `warming` | - |
| 4 | /judge in a **foreground** tab. Pick the mic and the AirPods explicitly in the drawer, press Start, then Enter | `judge_check.check_devices` against `/health` (`client_devices`) | mic / output labels from the page's `hello` |
| 5 | Joseph speaks alone 3 s | - | seconds to Isolated (first `acquisition.state == "locked"` after Enter; "already locked at mark" when the pill was Isolated before Enter), enrollment ready (negative = ready before Enter) |
| 6 | Re-learn live: press **1** while Aryan talks, ~5 s later **2** while Joseph talks (end on Joseph) | - | per swap: seconds from the `enroll` `profile` control to the first tick with the Enrolled tier playing that profile (`enrollment.profile`; the GPU session's while the GPU plays) |
| 7 | Aryan talks over Joseph for 20 s with café noise from laptop-b | - | share of ticks still Isolated; wrong-speaker share of `out.wav` |
| 8 | Space x3 (click the page first) | - | Raw/Selective flips (`ab` changes); per flip, ms from the `ab` control to the first tick with the new state |
| 9 | Name Call (**Shift+N**): say "Aryan" in a clean sentence 4 times, ~10 s apart, talk normally between | - | rise in `name_call.detections`: hits = min(rise, 4), false triggers = rise beyond 4 |
| 10 | Quiet (key **Q**) captions for 60 s; Joseph reads `captions.txt`. Until T-149/T-150 put Quiet on /judge, the step says Text is not on /judge and the caption rows print NOT MEASURED | - | final caption words, WER against `captions.txt` |
| 11 | Pull a bud out and put it back | asks y/n: did audio keep playing in both buds? | - |
| 12 | Sign the sheet | writes `results/rehearsal_<stamp>.md` | - |

## Bars
| Row | Bar | Source |
|---|---|---|
| Round trip median / p95 | <= 300 / <= 350 ms | `scripts/bt_latency.py` (T-144) |
| Mic in use / output in use | `MacBook Pro Microphone` / contains "AirPods" and does not start with "Default" (`Default - AirPods` is the system default: FAIL) | `config/demo-team.yaml` pins (T-143) |
| Mic peak / RMS (whole `in.wav`) | report-only (INFO). T-145's -48 dBFS is the /judge meter's short-window "too quiet" line, not a whole-file bar | T-145 |
| Seconds to Isolated | <= 6 s GPU, <= 8 s Mac (GPU when `hq_active` was mostly `remote`, the GPU played most ticks, or `--gpu`) | T-147 |
| Re-learn (profile swap 1/2) | every swap reaches the Enrolled tier with the new profile within 1.5 s | T-155 |
| Isolated during the over-talk | >= 90 % of ticks | humans (T-157) |
| Wrong speaker during the over-talk | report-only: the share of voiced 1 s windows of `out.wav` whose ECAPA embedding is closer to another saved profile than to the target's (Mac measured 6-25 %, best case 8.6-9.1 %). NOT MEASURED without profiles, `out.wav` or ECAPA weights | humans (T-157) |
| Space flips | >= 3; the A/B switch latency per flip is report-only | humans (T-157) |
| Name Call | >= 2 hits of 4 clean calls; false triggers 0 expected (FAIL otherwise) | humans (T-157) |
| Caption WER | <= 0.90, `TEXT_WER_BAR` imported from `scripts/judge_check.py`, so the sheet never fails a captions run judge_check passes | T-139 |
| Worklet underruns | 0 | always NOT MEASURED for now: /judge does not send the worklet's underrun count to the server |
| Enrollment ready, pill changes, HQ backend, HQ delay (median while the Enrolled tier plays, plus per tier), GPU played / tier / device / RTT p50 (T-156 `telemetry.remote`), caption words | no bar (INFO) | for the slides |

Each cell holds a number from this run or **NOT MEASURED**, next to its PASS / FAIL / INFO. The script never fills in a
default.
`Overall` is FAIL if any row failed, INCOMPLETE if a row is NOT MEASURED, and PASS otherwise. The exit code is 1 when
any row failed.

## Re-enroll the voice profiles (Sat Oct 10, after the mic gate; T-172)
The profiles must come from the demo mic in the demo room (decision Oct 9). The old ones are close-mic Voice Memos
(aryan 50.0 s, joseph 13.95 s; ECAPA cos(aryan, joseph) = 0.018). Each person, on the demo Mac:
1. **Record 3 takes of 25 s** through the demo input, 0.5-1 m from the mic at demo voice level, Harvard sentences plus
   the real demo lines (`scripts/record_take.py` prints them). Peaks about -6 dBFS, speech above -40 dBFS.
   Take 1 quiet room, take 2 with café noise on at about -20 dBFS, take 3 quiet again and softer. Then one extra
   20 s take kept as the held-out test file: **never pass it to the enroll step**.
   ```
   bash scripts/record_profile.sh --list-devices
   bash scripts/record_profile.sh aryan --device "MacBook Pro Microphone" --seconds 25    # x3, then x1 (held out)
   bash scripts/record_profile.sh joseph --device "MacBook Pro Microphone" --seconds 25
   ```
   Each take goes to `$HEARING_FIXTURES_DIR/team/<name>/profile-NN.wav` (next free NN).
2. **Enroll both in one command** (the three profile takes each, not the held-out one). `profile-01.wav` is the old
   Voice Memo that T-172 leaves out; the takes recorded at the gate are `profile-02..04`. Check first that the list
   shows them (25 s each, today's time) and which NN is the held-out take (the newest after the three):
   ```
   T=$HEARING_FIXTURES_DIR/team   # default: <main checkout>/eval/fixtures/generated/team
   ls -l $T/aryan/ $T/joseph/     # profile-02, -03, -04 = the gate takes; adjust the numbers below if they differ
   bash scripts/enroll_profiles.sh aryan $T/aryan/profile-0{2,3,4}.wav joseph $T/joseph/profile-0{2,3,4}.wav \
     --mic "MacBook Pro Microphone" --room "HackRU demo table"
   ```
   It checks every take (rate 16/44.1/48 kHz, at least 20 s, not clipped, speech at least -40 dBFS and 10 s) and
   writes nothing if one fails. It prints `cos(aryan, joseph)` and fails (exit 3) at 0.7 or above: the enrolled
   stage drops a negative profile that close (`SAME_VOICE`), so the other voice would leak. On success each name's
   takes become one new version `profiles/<name>/v<N>/` (with `manifest.json`: source clips, mic, room, date, cos)
   and `profiles/<name>/current` points at it; the old flat Voice Memo profile is kept as `v1`. A flat copy goes to
   `profiles-all/`, and the script prints the laptop command and the `ILAB_PROFILES=... scripts/gpu_up.sh ilab` command.
   **Rollback** (one command, then restart the server and re-run gpu_up):
   `bash scripts/enroll_profiles.sh --rollback aryan` (the version before current; `--to v1` names one;
   `--versions` lists them, `*` = current).
3. **Aryan runs the printed `gpu_up` command** (only Aryan runs it, from the main checkout). It needs T-129's
   `ILAB_PROFILES` option in `scripts/gpu_up.sh`; the script warns when the checkout does not have it.
4. Restart the laptop server with the printed `HEARING_PROFILES_DIR=...` command. `curl -s
   http://127.0.0.1:8000/health` must list `"profiles": ["aryan", "joseph"]` and `"profile_versions"` the new
   versions (e.g. `{"aryan": "v2", "joseph": "v2"}`).
5. **One live session:** /judge, Start, press key **2**. The pill must reach Isolated within **6 s on the GPU / 8 s on
   the Mac**. Session id: ________ · GPU or Mac: ________ · seconds to Isolated: ________
6. **Target absent:** one 25 s take where the target (key 2's person) stays silent and the other talker speaks for
   10 s. Print its row (pass `--delay-ms 950` on the A100):
   ```
   .venv/bin/python scripts/take_checks.py absent results/sessions/<dir> --start <s> --end <s>
   ```
   It prints the output level relative to the input and the wrong-speaker share over that stretch.
7. **Listening log:** after each take the listener answers wrong person / leak / robotic / delay (y/n):
   `.venv/bin/python scripts/take_checks.py listen --takes key2-live target-absent` writes `results/listening_<stamp>.md`.
