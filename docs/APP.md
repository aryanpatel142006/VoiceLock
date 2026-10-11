# App runbook (M14): HTTPS, install, stage and phone steps

The app is the web page at `/app/` (`web/app/`, a PWA). `/judge` and `/simple` are untouched and stay served: **`/judge` is the stage fallback**, one URL change away. This file is the human checklist; design is in `web/app/DESIGN_SPEC.md`, the stage script is `docs/STAGE_CARD.md`, the generic phone notes are `docs/DEMO.md#phones`.

Facts below were checked on laptop-a, Oct 10 2026 (LAN IP `10.75.113.217`, Mac name `Aryans-MacBook-Pro.local`). Steps that need the iPhone itself are not checked here (no phone in the loop); they are marked *(not run: needs the iPhone)*.

## 0. The rule that decides everything

**The phone is never the stage mic or the stage output.** The stage mic is the MacBook Pro Microphone (or the iPhone over USB, `--mic`, which is a *Mac* input and not this app) and the output is the AirPods paired to the Mac. The phone in this app is a **companion**: it mirrors state and captions and forwards Focus / Quiet / person / Release buttons into the stage session. The companion opens **no engine session** and **no microphone**, so it cannot disturb the stage tier. If you want audio on a phone, that is Solo mode (section 6), off stage only.

## 1. Start line

From the repo root, on the Mac that is on stage:

```bash
bash scripts/demo.sh --judge --team --lan --app
```

- `--judge --team` is the presenter path exactly as before; `--lan` = `hearing serve --host 0.0.0.0 --tls` (https + wss) and prints the phone URLs, a QR and the certificate path.
- **`--app` changes only the page that opens**: the laptop tab, the QR, and the URL on the `JUDGE ·` line become `https://localhost:8000/app/` instead of `/judge`. Same server, same flags, same preset. **Without `--app` the output is unchanged** (checked on an ephemeral port: `demo.sh --lan --no-open` with and without `--app` print the same lines, differing only in `/simple` vs `/app/` and the QR image; `--judge --team` was not run, the edit to its `JUDGE ·` line only swaps `/judge` for `/app/` under `--app`).
- Stage fallback if the app misbehaves: open `https://localhost:8000/judge` in a new tab; the same server is serving it. Or restart without `--app`: `bash scripts/demo.sh --judge --team --lan`.
- Wait for the `JUDGE · backend … · https://localhost:8000/app/ …` line before pressing Start (models are warm then).
- Keep the Mac awake: `caffeinate -dims bash scripts/demo.sh …`. Plain http works on `localhost` only; phones need `--lan`.
- iPhone over USB as the stage mic: add `--mic "Aryan patel's iPhone Microphone"` (the exact label from Setup's Microphone list).
- Checked Oct 10, 9:41 PM (T-405) on laptop-a: this line and the fine-tuned line below, each with `--no-open --port 0`, start healthy in about 6 s and print `JUDGE · backend Mac (measured: beats the A100 with profiles, T-129) · expected delay ~1.5 s … · https://localhost:<port>/app/ · profiles …/eval/fixtures/generated/profiles`; stopped with `demo.sh --stop --port <port>`. No browser session was run on them (no mic in agent runs).

## 1b. Base vs fine-tuned separator (the A/B)

Two start lines, same flags; only three environment variables differ. Restarting with the other line replaces the server (about 6 s); the phones reconnect by themselves.

```bash
# Base (shipped MossFormer2 weights, closed-set ECAPA pick)
bash scripts/demo.sh --judge --team --lan --app
# Fine-tuned (T-374 fixed-order checkpoint: slot 0 = Aryan, slot 1 = Joseph; gated windows duck the room 20 dB)
HEARING_SEP_CKPT=$PWD/models/T-374/fixed_best.pt HEARING_PICK=fixed HEARING_GATED_FLOOR_DB=20 \
  bash scripts/demo.sh --judge --team --lan --app
```

- Run it from the main checkout (`~/CODE/selective-hearing`); `models/` is gitignored and lives there (`$HEARING_MODELS_DIR`).
- `HEARING_PICK=fixed` needs `HEARING_SEP_CKPT` (base weights with the fixed pick play a random slot) and profiles named `aryan` / `joseph` (the team preset's default `HEARING_PROFILES_DIR` has them). A wrong path fails when the separator loads, with `HEARING_SEP_CKPT is not a file` (check the path with `ls` first).
- Why this arm: T-386 measured held-out wrong-speaker 1.6 % vs 15.7 % for base, leak -17.2 vs -8.8 dB, same compute (`docs/BENCHMARKS.md`). The adoption condition is Aryan's listen and a live room test; no default changed, so the base line still plays base.
- The morning task may swap the checkpoint for the T-387 / T-391 best one (`docs/T-387_fixed_wide_sir.md` recommends T-387's for the Aryan+Joseph fixed slot only). Only the `HEARING_SEP_CKPT` path changes; on Oct 10, 9:45 PM only `models/T-374/fixed_best.pt` existed in `$HEARING_MODELS_DIR`.
- `HEARING_FIXED_GATE` stays off (T-402: not adopted).

## 2. Trust the certificate once

`--lan` makes a self-signed CA certificate for every LAN IP, `Aryans-MacBook-Pro.local`, `hearing.local` and `localhost`, cached in `~/.hearing/tls/cert.pem`. It is **regenerated (new key) only when a new IP appears or it is within 30 days of expiry**, keeping the old IPs. Every regeneration invalidates both trusts below. So: **do the order below the evening before, with the venue network (or the hotspot, section 4) joined**, and do not change networks afterwards without repeating step 2b/2c.

### 2a. Start once, so the cert covers the IP you are on

Run the start line (section 1) once and check what it covers:

```bash
openssl x509 -in ~/.hearing/tls/cert.pem -noout -ext subjectAltName
```

It must list the IP you will use (`ifconfig | grep 'inet '`). On Oct 10 it lists `10.75.113.217`, `10.75.241.193`, `Aryans-MacBook-Pro.local`, `hearing.local`, `localhost`, `127.0.0.1`; fingerprint sha256 `9F:AE:44:6E:…:B6:F3:E0` (a regeneration that day, see `notes_for_reviewer` of T-355).

### 2b. Trust step 1: the Mac (so `https://localhost:8000/judge` and `/app/` show no warning)

```bash
security add-trusted-cert -d -r trustRoot -k ~/Library/Keychains/login.keychain-db ~/.hearing/tls/cert.pem
```

macOS asks for the login password (or Touch ID) once. Check:

```bash
security verify-cert -c ~/.hearing/tls/cert.pem     # "Cert Verify Result: No error."
curl -s https://localhost:8000/health                # works without -k
```

Before the step, `verify-cert` prints `CSSMERR_TP_NOT_TRUSTED` (checked Oct 10). Reload the stage tab afterwards; a tab opened before the step keeps the warning. Undo: `security remove-trusted-cert -d ~/.hearing/tls/cert.pem`. *(The add command itself was not run by T-355: it edits the user's keychain.)* mkcert is not installed and not needed.

### 2c. Trust step 2: the iPhone *(not run: needs the iPhone)*

1. On the Mac, with the server up, open `https://<mac-ip>:8000/connect` (or scan the QR printed by `--lan`, which opens `/app/`). The `/connect` page links the certificate; direct: `https://<mac-ip>:8000/hearing-ca.crt`.
2. On the iPhone in **Safari** (not Chrome, not the QR reader's preview): open `https://10.75.113.217:8000/hearing-ca.crt` → **Allow** (download the profile).
3. Settings → **Profile Downloaded** → **Install** (passcode) → Install again.
4. Settings → General → About → **Certificate Trust Settings** → switch **on** `hearing.local (selective hearing, self-signed)`. Without this Safari lets you "visit this website" but refuses the `wss://` socket.
5. Check: Safari opens `https://10.75.113.217:8000/connect` with no warning.

Android (Chrome): download `/hearing-ca.crt` → Settings → Security → Encryption & credentials → Install a certificate → CA certificate; details in `docs/DEMO.md#phones`.

## 3. LAN IP and the page the phone opens

- The QR / URLs on the terminal are the source of truth: `https://<ip>:8000/app/` and `https://Aryans-MacBook-Pro.local:8000/app/` (the `.local` name only where the network passes mDNS; prefer the IP).
- Reprint them any time: `.venv/bin/python -m hearing qr --port 8000 --page /app/`.
- Companion link: the laptop's **Listen** screen shows a QR to `/app/?watch=<session>` (M14 task 8, `web/app/DESIGN_SPEC.md`; not on main when this was written); scan that one for the companion role. The terminal QR opens `/app/` only (the app shell).

## 4. Hotspot fallback (venue Wi-Fi isolates clients)

Symptom: the phone cannot open `https://<mac-ip>:8000/connect` while the Mac can.

1. iPhone: Settings → **Personal Hotspot** → on. Mac: join that hotspot.
2. Start the server again (section 1). The cert picks up the hotspot's IP; if it had to regenerate (`openssl … subjectAltName` shows a new IP), **repeat 2b and 2c** (new key).
3. **Join the hotspot once the night before** (step 2a on it) so the cert already covers that IP: then the real run does not regenerate and nothing needs re-trusting.
4. The Mac then has no venue internet; nothing on stage needs it except the optional A100 (`--gpu`), which the team preset does not use. Stage timing over a hotspot was not measured (not measured: no hotspot run in T-355).

## 5. Add to Home Screen *(not run: needs the iPhone)*

Optional polish; the stage works from a Safari tab.

1. Safari on the iPhone (trust done, section 2c) → `https://<mac-ip>:8000/app/` → Share → **Add to Home Screen** → Add.
2. Open it from the home-screen icon: it runs standalone (no Safari bar; `display: standalone`, scope `/app/`, black-translucent status bar).
3. A home-screen app is a **separate storage and trust context** from the Safari tab: if the IP changed or the cert was regenerated, delete the icon, redo 2c, add again.
4. The service worker (`web/app/sw.js`) is network-first and versioned, so a new server build is picked up on the next open; never rely on the installed copy offline (the engine is on the Mac).

## 6. Solo mode caveat (NICE, off stage only)

Solo = the phone's own mic and its own engine session (the SDK path). It is a labelled extra, never the demo:

- A second session on the Mac loads the MPS again and can overrun the stage tier (`/health` shows `sessions_active`), so the app **refuses Solo while a judge session is live** and says why. Do not work around it during judging.
- iOS Safari limits: the AudioContext is created inside the Start tap; AudioWorklet needs iOS 14.5+; **no background audio** (screen lock or app switch suspends it; Solo asks for `navigator.wakeLock`, keep Safari in front); sound goes to the **phone's** audio route, not the AirPods that are paired to the Mac.
- Wear earphones on the phone: its speaker feeds back into its mic.
- Solo is only as good as the SDK path was checked (`scripts/phone_check.py`); it is not part of the stage claims in `docs/CLAIMS.md`.

## 7. The app on stage: screens, keys, pills

Screens (nav at the top): **Welcome** (press and hold the disc 0.9 s to start), **Setup** (Microphone and Output pickers, live level meter: Too quiet / Good / Clipping), **People** (the saved voices, Aryan = 1, Joseph = 2; "Following now." only while isolated and Selective plays), **Listen** (the stage screen), **About**. **Diagnostics** is a sheet over Listen (key **D**: backend, tier, rev, delay, RTT, recording). **Phone** is the companion screen the Listen QR opens (`/app/?watch=…`).

Keys on Listen are `/judge`'s (`web/app/keys.js`): **Space** Raw ↔ Selective, **Shift+F** Focus, **Shift+Q** Quiet, **1** / **2** saved person, **Shift+0** Release, **D** Diagnostics, **Shift+R** labelled replay, **Shift+C** calibrate (off stage). Bare F/Q/R/0 act only with Diagnostics open. Click the page first; an amber **Keys off: click the page** banner shows when it lost focus.

- **Status pill** (same words as `/judge`): Press Start → Loading voice — Joseph → **Isolated** (the big word; the line under it says "You hear Joseph"), Raw (amber), Searching after Release, Replay (violet, never live).
- **Gate pill** (T-382) under it, only while Isolated and Selective in Focus: **SOLO** (confident, full level), **OVERLAP-DUCKED** (an uncertain window played softened), **UNSURE** (gated window or confidence under 0.5). Hidden in Raw and Quiet and when the engine sends no gate fields.
- **Quiet**: nothing plays; the chosen person's words appear as captions under their name, 1-2 s behind. "Captions wait until Joseph is isolated." means the pick is not locked yet.

## 8. Checklist (the evening before, then the morning)

Evening, venue network or hotspot joined:
- [ ] `bash scripts/demo.sh --judge --team --lan --app` starts; `JUDGE ·` line shows `/app/`.
- [ ] The fine-tuned line (section 1b) starts too; base vs fine-tuned listened once each.
- [ ] `openssl x509 -in ~/.hearing/tls/cert.pem -noout -ext subjectAltName` lists the current IP.
- [ ] Mac trust: `security verify-cert -c ~/.hearing/tls/cert.pem` says `No error`.
- [ ] iPhone trust: `/connect` opens without a warning in Safari.
- [ ] iPhone companion pass: scan the Listen QR, Focus / Quiet / person / Release forward into the stage session.
- [ ] `bash scripts/demo.sh --judge --check --team --no-hw` still PASS (stage path untouched).
- [ ] `/judge` opens in a second tab (fallback).

Morning: restart once with the chosen line (base or fine-tuned) (`--check` first if time); do **not** change network; AirPods on the Mac, phone muted and face-down.
