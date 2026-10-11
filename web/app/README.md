# App browser checks

From the repository root:

```sh
.venv/bin/python scripts/app_check.py
```

The command starts `python -m hearing serve --tls --port 0` on loopback, reads
its assigned port, and stops the child server when finished. It creates a
throwaway certificate without changing your trusted certificates. Playwright
Chromium accepts that certificate for this run, supplies a fake microphone,
and mutes output. No microphone, speakers, GPU, model weights, or external API
is required. Playwright is already in the project's `dev` dependencies;
if Chromium is missing, install it once with
`.venv/bin/python -m playwright install chromium`.

Each fresh browser context walks Welcome → Setup → People → Listen using the
visible controls: holding Start on Welcome opens the fake mic (Setup's Start mic
is pressed only if it did not).
It checks incoming and outgoing audio on the real secure WebSocket, switches
Selective → Raw → Selective and Focus → Quiet → Focus, and checks the outgoing
control messages and Quiet telemetry. The no-profile People state is expected:
this fixture does not enroll or pretend to isolate a real person.

The check requires a secure context, the app manifest's names, start URL,
scope, standalone display, colours, 192/512 PNG icons and a maskable icon.
Every declared icon must return HTTP 200 and decode at its declared size.
The service worker must control the page before and after reload, with the
`/app/` scope and `/app/sw.js` script. JavaScript exceptions, console errors,
and server error messages fail the run. This checks PWA prerequisites in
Chromium; it does not certify iOS installation or exercise Add to Home Screen.

Exit status is **0 on success, 1 on failure**. Outputs default to
`.swarm-run/app-check/`; choose another directory with `--output PATH`:

- Six screenshots per viewport: Welcome, Setup, People, Selective, Raw, Quiet.
- Captures start at the top of the page after finite transitions finish; the live
  audio scene keeps running.
- Exact viewport sizes: 390×844, 1440×900, 1920×1080, at DPR 1.
- `results.json`: overall result, per-viewport checks, browser version and errors.
- `server.log` and `browser-WIDTHxHEIGHT.json`: server diagnostics, captured
  controls, console errors and audio frame counts.

Use a fresh output directory when archiving a run: reruns overwrite matching
files and may leave screenshots from an earlier successful run after a failure.
Read `results.json`, not the presence of screenshots, to determine success.
`HEARING_TEST_SLOWDOWN` extends readiness timeouts on busy machines.

The archived screenshots in [docs/app-shots](../../docs/app-shots/README.md)
show mock pipelines with genuine browser levels and server telemetry. They
are not evidence of voice isolation, delay, audio quality, or performance.

### Walking a server that is already running

With a server origin the same script runs the T-356 walk instead, against
whatever that server has loaded (for example `scripts/demo.sh --judge --team`):

```sh
.venv/bin/python scripts/app_check.py https://localhost:8000 [--live] [--shots DIR] [--wav FILE]
```

It visits every screen in the nav and the Diagnostics sheet at 390, 768, 1440
and 1920 px, and fails on console errors, sideways scroll, placeholders and green
pressed controls while nothing plays. `--live` presses Start listening with a
fake mic looping `--wav` (default: the team fixture's held-out clip) and walks
every Listen state the engine reaches, plus the phone companion. Its JSON
report goes to stdout. `--output` belongs to the self-starting journey; `--live`,
`--shots` and `--wav` need an origin.

Checker regression tests (including real-browser success and deliberately
broken manifest/console runs):

```sh
.venv/bin/python -m pytest web/app/tests/e2e -q
```

Browser cases are marked `slow`. The repository's fast suite does not discover
this directory, so run that command explicitly when changing the checker.
The small manifest validation cases can be run with `-m 'not slow'`.
