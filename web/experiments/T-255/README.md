# T-255 · Blind A/B listening harness (prototype)

## Review status — 2026-10-10

Parked post-hackathon per Joseph's latest triage; round-one source recovered
from `f8ac0192778706b98e68d346f90def275c3b7dfe` on `task/T-255`.
This is an incomplete experimental prototype, not approved for production or
private media. No new listening study or audio rendering was performed.

Review blockers before running with any imported manifest:

- Imported clip paths are interpolated into HTML and are not restricted to local
  paths. Use only the bundled trusted example; sanitized imports and network
  exclusion are not enforced. Example WAV files are absent, so it cannot produce
  a valid listening result as shipped.
- Ratings only cover X/Y although validation accepts more than two clips.
- Resume lacks manifest identity and does not preserve unfinished trial order.
- Playback gating checks a play event, not completion or successful decoding.
- Matched-level playback, six-stage target/oracle/picked/mixer/AGC/delivered
  comparison, missing metrics, latency tradeoffs, and post-rating label reveal
  are not implemented. Unblinding is in the export and final summary only.
- The Python tests mirror the schema; they do not exercise JavaScript or prove
  browser behavior. Human listening and visual review remain pending.

Schema is described below and illustrated in `trials.example.json`. For a
future authorized local review, serve this directory on loopback with an
ephemeral port using `.venv/bin/python -m http.server 0 --bind 127.0.0.1
--directory web/experiments/T-255` from the repository root; read the printed
port and open `/index.html` manually. Do not upload or publish results.
The source patch can be obtained with `git show f8ac019 --` followed by the
three scoped paths listed in the task; this recovery does not push or open a PR.

Static page, no server, no build. Open `index.html` **over HTTP**
(`python3 -m http.server` in this dir — `fetch()` for the example manifest
does not work over `file://`).

## Use

1. Put your clips in a folder (e.g. `clips/`), referenced by relative path.
2. Write `trials.json` (see `trials.example.json`):
   each trial = one input segment × ≥2 conditions, each clip needs
   `id`, `file`, `condition`. `condition` is the truth — the page never shows it.
3. Open the page, load the manifest, enter a listener name.
4. Per trial: clips appear as randomized **X / Y**. Play each at least once
   (gating enforced), then pick X / Y / tie, optional 1–5 ratings
   (clarity, leak suppression, naturalness), PLAN.md failure tags per clip,
   optional comment.
5. Progress auto-saves to localStorage — refresh-safe.
6. "Download results JSON" exports unblinded records
   (`choice_condition`, `dimensions_by_condition`, `tags_by_condition`).

## Files

- `index.html` — page structure
- `ab.css` — dark theme matching web/console
- `ab.js` — trial logic (also exports `validateManifest`/`shuffle` for tests)
- `trials.example.json` — manifest format example (audio files not included)

## Out of scope (follow-ups)

- Python aggregator over multiple listeners' JSONs (win rates, κ agreement)
- A/B/X hidden-reference variant
- Manifest-time alignment self-check (equal durations ±10 ms)
