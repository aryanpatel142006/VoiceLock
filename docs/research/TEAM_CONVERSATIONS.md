# Synthetic team conversation benchmark

`eval/prep_team_conversations.py` builds a deterministic, offline benchmark from
**disjoint** solo fixture takes:

```text
$HEARING_FIXTURES_DIR/team/
  aryan/fixture-*.wav
  joseph/fixture-*.wav
```

The generator rejects missing speakers, silent takes, wrong sample rates, and
byte-identical takes across speakers. It does not read `profile*.wav` or the
existing `team/conversation/` recording. Run it with:

```bash
.venv/bin/python -m eval.prep_team_conversations \
  --source-root "$HEARING_FIXTURES_DIR/team" \
  --manifest eval/fixtures/generated/team_conversations/manifest.json
```

Every rendered scene is exactly 10 seconds, mono 16 kHz. The manifest contains
one 200 ms-pause alternating scene, three simultaneous-overlap scenes at target
to interferer SIR `-3`, `0`, and `+3 dB`, and three quick-interruption scenes
where Joseph enters 500 ms after Aryan starts a turn. `mix.wav`, `target.wav`,
and `interferer.wav` are generated beside each scene directory; their paths are
relative to the manifest, and all rendered audio is gitignored.

The overlap SIR is measured over the rendered non-zero stems and is enforced by
scaling the interferer after placement. The fixed seed selects takes
reproducibly, so an overnight run can be compared byte-for-byte at the manifest
level. The committed manifest is an index/template; it is regenerated when the
team's disjoint takes are available.

## Data handling

Only the two named fixture directories are inputs. Profile recordings are kept
out of this benchmark to prevent enrollment/test leakage. No network, model
weights, or hardware devices are used.
