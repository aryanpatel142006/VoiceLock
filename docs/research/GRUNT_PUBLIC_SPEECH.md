# Overnight speech shard (T-218)

Source collection only: no mixtures, no inference.

## What was collected
- Source: LibriSpeech `dev-clean`, official `https://www.openslr.org/resources/12/dev-clean.tar.gz` (337,926,286 bytes full; not downloaded). Licence CC BY 4.0, stated on `https://www.openslr.org/12/` (fetched Oct 9 2026). Citation: Panayotov et al., ICASSP 2015.
- Method: `eval/overnight_speech_collect.py` streams the gzip tar over one HTTPS response and closes it at the byte cap (default 200 MiB, hard cap 256 MiB). Tar members are ordered by speaker, so a prefix gives complete speakers.
- Observed: **12 speakers, 96 utterances (8 each), 98,600,960 network bytes (94.0 MiB)**, aborted at budget: no; full tarball: no. 721.6 s of speech, 13,125,843 bytes of native FLAC. Existing caches (T-209's `public_voice/test-clean.tar.gz`) were not touched and no bytes were re-downloaded.
- Speakers (LibriSpeech ids): 84, 777, 1673, 1988, 2035, 2086, 2277, 5536, 6295, 6345, 7976, 8297. Sex (F/M) comes from the corpus's own `SPEAKERS.TXT` shipped inside the tarball; no other identity data is stored.
- Roles per speaker: 3 enrollment, 2 development, 3 holdout. Each is a whole, distinct utterance (never a crop of another), picked round-robin across chapters so roles rarely share a recording block. Utterances are 2–15 s.
- Native format: 16 kHz mono FLAC, written verbatim with SHA256. `derived_16k_mono/*.wav` is a PCM16 re-encode only; the collector refuses to derive anything whose native rate is not 16 kHz mono (no silent resampling). All 96 were already 16 kHz mono.
- Transcripts: the original corpus `*.trans.txt` (upper case), with a `transcript_normalized` field from `eval.metrics.normalize_text`. None invented; empty ones are rejected.

## Where things are
- Public manifest (commit-safe, relative paths, SHA256 per clip): `eval/manifests/overnight_grunt/speech.json`.
- Media (gitignored): `$HEARING_FIXTURES_DIR/overnight_grunt/speech/{native,derived_16k_mono}/` (main checkout: `/Users/aryanpatel/CODE/selective-hearing/eval/fixtures/generated/overnight_grunt/speech`, 35 MB).
- Full report with absolute paths: `$HEARING_RESULTS_DIR/overnight_grunt/speech/collect.json`.
- Re-run: `.venv/bin/python -m eval.overnight_speech_collect --speakers 12` (same bytes, deterministic selection). Needs `HEARING_FIXTURES_DIR` / `HEARING_RESULTS_DIR` (set by `scripts/_py.sh`).

## Usage for consumers
Enroll from `role == "enrollment"`, tune on `development`, report only on `holdout`. Load `native.path` relative to the media dir; use `derived_16k_mono` when a plain WAV is needed. Pair different `speaker_id`s for overlap; never score a speaker against its own enrollment utterances.

## Gaps
- LibriSpeech is read audiobook speech, not café or conversational speech; accents and recording quality vary by volunteer.
- 12 of dev-clean's 40 speakers only (budget-bounded prefix; speaker ids are not randomly sampled, they are the first in tar order).
- No licence/source was left unresolved, so there is no discovery-only list for this shard.
