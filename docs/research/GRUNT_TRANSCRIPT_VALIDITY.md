# Overnight transcript validity audit

Audit shard: `transcripts` (T-220), run 2026-10-09.  The audit is metadata-only:
it does not run ASR, forced alignment, speaker identification, or a browser, and
it never promotes an absent transcript to a reference transcript.

## Reproduction

```bash
.venv/bin/python -m eval.overnight_transcript_audit \
  eval/fixtures/tiny eval/fixtures/demo --limit 300 \
  --output "$HEARING_RESULTS_DIR/overnight_grunt/transcripts/transcript_validity.jsonl"
```

The JSONL output is an ignored result artifact.  Each record includes the
manifest and opaque fixture directory name, role, declared dataset/licence,
source-file existence and SHA256, audio sample rate/duration when readable,
transcript status, utterance status, word-boundary status, safe interval, and
issues.  `--limit` bounds manifests (not records) and defaults to 300.

## Observed availability

The command above saw **5 manifests and 81 transcript-bearing entries**:
55 target, 25 other, and 1 tiny interferer entry.  All 81 entries have exact
text fields and declared licence strings, but this is not equivalent to
hand-verified audio truth.  There are **0 known-good complete-word public
cases** in this checkout: the two source-backed tiny entries have no explicit
word boundaries, while the GRID/demo sentence metadata has boundaries but its
media is absent.

| audit field | count |
|---|---:|
| exact reference text | 81 |
| automatic/unverified text | 0 |
| full-reference candidates | 80 |
| truncated | 1 |
| repeated/looped interferer | 0 |
| aligned word boundaries | 0 |
| missing word boundaries | 81 |
| source files present | 2 |
| source files missing | 79 |
| source IDs agreeing with metadata | 2 |
| source IDs missing | 79 |

The two source-backed records are `eval/fixtures/tiny/manifest.json` entries
`1580-141084-0019` and `61-70968-0002`; both are 16 kHz, 3.0 s, 48,000-frame
WAVs.  Their SHA256 values are
`c157d8212cbb5f161eae2c0bdf989da22c6d7625f97006975cdf9f865a76112b` and
`266b20f2883bdb2039d177b5a0a6a3be69bf4857e6c7b72b0d9b23ee27b41e77b`.
The tiny babble file is also present (16 kHz, 3.0 s, 48,000 frames), with
SHA256 `2bb0f2e1b43ee3b9b0bae0159e901056f1a9918d23480eaaf941fd971920bad7`;
it is noise, not a transcript case.

## Provenance and validity

The tiny manifest declares `openslr/librispeech_asr`, `test-clean`, and
`CC BY 4.0 (LibriSpeech, Panayotov et al. 2015)`.  The demo manifests declare
GRID and `CC BY 4.0 (GRID corpus, Cooke et al. 2006,
doi:10.5281/zenodo.3625687)`; their noise provenance declares LibriSpeech
test-clean and the same licence.  These are declarations copied from the
manifests, not an independent licence adjudication.  The audit intentionally
does not download unresolved sources.

The one truncated record is the tiny `speech_a` reference whose declared
source duration is longer than the fixture duration.  Demo sentence metadata
contains sentence start/end spans and repeated GRID utterance IDs in the
scene, but without the corresponding WAV/MP4 stems it remains discovery-only.
No private team text is emitted; any future team records should be represented
by opaque fixture IDs and retain their text locally.

## Handoff and blockers

There are no 12+ verified complete-word public cases available in the checked-in
roots, so none are relayed as usable truth to T-211/T-213.  The blockers are
missing demo stems/media and missing actual word-boundary annotations for the
two available tiny WAVs.  A future collection run may use the same CLI against
the shared public/tiny/GRID roots after media and primary-source licence
evidence are present; unresolved licence records must remain discovery-only.
