# Overnight word-reference preparation

`eval/prep_word_refs.py` prepares a private, replayable scoring plan from
existing transcript/alignment manifests. It is preparation only: it performs
no ASR, does not infer word boundaries, and never treats an unverified
transcript as a scored reference.

## Input contract

Pass a JSON list of cases, or an object with a `cases` list and optional
`seed`. Each case may contain `id`, `crop: {"start", "end"}`, `source`, and
`target` / `interferer` records. A speaker record needs `speaker`,
`transcript`, and `alignment`; each alignment entry must contain a word (or
text), numeric `start`, and numeric `end` in seconds. Source provenance is
copied privately and hashed. The crop is applied conservatively: only words
whose actual boundaries are wholly inside the crop are emitted.

Transcript-only, crop-only, speaker-absent, and crop-mismatched cases are
explicitly unavailable (or contain only the complete words that remain).
Repeated words retain a 1-based occurrence number. Words present in both
target and interferer references are listed as ambiguous and are not assigned
to either speaker. The recipe scores target deletions and competitor-only
insertions; it does not claim a measured WER.

## Replay command

```bash
.venv/bin/python -m eval.prep_word_refs \
  --input "$HEARING_FIXTURES_DIR/<set>/word_manifest.json" \
  --output "$HEARING_FIXTURES_DIR/overnight_prep/T-237/word_refs.json" \
  --summary docs/research/word_refs_summary.json \
  --checkpoint-dir "$HEARING_FIXTURES_DIR/overnight_prep/T-237/checkpoints" \
  --max-cases 500
```

The private output may contain speaker names, utterance identifiers, source
intervals, and hashes; keep it under the ignored fixtures/results area. The
summary is safe to publish only when its input path and source data are
public-safe. Output writes are atomic. The 500-case cap is a deterministic
stop; the command reports planned, available, unavailable, and ambiguous
counts separately. The `seed` and configuration hash are recorded even
though no random operation is used.
With `--checkpoint-dir`, each case is atomically written under a
content-addressed filename and a valid matching checkpoint is reused on
restart; malformed or mismatched checkpoints are recomputed.

## Consumers

T211/T213/T217 can consume `cases[*].target.words` as reference boundaries,
use `cases[*].recipe` for deletion/insertion scoring, and exclude
`ambiguous_shared_words`. A consumer must stop or mark a case unavailable
when `status` is `unavailable`; it must not fill missing boundaries with ASR
timestamps.

The T220 audit confirms that delayed TSE fallback is a delayed mixture rather
than a clean interferer stem. This plan therefore records source references
only and does not derive a competing waveform or claim separator quality.
