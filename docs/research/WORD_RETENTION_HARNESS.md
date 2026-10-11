# Target word-retention harness

`eval.score_word_retention` scores ASR transcripts from separated/augmented
scenes against the clean target transcript. It is deliberately transcript-only:
ASR adapters and audio preparation run separately, so overnight scoring is
deterministic, offline, and does not need model weights or network access.

## Input contract

Pass one or more `.jsonl`/`.json` files, or directories containing them. JSONL
has one object per scene; JSON accepts either a list or
`{"scenes": [...]}`. Each scene requires:

| Field | Meaning |
|---|---|
| `scene_id` (or `id`) | Stable scene identifier |
| `interference_ratio` (or `sir_db`, `interference_db`, `ratio`) | Interference level, normally SIR in dB |
| `target_text` (or `reference`, `clean_target_text`) | Clean target transcript |
| `separated_transcript` (or `hypothesis`, `output_transcript`, `asr_text`) | ASR transcript of separated audio |

Transcript text is lowercased, punctuation-normalized, and whitespace-collapsed
using `eval.metrics.normalize_text`. Empty target transcripts are rejected
because word loss has no denominator.

## Metrics and output

Each scene emits:

- **WER**: substitutions + deletions + insertions divided by reference words.
- **Word loss percentage**: deletions divided by reference words, multiplied by
  100. Substitutions are not counted as lost words; this keeps retention
  distinct from intelligibility.

The report contains `scenes` and deterministic `by_interference_ratio` groups.
Group WER is corpus-independent mean scene WER, while `reference_words` and
`deletions` are also retained for a corpus-level loss calculation. The scorer
does not infer whether an ASR error came from interference.

```bash
.venv/bin/python -m eval.score_word_retention \
  eval/shards/word_retention \
  --output eval/results/word_retention.json
```

For each augmented scene, retain the clean target text and the exact
interference ratio in the same record. This prevents accidental comparison of
different target utterances and makes ratio sweeps reproducible.
