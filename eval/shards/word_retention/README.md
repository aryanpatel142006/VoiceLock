# Word-retention shards

This directory is reserved for small, checked-in transcript fixtures. Overnight
audio and ASR outputs belong in the generated/results directories and should
not be committed. Each JSONL line is one augmented scene:

```json
{"scene_id":"sir_-5_0001","interference_ratio":-5,"target_text":"the quick brown fox","separated_transcript":"the brown fox"}
```

`interference_ratio` is normally SIR in dB; the scorer also accepts `sir_db`,
`interference_db`, or `ratio`. The reference accepts `reference` and
`clean_target_text`; the hypothesis accepts `hypothesis`, `output_transcript`,
or `asr_text`. A `reference` plus `transcript` pair is also accepted for
manifests produced by the existing transcription tools.

Run the offline scorer on a shard or a directory of shards:

```bash
.venv/bin/python -m eval.score_word_retention eval/shards/word_retention \
  --output eval/results/word_retention.json
```
