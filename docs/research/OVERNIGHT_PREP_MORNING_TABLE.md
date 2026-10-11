# Overnight preparation: morning evidence table

`eval/prep_morning_table.py` is an offline, replayable reducer for result
JSON/JSONL produced by overnight jobs. It does not run ASR, GPU work, or
choose a production default. It groups actual rows by `stage`, `speaker`,
`sir_db`, and `condition`, and reports the number of rows received separately
from the number of rows containing each score.

```sh
.venv/bin/python -m eval.prep_morning_table overnight-results.json \
  --output morning-table.json --checkpoint morning-table.checkpoint.json
```

The reducer accepts a JSON object containing `rows`, `results`, or `records`,
a JSON array, or one JSON object per line. Output is public-safe metadata:
input filename/hash, stage dimensions, sample and missing counts, p50/p95
latency, and mean PESQ-WB/DNSMOS SIG when present. Missing scores are
`null` and produce `quality_gate: "unknown"`; no defaults are substituted.
The hard gate is `pass` only at PESQ-WB >= 2.8 and DNSMOS SIG >= 3.4, and is
`reject` below PESQ-WB 2.5 or DNSMOS SIG 3.2. Intermediate values are
`review`.

`comparison_eligible` is true only for target-present rows with exactly one
non-missing input/source hash, model revision, and config revision in a
group. Thus invalid or absent hashes cannot silently support a comparison.
The checkpoint is written atomically after the output and records the input
hash; rerunning the command safely replaces the output. Planned cases are
never counted: `samples_input` and each group's `samples` are actual rows
consumed.

Inputs must carry provenance from the producing job (`input_hash` or
`source_hash`, `model_revision`/`model_hash`, and `config_revision`/`config_hash`).
The script does not invent speaker labels, SIR values, transcripts, intervals,
scores, or licence claims. Source/utterance/interval/hash and licence
evidence remain the responsibility of the producer manifest; private paths
and transcripts must not be placed in the public result.

