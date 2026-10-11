# Overnight grunt split ledger

This ledger is the public-safe result of `eval/overnight_split_audit.py`. It records
metadata and exclusion reasons only; private absolute fixture paths are deliberately
omitted from `eval/manifests/overnight_grunt/splits.json`.

## Audit run

Command:

```text
.venv/bin/python -m eval.overnight_split_audit eval/fixtures \
  --output eval/manifests/overnight_grunt/splits.json
```

* Partition seed: **229**
* Input assets: **2**
* Eligible: **2** (1 enrollment, 1 development, 0 holdout)
* Excluded: **0**
* Media: 2 PCM16 mono WAV files, 16 kHz, 3.125 s and 2.970 s, 192,088 bytes total
* Transcript validity: 2/2 non-empty fixture transcripts; no ASR inference was run
* New acquisition: **0 bytes**; no network or private recordings were used

The two assets are the committed tiny LibriSpeech fixtures. Both have source IDs,
speaker IDs, dataset/split, and the fixture's recorded **CC BY 4.0** evidence.
The source material is LibriSpeech test-clean (OpenSLR SLR12), not a new overnight
collection. The audit intentionally reports no holdout-eligible asset because this
small bank does not provide enough speaker groups for a meaningful speaker-disjoint
holdout; it does not manufacture one.

Observed SHA256:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `eval/fixtures/tiny/speech_a.wav` | 96,044 | `c157d8212cbb5f161eae2c0bdf989da22c6d7625f97006975cdf9f865a76112b` |
| `eval/fixtures/tiny/speech_b.wav` | 96,044 | `266b20f2883bdb2039d177b5a0a6a3be69bf4857e6c7b72b0d9b23ee27b41e77` |
| `eval/fixtures/tiny/manifest.json` | — | `23a7d15d2f6ddbd1f8fb9f67d0627afaaf41b35782ad16d5d190e81e938bedbe` |
| `eval/manifests/overnight_grunt/splits.json` | — | `832b3cc21f8ba574abdd020c0010987176fd97284e0e9b4b814b916db7c03241` |

## Audit rules

Rows are grouped by `(speaker_id, source_id)` before deterministic assignment.
All crops and transformations of a source stay together. Positive-duration
overlapping intervals from one source are excluded, as are repeated SHA256 values
when they cross partitions or identify a transformed copy. A speaker assigned to
holdout cannot appear in another partition. Missing dataset or licence provenance
is an explicit `unknown_provenance` exclusion, never an implicit pass. The offline
tests cover both overlapping intervals and transformed same-source duplicates.

Completed grunt shards can be added by passing their manifest roots to the same
command. The script does not download, delete, or modify fixtures.

## Blockers and follow-up

This shard cannot establish a speaker-disjoint holdout until additional
provenance-complete speaker groups arrive. No data was counted as collected beyond
the two existing committed fixtures, and no licence was inferred for unknown
sources.
