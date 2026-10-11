# Overnight source inventory

`eval/overnight_source_inventory.py` is a replayable, offline-first inventory
for the local fixture and session roots. It reads audio headers and manifests
before decoding, hashes every accepted file, and bounds decoded signal metrics
to 60 minutes. The JSONL output is intentionally local/ignored because it
contains absolute source paths; no audio, video, weights, or private
transcripts are committed.

```bash
.venv/bin/python -m eval.overnight_source_inventory \
  --root "$HEARING_FIXTURES_DIR" \
  --output "$HEARING_RESULTS_DIR/overnight_grunt/inventory/source_catalog.jsonl" \
  --summary "$HEARING_RESULTS_DIR/overnight_grunt/inventory/summary.json"
```

Each row records path, bytes, SHA256, sample rate, channels, duration, role,
dataset/speaker/utterance IDs when explicitly declared, exact manifest
intervals, a transcript *reference* (never generated text), and peak/RMS/
silence fraction when within the decode budget. The summary separates missing
and corrupt files, duplicate hashes, decode totals, and `mix_checks`; a mix is
only marked consistent when the PCM16 mix differs from the declared stem sum
by at most two integer PCM units. Files not named by a manifest are classified
from their filename only and do not receive invented identity or transcript
metadata.

## Observed local availability

The shared generated fixture cache was present at
`/Users/joeroche/Developer/selective-hearing/eval/fixtures/generated` during
the inventory run. It contains 1272 files (audio plus manifests/video); 879
audio rows were accepted, 208 manifest mix groups were checked, 348 duplicate
hash groups were identified, and no missing or corrupt inputs were observed.
The run decoded exactly 3600 seconds before reaching the 60-minute cap. The
committed `eval/fixtures/tiny` set provides three usable audio rows offline.
Rows with missing stems or invalid headers are reported separately and are not
counted as usable. A rerun should record the exact JSON summary beside the
ignored JSONL catalog for T-211.

## Provenance and licence evidence

The inventory copies only licence/dataset fields already present in local
manifests. The public fixture sources identify LibriSpeech as CC BY 4.0
([OpenSLR SLR12](https://www.openslr.org/12/)) and GRID as CC BY 4.0
([Zenodo 3625687](https://zenodo.org/records/3625687)); these are the primary
source records to re-check before redistribution. Generated pink noise has no
upstream recording licence. An absent or unresolved licence remains
discovery-only and is never downloaded by this tool.

The tool does not transcribe, infer demographics, or validate transcript
content against audio. Existing manifest transcript references are therefore
validity evidence only when the producer supplies it; they are not new
measurements.
