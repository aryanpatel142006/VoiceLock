# Overnight public-data kit (T-209)

Scope: eval-only. No runtime, default, profile or checkpoint change; nothing here adopts a dataset in the demo. No audio or weights are committed. Code: `eval/public_voice_sources.py`; manifest: `docs/research/public_voice_manifest.json`.

## Shortlist (licence and links verified live on 2026-10-09)

| # | Dataset | Use | Licence | Direct source | Status |
|---|---|---|---|---|---|
| 1 | LibriSpeech test-clean | Target/interferer speech, speakers 6930 and 1320, transcripts | CC BY 4.0 (OpenSLR page, read this run) | https://www.openslr.org/resources/12/test-clean.tar.gz | downloaded, md5 verified |
| 2 | DEMAND PCAFETER (16 kHz, 16 ch) | Café noise bed | CC BY 4.0 (Zenodo API `license.id=cc-by-4.0`) | https://zenodo.org/records/1227121 (doi 10.5281/zenodo.1227121) | downloaded, md5 verified |
| 3 | DEMAND OMEETING (16 kHz, 16 ch) | Meeting-room babble/reverberant ambience | CC BY 4.0 (same record) | same record, `OMEETING_16k.zip` | downloaded, md5 verified |

Room impulse responses (OpenSLR 28 RIRS_NOISES, Apache 2.0 per the OpenSLR page, ~1.3 GB) were the natural 4th pick but were not downloaded: it would take the directory to ~1.8 GB, close to the 2 GB cap, and the 30-minute download budget was spent on 1–3. Not downloaded is a gap, not a decision against it.

## Download evidence (commands, actual bytes)

```
HEARING_FIXTURES_DIR=<shared fixtures dir> .venv/bin/python -m eval.public_voice_sources download
.venv/bin/python -m eval.public_voice_sources manifest --out docs/research/public_voice_manifest.json
```

Files live in `$HEARING_FIXTURES_DIR/public_voice/`; directory total 536,798,833 bytes (cap 2,000,000,000).

| File | Bytes | md5 (equals publisher) | sha256 |
|---|---|---|---|
| test-clean.tar.gz | 346,663,984 | 32fa31d27d2e1cad72775fee3f4849a9 | 39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23 |
| PCAFETER_16k.zip | 107,431,494 | 99927d148128254141a9417d051510bb | 89be7194b83cd583c4b12f4bf7c19e9cf0e612b95312e9f4aedc968cd811bcd5 |
| OMEETING_16k.zip | 82,703,355 | 62f7cfe7fe6d30b7d8a215fe37c2dfd2 | 512016cad3a42511a0fb7e3c74d6924e6efde4921a6101361f1306bf90bf2e5a |

An earlier attempt truncated OMEETING at 22,617,556 bytes; the md5 check caught it and the downloader now resumes with `Range` (resumed from that byte count, then verified). Python's urllib failed TLS verification on this Mac until it was given certifi's CA bundle; the script now does that.

## Manifest: speakers, transcripts, disjoint enrollment/eval

`public_voice_manifest.json` holds, per LibriSpeech utterance: speaker, utterance id, archive path, sha256 of the FLAC, seconds, transcript and role. Per speaker the first utterances in id order fill ~20 s of enrollment; the rest evaluate. `check_disjoint` fails if any recording hash is in both roles.

| Speaker | Enroll utts / s | Eval utts / s |
|---|---|---|
| 6930 | 3 / 22.8 | 75 / 457.3 |
| 1320 | 2 / 23.0 | 57 / 458.3 |

The manifest also lists the 16 channel files of each DEMAND zip. Extract audio from the archives yourself; paths are archive-relative.

## Other datasets considered (not downloaded)

Licences below are from memory and were **not re-checked this run**; confirm on the publisher page before use.

| Dataset | Relevance | Licence (unverified) | Why not now |
|---|---|---|---|
| OpenSLR 28 RIRS_NOISES | Simulated + real RIRs, noises | Apache 2.0 (page read) | ~1.3 GB, cap |
| LibriSpeech dev-clean/test-other | More speakers, harder audio | CC BY 4.0 | test-clean already enough speakers |
| VCTK (Edinburgh DataShare) | 110 speakers, accents | CC BY 4.0 | studio, ~10 GB |
| AMI meeting corpus | Real conversational overlap | CC BY 4.0 | large, per-meeting downloads |
| LibriMix / WHAM! | Ready mixes | LibriMix CC BY 4.0; WHAM! CC BY-NC 4.0 | non-commercial noise, mixes large |
| MUSAN | Speech/music/noise | CC BY 4.0 (OpenSLR 17) | ~11 GB |
| ACE Challenge / EARS-Reverb / BUT ReverbDB | Measured room responses | mixed, check each | not verified |
| VoxCeleb | Speaker ID | restricted/YouTube-derived | not redistributable, expiring-style sources |

## Limits and missing

- Conversational overlap coverage is **missing**: no licensed small conversational set was downloaded (AMI is the candidate). OMEETING is ambient, not conversation.
- Reverberant coverage is only OMEETING ambience; no measured RIRs yet.
- Only two speakers; any eval built from this is a demo-scale check, not a population result.
- No model was run. Nothing here supports a quality claim.

## Attribution

LibriSpeech: Panayotov, Chen, Povey, Khudanpur, ICASSP 2015 (CC BY 4.0). DEMAND: Thiemann, Ito, Vincent, 2013, doi:10.5281/zenodo.1227121 (CC BY 4.0). Changes: none to the archives; the manifest only indexes them.
