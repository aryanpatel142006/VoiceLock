# Muse grunt: transcript, alignment and speaker-metadata sources (T-226)

Run 2026-10-10 on the cloud Muse host (no GPU, no model inference, no force alignment run, no browser). Machine-readable rows: `eval/manifests/overnight_grunt/muse_metadata.json` (schema `muse_metadata/v1`, 12 source rows). Bulk files are in the shared gitignored roots `$HEARING_FIXTURES_DIR/overnight_grunt/muse_metadata/` and `$HEARING_RESULTS_DIR/overnight_grunt/muse_metadata/` (main checkout `/Users/aryanpatel/CODE/selective-hearing/eval/{fixtures/generated,results}/overnight_grunt/muse_metadata`).

## 1. Observed HTTP / download results (measured)

| item | command (all `curl`, UA `T-226-research`) | result |
|---|---|---|
| OpenSLR 12 page, licence | `curl -sL https://www.openslr.org/12/` | 200, 8245 B, sha256 `9b012c60f8fe6a5435f96cf034c042ffe4222e7936c88cb835671413e74a5301`; text says `License: CC BY 4.0` |
| `raw-metadata.tar.gz` | `curl -sL -o raw-metadata.tar.gz https://www.openslr.org/resources/12/raw-metadata.tar.gz` | **downloaded 33,373,768 B**, md5 `25eced105e10f4081585af89b8d27cd2` = official `md5sum.txt` |
| SPEAKERS.TXT / CHAPTERS.TXT / BOOKS.TXT (extracted) | `tar xzf` | sha256 `3350f093…8f76`, `2e9db25c…6e0`, `2e5a2772…ff0` (full values below); SPEAKERS has 2484 readers; dev-clean 20F/20M, test-clean 20F/20M |
| LibriSpeech Alignments (Zenodo 2619474) | HEAD + `curl -r` Range | licence `cc-by-4.0` in the API; zip 623,023,332 B (not downloaded: over budget); **central directory fetched by Range, 32,664,432 B, 300,642 entries** (per-utterance `.TextGrid`) |
| TextGrids for speech.json utterances | `curl -r <off>-<end>` per entry, deflate decoded, CRC32 checked | **20 of 96 fetched** (speakers 2035 ×8, 2277 ×8, 2086 ×4), 149,243 B decoded, all CRC OK; the other 76 were **not fetched** (Zenodo Range requests ran ~25 s each and the 35-min limit stopped the job) |
| GRID Zenodo 3625687 | `curl https://zenodo.org/api/records/3625687` | 200, licence `cc-by-4.0`; `alignments.zip` 20,091,485 B md5 `df892a5a…9f` matches the local cache (existing, not re-downloaded) |
| AMI annotations 1.6.2 | cached by another Muse shard | 22,887,865 B, md5 `b9db0918…d43d475`, sha256 `b56e5bab…f9`; licence page states CC BY 4.0 |
| VoxCeleb1 metadata | `curl -o vox1_meta.csv …/meta/vox1_meta.csv` | 40,782 B, sha256 `48bcb2f0…ab0892`; 1251 speakers |
| HEAD-only checks | | `original-mp3.tar.gz` 87,960,560,420 B; `train-clean-100.tar.gz` 6,387,309,499 B; `dev-clean` 337,926,286 B; Zenodo GRID `audio_25k.zip` 2,607,605,854 B; AMI zip 22,887,865 B |

**New network bytes this run: 66,078,982 (≈63 MiB: 33,373,768 + 32,664,432 + 40,782) plus ~0.2 MB of TextGrid Range data (not exactly logged)**, inside the 512 MiB cap (speech 256). The GRID and AMI caches predate this task and are not counted. Full sha256s: SPEAKERS `3350f093382f8baa1d589c9f5a5cee8ec3705e656f521d6eafbaa31a5fcd8f76`, CHAPTERS `2e9db25c250a143b031ca003180acfce84b364e31fbc79250e306303e6f25306`, BOOKS `2e5a277207aeecad9f85db53f97e46320419c3d80e330d6524193b283ff5edb0`, LICENSE.TXT `908b0b565ae9bf61b7b41447cb0b83564f320375723bcc884ec30e5537a0d5dc`.

## 2. What the data shows

**LibriSpeech.** Utterance id is `speaker-chapter-index` (e.g. `2277-149874-0001`). `CHAPTERS.TXT` maps each chapter to its LibriVox project id and Gutenberg book id: the 96 `speech.json` utterances (12 speakers) fall in **30 chapters**, all resolved (rows `librispeech_chapters`; reader names are deliberately not copied). Sex per speaker is the corpus's own `SPEAKERS.TXT` label (`librispeech_speakers`, 12 rows, with subset and corpus minutes). Official transcripts are complete per utterance but have no word times.

**MFA alignments (third party).** Of the 20 fetched TextGrids, all word tiers end within the audio duration (last word ends 0.155 s before to ≤0 s after the clip end, ratio ≥0.81). The aligned word list equals the normalised reference in **15/20**; the other 5 differ (4 `<unk>`/`spn` tokens across 3 of them; one differs with no unknown token, unexplained, not investigated). These are model output, not hand truth: use them as a timing prior, not as reference.

**GRID.** `alignments.zip` has 34 speakers × 1000 = 34,000 `.align` files, 0 malformed lines, 204,000 non-silence word tokens. The time unit is **1/25000 s** (the largest end value 74,500 = 2.98 s), matching the 25 kHz `audio_25k.zip`; convert to seconds by dividing by 25000, and resample on your own edge. The official page gives only "34 talkers (18 male, 16 female)": no per-speaker table was obtained, so **no per-speaker sex is recorded**.

**AMI.** `ES2008a` has four words XMLs (speaker slots A–D) with 2,508 non-punctuation words with start/end times; the cached `ES2008a.Headset-0.wav` and `Array1-01.wav` are 16 kHz mono, 1043.36 s, 33,387,564 B each. `meetings.xml` gives channel/role/anonymised speaker codes; sex was not read as a field and is not recorded.

**VoxCeleb1.** Metadata only: 690 m / 561 f, dev 1211, test 40; no transcripts, so it supports speaker-similarity metrics only. Names are not stored.

## 3. Which source enables which metric (recommendation)

| source | licence | target WER | competitor word leakage | other |
|---|---|---|---|---|
| LibriSpeech transcripts | CC BY 4.0 | yes, per utterance | yes if the competitor is also LibriSpeech with a distinct speaker | speaker sex, enrollment/eval by speaker |
| LibriSpeech + Zenodo MFA TextGrids | CC BY 4.0 | yes | yes, plus word-level windows (timing prior) | 76/96 still to fetch |
| GRID + alignments | CC BY 4.0 | yes (closed-vocab sentences) | yes, word windows exact to the corpus tool | no per-speaker sex |
| AMI (manual words) | CC BY 4.0 | yes (conversational, overlapping) | yes (per-speaker word times) | meeting-level speaker slots |
| VCTK (discovery only) | CC BY 4.0 | yes in principle | yes | ~11 GB, not downloaded |
| VoxCeleb1 meta | CC BY-SA 4.0 (audio unresolved) | **no** | **no** | sex/nationality, EER/similarity only |
| LibriCSS, CorentinJ alignments, MFA tool, `original-mp3` | unresolved / not run / over budget | not enabled here | not enabled here | discovery-only |

## 4. Re-run recipes
```bash
curl -sL -o raw-metadata.tar.gz https://www.openslr.org/resources/12/raw-metadata.tar.gz   # md5 25eced105e10f4081585af89b8d27cd2
tar xzf raw-metadata.tar.gz LibriSpeech/SPEAKERS.TXT LibriSpeech/CHAPTERS.TXT LibriSpeech/BOOKS.TXT
# SPEAKERS.TXT: ID | SEX | SUBSET | MINUTES | NAME ; CHAPTERS.TXT: ID | READER | MINUTES | SUBSET | PROJ | BOOK ID | ...
# MFA TextGrid for one utterance: read the zip central directory by Range (last 65,536 B give the EOCD at offset 590358900), then
curl -sL -r <local_header_offset>-<offset+30+len(name)+extra+compressed_size> \
  https://zenodo.org/api/records/2619474/files/librispeech_alignments.zip/content
# zlib.decompress(data, -15), check CRC32; path dev-clean/<spk>/<chapter>/<utt>.TextGrid, tier "words"
```
GRID: `unzip alignments.zip` then `alignments/sN/<utt>.align` lines `start end word` (units 1/25000 s). AMI: `words/<meeting>.<A-D>.words.xml`, `<w starttime endtime>`; skip `punc="true"`.

## 5. Consumers
Send actionable metadata to Copilot-b2, Copilot-b4 and T-213 (see `notes_for_reviewer` and `messages`): use `muse_metadata.json` rows, treat 76 TextGrids as missing, treat GRID time units as 1/25000 s, never use VoxCeleb meta for WER.

## 6. Blockers and not done
- 76 of 96 LibriSpeech TextGrids not fetched; Zenodo Range was slow. A follow-up can fetch them with the recipe above.
- No forced alignment was run (no MFA, by scope). No per-speaker GRID sex, no AMI sex field.
- VCTK, `original-mp3`, LibriCSS and CorentinJ are discovery-only (size, or licence unresolved/unverified).
- Board relays and attaching these files to the task page could not be done from this worker (no board API access); the harness relays `messages` in the report. Local commits are not transferred data.
