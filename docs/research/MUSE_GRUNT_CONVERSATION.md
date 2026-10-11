# Muse grunt: conversational and multi-speaker speech source catalog (T-224)

Date 2026-10-10. Machine-readable twin: `eval/manifests/overnight_grunt/muse_conversation.json`. Media lives in `$HEARING_FIXTURES_DIR/overnight_grunt/muse_conversation/` (gitignored). No GPU, no ASR, no model weights were used.

## A. Observed (actual HTTP and download results)

Pilot download from the official AMI host (meeting ES2008a, 4 participants), 89,662,993 bytes in total (85.5 MiB, inside the 128 MiB Muse pilot budget):

| File | Bytes | Rate / ch / depth | Duration | SHA256 |
|---|---|---|---|---|
| ES2008a.Array1-01.wav (far-field, reverberant) | 33,387,564 | 16 kHz / 1 / 16-bit | 1043.36 s | 2382b3b7eea81ff6393555680c660f4b00be3f0971aef2a0a71197bb5109eae1 |
| ES2008a.Headset-0.wav (close-talk, speaker FEE029) | 33,387,564 | 16 kHz / 1 / 16-bit | 1043.36 s | 9038704a0c2380b3abed17af3927e110a3783d3e8a0a7743af86275552f5588a |
| ami_public_manual_1.6.2.zip (annotations) | 22,887,865 | n/a | n/a | b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d |

- Speakers (corpusResources/meetings.xml): FEE029 (A, ch 0), FEE030 (B), MEE031 (C), FEE032 (D). Headset-0 therefore maps to FEE029.
- Transcript validity: the zip opens (5183 entries) and `words/ES2008a.{A,B,C,D}.words.xml` exist, with 1291 / 644 / 474 / 481 word tags (speakers A/B/C/D). This is a structural check only; no ASR, WER or alignment accuracy was measured.
- Caveats: ES2008a is a training-split meeting. Headset audio carries cross-talk bleed, so it is a near-clean reference, not a studio stem.
- Failed: `https://www.openslr.org/resources/119/Eval_Ali.tar.gz` returned 404 (AliMeeting files are on the aliyuncs host listed below); `https://s3.amazonaws.com/dipco/DiPCo.tgz` returned 403.
- HEAD/Range probes (no download): AliMeeting Eval 3,673,718,355 B and Test 9,555,497,666 B on the aliyuncs host, Accept-Ranges bytes, a 1 KiB Range returned 206; AISHELL-4 test.tar.gz 5,241,010,904 B, 206; VoxConverse dev zip 1,988,647,478 B, test zip 4,272,764,397 B; ICSI core annotations 19,488,722 B.

Commands (reproducible): `curl -O` of the three URLs in the manifest into one fresh directory; `sha256` and `wave` header read with `python -I` from a different cwd.

## B. Source catalog (13 rows; licence facts read from the cited primary page this run)

| # | Dataset | Status | Licence (primary) | Clean refs / transcripts | Subset size and recipe |
|---|---|---|---|---|---|
| 1 | AMI | **Downloaded pilot** | CC BY 4.0, https://groups.inf.ed.ac.uk/ami/corpus/license.shtml | per-speaker headset + word XML; Array1-01 reverberant | 33.4 MB per channel; `curl -O .../AMICorpusMirror/amicorpus/<MEETING>/audio/<MEETING>.{Array1-01,Headset-0..3}.wav` |
| 2 | ICSI Meeting Corpus | Discovery (annotations reachable) | CC BY 4.0 per https://groups.inf.ed.ac.uk/ami/icsi/download/ | per-channel headset SPH, NXT transcripts | core annotations 19.5 MB; audio per-meeting links not resolved this run |
| 3 | AliMeeting (Mandarin) | Reachable, not downloaded | CC BY-SA 4.0, https://www.openslr.org/119/ | headset near-field + transcripts; 2-4 speakers | Eval 3.67 GB; stream a prefix: `curl -s -r 0-134217727 <Eval_Ali url> \| tar xzf -` |
| 4 | AISHELL-4 (Mandarin) | Reachable, not downloaded | CC BY-SA 4.0, https://www.openslr.org/111/ | transcripts; 4-8 speakers; clean stems not stated | test 5.24 GB; same prefix trick |
| 5 | CHiME-6 | Reachable, too large | CC BY-SA 4.0, https://www.openslr.org/150/ | in-ear + arrays, transcripts | dev 11 GB, eval 12 GB (OpenSLR page) |
| 6 | VoxConverse | Reachable, not downloaded | CC BY 4.0, copyright stays with video owners | RTTM only, no transcripts, no stems | dev 1.99 GB, test 4.27 GB |
| 7 | LibriSpeech dev-clean | Cached by T-218 | CC BY 4.0, https://www.openslr.org/12/ | clean single-speaker references | 337,926,286 B; see speech.json |
| 8 | LibriCSS | **Discovery only: licence unresolved** | none stated in README; GitHub API reports no SPDX | playback sources included | Google Drive id 1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l, not fetched |
| 9 | DiPCo | **Discovery only: licence unresolved** | no primary page found; S3 403 | n/a | none |
| 10 | MISP | **Discovery only: licence unresolved** | page shows only a site-template licence | n/a | none |
| 11 | LibriMix generator | Recipe only | code MIT; data derives from LibriSpeech plus WHAM noise (WHAM licence not verified here) | clean sources | Libri2Mix ~430 GB per README |
| 12 | MMS-MSG / SMS-WSJ | Recipe only | code MIT; SMS-WSJ needs WSJ (LDC) | clean sources | generator; no data |
| 13 | Fisher / CALLHOME / LDC | **Gated, paywalled** | LDC membership | n/a | not acquirable |

Fields not checked (speaker counts for ICSI and CHiME-6, ICSI audio URLs) are marked "not extracted" in the manifest; none were guessed. Only 13 of up to 15 rows were researched.

## C. Recommendation: three to acquire tonight

1. **AMI** (done for ES2008a): repeat per meeting for more reverberant far-field plus per-speaker close-talk pairs. About 67 MB per meeting for one array and one headset; annotations are already in hand.
2. **AliMeeting Eval via ranged prefix**: real far-field meeting audio with headset references, 2-4 speakers, CC BY-SA 4.0. Mandarin only, so use it for separation or leakage tests, not English WER. Keep it under the remaining pilot budget (about 40 MiB left after AMI).
3. **ICSI**: CC BY 4.0, transcripts reachable (19.5 MB). The per-meeting audio URL pattern must still be read from the download page before use.

Do not acquire the three licence-unresolved rows or the LDC row. VoxConverse has no transcripts or stems and is large.

## D. Debts

- Row facts for ICSI audio, CHiME-6 speaker counts and WHAM noise licence were not verified.
- Only AMI was downloaded; no other row has downloaded evidence.
- Transcript validity is a structural count, not an accuracy check.
- verify_fast result: see the report.
