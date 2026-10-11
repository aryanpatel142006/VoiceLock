# Muse grunt: noise and room-response source catalog (T-225)

Run 2026-10-10 on a cloud/laptop host, CLI/HTTP only. Machine-readable twin: `eval/manifests/overnight_grunt/muse_acoustics.json`. Media is in the shared ignored shard `$HEARING_FIXTURES_DIR/overnight_grunt/muse_acoustics/native/` (not committed). No inference, no transcripts, no existing shard touched. Complements T-219's `GRUNT_NOISE_RIRS.md` (ESC-50 + OpenSLR28 RIRs), which already filled the ESC-50 / recorded-RIR rows.

## Actual downloads (observed)
Command: HTTP range read of the zip central directory plus one member (`ch01.wav`) from Zenodo record 1227121 (DEMAND, CC BY 4.0; helper kept in `/tmp`, not committed).

| file | bytes | SHA256 | rate / dur |
|---|---|---|---|
| demand_pcafeter/ch01.wav (cafeteria) | 9,600,172 | 854dd0e5ef334a66abdd02d6d2b3e9d01b685df033d0acd1cd06e471c062550b | 16 kHz mono, 300.004 s |
| demand_straffic/ch01.wav (street traffic) | 9,600,172 | 8fa49dcf3996af655a3ecef8d06f0d9ef217c35410e3adce3d461da797987dbb | 16 kHz mono, 300.004 s |
| demand_dwashing/ch01.wav (appliance) | 9,600,172 | da02faca8c6523d24ee86aae6c6889c950bb34d559f1379cc87e7e3b75690f4d | 16 kHz mono, 300.004 s |

Network bytes ≈ 20.7 MB (6,790,961 + 7,493,929 + 6,424,352) against the 128 MiB noise budget; Muse-pilot and speech budgets unused. **Failed:** OOFFICE ch01 attempt produced no output; not retried, not counted. Zero transcripts (noise only). Stopped well inside 35 min.

## Catalog (15 sources; full fields in the JSON)
Status `available` = licence verified from the primary page/record and a small-subset path exists; `discovery-only` = no download done.

| source | kind | recorded/sim | licence (evidence) | status |
|---|---|---|---|---|
| DEMAND | noise: cafeteria, traffic, appliance, office, metro | recorded | CC BY 4.0 (Zenodo 1227121, HTTP 200) | available, 3 clips downloaded |
| OpenSLR 28 RIRS_NOISES | RIR + noise | real recorded + simulated | Apache 2.0 (openslr.org/28; range OK, 1,311,166,223 B) | available (T-219 has 11 RIRs) |
| OpenSLR 26 sim_rir | RIR | simulated | Apache 2.0 | available, not downloaded |
| ESC-50 | noise | recorded | CC BY-NC 3.0 | available (T-219 has 33 clips) |
| MUSAN | noise/music/speech | recorded | CC BY 4.0 (openslr.org/17) | discovery-only: 11 GB tar.gz |
| BUT ReverbDB | RIR + room noise | recorded | CC BY 4.0 (vutbr page; 9 rooms, volumes given, no RT60) | discovery-only: 8.7 GB tgz |
| dEchorate | RIR, babble | recorded | CC BY 4.0 (Zenodo 6576203) | discovery-only: HDF5 0.7–11 GB |
| MeshRIR | RIR grid | recorded | CC BY 4.0 (Zenodo 10852693) | discovery-only |
| Aalto ARNI | RIR | recorded | CC BY 4.0 (Zenodo 6985104) | discovery-only: 4–10 GB zips |
| FSD50K | noise | recorded | per-clip CC, Zenodo "other-at" | discovery-only: needs per-clip filter |
| WHAM! noise | cafe babble | recorded | not verified (wham.whisper.ai HTTP 000, guessed URLs 404) | discovery-only |
| DNS-Challenge | noise + RIR | mixed | per-set unresolved | discovery-only |
| TAU Urban 2019/2020 | noise | recorded | other-nc, 1.6 GB parts | rejected |
| MIT IR Survey | RIR | recorded | no licence on page | rejected |
| RWTH AIR | RIR | recorded | none found on page | rejected |

Measured RT60/room/device metadata: only BUT ReverbDB (room sizes) and dEchorate (database CSV, not fetched) supply any; DEMAND supplies none; nothing was invented.

## Top 5 for T-211 (ranked, usable now)
1. DEMAND PCAFETER ch01 — the only licence-clean cafeteria babble; use as the café-noise layer. Path above. Unlabeled speech inside: never a clean target.
2. DEMAND STRAFFIC ch01 — street traffic.
3. DEMAND DWASHING ch01 — steady appliance/HVAC-like hum.
4. OpenSLR 28 real RIRs (already in T-219 `noise.json`) — recorded rooms; keep separate from simulated.
5. ESC-50 clips (T-219 `noise.json`) — short keyboard/ventilation/weather; non-commercial.

## Recipes for Copilot-b3 (exact)
- More DEMAND: `GET https://zenodo.org/api/records/1227121/files/<CODE>_16k.zip/content` (CODE ∈ PCAFETER, STRAFFIC, TBUS, TCAR, TMETRO, OOFFICE, OMEETING, DKITCHEN, DLIVING, DWASHING, NFIELD, NPARK, NRIVER, PSTATION, PRESTO, SPSQUARE, OHALLWAY). Read the last 128 KiB with a `Range` header, parse the zip central directory, then range-read one `<CODE>/chNN.wav` (≈6.5–7 MB deflated, 9.6 MB raw). Verify md5 of whole zips only if downloaded in full (md5s in the Zenodo record). Python SSL on this Mac needs `SSL_CERT_FILE=<certifi cacert.pem>`; curl works.
- More recorded RIRs: range-read members of `https://www.openslr.org/resources/28/rirs_noises.zip` (Accept-Ranges: bytes).
- Anything marked discovery-only: do not fetch until a licence/size check clears it.

## Verification
`bash scripts/verify_fast.sh` result: see the report (recorded after the run).

## Blockers and temporary assumptions
- Licences for ESC-50-clip/OpenSLR upstream sets are as stated on the host page; upstream AIR/REVERB terms unverified.
- WHAM! unresolved; no café source beyond DEMAND PCAFETER.
- Board-page attachment and relays to T-209/T-211/T-210/T-212–215 could not be done from this sandbox (no board API access here); the harness relays via the report `messages`.
