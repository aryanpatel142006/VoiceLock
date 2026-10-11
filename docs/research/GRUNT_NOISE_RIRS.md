# Grunt: noise and room-response mini-bank (T-219)

Collected 2026-10-09 by `python -m eval.overnight_noise_collect`; machine-readable manifest `eval/manifests/overnight_grunt/noise.json` (SHA256, native rate, duration, licence URL, author, measured vs simulated per item). Media (native + 16 kHz mono derivative, 88 files, 22 MB on disk) lives in the shared ignored shard `$HEARING_FIXTURES_DIR/overnight_grunt/noise/{native,derived_16k_mono}`. No reverb applied, nothing transcribed. T-211 owns augmentation and may consume the manifest.

## Counts (observed)
- Network bytes: 24,028,581 (budget 128 MiB). Plus one 11,660,005-byte MIT zip probe downloaded earlier to read its licence, then deleted (see below).
- Noise: 33 ESC-50 clips, 5 s each = 165 s decoded (limit 20 min). Four coarse categories (keyboard 6, traffic_vehicle 15, ventilation_appliance 6, weather 6; nine fine `source_label`s). **Target of 5 categories and cafe/crowd was NOT met**: no licence-verifiable, decodable cafe/crowd source was found.
- RIRs: 11 measured, 11 rooms (AIR: bathroom, booth, corridor, kitchen, lecture, meeting, office, stairway; REVERB-2014: small/medium/large room 1), all 16 kHz native.

## Licence evidence
- ESC-50: `https://raw.githubusercontent.com/karolpiczak/ESC-50/master/LICENSE`. Dataset CC BY-NC 3.0 (non-commercial); per-clip origin, author and tag are in the manifest. Fine for the hackathon, not for commercial use.
- RIRs: OpenSLR SLR28 page (https://www.openslr.org/28/) declares Apache 2.0. Members were range-fetched from `rirs_noises.zip` (the 8 MB cost is the zip's central directory). Upstream AIR / REVERB terms were not independently verified. Room dimensions, RT60 and author are `null` (unknown).
- Noise speech content: ESC-50 category clips carry no speech label; none are a clean target (`usable_as_clean_target: false`). No interfering-speaker provenance exists for any item.

## Blockers and discovery-only
- MIT IR Survey (271 measured IRs, 11.7 MB zip): the official page and its sibling pages state no licence, so it is discovery-only. Honest note: the zip was fetched once to inspect contents before the licence was found missing; deleted, not in the bank.
- Wikimedia Commons crowd/cafe: one attempt (`--with-commons`): 2 Ogg files undecodable by libsndfile, 2 HTTP 429; nothing collected. Files would carry unlabeled speech.
- Freesound: needs an API key.
- T-209's shortlist is not on main; not read.
