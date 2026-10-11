# Team audio audit

Generated from local recordings; speech is estimated from 10 ms frame energy.

| speaker | recording | duration (s) | sample rate | peak | RMS | SNR vs silence (dB) | speech fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| shared | `00_AB_DEMO_COMPARISON.wav` | 20.000 | 16000 | 0.9000 | 0.0730 | 53.45 | 0.388 |
| shared | `01_raw_overlap_mix.wav` | 10.000 | 16000 | 0.9000 | 0.0891 | 15.18 | 0.520 |
| shared | `02_isolated_aryan.wav` | 10.000 | 16000 | 0.6943 | 0.0720 | 52.00 | 0.433 |
| shared | `03_isolated_joseph.wav` | 10.000 | 16000 | 0.4547 | 0.0404 | 70.76 | 0.372 |
| aryan | `fixture-01.wav` | 58.693 | 16000 | 0.2425 | 0.0150 | 16.35 | 0.304 |
| aryan | `profile-01.wav` | 50.000 | 16000 | 0.1766 | 0.0161 | 16.51 | 0.460 |
| aryan | `source-voicememo.m4a` | 108.693 | 44100 | 0.2432 | 0.0155 | 16.46 | 0.331 |
| joseph | `fixture-01-20261009-1353.wav` | 48.150 | 16000 | 0.7079 | 0.0649 | 25.02 | 0.381 |
| joseph | `fixture-02-namecall-20261009-1353.wav` | 27.800 | 16000 | 0.7079 | 0.0773 | 31.23 | 0.313 |
| joseph | `profile-01-20261009-1353.wav` | 15.700 | 16000 | 0.7079 | 0.0651 | 24.85 | 0.412 |
| aryan | `profile-01.wav` | 50.000 | 16000 | 0.7079 | 0.0644 | 16.50 | 0.460 |
| joseph | `profile-01.wav` | 15.700 | 16000 | 0.7079 | 0.0651 | 24.85 | 0.412 |

SNR uses the ratio of whole-recording RMS to the 10th-percentile 10 ms frame RMS. A floor of 1e-12 avoids undefined values for digitally silent recordings; speech fraction counts frames above max(3x noise floor, 5% of peak, 1e-4).
