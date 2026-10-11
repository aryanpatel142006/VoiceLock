# T-383: does loudness explain wrong-speaker windows?

Source: `wrong_speaker_audit_20261010-015949.json` (T-323, 236 windows, 42 wrong). Ratio = target stem RMS dB minus other stem RMS dB over the window's 1 s span (clean stems, before noise). Reproduce: `python -m eval.loudness_wrong_windows <audit.json>`.

| cell | windows | wrong | <-6 | -6..0 | 0..6 | >6 | median wrong (dB) | median all (dB) | target active in wrong |
|---|---|---|---|---|---|---|---|---|---|
| aryan easy | 59 | 5 | 2 | 2 | 1 | 0 | -1.7 | +10.8 | 4/5 |
| aryan hard | 59 | 6 | 3 | 2 | 1 | 0 | -47.2 | +10.8 | 3/6 |
| joseph easy | 59 | 15 | 5 | 7 | 3 | 0 | -2.1 | +20.4 | 11/15 |
| joseph hard | 59 | 16 | 6 | 7 | 3 | 0 | -2.9 | +20.4 | 12/16 |
| all windows |  | 236 | 28 | 32 | 42 | 134 | | | |
| wrong rate | | | 57% | 56% | 19% | 0% | | | |
| total | | 42 | 16 | 18 | 8 | 0 | | | |

**Conclusion:** Relative level matters (wrong rate 57% / 56% when the other voice is louder or within 6 dB below the target level, 0% when the target is >6 dB louder), but it is not a "quiet target" effect: the target was VAD-active in 30 of 42 wrong windows, and the jump comes from windows where the other talker is present at all (the >6 dB bin is mostly the other talker silent). Wrong windows are an other-speaker-louder-in-overlap problem, so level normalisation alone will not fix them. Caveat: 4 cells, one audit, 1 s windows; aryan-hard median is dominated by a near-silent target.
