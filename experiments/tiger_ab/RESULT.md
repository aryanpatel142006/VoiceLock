# T-372: TIGER-speech vs MossFormer2 as the blind separator (offline A/B, Oct 10 2026)

Verdict: **do not adopt.** Default stays MossFormer2 (`HEARING_SEPARATOR` unset). Select TIGER with
`HEARING_SEPARATOR=tiger|tiger-small|tiger-large` (weights in `$HEARING_MODELS_DIR/TIGER-speech[-small]`,
`hearing.tse.blind.download_tiger()`).

Weights: `JusperLee/TIGER-speech-small` @ `46f86be1a7255790e5c6d3a06d1fc4c94dee74cc`, `JusperLee/TIGER-speech`
(large) @ `f0340340b2d9bbf72074edf8c076dcab59a10ba2`. Apache-2.0 card, MIT code (vendored `hearing/tse/_tiger`).

Command (`eval/tiger_ab.py`, wraps `eval.wrong_speaker_audit.run`: same ECAPA/LDA pick, 1.0 s windows every 0.5 s,
same 4 real-pair rows of T-323; wrong-speaker % = other-dominant s / clip s, session_score rule):

    .venv/bin/python -m eval.tiger_ab --profiles $F/profiles --backends mossformer2 tiger tiger-large \
        --scene joseph:aryan:$F/t141/joseph/demo-team-easy [--scene ...]

| row (target) | MossFormer2 wrong % | TIGER-small wrong % | TIGER-large wrong % |
|---|---|---|---|
| joseph easy | 15.3 | 17.3 | 19.3 |
| aryan easy | 6.0 | 11.3 | (not run) |
| joseph hard | 18.7 | 18.0 | (not run) |
| aryan hard | 7.9 | 15.9 | (not run) |

SI-SDRi (dB, played output over mix) / leak dB (other stem, output vs mix): MossFormer2 joseph-easy +3.87 / -8.6,
aryan-easy +2.27 / -9.8, joseph-hard +1.98 / -5.1, aryan-hard +0.03 / -10.5. TIGER-small +1.72 / -7.0, -0.11 / -4.5,
+0.21 / -7.2, -0.49 / -4.1. TIGER-large (joseph easy only) +1.63 / -7.3.

Per 1.0 s window separator time, M4 Pro, joseph-easy, under `swarm-lock --exclusive` (load avg 4.4), deadline = 0.5 s hop:

| backend | device | p50 ms | p95 ms | late windows |
|---|---|---|---|---|
| MossFormer2 | MPS | 205 | 207 | 0 / 116 |
| TIGER-small | MPS | 493 | 499 | 6 / 116 |
| TIGER-large | MPS | 983 | 1003 | 116 / 116 |
| MossFormer2 | CPU | 4362 | 4435 | n/a (8 calls) |
| TIGER-small | CPU | 4011 | 4043 | n/a (8 calls) |

Reading: TIGER-small does not improve wrong-speaker on the Joseph rows (+2.0 pp on easy, -0.7 pp on hard, noise
level) and is worse on the Aryan rows and in SI-SDRi/leak; it sits on the window deadline. Large is worse still
and 2x too slow. ADOPT needs wrong-speaker to improve without missing the deadline: neither holds.

Caveats: single run per row (deterministic replay); TIGER-large only on one row (time); CPU numbers are
shared-machine CPU under the exclusive lock; the first-run MossFormer2 timing under load (350-450 ms) is not
used. Two porting fixes were needed for the vendored model: NNPACK disabled (CPU 69 s -> ~4 s per window) and
non-divisible `adaptive_avg_pool1d` hops to CPU on MPS.
