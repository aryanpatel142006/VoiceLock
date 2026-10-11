# Scenes

Scripted scenes for scoring target selection (M2). Plain data only; the scorer in `eval/scene_score.py` never imports `hearing/`. Field names follow `docs/CONTRACTS.md` (`track_id`, `box` normalised x,y,w,h, `p_speaking`, `last_seen_ts`).

## Schema
| Key | Meaning |
|---|---|
| `name`, `description` | free text |
| `fps` | frames per second (25) |
| `duration_s` | scene length; `fps * duration_s` frames at t = i / fps |
| `actors[].id` | integer, becomes `track_id` |
| `actors[].box` | keyframes `[t, x, y, w, h]`, 0-1 image coords (x,y top-left), linearly interpolated, held before the first and after the last keyframe |
| `actors[].speaking` | list of `[t0, t1]` intervals; `p_speaking` is 1.0 inside (t0 <= t < t1), else 0.0 |
| `camera` | heading keyframes `[t, actor_id]`: which actor the wearer faces (centred) from t on. Documentation of intent; boxes already encode it |
| `occlusions` | optional `{actor, t0, t1}`; the actor's track is omitted for those frames |
| `expected` | contiguous segments `{t0, t1, target}` covering `[0, duration_s)`; `target` is an actor id or `null` (no target) |

Each expected change of target between consecutive segments is an *expected switch* at the later segment's `t0`.

## Example
```yaml
name: a_turn_to_b
fps: 25
duration_s: 8
actors:
  - id: 1
    box: [[0, 0.44, 0.3, 0.12, 0.2], [4, 0.44, 0.3, 0.12, 0.2], [4.5, 0.14, 0.3, 0.12, 0.2]]
    speaking: [[0, 8]]
  - id: 2
    box: [[0, 0.74, 0.3, 0.12, 0.2], [4, 0.74, 0.3, 0.12, 0.2], [4.5, 0.44, 0.3, 0.12, 0.2]]
    speaking: [[0, 8]]
camera: [[0, 1], [4.5, 2]]
occlusions: []          # e.g. [{actor: 1, t0: 3.0, t1: 3.5}]
expected:
  - {t0: 0, t1: 4, target: 1}
  - {t0: 4, t1: 8, target: 2}
```

## Scenes
a_turn_to_b, b_talk_over, c_occlusion, d_moving_target, e_silence_then_b, f_three_people, g_look_down.

## Scoring
`accuracy`: fraction of frames with the right target, skipping 1 s after each expected switch. `switch_latency_s`: time from each expected switch to the first correct prediction (`None` if never). `false_switches`: predicted changes into a wrong target outside grace windows.
Run `python -m eval.scene_score eval/scenes/ --oracle`.
