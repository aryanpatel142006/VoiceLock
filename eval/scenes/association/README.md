# Association scenes (T-061)

Two-face scenes for `eval/association_eval.py`: does the Enrolled tier, auto-enrolled from the
**mixed** signal through real ASD, keep extracting the person the wearer chose? Each scene is
rendered from cached GRID clips (`eval/fixtures/generated/_sources/grid`, talkers s1 and s4):
person A sits on one half of a 640x360 frame, person B on the other, their audio is summed with
noise. Not read by `eval/scene_score.py` (different schema; that scorer only globs `eval/scenes/*.yaml`).

| Key | Meaning |
|---|---|
| `name`, `description` | free text |
| `duration_s` | scene length (a multiple of 0.04 s) |
| `persons.A`, `persons.B` | segments `[t0, t1, kind]` covering `[0, duration_s)` |
| `target` | `[t, person]` keyframes: who the wearer has chosen from `t` on (the oracle lock) |

Segment kinds (frames come from the talker's GRID clips, speech parts and idle parts cut by the alignments):

| kind | face | audio |
|---|---|---|
| `speak` | talking face | the talker's speech |
| `silent` | idle face (mouth closed, small natural motion) | none |
| `mouth` | talking face | none (the mouth moves, the track is muted) |
| `away` | none (turned away: background where the face was) | the talker's speech |
| `occluded` | talking face under a dark box (hand, passer-by) | the talker's speech |
| `absent` | none (left the frame) | none |

The end of an `away` / `occluded` / `absent` segment of the current target is a re-acquisition
event. Variant `ab` puts A = s1 on the left; variant `ba` puts A = s4 on the right.
