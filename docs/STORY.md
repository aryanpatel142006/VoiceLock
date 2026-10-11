# Selective Hearing: story and demo narrative

> Product story for the pitch and presentation. `PLAN.md` owns the technical acceptance checks. Confirm every stage claim against the final measured run.

## The person and the problem

In a café or conference, a listener may want to hear one person without taking in every conversation and background sound around them. For some sound-sensitive people, including some autistic people, that background can be overwhelming. The listener should decide whose voice matters and when to return to the room.

**One sentence:** Choose a person to hear; turn down the rest of the room. We are testing whether, with consent, more confirmed examples of a person's voice can improve that person's profile over time.

## What to show

Use one microphone location, one target speaker, one café sound bed, and the same headphones throughout. Record the device, settings and levels. Let the audience hear three conditions in order:

1. **Headphones with noise control off:** the ordinary room sound and conversation.
2. **Headphones with ANC on:** what noise cancellation alone changes for this exact setup.
3. **ANC on plus Selective Hearing Focus:** the chosen person's real voice remains available while the room is reduced.

Match the apparent loudness of the chosen voice as fairly as practical, keep the target and café positions fixed, and say which condition is live or replay. This is a demonstration of a measured setup, not a claim that every device or room behaves the same way. If a direct three-way live switch is unstable, use clearly labelled recordings of the same scenario and explain how they were captured.

Next, while the chosen person keeps talking, have another person farther to the side read **one sentence**. They should speak naturally; the effect does not depend on shouting. Do not stage full simultaneous start. T-141 measured **6–25% wrong-speaker time** in tested two-male-voice overlap, so show the exact run's remaining bleed and wrong-speaker rate. Reset returns the listener to Raw/Open; Quiet silences the room audio. Never call a replay live.

## Two honest ways to hear the same moment

**Live Focus** is what the listener hears while the conversation happens. It processes short incoming windows with a measured delay. A recording of both the raw microphone feed and the live processed feed becomes available in the demo UI as the take is captured.

**Enhanced Replay** reprocesses that *same raw take* after it has been recorded. It can use more context, stronger separation and cleanup, then produce a second audio file that may sound clearer. A full minute of context is available only after that minute has been captured and processed; it cannot be presented as zero-delay audio. A short completed excerpt could become a preview sooner. “Enhanced” is a goal to test by listening, not a promise of perfect reconstruction.

The judge first hears **Live Focus** through the headphones. Then the UI shows the newly captured run and offers level-matched **Raw / Live recording / Enhanced Replay** buttons, a waveform or timeline, and a visible **Rendering → Ready** state. The presenter says, “You heard the live version. Now here is the same recorded moment with the extra processing we can do when we have more time.” The change should be obvious and impressive without concealing which version the judge is hearing.

## The control that makes the idea personal

The intended interaction is simple: tap **Calibrate** while the person you want to hear speaks, then stay locked to that person until **Reset**. The button must verify that it captured enough of the intended person's voice; it cannot equate the loudest sound with identity. This exact interaction is still a build and real-mic acceptance gap. If it is not ready for the stage, use **Choose saved person** and **Release/Raw** and say that live calibration is the next step.

## Why this is more than noise reduction

ANC reduces incoming sound. The proposed extra control is **who remains audible**: the listener chooses a person, and the system uses a voice profile to preserve that person's speech while reducing unrelated sound and, when it works, another nearby voice. The three-way demonstration and one-sentence reveal should show the measured difference, including T-141's **6–25% wrong-speaker time** on prior two-male-voice overlap, rather than relying on a claim about what ANC can or cannot do.

## How it could get better over time

With consent, the listener could confirm clean examples of a person's voice across sessions. The product could keep a versioned voice profile, test whether the new examples improve recognition on separate recordings, and roll back when they do not. This is a credible personalization path, **not a feature to claim as working today**. More examples might improve speaker matching; they do not automatically teach the current separator to remove all other voices. `docs/VOICE_EMBEDDINGS.md` describes the research and proof needed.

Later choices may include a single familiar synthetic voice, captions, dedicated microphones, glasses or other hardware. Those are ways the experience could adapt to different needs after the phone/microphone path is proven.

## Closing line

“A conversation should not disappear just because the room is overwhelming. You choose the person; you choose how much of the room stays.”


## Optional stronger-overlap result and personalization — October 9 follow-up

The cafe-first scene above remains the core story. A quieter selected voice surviving a louder overlapping person would strengthen it, but is an optional measured reveal, not a new stage prerequisite. Describe the exact microphone positions, level ratios, enrollment and delay that passed. If the bonus fails, keep the working scene and explain the remaining overlap limitation.

Saved profiles and ECAPA matching already exist. The proposed new experiment compares 1, 2, 4 and 8 clean, consented, confirmed examples on different held-out speech; original profiles remain available for rollback. Until that curve demonstrates improvement, say: “It uses a saved voice profile for the person you chose. We are testing whether confirmed examples can make that profile better over time.” If an update passes, state the actual held-out gain and profile versions. A saved embedding alone, or a separator fine-tune, does not justify saying it continuously personalizes to the listener. The listener's mode preferences and the selected speaker's identity profile are different kinds of personalization; keep them distinct. Fine-tuning remains an optional research tool, not a claim on stage.


**Confirmed research target (Joseph, October 9):** strongly separate either of the actual two male speakers with different voices, at equal and unequal levels and swapped positions. Male/female was an illustrative example, not a new participant or demo requirement. Preserve the working cafe-first core if this bonus does not pass.
