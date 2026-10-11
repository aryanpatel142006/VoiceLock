# Pitch draft: 2-minute pitch and 4-minute demo script (judging Sun Oct 11, 2026)

Draft for Aryan and Joseph to say out loud and cut down. It follows `docs/STAGE_CARD.md` (keys, step order, the three spoken lines) and `docs/FRONTEND_PLAN.md` (the `/app/` Listen screen uses the same keys and amber/green states as `/judge`).

**Number rule.** Every number below comes from `docs/CLAIMS.md` section 1 and is said with its condition. The line it comes from is tagged like `[C1.6]`. Nothing from CLAIMS sections 1b, 2 or 3 is said. If rehearsal 2 adds a team-voice number to section 1 (wrong-speaker time N, caption error X on the stage card), swap it in for the stand-in number and say its condition. If it is not in section 1, it is not said.

**Honesty rule.** Say "delay", never "real-time" or "low latency". Say "saved voice profile", never "learns your voice". A replay is always called a replay. Nothing about the benefit for people with sensory overload is measured: it is why we built this, not a result.

Speaking pace: about 130 words a minute. Word counts are given per section so the timing holds.

---

## Part 1: the 2-minute pitch (spoken, no demo)

### 0:00-0:25 · The problem (about 55 words)

> Think of a loud café. You want to hear one person. But you hear everyone: the espresso machine, the next table, the music.
>
> For a lot of people that is tiring. For some sound-sensitive people, including some autistic people, it is too much. So they leave, or they put on headphones and lose the person too.

### 0:25-0:55 · What it does (about 65 words)

> Selective Hearing lets you pick a person and turn down the room.
>
> You choose whose voice to hear. That voice plays in your earbuds. The rest of the room gets quieter, as much as you want.
>
> One tap brings the whole room back. Another mode goes silent and just shows captions.
>
> It plays about 1.5 seconds behind the room, like live captions. [C1.6]

### 0:55-1:20 · How it works (about 45 words)

> A microphone near the person you want feeds our engine on the laptop.
>
> First, a noise filter. The filter itself adds 13 milliseconds, engine-side, not counting Bluetooth or the Focus delay. [C1.7]
>
> Then a voice match against a saved voice profile picks out the person you chose. It keeps that voice and turns everything else down.


### 1:20-1:45 · What we measured, honestly (about 60 words)

> With our two real voices, the chosen speaker is followed correctly when they talk alone. [C1.1]
>
> When both of us talk at once, the wrong voice is heard 6 to 25 percent of the time. That is the hard case. [C1.2]
>
> Our best run: the other voice 21 dB quieter, and 4.6 dB cleaner. Two male voices, on a Mac. [C1.3]

### 1:45-2:00 · What is next (about 35 words)

> Next: one-tap calibrate for a new person, a clip mic for the person you want, and testing with the people this is for.
>
> Turn down the world without disconnecting from it.

---

## Part 2: the 4-minute demo script (live, page `/judge` or the `/app/` Listen screen)

Keys are the ones on the stage card: **Start**, **2** (Joseph), **Space** (Raw amber / Selective green), **Shift+Q** (Quiet), **Shift+F** (back to Focus), **D** (drawer), **Shift+R** (labelled replay, only if live failed). Click the page first. Never Shift+N or Shift+C.

### 0:00-0:30 · Setup, said out loud (stage card step 1)

Before the judge puts the AirPods in: seal test, volume set low together, mic pointed at the talkers.

> Joseph holds an iPhone as his mic, cabled to the laptop, like a hearing-aid remote mic. Our two voices were saved earlier as profiles from a voice memo. You wear AirPods Pro with noise cancelling. That speaker plays café noise.
>
> Please listen, don't talk. The sound is about one and a half seconds behind, like live captions. [C1.6]

(If the drawer shows the GPU backend, say "about 1 second behind" instead. [C1.6])

### 0:30-1:00 · Pick a person (step 2)

Press **Start**, then **2**. Joseph talks alone for about 5 seconds until the pill reads **Isolated**.

> This is Focus. You hear the one person you choose. Right now that is Joseph.

### 1:00-1:50 · Café noise and the A/B (step 3, the headline)

Café noise on (laptop-b). Joseph keeps talking. Press **Space** three times, mid-sentence, a few seconds apart. Amber is the raw room. Green is Selective Hearing.

Say nothing between the flips. Let the judge hear it. After the third flip, leave it on green.

> That was the room, and that was Joseph with the room turned down. Same mic, same moment.

### 1:50-2:30 · Quiet captions (step 4)

Press **Shift+Q**. The audio goes silent. Joseph reads two sentences of his own choice. Captions appear about 1-2 seconds behind.

> Quiet mode. No sound at all, just captions of what was said. No prompt, no script: this is what Whisper heard. Our captions are rough: 71 to 83 percent word error on a hard test clip with a second talker. [C1.9]

Press **Shift+F** to go back to Focus.

### 2:30-3:05 · The hard case (step 5, the reveal)

Aryan says ONE sentence over Joseph, about 5 seconds, then stops. Joseph keeps going.

> When we both talk at once, the wrong voice gets through 6 to 25 percent of the time. [C1.2] That is the hard case: two similar voices at the same time. The research agrees.
>
> From one laptop mic across a room, picking one of two similar male voices did not work well enough. Products that do pick a voice have the talker at the mic. So the person you want to hear wears a clip mic, and the app does the part no product does: pick who, mute the rest, caption only them.

(The second paragraph is the stage card's far-field honesty line, shortened. Use the full line if there is time.)

### 3:05-3:35 · Show the numbers (step 6)

Press **D** to open the drawer: backend, delay, round trip, "this session is being recorded".

> This is the backend and the delay, live. Every run is logged, so numbers come from recordings, not memory. [C1.10]

Only say the C1.10 line if the drawer really shows the recording on stage.

### 3:35-4:00 · Close (step 7)

> Pick a person. Turn down the room. Get the room back with one tap.
>
> Turn down the world without disconnecting from it.

### Only if steps 2-5 failed (step 8)

Press **Shift+R**. The REPLAY badge is on screen. Say:

> This is a replay. It was recorded earlier, mixed digitally, and processed now.

Never present it as live. If Focus is wrong or stuck for more than 10 seconds, go to Quiet captions first (fallback a). If captions are bad too, open `/simple` and A/B the noise filter (fallback b), and say: "noise suppression, not person selection."

---

## Questions the judges may ask (short answers, section 1 only)

| Question | Answer |
|---|---|
| How fast is it? | About 1.5 s behind on the Mac, about 1 s on a GPU. The noise filter alone is 13 ms, engine-side. Delay, not real-time. [C1.6, C1.7] |
| Does it work on any two voices? | No. Two similar voices talking at once is the hard case: 6 to 25 percent wrong-speaker time with our two voices. [C1.2] |
| What is your best result? | Joseph as the target: the other voice 21 dB quieter, 4.6 dB cleaner, two male voices, Mac. [C1.3] |
| With two very different voices? | On a test clip with two different voices, separation was +8.3 dB with the other voice 18 dB down. That clip is not our voices. [C1.8] |
| Can I just say a name to switch? | Saying the name worked 2 of 4 times in quiet, 0 of 4 with noise or overlap, and never fired by mistake. That is why it is off the stage script. [C1.4] On other speech it had 0 false triggers in 5 minutes. [C1.5] |
| Are the captions accurate? | No. 71 to 83 percent word error on a hard test clip. [C1.9] |
| Does it learn my voice? | It uses a saved voice profile for the person you choose. We do not claim it improves with use. |
| Does it help people with sensory overload? | That is why we built it. We have not run a user study yet, so we do not claim a benefit. |
| Is that live? | Yes, unless the REPLAY badge is on screen. |

## What is next (for the pitch close or Q&A; plans, not results)

- **One-tap Calibrate** while the person talks. Built, but not yet shown live; on stage we pick a saved voice profile.
- **A clip mic for the person you want to hear**, so the voice you want is the loudest one at the mic.
- **The phone app** (`/app/`): the stage laptop shows a QR, the phone follows the same session as a companion.
- **Testing with sound-sensitive people.** No user study yet.
- **Later:** a camera to pick a person by looking at them, and a familiar synthetic voice reading the captions. Both are after HackRU.

## Do not say

"Real-time", "low latency", "under 100 ms", "works on any voices", "accurate captions", "learns your voice", "the GPU is better", any number with "+13.8 dB" or similar, or anything from a replay as live. Reasons: `docs/CLAIMS.md` section 3.
