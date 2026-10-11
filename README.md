<div align="center">

# VoiceLock

**Turn down the world without disconnecting from it.**

![HackRU Fall 2026](https://img.shields.io/badge/HackRU-Fall%202026-1e1e1e)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab)
![PyTorch](https://img.shields.io/badge/PyTorch-pipeline-ee4c2c)
![PWA](https://img.shields.io/badge/web%20app-PWA-5a0fc8)

</div>

## Demo

[![VoiceLock demo preview: Auto voice focus switches between the isolated closest voice and the whole room](docs/media/VoiceLock_demo_preview.gif)](docs/media/VoiceLock_demo.mp4)

▶ **[Watch the full demo (2 min 25 s, mp4)](docs/media/VoiceLock_demo.mp4)**

What the video shows, as labelled on screen: the before/after café clip was **recorded earlier and processed by VoiceLock**; on the Listen screen the **mic input is a pre-recorded test voice**; Who's speaking runs on **recorded clips of Aryan, then Joseph**. The SOS, Talk to Me and profile screens are the live app.

<p align="center">
  <img src="docs/media/welcome.webp" alt="VoiceLock welcome screen: Turn down the world without disconnecting from it, with a Hold to start button" width="49%">
  <img src="docs/media/listen.webp" alt="Listen screen, idle: Auto voice focus switch, Focus and Quiet modes, room dial, SOS button and Start listening" width="49%">
</p>

## Why we built it

Some of the people closest to us are autistic. We've seen how overwhelming an everyday place can become when many conversations happen at once: a café, a conference hall, a hackathon like this one. For many autistic people with sensory overload, a room like that isn't just hard. It's off-limits, and staying home is easier.

Noise cancelling doesn't fix this. It turns *everyone* off, including the one person you came to talk to. We wanted the opposite: let someone choose what they hear, and remember what helps them when it's still too much. Our line for it: **turn down the world without disconnecting from it.**

## Features

VoiceLock is a web app you can install on your phone. You listen through your earbuds (AirPods Pro in our demo), and it gives you control over the room:

- **Focus (Auto voice focus).** VoiceLock cleans the café noise out of the room, splits what's left into separate voices, and plays the loudest, closest one: the person talking to you. Everything else is left out, at one steady volume. Press Space (or flip the switch) and you hear the whole room again.
- **Quiet mode.** When even one voice is too much, VoiceLock's own audio pauses and the person's words keep coming as captions (ElevenLabs speech-to-text). You're still in the conversation.
- **One-press SOS.** One press cuts the room and fades in your favourite calm sound. Then you choose: **Stay Quiet** (silence plus captions), **Play My Sounds** (rain, brown or white noise, a soft pad) or **Talk to Me**.
- **Talk to Me.** A calm spoken companion. Hold to talk (or type) and it answers out loud. Gemini writes the reply from your saved profile (what helps, what to avoid) and ElevenLabs gives it a voice. It offers to remember what helped, and saves it **only when you confirm**. Ask it to "call Jordan" and it opens a call to your saved person: the real dialer on a phone, and a labelled demo call screen on the laptop.
- **What I've learned about you.** Your profile, in plain words. Open it, change it, remove anything.
- **Phone companion.** Scan a QR code to follow along on your phone.
- **One-button remote.** A presenter remote stands in for the wearable band we want to build: one press for SOS, hold to come back.
- **Who's speaking.** A separate page that shows our fine-tuned speaker recognition live: it tells which of two teammates is talking.

VoiceLock is support for sensory overload, not treatment or medical advice. It never guesses how you feel: you press the button. If someone mentions being in danger, Talk to Me points them to the **988** Suicide and Crisis Lifeline (or 911) instead of giving advice.

<p align="center">
  <img src="docs/media/sos-choices.webp" alt="SOS screen: You're safe. What would help? Stay Quiet, Play My Sounds, Talk to Me, and a Hold to go back button" width="49%">
  <img src="docs/media/talk-to-me.webp" alt="Talk to Me after one typed exchange: the user says it's loud and they feel overwhelmed; the reply offers a slow breath together. Chips read Personal · Gemini and Voice: ElevenLabs" width="49%">
</p>
<p align="center">
  <img src="docs/media/learned.webp" alt="What I've learned about you, demo profile Alex: what helps, what to avoid, sound preferences and confirmed memories, each removable" width="49%">
  <img src="docs/media/who.webp" alt="Who's speaking page before start: Aryan and Joseph cards with match bars, fine-tuned on our two voices, speaker ID only, no audio output" width="49%">
</p>
<p align="center">
  <img src="docs/media/phone.webp" alt="Phone companion at 390 px: Follow the stage, a list of laptop sessions to follow, and Solo mode" width="28%">
</p>
<p align="center"><sub>Screenshots of the running app. The profile shown is Alex, a made-up demo profile. Talk to Me's reply came from the live Gemini and ElevenLabs path.</sub></p>

## How it works

![The live pipeline: iPhone mic, DeepFilterNet3 denoise, MossFormer2 separation, lock on the loudest closest voice, a Silero VAD foreground-speech gate on the raw mic and a steady output level, then AirPods](docs/media/diagram-pipeline.svg)

**The app** is a web app / PWA (HTML, CSS and JavaScript in `web/app`), installable on a phone. A Python server streams microphone audio over WebSockets into a PyTorch pipeline and streams the processed voice back.

**The listening pipeline.** The microphone hears a mixture of voices and noise:

$$x(t) = \sum_{i=1}^{N} s_i(t) + n(t)$$

1. **Denoise first.** DeepFilterNet3 (low-latency variant, 13 ms algorithmic delay) removes the café noise $n(t)$ *before* separation. We measured both orders offline, and denoising first won on every count: after separation, the denoiser treated separation artifacts as noise and clipped words after a pause.
2. **Separate.** A pretrained MossFormer2 separator splits each 1 s window into streams $\hat s_1, \dots, \hat s_K$.
3. **Select.** This is where our two modes differ.
   - **Live Focus (what we demo)** plays the loudest stream:
     $$k^* = \arg\max_k \lVert \hat s_k \rVert^2$$
     Another voice takes over only when it is 6 dB louder for a full second, so the lock doesn't flicker, and short pauses don't break it.
   - **Profile mode (our research path)** picks the stream that sounds most like a saved voice sample, using ECAPA speaker embeddings $e(\cdot)$:
     $$k^* = \arg\max_k \cos\big(e(\hat s_k),\, e_{\text{target}}\big)$$
4. **Gate and level.** A Silero VAD foreground-speech gate, run on the raw input, mutes the output when nobody nearby is talking, so café babble doesn't leak through in pauses. A steady output gain keeps the voice at one level.

The output plays about **1.5 s** behind the room (1.0 s window + 0.5 s hop), like live captions.

![SOS and Comfort flow: one press cuts the room and fades in a calm sound; Stay Quiet, Play My Sounds or Talk to Me (ElevenLabs speech-to-text, Gemini with your profile, ElevenLabs voice); Remember this is saved only on confirm into What I've learned about you, which Gemini reads](docs/media/diagram-sos.svg)

**Personalization (Gemini).** Talk to Me sends Gemini the profile, what the user has confirmed helps, and what to avoid. Gemini returns a short reply as schema-constrained JSON, plus whether to offer a "Remember this?" chip or start a call. Without a key or a connection it falls back to a gentle default, and the screen says so.

**Voice (ElevenLabs).** Text-to-speech gives Talk to Me its calm voice. Speech-to-text handles hold-to-talk and the Quiet mode captions. Both sit behind adapters with offline fallbacks.

**Fine-tuning.** We fine-tuned MossFormer2 on our own voices on Rutgers iLab A100 GPUs, with a fixed output order: slot 0 is always one teammate, slot 1 the other.

**How we worked.** Three of us coordinated a swarm of AI coding agents (Claude Code, Codex and Gemini agents) through a Notion task board and GitHub. Each agent took a task, worked in its own git worktree and reported measured numbers, and we reviewed and merged. Every live session is recorded, so our numbers come from recordings, not memory.

## Built with

| Part | What we used |
|---|---|
| App | HTML, CSS, JavaScript (PWA, installable on a phone), WebSockets |
| Server | Python, FastAPI, PyTorch |
| Denoise | DeepFilterNet3 (low-latency variant) |
| Separate | MossFormer2 (pretrained; fine-tuned on Rutgers iLab A100 GPUs) |
| Voice match and gate | ECAPA speaker embeddings, Silero VAD |
| Talk to Me | Gemini (schema-constrained JSON), ElevenLabs text-to-speech and speech-to-text |
| Demo hardware | A MacBook, an iPhone cabled to it as the mic, AirPods Pro, a presenter remote |
| How we worked | Claude Code, Codex and Gemini agents, Notion, GitHub |

## Research

![Two research halves: Separate (live, pretrained MossFormer2 with the loudest-voice lock) and Recognize (fine-tuned on our two voices, the Who's speaking page), to be fused in the next training run](docs/media/diagram-research.svg)

### What we measured

- **99.2% correct-speaker selection** in profile mode, from a 2 s voice sample, on a benchmark of 120 equal-loudness two-speaker mixtures (frozen speaker encoder).
- **+17.6 dB SI-SDR improvement**, with the other voice **52 dB** down while only they talk, on a recorded clip: two team voices at equal loudness with real café noise, processed afterwards.
- **+8.1 to +14.3 dB SI-SDRi** in profile mode across all six pairs of our four recorded voices (offline mixtures, including a competing voice 6 dB louder and café noise).
- **7× fewer wrong-speaker mistakes** for familiar voices after fine-tuning (31.6% → 4.5%, offline).
- **Who's speaking: 100% correct labels** on held-out MacBook-mic recordings (311/311 and 307/307 windows), switching to the new speaker in a median of 0.5 s.
- **A product, not just a model.** Focus, Quiet, SOS, Talk to Me, a learned profile and a phone companion all run end to end, live, on a laptop.

We measured improvement in decibels with the scale-invariant signal-to-distortion ratio:

$$\text{SI-SDR}(\hat s, s) = 10\log_{10}\frac{\lVert \alpha s\rVert^2}{\lVert \alpha s - \hat s\rVert^2}, \qquad \alpha = \frac{\langle \hat s, s\rangle}{\lVert s\rVert^2}$$

$$\text{SI-SDRi} = \text{SI-SDR}(\hat s, s) - \text{SI-SDR}(x, s)$$

That's how much closer the output $\hat s$ is to the clean voice $s$ than the raw mixture $x$ was. Every number above was measured with its condition written down. We'd rather show a real 4.5% than an impressive guess.

### What was hard

**The hard case is a quiet voice under a louder one.** Today's models separate two voices well. Picking the *right* one when it's quieter than its neighbour is not, and that's the case that matters most to our users. Similar voices make it worse. Offline, with only 2 s of calibration, the wrong person came through **1.0%** of the overlap time for male–female pairs, but **9.1%** for male–male pairs.

**Making the model smaller and faster.** We tried distilling the separator into a smaller streaming student, but in our time budget the student didn't learn enough to replace the teacher.

**Fine-tuning works for familiar voices, not strangers.** Fine-tuned on two teammates, wrong-speaker time fell from **31.6%** with the stock model to **4.5%** (offline, 14 held-out team rows). On voices it had never heard, the same models picked the wrong person **62–93%** of the time. Our users will meet strangers, so the app ships the general model. The fine-tuned model lives on the Who's speaking page, where it does what it's good at.

**A learned projection head made things worse.** We added a small learned layer on top of the speaker embeddings to sharpen the match. On synthetic test embeddings, it accepted the wrong voice in 52% of windows where the target wasn't speaking (plain cosine: 0%), so we kept the simpler scorer.

**The real world.** Our first live test at 4 a.m. on judging day went badly, and the recordings showed exactly why:
- Voice profiles recorded on the MacBook mic didn't match the iPhone mic we used on stage. Similarity dropped from 0.52–0.55 to 0.32–0.35, and the system muted the target speaker for 69–92% of the time it was locked on. That is why live Focus now follows the loudest, closest voice instead of a profile.
- The output volume pumped by about 28 dB, many times a minute. We replaced it with a steady output gain and the VAD gate.
- Agent jobs on the same Mac overloaded the pipeline, so demos now run on an idle machine.
- Bluetooth: the AirPods mic forces call-quality audio, so the mic is an iPhone cabled to the laptop. Bluetooth playback adds a delay we haven't measured on top of the 1.5 s.
- Café noise landed in whichever voice was picked, which is what led to denoise-first and the VAD gate.
- macOS Python's default TLS context failed every Gemini and ElevenLabs call with `CERTIFICATE_VERIFY_FAILED` until we pointed it at a proper certificate bundle.

### What we learned

- **Measure before you believe.** Several ideas that looked great on paper (a projection head, stricter voice matching, more enrollment examples) didn't survive measurement, and the recordings told us more than our intuitions did.
- **Offline and live are different worlds.** A model can score well on clean mixtures and still fail on a different microphone in a noisy room. We learned to test on the hardware we'd demo on.
- **Order matters in a pipeline.** Denoising before separation, not after, fixed problems no amount of tuning afterwards could.
- **Design for the person, not the benchmark.** The features that matter most (one press to make it stop, a calm voice, captions instead of sound) came from asking what someone overwhelmed actually needs.
- **Coordinating many AI agents is its own engineering problem.** It needs clear tasks, isolated workspaces and honest reports.

### What's next

- **Fuse the two halves.** Live separation and fine-tuned recognition each work on their own. One model that does both, picking *your* person, loud or quiet, needs a fresh training run with more GPU time.
- **Quieter voices in louder rooms**, the hardest case, and detecting when your person isn't talking at all.
- **Lower latency:** on-device processing on the phone, and wired audio instead of Bluetooth.
- **A wearable band:** one press on the wrist, replacing today's remote.
- **Testing with the people it's for**, autistic users and their families, in real places.

We want VoiceLock to be an everyday tool that gives people more independence and more control over what they hear.

## Run it locally

Requirements: macOS or Linux, Python 3.11+, `ffmpeg`, `libsndfile`. Model weights download on first use and are not in this repository; neither are voice recordings or profiles.

```bash
bash scripts/setup.sh                        # creates .venv and installs the package
.venv/bin/pip install -e ".[ml]"             # the model stack (PyTorch, ONNX Runtime, SpeechBrain, ...)
.venv/bin/pip install --no-deps clearvoice   # MossFormer2; it pins numpy<2, so no deps (see pyproject.toml)

# optional: Talk to Me and captions (without keys it falls back to a gentle default and says so)
export GEMINI_API_KEY=...  ELEVENLABS_API_KEY=...

# the demo: opens the app at https://localhost:8000/app/ (self-signed certificate);
# phones on the same Wi-Fi get a QR code for the companion
HEARING_COMFORT=1 HEARING_PICK=loudest bash scripts/demo.sh --judge --team --lan --app
```

- `--mic "<label>"` picks the input device by its exact name (we use an iPhone cabled to the laptop).
- `--port 0` picks a free port and prints it, `--no-open` skips the browser, and `bash scripts/demo.sh --stop --port <port>` stops the server.
- `--team` loads the two-voice preset in `config/demo-team.yaml`, which expects saved voice profiles (`scripts/enroll_profiles.sh`).
- Keys in the app: **Space** turns Auto voice focus on or off, **Shift+F** / **Shift+Q** switch Focus and Quiet, **Shift+S** (or a paired presenter remote) is SOS, and holding 1.5 s brings you back to the room.
- More detail in `docs/DEMO.md`, `docs/APP.md` and `docs/STAGE_CARD.md`.

> **Note on this repository.** The code here is a snapshot synced from our working repository on Oct 11, 2026 (09:21 EDT). The SOS Comfort screens, Talk to Me, What I've learned about you and the Who's speaking page shown above were finished after that snapshot and are not in this code yet.

## Team

Built at HackRU Fall 2026 by **Aryan Patel**, **Joseph Roche** and **Ahmad Abbadi**.
