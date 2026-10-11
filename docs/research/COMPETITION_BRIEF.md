# Competition brief for judging Q&A (HackRU F26)

T-362, written Oct 10, 2026. This is for answering judges' questions, not for slides. Vendor facts come from the linked primary pages, checked Oct 10, 2026. Every number about **our** system comes from `docs/CLAIMS.md` section 1, with the line number given, and keeps the condition stated there. If a vendor does not publish a figure, the table says "not published". We did not measure any competitor.

## What exists today

| Product | What it does | Latency | Isolates ONE chosen voice? | Price / availability |
|---|---|---|---|---|
| **Apple Conversation Boost** (AirPods Pro, inside Transparency) | Apple says it lets you "focus on the person talking in front of you" ([Apple AirPods guide](https://support.apple.com/guide/airpods/dev966f5f818)). It "will automatically amplify the voice of the person you're talking to" and reduces background sound ([apple.com/airpods-pro](https://www.apple.com/airpods-pro/)). | Not published. | **No.** It picks by direction: whoever is in front of you. It does not use who the person is. | Comes with AirPods Pro. AirPods Pro 3 start at $249 (US) ([Apple Newsroom, Sep 9 2025](https://www.apple.com/newsroom/2025/09/introducing-airpods-pro-3-the-ultimate-audio-experience/)). |
| **Apple Live Listen** | The iPhone or iPad acts as a remote microphone and streams to AirPods or Beats. Apple's instruction is "Place your iPhone or iPad in front of the person that you want to hear" ([Apple Support 102479](https://support.apple.com/en-qa/102479)). | Not published. | **No.** It picks by where the phone sits. Anyone near the phone is heard too. | Free on iOS/iPadOS 14.3 or later with supported AirPods or Beats ([Apple Support 102479](https://support.apple.com/en-qa/102479)). |
| **Apple Hearing Aid Feature** (AirPods Pro) | FDA-authorized OTC hearing-aid software for adults 18+ with perceived mild to moderate loss. It is fitted from the user's hearing levels ([FDA, Sep 12 2024](https://www.fda.gov/news-events/press-announcements/fda-authorizes-first-over-counter-hearing-aid-software)). | Not published. | **No.** It amplifies and shapes the sound for the user's hearing. | Comes with compatible AirPods Pro ($249 for Pro 3, link above). |
| **Google Sound Amplifier** (Android) | Filters, boosts and amplifies surrounding sound through headphones. Its Conversation mode (Pixel 3 and later) is meant to "amplify your conversation partner's voice and block out other noises": you point the phone's camera at the speaker ([Google Accessibility Help](https://support.google.com/accessibility/android/answer/9157755?hl=en)). | Not published. Google warns that "with some Bluetooth headphones, sound might be delayed" (same page). | **Partly.** Conversation mode is aimed at one partner with the camera, but it picks by direction, not by voice identity. | Built into Pixel 3 and later and some other Android phones, and on Google Play (same page). The help page lists no price. |
| **Premium AI hearing aids** (e.g. Phonak Audéo Sphere Infinio) | A dedicated DEEPSONIC chip runs a deep neural network "dedicated solely to separating speech from noise … from any direction". Phonak claims 10 dB SNR improvement (vendor figure) ([Phonak Audéo Sphere](https://phonak.com/en-ca/hearing-devices/hearing-aids/audeo-sphere)). | Phonak does not publish a figure. For hearing aids generally, Stone & Moore found delays above about 20 ms start to disturb normal-hearing listeners ([Hearing Review summary](https://hearingreview.com/2010/01/acceptable-processing-delay-in-digital-hearing-aids/)). | **No.** It separates all speech from noise. It does not choose one talker over another. | Prescription devices fitted by a hearing-care professional. Phonak does not list a price. OTC hearing aids for mild to moderate loss have been legal in the US since FDA's 2022 rule, which FDA's 2024 announcement references ([FDA](https://www.fda.gov/news-events/press-announcements/fda-authorizes-first-over-counter-hearing-aid-software)). |
| **Krisp-style noise apps** (Krisp) | AI noise and "background voice removal" on calls ("only active voices on calls are heard"). It works on your outgoing side and on other participants' audio, and runs with Zoom, Teams, Meet, Webex and Slack ([Krisp noise cancellation](https://krisp.ai/noise-cancellation/)). | Not published on the product page. | **For calls only.** It keeps the active or primary speaker on a call. It does not let a listener in a room choose which person they hear. | Free trial. Core is $8 per user per month billed annually, or $16 monthly ([Krisp pricing](https://krisp.ai/pricing/)). |

**Our position in one line:** everything above either cleans noise (hearing aids, Krisp, Sound Amplifier) or picks by direction or placement (Conversation Boost, Live Listen, Sound Amplifier's Conversation mode). Selective Hearing picks by **whose voice it is**: the listener chooses a saved voice profile. That choice costs delay (Q2) and is not reliable when two people overlap (Q3).

## Five likely judge questions (answers use only CLAIMS.md section 1)

**Q1. AirPods already have Conversation Boost. What's new?**
Conversation Boost amplifies whoever is in front of you, while we follow one chosen person's saved voice profile wherever they are. With our two real voices, the chosen speaker is followed correctly when they talk alone; when both of us talk at once, the wrong voice is heard 6–25% of the time (CLAIMS 1.1, 1.2: T-141, 12 runs, Mac).

**Q2. Hearing aids work in a few milliseconds. What's your latency?**
The isolated voice plays about 1.5 s behind the room on the Mac (Balanced preset, by design), and about 1 s on a GPU (950 ms measured), so this is a delayed focus stream, not real-time (CLAIMS 1.6). Only the noise filter itself adds 13 ms engine-side, and that number excludes Bluetooth and the Focus delay (CLAIMS 1.7: M4 Pro, 0.7 ms p95 compute per 10 ms chunk).

**Q3. How well does it actually separate two people?**
In our best run with our own two male voices and Joseph as target, the other voice was 21 dB quieter, SI-SDRi was +4.6 dB and the wrong voice was heard 8.6–9.1% of the time (CLAIMS 1.3: T-141, Mac). On a test clip with two different-sounding stand-in voices (not ours), separation was +8.3 dB with the other voice 18 dB down (CLAIMS 1.8: GRID male/female replay clip, Mac).

**Q4. Can I just say someone's name and have it switch to them?**
Name Call worked 2 of 4 times in quiet and 0 of 4 times with noise or overlap, and it never fired by mistake (CLAIMS 1.4: Joseph's takes). On other people's speech without the name it had 0 false triggers in 5 min (CLAIMS 1.5: GRID + 4 Kokoro voices, not our voices).

**Q5. How do we know these numbers aren't cherry-picked, and are the captions reliable?**
Every session is recorded (in.wav, out.wav, events.jsonl), so the numbers come from recordings, not memory (CLAIMS 1.10). The live captions are rough: 71–83% word error on a hard test clip with a competing talker (CLAIMS 1.9: GRID stand-in, 3 runs, Mac), and they show whatever Whisper heard.

## What not to say when comparing (from CLAIMS.md section 3)
- Do not say "lower latency than X" or "real-time". Our delay is about 1.5 s, and the competitors' latencies are not published.
- Do not say "better than hearing aids" or "better than Krisp". We never measured them side by side.
- Do not say "works on any voice" or that it isolates the judges' voices. Only Aryan's and Joseph's profiles are tuned.
- Do not claim a benefit for people with hearing loss or sensory overload. There has been no user study (CLAIMS section 2).
