// About screen copy (T-352; regrouped T-463). Every number is from docs/CLAIMS.md section 1, with its condition.
// Nothing from sections 1b, 2 or 3. Change a figure only after CLAIMS.md changes. T-463: what works first, then where
// it breaks; no unsourced claims about other products ("Nobody ships that" was removed: nothing in CLAIMS.md backs it);
// the delay is the server's expected figure (/health tse_window), filled in by screens/about.js. T-488: the app no
// longer has anyone choose a person; it lets the loudest voice (the one closest to the mic) through, so the story
// copy says that and the figures keep their measured conditions.

export const ABOUT = {
  title: "The voice closest to you comes through.",
  /** `delay` is store.js expectedDelayCopy(health, {capital: false}): "about 1.5 s behind" or "a short delay behind". */
  lede: (delay) => `VoiceLock keeps the voice closest to you and turns everything else down. It plays ${delay} the room.`,
  worksHeading: "What works",
  worksNote: "With our two real voices, it follows the right voice when only one person talks (12 tries, on a Mac). Each number says how we got it.",
  works: [
    { value: "21 dB", label: "Best run: the other voice 21 dB quieter, and the kept voice 4.6 dB clearer", condition: "Joseph's voice kept; the wrong voice heard 8.6-9.1% of the time; two men talking; on a Mac." },
    { value: "+8.3 dB", label: "Two voices pulled apart on a test recording; the other voice 18 dB quieter", condition: "A recording of two other people, not us, played on the Mac." },
    { value: "13 ms", label: "Added by the noise filter itself", condition: "The filter alone, on a laptop. Bluetooth and the Focus delay come on top." },
    // value and condition are filled from /health by screens/about.js (delayClaim)
    { key: "delay", value: "—", label: "Delay behind the room (about 1 s on a faster server)", condition: "The normal setting on the Mac. A short delay on purpose, not instant." },
  ],
  breaksHeading: "Where it breaks",
  breaks: [
    { value: "6-25%", label: "Wrong voice heard while both talk at once", condition: "The hard case. Our two voices (two men), 12 tries, on a Mac." },
    { value: "71-83%", label: "Words the captions get wrong: captions are rough", condition: "A hard test recording with someone talking over it, on a Mac." },
  ],
  farField: "From one laptop mic across a room, telling two similar men's voices apart is where it breaks: the wrong voice 6-25% of the time while both talked. So the app listens through a close mic and keeps only the loudest voice, the one closest to it; everything else goes quiet.",
  // the technical terms behind the plain lines above, in a closed "How we measured" disclosure (same figures, CLAIMS.md 1)
  measuredHeading: "How we measured",
  measured: [
    "21 dB: other voice suppressed 21 dB; SI-SDRi (scale-invariant signal-to-distortion ratio improvement) +4.6 dB; Joseph as target, 8.6-9.1% wrong-speaker time; two male voices; Mac.",
    "+8.3 dB: SI-SDRi on a stand-in clip with two different voices (not ours), other talker 18 dB down, replayed on the Mac.",
    "13 ms: algorithmic delay of a noise-only denoiser, engine-side only, on an Apple M4 Pro. Excludes Bluetooth and the Focus delay.",
    "Delay: the engine's balanced preset, its window plus hop, as reported by the server. A delay, not real-time; about 1 s on a GPU.",
    "6-25%: wrong-speaker time during overlap, our two male voices, 12 runs, Mac.",
    "71-83%: caption word error rate (WER) on a stand-in clip with a competing talker, no caption prompt, Mac.",
  ],
  logged: "Every run is logged, so numbers come from recordings, not memory.",
  footer: "Auto voice focus needs no setup and no saved voices: the loudest voice comes through on its own.",
  back: "Back to the start",
};
