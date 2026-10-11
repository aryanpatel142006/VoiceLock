// About screen copy (T-352; regrouped T-463). Every number is from docs/CLAIMS.md section 1, with its condition.
// Nothing from sections 1b, 2 or 3. Change a figure only after CLAIMS.md changes. T-463: what works first, then where
// it breaks; no unsourced claims about other products ("Nobody ships that" was removed: nothing in CLAIMS.md backs it);
// the delay is the server's expected figure (/health tse_window), filled in by screens/about.js.

export const ABOUT = {
  eyebrow: "About",
  title: "Choose whose voice you hear.",
  /** `delay` is store.js expectedDelayCopy(health, {capital: false}): "about 1.5 s behind" or "a short delay behind". */
  lede: (delay) => `Selective Hearing turns down a noisy room and keeps one person you chose. It plays ${delay} the room.`,
  worksHeading: "What works",
  worksNote: "With our two real voices, the chosen person is followed correctly when they talk alone (12 runs, Mac). Each figure comes with the condition it was measured under.",
  works: [
    { value: "21 dB", label: "Best run: the other voice quieter, and 4.6 dB cleaner (SI-SDRi)", condition: "Joseph as target, 8.6-9.1% wrong, two male voices, Mac." },
    { value: "+8.3 dB", label: "Separation on a test clip, other voice 18 dB down", condition: "Stand-in clip with two different voices, not ours, replayed on the Mac." },
    { value: "13 ms", label: "Added by the noise filter itself", condition: "Engine-side only: a noise-only denoiser on an M4 Pro. Excludes Bluetooth and the Focus delay." },
    { value: "0", label: "Name Call false triggers in 5 min of other speech", condition: "Not our voices: GRID plus 4 Kokoro voices (5.06 min)." },
    // value and condition are filled from /health by screens/about.js (delayClaim)
    { key: "delay", value: "—", label: "Delay behind the room (about 1 s on a GPU)", condition: "Balanced preset on the Mac, by design. A delay, not real-time." },
  ],
  breaksHeading: "Where it breaks",
  breaks: [
    { value: "6-25%", label: "Wrong voice heard while both talk at once", condition: "The hard case. Our two male voices, 12 runs, Mac." },
    { value: "0 of 4", label: "Name Call with noise or overlap (2 of 4 in quiet)", condition: "Joseph's takes. It never fired by mistake in them." },
    { value: "71-83%", label: "Caption word error: captions are rough", condition: "A hard test clip with a competing talker, no caption prompt, Mac." },
  ],
  farField: "From one laptop mic across a room, picking one of two similar male voices is where it breaks: 6-25% wrong while both talked. So the person you want to hear talks into a close mic (a phone over USB, or the laptop up close), and the app picks who, turns down the rest, and captions only them.",
  logged: "Every run is logged, so numbers come from recordings, not memory.",
  footer: "Saved voice profiles are cuts of recordings, not learned just now. Only our two saved profiles are tuned.",
  back: "Back to the start",
};
