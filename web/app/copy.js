// About screen copy (T-352). Every number is from docs/CLAIMS.md section 1, with its condition. Nothing from
// sections 1b, 2 or 3. Change a figure only after CLAIMS.md changes.

export const ABOUT = {
  eyebrow: "About",
  title: "Choose whose voice you hear.",
  lede: "Selective Hearing turns down a noisy room and keeps one person you chose. It plays about 1.5 s behind the room.",
  // Verbatim from docs/STAGE_CARD.md section 4. Do not reword.
  farField: "From one laptop mic across a room, picking one of two similar male voices did not work in our tests: 6-25% wrong while both talked. Nobody ships that; products that select a voice have the talker at the mic, and products that hear across a table use several mics and direction. So the person you want to hear wears a clip mic, and the app does the part no product does: pick who, mute the rest, caption only them.",
  claimsHeading: "What we measured",
  claimsNote: "Each figure comes with the condition it was measured under. Recorded sessions, not memory.",
  claims: [
    { value: "Solo", label: "Followed correctly when they talk alone", condition: "Our two real voices, 12 runs, Mac." },
    { value: "6-25%", label: "Wrong voice heard while both talk at once", condition: "The hard case. Same two voices, 12 runs, Mac." },
    { value: "21 dB", label: "Best run: other voice quieter, 4.6 dB cleaner (SI-SDRi)", condition: "Joseph as target, 8.6-9.1% wrong, two male voices, Mac." },
    { value: "2 of 4", label: "Name Call worked in quiet; 0 of 4 with noise or overlap", condition: "Never fired by mistake in those takes." },
    { value: "0", label: "Name Call false triggers in 5 min of other speech", condition: "Not our voices: GRID plus 4 Kokoro voices (5.06 min)." },
    { value: "~1.5 s", label: "Delay behind the room (about 1 s on a GPU)", condition: "Balanced preset on the Mac, by design. A delay, not real-time." },
    { value: "13 ms", label: "Added by the noise filter itself", condition: "Engine-side only: a noise-only denoiser on an M4 Pro. Excludes Bluetooth and the Focus delay." },
    { value: "+8.3 dB", label: "Separation on a test clip, other voice 18 dB down", condition: "Stand-in clip with two different voices, not ours, replayed on the Mac." },
    { value: "71-83%", label: "Caption word error", condition: "A hard test clip with a competing talker, no caption prompt, Mac. Captions are rough." },
  ],
  logged: "Every run is logged, so numbers come from recordings, not memory.",
  footer: "Saved voice profiles are cuts of recordings, not learned just now. Only our two saved profiles are tuned.",
  back: "Back to the start",
};
