// Fake camera and mic for the headless console checks (Chromium's --use-file-for-fake-*):
// two green marker "faces", the left one's black mouth opening at 4 Hz, and noise bursts on the
// mic while it is open. The server's mock detector (HEARING_FACES=mock) finds the boxes and the
// lip-motion ASD (HEARING_ASD=lipmotion) pairs the mouth with the bursts.

import { writeFileSync } from "node:fs";

export const W = 640, H = 360, FPS = 25, SECONDS = 4, SR = 48000;
export const talking = (t) => Math.floor(t * 8) % 2 === 0; // 125 ms open/loud, 125 ms closed/quiet
export const FACES = [
  { x: 0.08, y: 0.25, w: 0.16, h: 0.45, talks: true },
  { x: 0.42, y: 0.25, w: 0.16, h: 0.45, talks: false },
];

/** Y4M (4:2:0) clip: grey background, green faces, black mouths. */
export function writeY4m(path) {
  const header = `YUV4MPEG2 W${W} H${H} F${FPS}:1 Ip A1:1 C420jpeg\n`;
  const frames = [Buffer.from(header)];
  for (let i = 0; i < FPS * SECONDS; i++) {
    const y = new Uint8Array(W * H).fill(128);
    const u = new Uint8Array((W / 2) * (H / 2)).fill(128);
    const v = new Uint8Array((W / 2) * (H / 2)).fill(128);
    for (const f of FACES) {
      const x0 = Math.round(f.x * W), y0 = Math.round(f.y * H);
      const x1 = Math.round((f.x + f.w) * W), y1 = Math.round((f.y + f.h) * H);
      const mouth = f.talks && talking(i / FPS) ? Math.round(0.15 * (y1 - y0)) : 0;
      const my = y0 + Math.round(0.65 * (y1 - y0));
      const mx0 = x0 + Math.round((x1 - x0) / 4), mx1 = x1 - Math.round((x1 - x0) / 4);
      for (let r = y0; r < y1; r++) {
        for (let c = x0; c < x1; c++) {
          const dark = mouth && r >= my && r < my + mouth && c >= mx0 && c < mx1;
          y[r * W + c] = dark ? 0 : 150; // full-range BT.601 pure green: Y 150, U 44, V 21
          if (r % 2 === 0 && c % 2 === 0) {
            u[(r / 2) * (W / 2) + c / 2] = dark ? 128 : 44;
            v[(r / 2) * (W / 2) + c / 2] = dark ? 128 : 21;
          }
        }
      }
    }
    frames.push(Buffer.from("FRAME\n"), y, u, v);
  }
  writeFileSync(path, Buffer.concat(frames));
}

/** Mono PCM16 WAV: noise bursts while the left face's mouth is open. */
export function writeWav(path) {
  const n = SR * SECONDS;
  const data = Buffer.alloc(n * 2);
  let seed = 1;
  for (let i = 0; i < n; i++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    const noise = seed / 0x7fffffff - 0.5;
    const amp = talking(i / SR) ? 0.5 : 0.004;
    data.writeInt16LE(Math.round(noise * amp * 32767), i * 2);
  }
  const h = Buffer.alloc(44);
  h.write("RIFF", 0); h.writeUInt32LE(36 + data.length, 4); h.write("WAVE", 8);
  h.write("fmt ", 12); h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20); h.writeUInt16LE(1, 22);
  h.writeUInt32LE(SR, 24); h.writeUInt32LE(SR * 2, 28); h.writeUInt16LE(2, 32); h.writeUInt16LE(16, 34);
  h.write("data", 36); h.writeUInt32LE(data.length, 40);
  writeFileSync(path, Buffer.concat([h, data]));
}

