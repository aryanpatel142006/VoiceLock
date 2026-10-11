// T-138: camera capture clock. Chrome throttles a hidden tab's setInterval to 1 Hz (live session
// 20261009T174419Z-2: video at exactly 1.0 s spacing while the presenter looked at another window,
// vision flapped and the lock dropped). A dedicated Worker's timers are not throttled that way, so
// the worker ticks and the page captures on each tick. Falls back to setInterval without Workers.

const WORKER_SRC = "let t=null;onmessage=(e)=>{clearInterval(t);t=e.data>0?setInterval(()=>postMessage(0),e.data):null;};";

/** Call ``fn`` every ``ms`` milliseconds, unthrottled when possible; returns ``{ stop() }``. */
export function frameClock(ms, fn) {
  let worker = null;
  try {
    const url = URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" }));
    worker = new Worker(url);
    URL.revokeObjectURL(url);
  } catch {
    worker = null;  // CSP or no Worker: plain (throttled) timer
  }
  if (!worker) {
    const timer = setInterval(fn, ms);
    return { stop: () => clearInterval(timer) };
  }
  worker.onmessage = () => fn();
  worker.postMessage(ms);
  return { stop: () => worker.terminate() };
}

/**
 * Watches the gap between captured frames while the page is hidden: ``onChange(true)`` once frames
 * stop arriving at ``minFps`` (the browser paused the camera or throttled capture), ``onChange(false)``
 * when they are back. ``frame(nowMs)`` per sent frame,
 * ``check(nowMs, hidden)`` on every capture tick and on every render.
 */
export class BackgroundWatch {
  constructor(minFps, onChange) {
    this.maxGapMs = 1000 / minFps;
    this.onChange = onChange;
    this.lastFrameMs = null;
    this.paused = false;
  }

  frame(nowMs) {
    this.lastFrameMs = nowMs;
    if (this.paused) this.#set(false);
  }

  check(nowMs, hidden) {
    const late = this.lastFrameMs !== null && nowMs - this.lastFrameMs > this.maxGapMs;
    if (late && hidden && !this.paused) this.#set(true);
  }

  #set(paused) {
    this.paused = paused;
    this.onChange(paused);
  }
}
