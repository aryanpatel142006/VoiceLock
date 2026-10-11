// Selective Hearing app service worker (T-345). Scope /app/.
// Network-first: the demo always runs the JS on the server; the cache is only an offline fallback for
// the shell. Never touches the engine (/ws, /health), the console files or the SDK: those go straight
// to the network. VERSION changes with every build: the server (hearing/server/static.py) replaces
// BUILD with the git rev + a hash of web/app/ (T-452), so a new build drops the old caches on activate
// instead of serving stale modules (Safari did). "dev" only when sw.js is served by something else.
const BUILD = "dev";
const VERSION = `hearing-app-${BUILD}`;
const SHELL = [
  "./", "index.html", "app.js", "app.css", "manifest.webmanifest",
  "icons/icon-192.png", "icons/icon-512.png", "icons/maskable-512.png",
  "icons/apple-touch-icon.png", "icons/favicon-32.png",
];
const SCOPE = new URL("./", self.location).pathname; // "/app/"

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(VERSION);
    // one missing file must not fail the install (the shell still works online)
    await Promise.all(SHELL.map((path) => cache.add(path).catch(() => undefined)));
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names.filter((n) => n.startsWith("hearing-app-") && n !== VERSION).map((n) => caches.delete(n)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== self.location.origin || !url.pathname.startsWith(SCOPE)
      || req.headers.has("range")) {
    return; // engine, console, SDK, uploads: the browser handles them, untouched
  }
  event.respondWith((async () => {
    try {
      const res = await fetch(req);
      if (res.ok && res.type === "basic") {
        const copy = res.clone();
        event.waitUntil(caches.open(VERSION).then((c) => c.put(req, copy)));
      }
      return res;
    } catch (err) {
      const hit = await caches.match(req, { ignoreSearch: req.mode === "navigate" });
      if (hit) return hit;
      if (req.mode === "navigate") {
        const shell = await caches.match("./");
        if (shell) return shell;
      }
      throw err;
    }
  })());
});
