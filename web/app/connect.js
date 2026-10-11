// Phone link (M14, T-351): the laptop shows a QR code that opens the companion, `/app/?watch=<session>`.
// The code and the URLs come from the server's `/connect?page=` (hearing/server/connect.py): it knows the LAN
// addresses `hearing serve --tls` printed (a laptop tab on https://localhost would otherwise encode "localhost",
// which a phone cannot reach). No QR library in the page; the server draws the SVG.
//
// mountPhoneLink(host, store) adds a small "Phone" toggle; the code is only fetched for a real session id and
// redrawn when the id changes (reconnect = new session = new code). Without a session it says so.

/** The origin-relative companion path for a session id. */
export function companionPath(sessionId) {
  return `/app/?watch=${encodeURIComponent(sessionId)}`;
}

/** `/connect?page=...` for a session id. */
export function connectUrl(sessionId) {
  return `/connect?page=${encodeURIComponent(companionPath(sessionId))}`;
}

/** Pull the QR SVG and the URLs out of the `/connect` page (HTML text). */
export function parseConnect(html, parser = new DOMParser()) {
  const doc = parser.parseFromString(html, "text/html");
  const svg = doc.querySelector(".qr svg");
  const urls = [...doc.querySelectorAll("ul a[href]")].map((a) => a.getAttribute("href"));
  const tls = Boolean(doc.querySelector('a[href="/hearing-ca.crt"]'));
  return { svg: svg ? svg.outerHTML : "", urls, tls };
}

/** Fetch the QR for a session; rejects when the server cannot draw one. */
export async function fetchPhoneLink(sessionId, fetchImpl = globalThis.fetch) {
  const res = await fetchImpl(connectUrl(sessionId), { cache: "no-store" });
  if (!res.ok) throw new Error(`/connect answered ${res.status}`);
  const out = parseConnect(await res.text());
  if (!out.svg || !out.urls.length) throw new Error("/connect returned no code");
  return out;
}

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  node.append(...children.filter((c) => c != null));
  return node;
}

/**
 * A "Phone" toggle and its QR card inside `host`. `store` is the app store (store.js): its `sessionId` is the
 * stage session the phone will mirror. Returns cleanup.
 */
export function mountPhoneLink(host, store, { signal } = {}) {
  // the card's styles live with the phone screen's (one owner); same data-href as app.js's linkStyle, so no duplicate
  if (!document.querySelector('link[data-href="screens/phone.css"]')) {
    document.head.append(h("link", { rel: "stylesheet", href: new URL("./screens/phone.css", import.meta.url).href, "data-href": "screens/phone.css" }));
  }
  const toggle = h("button", { type: "button", class: "pl-toggle", "aria-expanded": "false", "aria-controls": "pl-card" },
    h("span", { class: "pl-toggle-glyph", "aria-hidden": "true" }), h("span", { text: "Phone" }));
  const code = h("div", { class: "pl-code", "aria-hidden": "true" });
  const status = h("p", { class: "pl-status", role: "status" });
  const link = h("a", { class: "pl-url", target: "_blank", rel: "noopener" });
  // the full link is long (a 32-hex session id): show only the host on one line; Copy and the href keep all of it
  const copy = h("button", { type: "button", class: "pl-copy", text: "Copy link" });
  const linkRow = h("div", { class: "pl-link" }, link, copy);
  const card = h("aside", { id: "pl-card", class: "pl-card", hidden: true, "aria-label": "Follow this session on a phone" },
    h("p", { class: "pl-eyebrow", text: "Companion" }),
    h("p", { class: "pl-title", text: "Scan to follow on a phone" }),
    code,
    status,
    linkRow,
    h("p", { class: "pl-fine", text: "The phone mirrors this session and can press Focus, Quiet, a person or Release. It never becomes the mic or the speaker." }));
  const wrap = h("div", { class: "pl", "data-open": "false" }, toggle, card);
  host.append(wrap);

  let shownFor = undefined;
  let token = 0;
  async function draw() {
    const id = store.state.sessionId || null;
    if (id === shownFor) return;
    shownFor = id;
    const mine = ++token;
    wrap.dataset.session = id || "";
    if (!id) {
      code.replaceChildren();
      linkRow.hidden = true;
      status.textContent = "No session yet: the code appears once the engine answers.";
      return;
    }
    status.textContent = "Drawing the code…";
    try {
      const out = await fetchPhoneLink(id);
      if (mine !== token) return;
      code.innerHTML = out.svg; // the server's own SVG (hearing.server.tls.qr_svg), no user input in it
      link.href = out.urls[0];
      link.title = out.urls[0];
      link.textContent = new URL(out.urls[0]).host; // the session id is on the status line above
      linkRow.hidden = false;
      status.textContent = out.tls ? `Session ${id.slice(0, 8)}` : `Session ${id.slice(0, 8)} · plain http: a phone needs --tls`;
    } catch (err) {
      if (mine !== token) return;
      code.replaceChildren();
      linkRow.hidden = true;
      status.textContent = `No code: ${err?.message || err}`;
    }
  }
  function setOpen(open) {
    wrap.dataset.open = String(open);
    toggle.setAttribute("aria-expanded", String(open));
    card.hidden = !open;
    if (open) draw();
  }
  toggle.addEventListener("click", () => setOpen(card.hidden), { signal });
  copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(link.href); copy.textContent = "Copied"; }
    catch { copy.textContent = "Copy failed"; }
    setTimeout(() => { copy.textContent = "Copy link"; }, 1500);
  }, { signal });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !card.hidden) setOpen(false); }, { signal });
  const off = store.on("session", () => { if (!card.hidden) draw(); });
  const cleanup = () => { off(); wrap.remove(); };
  signal?.addEventListener("abort", cleanup, { once: true });
  return cleanup;
}
