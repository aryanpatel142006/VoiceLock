// About screen (T-352): pitch, the measured claims with their conditions, then where it breaks and the far-field line.
// Copy lives in ../copy.js. Static text plus one server fact: the expected delay from /health tse_window (T-463).
import { ABOUT } from "../copy.js";
import { expectedDelayCopy, expectedDelayMs, fetchHealth, fmtDelay } from "../store.js";

export const title = "About";

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  node.append(...children.filter((c) => c != null));
  return node;
}

/** The delay row from /health (T-419 window + hop): "~1.5 s", or "—" while the server has not said. */
export function delayClaim(health) {
  const ms = expectedDelayMs(health);
  return ms == null ? { value: "—", condition: "Waiting for the engine to report its window." }
    : { value: `~${fmtDelay(ms)}`, condition: "This engine's balanced preset: its window plus hop, by design. A delay, not real-time." };
}

function claimList(items, offset = 0) {
  return el("ul", { class: "about-list" }, ...items.map((k, i) =>
    el("li", { class: "claim", style: `--i:${i + offset}`, ...(k.key ? { "data-claim": k.key } : {}) },
      el("span", { class: "claim-value", text: k.value }),
      el("span", { class: "claim-label", text: k.label }),
      el("span", { class: "claim-cond", text: k.condition }))));
}

export function mount(root, { signal } = {}) {
  const c = ABOUT;
  const article = el("article", { class: "about", "aria-labelledby": "about-title" },
    el("p", { class: "eyebrow", text: c.eyebrow }),
    el("h1", { id: "about-title", class: "about-title", text: c.title }),
    el("p", { class: "about-lede", text: c.lede(expectedDelayCopy(null, { capital: false })) }),
    el("section", { class: "about-claims", "aria-labelledby": "about-works-h" },
      el("h2", { id: "about-works-h", class: "about-h", text: c.worksHeading }),
      el("p", { class: "about-note", text: c.worksNote }),
      claimList(c.works)),
    el("section", { class: "about-claims about-breaks", "aria-labelledby": "about-breaks-h" },
      el("h2", { id: "about-breaks-h", class: "about-h", text: c.breaksHeading }),
      claimList(c.breaks, c.works.length),
      el("aside", { class: "about-honesty", "aria-label": "Far-field honesty" },
        el("p", { class: "about-honesty-text", text: c.farField }))),
    el("p", { class: "about-note", text: c.logged }),
    el("p", { class: "about-note", text: c.footer }),
    el("p", {}, el("a", { href: "#sos", class: "link", text: "SOS: pair the band" })), // T-461
    el("p", {}, el("a", { href: "#welcome", class: "link", text: c.back })));
  root.append(article);
  fetchHealth().then((health) => {
    if (signal?.aborted || !health) return;
    article.querySelector(".about-lede").textContent = c.lede(expectedDelayCopy(health, { capital: false }));
    const row = article.querySelector('[data-claim="delay"]');
    const d = delayClaim(health);
    row.querySelector(".claim-value").textContent = d.value;
    row.querySelector(".claim-cond").textContent = d.condition;
  });
}
