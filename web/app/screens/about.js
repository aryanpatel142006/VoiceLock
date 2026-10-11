// About screen (T-352): pitch, the far-field honesty line and the measured claims with their conditions.
// Copy lives in ../copy.js. Static text only: no telemetry, nothing simulated.
import { ABOUT } from "../copy.js";

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

export function mount(root) {
  const c = ABOUT;
  root.append(el("article", { class: "about", "aria-labelledby": "about-title" },
    el("p", { class: "eyebrow", text: c.eyebrow }),
    el("h1", { id: "about-title", class: "about-title", text: c.title }),
    el("p", { class: "about-lede", text: c.lede }),
    el("aside", { class: "about-honesty", "aria-label": "Far-field honesty" },
      el("p", { class: "about-honesty-text", text: c.farField })),
    el("section", { class: "about-claims", "aria-labelledby": "about-claims-h" },
      el("h2", { id: "about-claims-h", class: "about-h", text: c.claimsHeading }),
      el("p", { class: "about-note", text: c.claimsNote }),
      el("ul", { class: "about-list" }, ...c.claims.map((k, i) =>
        el("li", { class: "claim", style: `--i:${i}` },
          el("span", { class: "claim-value", text: k.value }),
          el("span", { class: "claim-label", text: k.label }),
          el("span", { class: "claim-cond", text: k.condition }))))),
    el("p", { class: "about-note", text: c.logged }),
    el("p", { class: "about-note", text: c.footer }),
    el("p", {}, el("a", { href: "#welcome", class: "link", text: c.back }))));
}
