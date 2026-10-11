"""LockedIn Comfort Companion: local profiles, brief consented guidance, and the SOS check-in plan.

Two layers share one Gemini adapter:
- Joseph's profile layer (``/comfort/profiles``, ``/comfort/guidance``, ``/comfort/feedback``): per-person preferences
  (helps / avoid / sounds / style), one strategy from the catalog's ``strategies`` that the profile does not avoid
  (its helps ranked first, then the rest of the catalog; the client's ``recent`` ids are skipped so "Another idea"
  varies),
  optional ElevenLabs speech (off unless ``speak`` is true and a key exists). Confirmed feedback is the learning loop:
  "not helpful" + confirm moves a strategy into avoid and writes ``history``; nothing is saved without confirm.
- The plan layer (``/comfort/plan``): after SOS cuts the room the app asks "What's too much right now?" and sends one
  of four button ids. The answer is a structured plan built only from ``comfort_catalog.json`` ids (a calm sound, a
  breathing pattern, three grounding lines, an optional message to a trusted person). Gemini may pick the ids and word
  the lines; its output is schema-constrained and validated, and anything off-format, slow (> ``GEMINI_TIMEOUT_S``,
  default 2 s), or unavailable falls back to the catalog default, reported honestly as ``source: default``.

Only button taps ever reach Gemini: never audio, never a transcript. ``GEMINI_API_KEY`` is read from the environment
and never sent to the client. ``HEARING_COMFORT=1`` enables the Gemini path; unset, every plan is the default plan.
Support for sensory overload, not medical advice.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import ssl
import time
import wave
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from fastapi import FastAPI, HTTPException

from hearing.audio.io import float_to_pcm16
from hearing.synth.tts import ElevenLabsTTS, TtsError

DEFAULT_MODEL = "gemini-3.5-flash-lite"
# Spoken guidance: a calm, warm premade voice that this restricted key can use (Sarah; checked 200 on Oct 11, 2026).
DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
DEFAULT_TTS_MODEL = "eleven_flash_v2_5"
DEFAULT_TIMEOUT_S = 2.0
CATALOG_PATH = Path(__file__).with_name("comfort_catalog.json")
TRIGGERS = ("noise", "people", "lights", "unknown")
MAX_LINE_WORDS = 14
MAX_MESSAGE_WORDS = 30
log = logging.getLogger(__name__)
_SSL_CONTEXT: ssl.SSLContext | None = None


def ssl_context() -> ssl.SSLContext:
    """Verified TLS with certifi's CA bundle. The python.org macOS build ships without system CAs, so the bare
    default context fails every Gemini call with CERTIFICATE_VERIFY_FAILED."""
    global _SSL_CONTEXT
    if _SSL_CONTEXT is None:
        try:
            import certifi
            _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            _SSL_CONTEXT = ssl.create_default_context()
    return _SSL_CONTEXT


def _redact(text: str, api_key: str) -> str:
    return text.replace(api_key, "<redacted>") if api_key else text


# words that make a grounding line or message medical or alarming; a plan containing one is replaced by the default
BLOCKED_WORDS = ("diagnos", "disorder", "medic", "therap", "emergency", "911", "doctor", "panic attack", "http", "www.")


def gemini_timeout_s() -> float:
    try:
        return max(0.2, float(os.environ.get("GEMINI_TIMEOUT_S", DEFAULT_TIMEOUT_S)))
    except ValueError:
        return DEFAULT_TIMEOUT_S


def gemini_model() -> str:
    return os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


def comfort_enabled() -> bool:
    return os.environ.get("HEARING_COMFORT", "").strip().lower() in {"1", "true", "yes", "on"}


def gemini_json(api_key: str, model: str, system: str, user: dict, schema: dict, *, timeout_s: float,
                max_tokens: int = 320, temperature: float = 0.3) -> Any:
    """One ``generateContent`` call returning the parsed JSON the response schema forced. Raises on any failure."""
    request_body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(user)}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": temperature,
                             "maxOutputTokens": max_tokens, "responseSchema": schema},
    }
    request = Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(request_body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s, context=ssl_context()) as response:
            result = json.load(response)
    except HTTPError as exc:
        try:
            message = json.loads(exc.read() or b"{}").get("error", {}).get("message", "")
        except (OSError, ValueError, AttributeError):
            message = ""
        log.warning("Gemini %s HTTP %s: %s", model, exc.code, _redact(str(message or exc.reason)[:300], api_key))
        raise
    except (URLError, OSError) as exc:
        log.warning("Gemini %s request failed: %s", model, _redact(str(exc)[:300], api_key))
        raise
    try:
        return json.loads(result["candidates"][0]["content"]["parts"][0]["text"])
    except (KeyError, IndexError, TypeError, ValueError):
        log.warning("Gemini %s returned no JSON text (finishReason %s)", model,
                    (result.get("candidates") or [{}])[0].get("finishReason") if isinstance(result, dict) else None)
        raise
MODES = {"silence", "music", "speech"}
TALK = {"little", "some"}
STYLES = {"short", "normal"}
QUESTIONS = {"minimal", "normal"}
SPOKEN = {"only when requested", "never"}
FAVORITE_SOUNDS = {"brown", "rain", "white", "pad"}
PROFILE_ID = r"[a-z0-9-]{1,48}"
MAX_HISTORY = 50

DEMO_PROFILES = (
    {"id": "alex-demo", "preferred_name": "Alex", "demo": True, "fictional": True, "preferred_mode": "speech",
     "talk_amount": "little", "activities": ["drawing", "listening to music"],
     "helps": ["breathing", "lofi"], "avoid": ["loud noise"], "favorite_sound": "pad", "volume": 40,
     "sound_volume": 40, "guidance_style": "short", "questions": "normal",
     "spoken_guidance": "only when requested", "history": []},
    {"id": "sam-demo", "preferred_name": "Sam", "demo": True, "fictional": True, "preferred_mode": "silence",
     "talk_amount": "little", "activities": ["holding a familiar object"],
     "helps": ["silence", "sensory-object"], "avoid": ["breathing", "loud noise"], "favorite_sound": None,
     "volume": 0, "sound_volume": 0, "guidance_style": "short", "questions": "minimal",
     "spoken_guidance": "only when requested", "history": []},
)


def _strings(value: Any, key: str, limit: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise HTTPException(422, f"{key} must be a list of short strings")
    out = [str(x).strip()[:80] for x in value if isinstance(x, str | int | float) and str(x).strip()]
    return list(dict.fromkeys(out))[:limit]


def _history(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    entries = []
    for item in value[-MAX_HISTORY:]:
        if isinstance(item, dict) and isinstance(item.get("strategy"), str) and isinstance(item.get("helpful"), bool):
            entries.append({"ts": str(item.get("ts", ""))[:40], "strategy": item["strategy"][:48],
                            "label": str(item.get("label", item["strategy"]))[:48], "helpful": item["helpful"],
                            "text": str(item.get("text", ""))[:120]})
    return entries


def normalize_profile(profile_id: str, data: dict) -> dict:
    """The stored shape. Accepts new fields and the PR #402 ones (dislikes/worked/talk_amount), raises 422 on bad input.

    ``avoid``/``helps`` are the source of truth; ``dislikes``/``worked``/``breathing_helps``/``movement_helps`` are kept
    as mirrors so the plan layer and older clients still read sensible values."""
    if not isinstance(data, dict):
        raise HTTPException(422, "Profile must be an object")
    name = data.get("preferred_name")
    if not isinstance(name, str) or not name.strip():
        raise HTTPException(422, "preferred_name is required")
    helps = _strings(data["helps"], "helps", 12) if "helps" in data else [
        *(["breathing"] if data.get("breathing_helps") else []), *(["movement"] if data.get("movement_helps") else []),
        *_strings(data.get("worked"), "worked", 10)]
    avoid = _strings(data["avoid"] if "avoid" in data else data.get("dislikes"), "avoid", 12)
    helps = [h for h in dict.fromkeys(helps) if h.lower() not in {a.lower() for a in avoid}]
    style = data.get("guidance_style") or ("short" if data.get("talk_amount", "little") == "little" else "normal")
    if style not in STYLES:
        raise HTTPException(422, "guidance_style must be short or normal")
    mode = data.get("preferred_mode", "speech")
    if mode not in MODES:
        raise HTTPException(422, "preferred_mode must be silence, music or speech")
    questions = data.get("questions", "normal")
    if questions not in QUESTIONS:
        raise HTTPException(422, "questions must be minimal or normal")
    spoken = data.get("spoken_guidance", "only when requested")
    if spoken not in SPOKEN:
        raise HTTPException(422, "spoken_guidance must be 'only when requested' or 'never'")
    favorite = data.get("favorite_sound")
    if favorite is not None and favorite not in FAVORITE_SOUNDS:
        raise HTTPException(422, f"favorite_sound must be one of {', '.join(sorted(FAVORITE_SOUNDS))} or null")
    # one stored value, 0-100; ``volume`` and ``sound_volume`` are two names for it (both accepted, both returned)
    volume = data.get("sound_volume") if data.get("sound_volume") is not None else data.get("volume")
    if volume is not None:
        if isinstance(volume, bool) or not isinstance(volume, int | float) or not 0 <= volume <= 100:
            raise HTTPException(422, "volume / sound_volume must be a number 0-100 or null")
        volume = round(volume)
    lowered = {h.lower() for h in helps}
    return {"id": profile_id, "preferred_name": name.strip()[:40], "demo": bool(data.get("demo", False)),
            "fictional": bool(data.get("fictional", data.get("demo", False))), "preferred_mode": mode,
            "talk_amount": "little" if style == "short" else "some", "activities": _strings(data.get("activities"), "activities", 6),
            "helps": helps, "avoid": avoid, "favorite_sound": favorite,
            "volume": volume, "sound_volume": volume,
            "guidance_style": style, "questions": questions, "spoken_guidance": spoken,
            "history": _history(data.get("history", [])),
            # mirrors for the plan layer / PR #402 clients
            "dislikes": list(avoid), "worked": list(helps),
            "breathing_helps": "breathing" in lowered, "movement_helps": "movement" in lowered}


class ComfortStore:
    """Profiles in one private local JSON file (mode 0600). Seeds the two fictional demo profiles."""

    def __init__(self, directory: Path):
        self.path = directory / "profiles.json"
        directory.mkdir(parents=True, exist_ok=True)
        saved = self._read() if self.path.exists() else {}
        for seed in DEMO_PROFILES:  # seed, or upgrade a pre-learning-loop demo profile
            if seed["id"] not in saved or "helps" not in saved[seed["id"]]:
                saved[seed["id"]] = normalize_profile(seed["id"], seed)
        self._write(saved)

    def reset_demo(self) -> list[str]:
        """Put the fictional demo profiles (Alex, Sam) back to their seeds; every other profile is left alone."""
        saved = self._read()
        for seed in DEMO_PROFILES:
            saved[seed["id"]] = normalize_profile(seed["id"], seed)
        self._write(saved)
        return [seed["id"] for seed in DEMO_PROFILES]

    def _read(self) -> dict:
        return json.loads(self.path.read_text())

    def _write(self, data: dict) -> None:
        pending = self.path.with_suffix(".tmp")
        pending.write_text(json.dumps(data, indent=2) + "\n")
        pending.chmod(0o600)
        pending.replace(self.path)

    def all(self) -> list[dict]:
        return [normalize_profile(pid, p) for pid, p in self._read().items()]

    def get(self, profile_id: str) -> dict:
        if not isinstance(profile_id, str) or not re.fullmatch(PROFILE_ID, profile_id):
            raise HTTPException(422, "Invalid profile ID")
        profile = self._read().get(profile_id)
        if profile is None:
            raise HTTPException(404, "Comfort profile not found")
        return normalize_profile(profile_id, profile)

    def put(self, profile_id: str, data: dict, *, history: list[dict] | None = None) -> dict:
        """Save a profile. ``history`` is kept from the stored copy: only confirmed feedback (``history=``) writes it."""
        if not re.fullmatch(PROFILE_ID, profile_id):
            raise HTTPException(422, "Invalid profile ID")
        saved = self._read()
        kept = history if history is not None else saved.get(profile_id, {}).get("history", [])
        profile = normalize_profile(profile_id, {**data, "history": kept}) if isinstance(data, dict) \
            else normalize_profile(profile_id, data)
        saved[profile_id] = profile
        self._write(saved)
        return profile

    def remove_entry(self, profile_id: str, field: str, entry: str) -> dict:
        if field not in {"helps", "avoid"}:
            raise HTTPException(422, "field must be helps or avoid")
        profile = self.get(profile_id)
        before = profile[field]
        profile[field] = [x for x in before if x.lower() != entry.lower()]
        if profile[field] == before:
            raise HTTPException(404, f"{entry} is not in {field}")
        profile.pop("dislikes"), profile.pop("worked")
        return self.put(profile_id, profile)

    def feedback(self, catalog: dict, profile_id: str, strategy_id: str, helpful: bool, confirm: bool) -> dict:
        profile = self.get(profile_id)
        strategy = find_strategy(catalog, strategy_id)
        if strategy is None:
            raise HTTPException(422, "Unknown strategy")
        label, name = strategy["label"], profile["preferred_name"]
        if not confirm:
            return {"saved": False, "profile": profile, "change": None,
                    "ask": f"Remember that {label.lower()} wasn't helpful?" if not helpful
                    else f"Remember that {label.lower()} helped?"}
        sid = strategy["id"]

        def matches(entry: str) -> bool:
            return entry.lower() in {sid, label.lower()}

        if helpful:
            profile["helps"] = [*[x for x in profile["helps"] if not matches(x)], sid]
            profile["avoid"] = [x for x in profile["avoid"] if not matches(x)]
            if strategy.get("favorite"):
                profile["favorite_sound"] = strategy["favorite"]
            text = f"{label} helped last time."
            change = f"Saved: {label.lower()} is now on {name}'s list of things that help."
        else:
            profile["avoid"] = [*[x for x in profile["avoid"] if not matches(x)], sid]
            profile["helps"] = [x for x in profile["helps"] if not matches(x)]
            if strategy.get("favorite") and profile["favorite_sound"] == strategy["favorite"]:
                profile["favorite_sound"] = None
            text = f"{label} wasn't helpful last time."
            change = f"Saved: {label.lower()} won't be suggested to {name} again."
        profile["history"] = [*profile["history"], {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                    "strategy": sid, "label": label, "helpful": helpful,
                                                    "text": text}][-MAX_HISTORY:]
        history = profile.pop("history")
        profile.pop("dislikes"), profile.pop("worked")
        return {"saved": True, "profile": self.put(profile_id, profile, history=history), "change": change,
                "ask": None}


# ---------------------------------------------------------------- guidance: one strategy, learned from feedback

def find_strategy(catalog: dict, key: Any) -> dict | None:
    """A catalog strategy by id or (case-insensitive) label."""
    if not isinstance(key, str):
        return None
    low = key.strip().lower()
    return next((s for s in catalog["strategies"] if low in {s["id"], s["label"].lower()}), None)


def is_avoided(strategy: dict, avoid: list[str]) -> bool:
    terms = [a.strip().lower() for a in avoid if a.strip()]
    label = strategy["label"].lower()
    return any(t == strategy["id"] or t == label or (len(t) >= 4 and t in label) for t in terms)


MAX_RECENT = 3


def clean_recent(value: Any) -> list[str]:
    """The ``recent`` request field: up to the last three strategy ids suggested this SOS session (oldest first)."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise HTTPException(422, "recent must be a list of strategy ids")
    ids = [x.strip().lower() for x in value if isinstance(x, str) and re.fullmatch(PROFILE_ID, x.strip().lower())]
    return list(dict.fromkeys(reversed(ids)))[:MAX_RECENT][::-1]


def ranked_strategies(catalog: dict, profile: dict) -> list[dict]:
    """Every strategy the profile does not avoid, ranked: its helps first (in its order), then the rest of the
    catalog, then the quiet fallback. Avoided strategies never appear. Never empty."""
    avoid = profile.get("avoid", [])
    helps = [s for s in (find_strategy(catalog, h) for h in profile.get("helps", [])) if s is not None]
    rest = [s for s in catalog["strategies"] if not s.get("fallback")]
    tail = [s for s in catalog["strategies"] if s.get("fallback")]
    ranked = [s for s in {s["id"]: s for s in [*helps, *rest, *tail]}.values() if not is_avoided(s, avoid)]
    return ranked or tail


def allowed_strategies(catalog: dict, profile: dict, recent: list[str] | None = None) -> list[dict]:
    """The ranked set minus what was suggested recently, unless that would leave nothing."""
    ranked = ranked_strategies(catalog, profile)
    fresh = [s for s in ranked if s["id"] not in set(recent or [])]
    return fresh or ranked


def rotate_pick(catalog: dict, profile: dict, recent: list[str] | None = None) -> dict:
    """The offline pick: the next ranked strategy after the most recent one, skipping anything recent. With no
    history it is the first of the profile's helps; repeated "Another idea" walks the whole allowed set."""
    ranked = ranked_strategies(catalog, profile)
    recent = list(recent or [])
    ids = [s["id"] for s in ranked]
    start = ids.index(recent[-1]) + 1 if recent and recent[-1] in ids else 0
    order = ranked[start:] + ranked[:start]
    return next((s for s in order if s["id"] not in set(recent)), order[0])


def strategy_text(strategy: dict, profile: dict) -> str:
    return strategy[profile.get("guidance_style", "short")].format(name=profile["preferred_name"])


GUIDANCE_SYSTEM = (
    "You are LockedIn's optional Comfort Companion for someone in sensory overload. Pick the single best NEW strategy "
    "id for this person from the allowed list. The list is ranked: entries with from_helps true are the person's own "
    "list of what helps, so prefer those, but the person asked for another idea, so never repeat anything in "
    "recently_suggested and do pick a different kind of strategy when that fits better. Phrase it freshly in your own "
    "words: vary the wording each time, do not copy the example. Address the person by their preferred name, at most "
    "two short, gentle sentences, in their style: 'short' means one sentence of at most 14 words; 'normal' at most two "
    "sentences and 30 words. Never mention anything in avoid. No questions if questions is 'minimal'. Do not assume a "
    "diagnosis, give medical advice, force an exercise, claim to know feelings, promise relief or mention emergencies. "
    "Nothing plays or starts until the person taps: never say something is playing, is ready to play, or that you "
    "started it. Only entries with app_action can be started from the screen; for those, invite instead of claiming "
    "(e.g. 'Want some soft music? Tap play.'). For entries without app_action give plain advice and never mention "
    "tapping, a button or the app playing anything. "
    "Return only JSON with strategy and text."
)

# what a strategy can start on the Talk card: sound -> Play <label> (sounds.js), breathing -> the breathing circle,
# silence -> the Stay Quiet card; everything else (an object, movement, a quieter spot, a pause) is advice only
ACTIONS = {"sound": "play", "breathing": "breathe", "silence": "quiet"}
CLAIMS = re.compile(r"\b(?:is|are|now|already) playing\b|\bready (?:to play|for you)\b|\bplaying (?:now|for you)\b"
                    r"|\bi(?:'ve| have| just)? ?(?:started|put on|turned on|queued)\b|\bi(?:'ll| will) (?:play|start)\b",
                    re.I)
TAP = re.compile(r"\b(?:tap|press|click|button)\b", re.I)


class WordingError(ValueError):
    """Gemini's text claims an app action that has not happened (or one the strategy has none of): the pick stands,
    the catalog's own text replaces the wording."""

    def __init__(self, strategy: dict, why: str):
        super().__init__(why)
        self.strategy = strategy


def check_claims(text: str, strategy: dict) -> str:
    if CLAIMS.search(text):
        raise WordingError(strategy, "text claims something is already playing or started")
    if strategy_action(strategy) is None and TAP.search(text):
        raise WordingError(strategy, "text offers a tap for a strategy with no app action")
    return text


def strategy_action(strategy: dict) -> str | None:
    return "quiet" if strategy.get("sound") == "silence" else ACTIONS.get(strategy.get("kind"))


def strategy_body(catalog: dict, strategy: dict) -> dict:
    """The strategy as the client gets it, with its app action: play (and the sounds.js id), breathe (and the
    pattern), quiet, or none."""
    action = strategy_action(strategy)
    body = {"id": strategy["id"], "label": strategy["label"], "kind": strategy["kind"], "sound": strategy.get("sound"),
            "favorite_sound": strategy.get("favorite"), "action": action}
    if action == "play":
        body["play"] = strategy.get("favorite")
    if action == "breathe":
        pattern = next((b for b in catalog.get("breathing", []) if b["id"] == strategy.get("breathing")), None)
        if pattern:
            body["pattern"] = {k: pattern[k] for k in ("id", "label", "inhale_s", "hold_s", "exhale_s", "rest_s", "cycles")}
    return body
GUIDANCE_TEMPERATURE = 0.9


def check_guidance_text(text: Any, profile: dict) -> str:
    line = clean_line(text, 20 if profile.get("guidance_style") == "short" else 30)
    if len(re.findall(r"[.!?](?:\s|$)", line)) > 2 or "\n" in str(text):
        raise ValueError("more than two sentences")
    low = line.lower()
    if any(a.strip().lower() in low for a in profile.get("avoid", []) if a.strip()):
        raise ValueError("guidance mentions something the person avoids")
    if profile.get("questions") == "minimal" and "?" in line:
        raise ValueError("questions are minimal for this person")
    return line


def gemini_guidance(catalog: dict, profile: dict, allowed: list[dict], *, api_key: str, model: str,
                    timeout_s: float, recent: list[str] | None = None) -> tuple[dict, str]:
    """Gemini picks one allowed id (schema enum) and phrases it; validated, raises on anything off-format."""
    by_id = {s["id"]: s for s in allowed}
    helps = {s["id"] for s in (find_strategy(catalog, h) for h in profile.get("helps", [])) if s is not None}
    labels = {s["id"]: s["label"] for s in catalog["strategies"]}
    user = {"preferred_name": profile["preferred_name"], "guidance_style": profile["guidance_style"],
            "questions": profile["questions"], "helps": profile["helps"][:12], "avoid": profile["avoid"][:12],
            "favorite_sound": profile.get("favorite_sound"), "preferred_mode": profile.get("preferred_mode"),
            "recently_suggested": [labels.get(r, r) for r in (recent or [])],
            "recent_feedback": [h["text"] for h in profile["history"][-5:]],
            "allowed": [{"id": s["id"], "label": s["label"], "from_helps": s["id"] in helps,
                         "app_action": strategy_action(s), "example": strategy_text(s, profile)} for s in allowed]}
    schema = {"type": "OBJECT", "properties": {"strategy": {"type": "STRING", "enum": list(by_id)},
                                               "text": {"type": "STRING"}}, "required": ["strategy", "text"]}
    picked = gemini_json(api_key, model, GUIDANCE_SYSTEM, user, schema, timeout_s=timeout_s, max_tokens=120,
                         temperature=GUIDANCE_TEMPERATURE)
    if not isinstance(picked, dict) or picked.get("strategy") not in by_id:
        raise ValueError("strategy outside the allowed list")
    strategy = by_id[picked["strategy"]]
    return strategy, check_claims(check_guidance_text(picked.get("text"), profile), strategy)


def failure_reason(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, URLError):  # HTTPError is a URLError too
        return "timeout" if isinstance(exc.reason, TimeoutError) else "error"
    if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError):
        return "invalid"
    return "error"


def make_guidance(catalog: dict, profile: dict, *, env: dict | None = None, planner=gemini_guidance,
                  recent: list[str] | None = None) -> dict:
    """The /comfort/guidance body (without audio): Gemini when a key exists and it answers in time, else the next
    allowed strategy in rotation (skipping ``recent``) with catalog text. ``source`` and ``reason`` say which."""
    env = os.environ if env is None else env
    recent = list(recent or [])
    allowed = allowed_strategies(catalog, profile, recent)
    api_key = env.get("GEMINI_API_KEY") or ""
    model = env.get("GEMINI_MODEL") or DEFAULT_MODEL
    try:
        timeout_s = max(0.2, float(env.get("GEMINI_TIMEOUT_S", DEFAULT_TIMEOUT_S)))
    except ValueError:
        timeout_s = DEFAULT_TIMEOUT_S
    t0 = time.perf_counter()
    source, reason, choice = "default", None, None
    if not api_key:
        reason = "no_key"
    else:
        try:
            choice = planner(catalog, profile, allowed, api_key=api_key, model=model, timeout_s=timeout_s,
                             recent=recent)
            source = "gemini"
        except WordingError as exc:  # Gemini's pick, the catalog's honest words; labelled as the default text
            reason = "invalid"
            choice = (exc.strategy, strategy_text(exc.strategy, profile))
            log.warning("Comfort guidance used the catalog text for %s: %s", exc.strategy["id"], exc)
        except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
            reason = failure_reason(exc)
            log.warning("Comfort guidance fell back to default (%s): %s: %s", reason, type(exc).__name__,
                        _redact(str(exc)[:200], api_key))
    if choice is None:
        pick = rotate_pick(catalog, profile, recent)
        choice = (pick, strategy_text(pick, profile))
    strategy, text = choice
    return {"profile_id": profile["id"], "source": source, "reason": reason,
            "model": model if source == "gemini" else None, "latency_ms": int((time.perf_counter() - t0) * 1000),
            "strategy": strategy_body(catalog, strategy),
            "text": text, "ask": f"Remember that {strategy['label'].lower()} wasn't helpful?",
            # PR #402 names, kept for older clients
            "suggestion": strategy["id"], "planner": "gemini" if source == "gemini" else "local"}


# ---------------------------------------------------------------- the SOS check-in plan (catalog-driven)

def load_catalog(path: Path = CATALOG_PATH) -> dict:
    """The catalog, checked once: every default must name ids that exist and every trigger must have lines."""
    catalog = json.loads(path.read_text())
    ids = {key: {item["id"] for item in catalog[key]} for key in ("sounds", "breathing", "messages")}
    for trigger in TRIGGERS:
        default = catalog["defaults"][trigger]
        if default["sound"] not in ids["sounds"] or default["breathing"] not in ids["breathing"] \
                or default["message"] not in ids["messages"]:
            raise ValueError(f"comfort catalog: default for {trigger} names an unknown id")
        lines = catalog["grounding"][trigger]
        if len(lines) != 3 or not all(isinstance(line, str) and line.strip() for line in lines):
            raise ValueError(f"comfort catalog: grounding for {trigger} must be three lines")
    seen = set()
    for strategy in catalog["strategies"]:
        sid = strategy["id"]
        if sid in seen or not re.fullmatch(PROFILE_ID, sid) or not strategy.get("label"):
            raise ValueError(f"comfort catalog: strategy {sid!r} needs a unique id and a label")
        seen.add(sid)
        if strategy.get("sound") is not None and strategy["sound"] not in ids["sounds"]:
            raise ValueError(f"comfort catalog: strategy {sid} names an unknown sound")
        if strategy.get("favorite") is not None and strategy["favorite"] not in FAVORITE_SOUNDS:
            raise ValueError(f"comfort catalog: strategy {sid} names an unknown favorite sound")
        for style in STYLES:
            if "{name}" not in strategy.get(style, ""):
                raise ValueError(f"comfort catalog: strategy {sid} needs {style} text with {{name}}")
    if not any(s.get("fallback") for s in catalog["strategies"]):
        raise ValueError("comfort catalog: one strategy must be the fallback")
    return catalog


def _by_id(items: list[dict], item_id: str) -> dict:
    return next(item for item in items if item["id"] == item_id)


def build_plan(catalog: dict, trigger: str, sound: str, breathing: str, grounding: list[str], message: str) -> dict:
    """A plan from catalog ids (both the default and a validated Gemini pick go through here)."""
    sound_item = _by_id(catalog["sounds"], sound)
    breath = _by_id(catalog["breathing"], breathing)
    note = _by_id(catalog["messages"], message)
    return {
        "trigger": trigger,
        "sound": {"id": sound_item["id"], "kind": sound_item["kind"], "label": sound_item["label"], "note": sound_item.get("note", "")},
        "breathing": {k: breath[k] for k in ("id", "label", "inhale_s", "hold_s", "exhale_s", "rest_s", "cycles")},
        "grounding": [line.strip() for line in grounding],
        "message": None if note.get("text") is None else {"id": note["id"], "text": note["text"]},
    }


def default_plan(catalog: dict, trigger: str) -> dict:
    d = catalog["defaults"][trigger]
    return build_plan(catalog, trigger, d["sound"], d["breathing"], catalog["grounding"][trigger], d["message"])


def clean_line(text: Any, max_words: int) -> str:
    """One gentle line: a string, one line, 1..max_words words, nothing medical or alarming, no links."""
    line = " ".join(text.split()) if isinstance(text, str) else ""
    words = line.split(" ")
    if not line or len(words) > max_words:
        raise ValueError("line length")
    low = line.lower()
    if any(word in low for word in BLOCKED_WORDS):
        raise ValueError("blocked word")
    return line


def plan_schema(catalog: dict) -> dict:
    return {"type": "OBJECT", "properties": {
        "sound": {"type": "STRING", "enum": [s["id"] for s in catalog["sounds"]]},
        "breathing": {"type": "STRING", "enum": [b["id"] for b in catalog["breathing"]]},
        "message": {"type": "STRING", "enum": [m["id"] for m in catalog["messages"]]},
        "grounding": {"type": "ARRAY", "items": {"type": "STRING"}, "minItems": 3, "maxItems": 3},
    }, "required": ["sound", "breathing", "message", "grounding"]}


PLAN_SYSTEM = (
    "You are the Comfort Companion inside LockedIn, a hearing app for people who get sensory overload (many are "
    "autistic). The room's sound has just been cut off at the person's request, and they tapped one button saying "
    "what is too much. Build a short plan from the catalog only: pick one sound id, one breathing id, one message id "
    "(or 'none'), and write exactly three grounding lines of at most 14 plain, gentle words each, present tense, no "
    "commands shouted, no exclamation marks, no emojis. Respect the person's saved dislikes and preferences. Never "
    "give medical advice, never name a diagnosis, never promise relief, never mention emergencies or services, never "
    "claim to know how they feel. If unsure, keep it quiet and simple. Return only the JSON."
)


def preferences_for(profile: dict | None) -> dict:
    """The only profile fields Gemini sees: bounded, user-owned preference flags and short strings."""
    if not profile:
        return {}
    return {"preferred_mode": profile.get("preferred_mode"),
            "breathing_helps": bool(profile.get("breathing_helps")),
            "movement_helps": bool(profile.get("movement_helps")),
            "dislikes": [str(x)[:40] for x in profile.get("dislikes", [])[:10]],
            "worked_before": [str(x)[:40] for x in profile.get("worked", [])[:10]],
            "helps": [str(x)[:40] for x in profile.get("helps", [])[:12]],
            "avoid": [str(x)[:40] for x in profile.get("avoid", [])[:12]]}


def gemini_plan(catalog: dict, trigger: str, *, sound_now: str | None, profile: dict | None, api_key: str,
                model: str, timeout_s: float) -> dict:
    """Gemini's pick, validated against the catalog; raises ValueError when anything is off-format."""
    user = {"what_is_too_much": trigger, "sound_playing_now": sound_now, "preferences": preferences_for(profile),
            "catalog": {"sounds": [{"id": s["id"], "label": s["label"]} for s in catalog["sounds"]],
                        "breathing": [{"id": b["id"], "label": b["label"]} for b in catalog["breathing"]],
                        "messages": [{"id": m["id"], "text": m["text"]} for m in catalog["messages"]],
                        "example_grounding": catalog["grounding"][trigger]}}
    picked = gemini_json(api_key, model, PLAN_SYSTEM, user, plan_schema(catalog), timeout_s=timeout_s)
    picked = picked if isinstance(picked, dict) else {}
    ids = {key: {item["id"] for item in catalog[key]} for key in ("sounds", "breathing", "messages")}
    sound, breathing, message = picked.get("sound"), picked.get("breathing"), picked.get("message")
    if sound not in ids["sounds"] or breathing not in ids["breathing"] or message not in ids["messages"]:
        raise ValueError("plan names an unknown id")
    lines = picked.get("grounding")
    if not isinstance(lines, list) or len(lines) != 3:
        raise ValueError("plan needs three grounding lines")
    grounding = [clean_line(line, MAX_LINE_WORDS) for line in lines]
    if profile and any(d.lower() in " ".join(grounding).lower() for d in profile.get("dislikes", []) if d):
        raise ValueError("plan mentions a dislike")
    return build_plan(catalog, trigger, sound, breathing, grounding, message)


def make_plan(catalog: dict, trigger: str, *, sound_now: str | None = None, profile: dict | None = None,
              env: dict | None = None, planner=gemini_plan) -> dict:
    """The /comfort/plan body: Gemini when enabled and reachable within the timeout, else the honest default."""
    env = os.environ if env is None else env
    enabled = str(env.get("HEARING_COMFORT", "")).strip().lower() in {"1", "true", "yes", "on"}
    api_key = env.get("GEMINI_API_KEY") or ""
    model = env.get("GEMINI_MODEL") or DEFAULT_MODEL
    try:
        timeout_s = max(0.2, float(env.get("GEMINI_TIMEOUT_S", DEFAULT_TIMEOUT_S)))
    except ValueError:
        timeout_s = DEFAULT_TIMEOUT_S
    t0 = time.perf_counter()
    source, reason, plan = "default", None, None
    if not enabled:
        reason = "disabled"
    elif not api_key:
        reason = "no_key"
    else:
        try:
            plan = planner(catalog, trigger, sound_now=sound_now, profile=profile, api_key=api_key, model=model,
                           timeout_s=timeout_s)
            source = "gemini"
        except TimeoutError:
            reason = "timeout"
        except URLError as exc:  # HTTPError is a URLError too
            reason = "timeout" if isinstance(exc.reason, TimeoutError) else "error"
        except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError):
            reason = "error"
        except ValueError:
            reason = "invalid"
        if reason:
            log.warning("Comfort plan fell back to default (%s)", reason)
    if plan is None:
        plan = default_plan(catalog, trigger)
    return {"source": source, "model": model if source == "gemini" else None, "reason": reason,
            "latency_ms": int((time.perf_counter() - t0) * 1000), "plan": plan}


def wav_base64(samples: np.ndarray) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(float_to_pcm16(np.asarray(samples, dtype=np.float32)))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def comfort_root() -> Path:
    return Path(os.environ.get("HEARING_COMFORT_DIR", Path.home() / ".local/share/LockedIn/comfort"))


def mount_comfort(app: FastAPI) -> None:
    root = comfort_root()
    store: ComfortStore | None = None
    store_lock = Lock()

    def get_store() -> ComfortStore:
        nonlocal store
        with store_lock:
            if store is None:
                store = ComfortStore(root)
            return store

    catalog = load_catalog()

    @app.get("/comfort/status")
    def status() -> dict:
        """What the client may claim: the Gemini path is live only when enabled and a key exists. Never the key."""
        gemini = comfort_enabled() and bool(os.environ.get("GEMINI_API_KEY"))
        return {"enabled": comfort_enabled(), "gemini": gemini, "model": gemini_model() if gemini else None,
                "timeout_s": gemini_timeout_s()}

    @app.get("/comfort/catalog")
    def get_catalog() -> dict:
        return {k: v for k, v in catalog.items() if not k.startswith("_")}

    @app.post("/comfort/plan")
    def plan(data: dict) -> dict:
        trigger = data.get("trigger")
        if trigger not in TRIGGERS:
            raise HTTPException(400, f"trigger must be one of {', '.join(TRIGGERS)}")
        sound_now = data.get("sound_now")
        if sound_now is not None and (not isinstance(sound_now, str) or sound_now not in {"noise", "music"}):
            raise HTTPException(400, "sound_now must be noise or music")
        profile = None
        profile_id = data.get("profile_id")
        if profile_id is not None:
            if not isinstance(profile_id, str) or not re.fullmatch(r"[a-z0-9-]{1,48}", profile_id):
                raise HTTPException(400, "Invalid profile ID")
            profile = get_store().get(profile_id)
        return make_plan(catalog, trigger, sound_now=sound_now, profile=profile)

    @app.get("/comfort/profiles")
    def profiles() -> list[dict]:
        return get_store().all()

    @app.put("/comfort/profiles/{profile_id}")
    def save_profile(profile_id: str, data: dict) -> dict:
        data.pop("history", None)  # history is written only by confirmed feedback
        return get_store().put(profile_id, data)

    @app.delete("/comfort/profiles/{profile_id}/{field}/{entry}")
    def remove_entry(profile_id: str, field: str, entry: str) -> dict:
        """Undo one learned or stated item, e.g. DELETE /comfort/profiles/alex-demo/avoid/breathing."""
        return get_store().remove_entry(profile_id, field, entry)

    @app.post("/comfort/feedback")
    def save_feedback(data: dict) -> dict:
        """Saves only when confirm is true. {profile_id, strategy (or the PR #402 name suggestion), helpful, confirm}."""
        if not isinstance(data.get("helpful"), bool) or not isinstance(data.get("confirm"), bool):
            raise HTTPException(422, "helpful and confirm must be booleans")
        strategy = data.get("strategy", data.get("suggestion"))
        if not isinstance(strategy, str):
            raise HTTPException(422, "strategy is required")
        return get_store().feedback(catalog, data.get("profile_id"), strategy, data["helpful"], data["confirm"])

    @app.post("/comfort/guidance")
    def speak(data: dict) -> dict:
        """One strategy for the profile, never one it avoids, not one of ``recent`` (the last three suggested this
        SOS session) while others remain. Speech only when speak is true and a key exists."""
        if not isinstance(data.get("speak", False), bool):
            raise HTTPException(422, "speak must be a boolean")
        recent = clean_recent(data.get("recent"))
        profile = get_store().get(data.get("profile_id"))
        result = {**make_guidance(catalog, profile, recent=recent), "audio_base64": None, "engine": None,
                  "error": None}
        if data.get("speak") is not True:
            return result
        if not os.environ.get("ELEVENLABS_API_KEY"):
            return {**result, "error": "ElevenLabs key is not configured"}
        api_key = os.environ["ELEVENLABS_API_KEY"]
        voice_id = os.environ.get("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE_ID
        model_id = os.environ.get("ELEVENLABS_MODEL_ID") or DEFAULT_TTS_MODEL
        t0 = time.perf_counter()
        try:
            samples = ElevenLabsTTS(model_id=model_id).synthesize(result["text"], voice_id)
            if samples.size == 0:
                raise TtsError("no audio was returned")
            result.update(audio_base64=wav_base64(samples), engine="elevenlabs")
        except (TtsError, OSError, ValueError) as exc:
            log.warning("comfort: ElevenLabs speech failed (voice %s, model %s): %s", voice_id, model_id,
                        _redact(str(exc), api_key)[:300])
            result["error"] = "ElevenLabs unavailable"
        result["tts_ms"] = int((time.perf_counter() - t0) * 1000)
        return result
