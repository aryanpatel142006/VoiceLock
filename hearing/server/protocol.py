"""Wire format of ``/ws/session`` (docs/CONTRACTS.md#events).

Binary messages: ``u8 kind · u32 seq · f64 capture_ts`` little endian, then the payload
(PCM16 mono audio or a JPEG). Text messages: JSON objects with a ``type`` field.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

KIND_AUDIO_IN = 1
KIND_VIDEO_IN = 2
KIND_AUDIO_OUT = 3
KINDS = (KIND_AUDIO_IN, KIND_VIDEO_IN, KIND_AUDIO_OUT)

HEADER = struct.Struct("<BId")  # 13 bytes, no padding


class ProtocolError(ValueError):
    """A malformed client message; the session reports it and carries on."""


@dataclass(frozen=True)
class Frame:
    kind: int
    seq: int
    capture_ts: float
    payload: bytes


def pack_frame(kind: int, seq: int, capture_ts: float, payload: bytes) -> bytes:
    return HEADER.pack(kind, seq & 0xFFFFFFFF, capture_ts) + bytes(payload)


def unpack_frame(data: bytes) -> Frame:
    if len(data) < HEADER.size:
        raise ProtocolError(f"binary frame is {len(data)} bytes; the header alone is {HEADER.size}")
    kind, seq, capture_ts = HEADER.unpack_from(data)
    if kind not in KINDS:
        raise ProtocolError(f"unknown binary frame kind {kind}; expected one of {KINDS}")
    return Frame(kind, seq, capture_ts, bytes(data[HEADER.size :]))


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Hello(_Message):
    type: Literal["hello"]
    sample_rate: int = Field(16000, ge=8000, le=96000)
    frame_ms: float = Field(20.0, gt=0, le=200)
    video: bool = False
    client: str = "unknown"
    input_device: str | None = None  # the mic's label, so recordings show which was used (T-121)
    output_device: str | None = None
    devices_missing: list[str] = Field(default_factory=list, max_length=4)  # "mic"/"output" saved but not found (T-133)


class TargetLock(_Message):
    lock_track_id: int


class Attention(_Message):
    """Sticky-attention tunables; ranges live in ``hearing.vision.target.ATTENTION_RANGES``."""

    dwell_s: float | None = Field(None, ge=0.1, le=3.0)
    hold_s: float | None = Field(None, ge=0.2, le=10.0)
    switch_margin: float | None = Field(None, ge=0.0, le=0.5)
    tie_margin: float | None = Field(None, ge=0.0, le=0.5)
    hold_lock_s: float | None = Field(None, ge=1.0, le=60.0)  # explicit lock survives a lost face (T-107)
    preset: Literal["glance", "normal", "sticky"] | None = None  # logged to prefs only


class Enroll(_Message):
    """Manual voice calibration (T-085): ``start`` keeps every voiced chunk of the next ``seconds``
    as the target's enrollment; ``clear`` forgets the clips. ``profile`` (T-125) enrolls the voice
    profile ``name`` from ``$HEARING_PROFILES_DIR`` with ``other`` (default: every other profile) as
    the known negative."""

    action: Literal["start", "clear", "profile"]
    seconds: float = Field(3.0, ge=1.0, le=10.0)
    name: str | None = Field(None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    other: str | None = Field(None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class NameCall(_Message):
    """The wearer's name for ``mode: "name_call"`` (T-120): whoever says it becomes the target."""

    text: str = Field(..., min_length=1, max_length=64)
    aliases: list[str] = Field(default_factory=list, max_length=8)
    # T-123 false-trigger guards; absent = keep the session's (default "normal", no extra exclusions)
    match: Literal["strict", "normal", "loose"] | None = None
    exclude: list[str] | None = Field(default=None, max_length=16)


class Control(_Message):
    """Partial update: only the fields the client sent are applied (``model_fields_set``)."""

    type: Literal["control"]
    mode: Literal["natural", "focus_hq", "synthetic", "text", "quiet", "auto", "name_call"] | None = None
    suppression: float | None = Field(None, ge=0, le=100)
    target: TargetLock | None = None
    voice_override: str | None = None
    hq_backend: Literal["local", "remote", "auto"] | None = None  # where Focus HQ runs
    attention: Attention | None = None
    ab: Literal["raw", "processed"] | None = None  # what the listener hears (client-side A/B); echoed
    latency_preset: Literal["speed", "balanced", "accuracy"] | None = None  # Focus HQ latency/quality preset
    synth_feed: Literal["auto", "mixture", "natural", "focus"] | None = None  # what feeds Synthetic/Text ASR
    enroll: Enroll | None = None  # manual voice calibration (T-085)
    isolation: Literal["standard", "strict"] | None = None  # enrolled-tier strictness (T-085)
    profile: Literal["judge", "team"] | None = None  # locked demo defaults + auto-calibration (T-088); team = judge + config/demo-team.yaml (T-130)
    name: NameCall | None = None  # Name Call: the wearer's name and aliases (T-120)
    name_call: Literal["release"] | None = None  # Name Call: drop the current target, listen again
    agc: bool | None = None  # judge-profile output AGC; false = off (the remote leg of the remote path, T-169)
    sos: bool | None = None  # T-462: true = silence the output, captions off (mode kept); false = back as before
    sos_captions: bool | None = None  # Comfort's Stay Quiet: true = Quiet's captions keep coming under SOS (output stays silent); SOS off resets it


def parse_message(data: dict) -> Hello | Control:
    """Validate a decoded JSON client message; raises :class:`ProtocolError`."""
    kind = data.get("type") if isinstance(data, dict) else None
    model = {"hello": Hello, "control": Control}.get(kind)
    if model is None:
        raise ProtocolError(f"unknown message type {kind!r}; expected 'hello' or 'control'")
    try:
        return model.model_validate(data)
    except ValidationError as e:
        details = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        raise ProtocolError(f"invalid {kind}: {details}") from None


def error_message(stage: str, message: str, degraded_to: str | None = None) -> dict:
    return {"type": "error", "stage": stage, "message": message, "degraded_to": degraded_to}
