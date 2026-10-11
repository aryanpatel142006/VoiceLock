"""Sound device discovery and resolution for ``hearing live`` / ``hearing latency-test``.

``sounddevice`` (PortAudio) is imported lazily so the rest of the package, and the unit
tests, work on machines without an audio stack. Resolution itself is pure: pass your own
device list to test it.

Every PortAudio import goes through :func:`sounddevice`, which refuses inside pytest unless
``HEARING_HW_TESTS=1``: initialising CoreAudio from a headless process without microphone
permission aborts it (SIGABRT) and pops a crash dialog on macOS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

Kind = Literal["input", "output"]


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    inputs: int
    outputs: int
    default_samplerate: float
    hostapi: str = ""

    def supports(self, kind: Kind) -> bool:
        return (self.inputs if kind == "input" else self.outputs) > 0


def sounddevice():
    """The ``sounddevice`` module, unless we are under pytest without ``HEARING_HW_TESTS=1``."""
    if "PYTEST_CURRENT_TEST" in os.environ and os.environ.get("HEARING_HW_TESTS") != "1":
        raise RuntimeError("audio hardware is off in tests; set HEARING_HW_TESTS=1 to open devices")
    import sounddevice as sd

    return sd


def list_devices() -> list[DeviceInfo]:
    """Every PortAudio device on this machine (needs ``sounddevice``)."""
    sd = sounddevice()

    apis = {i: api["name"] for i, api in enumerate(sd.query_hostapis())}
    return [
        DeviceInfo(
            index=d["index"],
            name=d["name"],
            inputs=int(d["max_input_channels"]),
            outputs=int(d["max_output_channels"]),
            default_samplerate=float(d["default_samplerate"]),
            hostapi=apis.get(d["hostapi"], str(d["hostapi"])),
        )
        for d in sd.query_devices()
    ]


def default_device(kind: Kind) -> int | None:
    """PortAudio's default device index for ``kind`` (``None`` when there is none)."""
    sd = sounddevice()

    index = sd.default.device[0 if kind == "input" else 1]
    return None if index is None or index < 0 else int(index)


def resolve_device(
    spec: str | int | None, kind: Kind, devices: list[DeviceInfo] | None = None
) -> int | None:
    """``None`` -> system default; an index (or digit string) -> that index; otherwise a
    case-insensitive name match (exact first, then unique substring) among ``kind`` devices.
    """
    if spec is None or spec == "":
        return None
    if devices is None:
        devices = list_devices()
    by_index = {d.index: d for d in devices}
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
        index = int(spec)
        if index not in by_index:
            raise ValueError(f"Device index {index} is out of range (0-{max(by_index, default=-1)})")
        if not by_index[index].supports(kind):
            raise ValueError(f"Device {index} ({by_index[index].name!r}) has no {kind} channels")
        return index
    needle = str(spec).strip().lower()
    candidates = [d for d in devices if d.supports(kind)]
    exact = [d for d in candidates if d.name.lower() == needle]
    if len(exact) == 1:
        return exact[0].index
    partial = [d for d in candidates if needle in d.name.lower()]
    if len(partial) == 1:
        return partial[0].index
    if not partial:
        names = ", ".join(f"{d.index}: {d.name}" for d in candidates) or "none"
        raise ValueError(f"No {kind} device matches {spec!r}; {kind} devices: {names}")
    names = ", ".join(f"{d.index}: {d.name}" for d in partial)
    raise ValueError(f"{kind} device {spec!r} is ambiguous: {names}")


def format_devices(devices: list[DeviceInfo] | None = None) -> str:
    """One line per device: ``index  name  in=N out=M  default_sr  hostapi``."""
    if devices is None:
        devices = list_devices()
    width = max((len(d.name) for d in devices), default=4)
    lines = [
        f"{d.index:<3} {d.name:<{width}}  in={d.inputs} out={d.outputs}  "
        f"{int(d.default_samplerate)} Hz  {d.hostapi}".rstrip()
        for d in devices
    ]
    return "\n".join(lines) + ("\n" if lines else "")
