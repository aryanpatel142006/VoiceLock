"""Named stage factories and a scalar-only, non-evaluating pipeline grammar."""

import ast
import re
from collections.abc import Callable
from itertools import pairwise
from typing import TypeVar

from hearing.types import AudioChunk, Stage, StageContext

Factory = TypeVar("Factory", bound=Callable[..., Stage])
_factories: dict[str, Callable[..., Stage]] = {}
_NAME = r"[A-Za-z_][A-Za-z0-9_-]*"
_INTEGER = re.compile(r"[+-]?\d+")
_FLOAT = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def register(name: str) -> Callable[[Factory], Factory]:
    """Register a class/factory; duplicate names are configuration errors."""
    if not re.fullmatch(_NAME, name):
        raise ValueError(f"Invalid stage name: {name!r}")

    def decorate(factory: Factory) -> Factory:
        old = _factories.get(name)
        if old is not None:
            # `python -m hearing.server.remote` runs a stage module a second time as __main__
            # (the package imported it first): keep the package's factory, not an error.
            if getattr(factory, "__module__", None) == "__main__" and \
                    getattr(old, "__qualname__", None) == getattr(factory, "__qualname__", None):
                return factory
            raise ValueError(f"Stage already registered: {name!r}")
        _factories[name] = factory
        return factory

    return decorate


def names() -> list[str]:
    return sorted(_factories)


def get(name: str, **kwargs) -> Stage:
    """Construct fresh state for one pipeline/session."""
    try:
        factory = _factories[name]
    except KeyError:
        raise KeyError(
            f"Unknown stage {name!r}; known stages: {', '.join(names())}"
        ) from None
    return factory(**kwargs)


def _split(text: str, separator: str) -> list[str]:
    """Split only outside quotes, retaining escapes for literal_eval."""
    parts, start, quote, escaped = [], 0, None, False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == separator:
            parts.append(text[start:index].strip())
            start = index + 1
    if quote is not None:
        raise ValueError("Unterminated quoted pipeline value")
    parts.append(text[start:].strip())
    return parts


def _value(text: str) -> int | float | bool | str:
    if text.startswith(("'", '"')):
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            raise ValueError(f"Invalid quoted pipeline value: {text!r}") from None
        if isinstance(value, str):
            return value
    elif text.lower() in {"true", "false"}:
        return text.lower() == "true"
    elif _INTEGER.fullmatch(text):
        return int(text)
    elif _FLOAT.fullmatch(text):
        return float(text)
    elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_./:-]*", text):
        return text
    raise ValueError(f"Invalid pipeline value: {text!r}; quote string values")


def _parse_stage(text: str) -> tuple[str, dict]:
    match = re.fullmatch(rf"({_NAME})\s*(?:\((.*)\))?", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Invalid pipeline stage: {text!r}")
    name, arguments = match.groups()
    kwargs = {}
    if arguments is not None and arguments.strip():
        for argument in _split(arguments, ","):
            key, equals, value = argument.partition("=")
            key = key.strip()
            if not equals or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid pipeline argument: {argument!r}")
            if key in kwargs:
                raise ValueError(f"Duplicate pipeline argument: {key!r}")
            kwargs[key] = _value(value.strip())
    return name, kwargs


class Pipeline:
    """A streaming Stage composed of stages with a common external rate."""

    def __init__(self, spec: str, stages: list[Stage]):
        if not stages:
            raise ValueError("A pipeline needs at least one stage")
        for left, right in pairwise(stages):
            if left.sample_rate != right.sample_rate:
                raise ValueError(
                    f"Sample-rate mismatch: {left.name} ({left.sample_rate}) > "
                    f"{right.name} ({right.sample_rate}); stages must resample on their own edges"
                )
        self.name = spec
        self.stages = tuple(stages)
        self.sample_rate = stages[0].sample_rate

    @property
    def latency_ms(self) -> float:
        # Read live: a ``tiers`` ladder or ``remote`` stage changes its delay with what it plays.
        return sum(stage.latency_ms for stage in self.stages)

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        for stage in self.stages:
            chunk = stage.process(chunk, ctx)
        return chunk

    def reset(self) -> None:
        for stage in self.stages:
            stage.reset()


def build_pipeline(spec: str) -> Pipeline:
    """Build ``a>b(key=value)>c`` without executing expressions from config."""
    parsed = [_parse_stage(part) for part in _split(spec, ">")]
    return Pipeline(spec, [get(name, **kwargs) for name, kwargs in parsed])
