"""Turn the DeepFilterNet3 ONNX export into per-hop streaming graphs with explicit state.

Upstream ships ``enc.onnx``, ``erb_dec.onnx`` and ``df_dec.onnx`` as *whole-sequence*
graphs (dynamic time axis ``S``); the Rust real-time plugin makes them streaming with
tract's pulse transform, which onnxruntime cannot do. This module rewrites the graphs so
that every piece of time context is an explicit input/output pair:

* each ``GRU`` gets its ``initial_h`` from a graph input and exposes ``Y_h`` as an output;
* each time-axis ``Pad`` in front of a causal ``Conv`` (kernel ``kt`` over time) becomes a
  ``Concat`` of a ``[1, C, kt-1, F]`` state input with the new frames, and the last ``kt-1``
  frames of that concat are returned as the next state.

The rewritten graphs accept any number of frames ``S >= 1`` per call, so feeding one 10 ms
hop at a time gives the same result as feeding the whole file. Nothing here touches
audio; see :mod:`hearing.enhance.dfn` for the stage.

Weights are pinned to a commit of ``Rikorose/DeepFilterNet`` by SHA-256 and provisioned
explicitly with :func:`provision`; processing never downloads.
"""

from __future__ import annotations

import configparser
import hashlib
import json
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hearing import models

GRAPHS = ("enc", "erb_dec", "df_dec")
REVISION = "d375b2d8309e0935d165700c91da9de862a99c31"
SURGERY_VERSION = (
    1  # bump when the rewrite changes so cached streaming graphs regenerate
)
TIME_AXIS = 2  # upstream feature layout is [batch, channel, time, freq]


@dataclass(frozen=True)
class Variant:
    key: str
    archive: str
    sha256: str

    @property
    def url(self) -> str:
        return f"https://github.com/Rikorose/DeepFilterNet/raw/{REVISION}/models/{self.archive}"


VARIANTS = {
    "dfn": Variant(
        "dfn",
        "DeepFilterNet3_onnx.tar.gz",
        "c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616",
    ),
    "dfn_ll": Variant(
        "dfn_ll",
        "DeepFilterNet3_ll_onnx.tar.gz",
        "5998e58e8ba0e09bb76986ef97b84afa065a571ef282d4a1222f341e3251cf3a",
    ),
}


@dataclass(frozen=True)
class StateSpec:
    """One recurrent state: graph input ``name``, graph output ``out`` and its shape."""

    name: str
    out: str
    shape: tuple[int, ...]


def model_dir(variant: str) -> Path:
    return models.cache_dir() / "dfn" / variant


def archive_path(variant: str) -> Path:
    return models.cache_dir() / "dfn" / VARIANTS[variant].archive


def _verify(data: bytes, expected: str, what: str) -> None:
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"{what}: checksum mismatch; re-provision the pinned weights")


def download_archive(variant: str) -> Path:
    """Explicit network adapter: fetch the pinned tarball once, verify, install atomically."""
    spec = VARIANTS[variant]
    path = archive_path(variant)
    if path.exists():
        _verify(path.read_bytes(), spec.sha256, path.name)
        return path
    import httpx

    response = httpx.get(spec.url, timeout=120, follow_redirects=True)
    response.raise_for_status()
    data = response.content
    _verify(data, spec.sha256, spec.archive)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        f.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_archive(path: Path) -> tuple[dict[str, bytes], configparser.ConfigParser]:
    """Return ``{"enc": bytes, ...}`` and the parsed ``config.ini`` from the tarball."""
    graphs: dict[str, bytes] = {}
    config = configparser.ConfigParser()
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            handle = tar.extractfile(member)
            if handle is None:
                continue
            if name == "config.ini":
                config.read_string(handle.read().decode())
            elif name.endswith(".onnx") and name[:-5] in GRAPHS:
                graphs[name[:-5]] = handle.read()
    missing = [g for g in GRAPHS if g not in graphs]
    if missing or not config.sections():
        raise ValueError(f"{path}: archive is missing {missing or ['config.ini']}")
    return graphs, config


# --- graph surgery ------------------------------------------------------------------------


def _consumers(graph, name: str) -> list:
    return [n for n in graph.node if name in n.input]


def _static_shape(value_info) -> tuple[int, ...] | None:
    dims = value_info.type.tensor_type.shape.dim
    if not dims:
        return None
    return tuple(d.dim_value if d.HasField("dim_value") else -1 for d in dims)


def _prune_dead_nodes(graph) -> None:
    """Drop nodes whose outputs nobody consumes (the old zero-state producers)."""
    outputs = {o.name for o in graph.output}
    while True:
        used = set(outputs)
        for node in graph.node:
            used.update(node.input)
        dead = [n for n in graph.node if not any(o in used for o in n.output if o)]
        if not dead:
            return
        for node in dead:
            graph.node.remove(node)


def _probe_shapes(model, names: set[str]) -> dict[str, tuple[int, ...]]:
    """Run the untouched graph once with one frame to learn the shapes of ``names``.

    The upstream export declares intermediate tensors (``e0``, ``c0``, …) with symbolic
    frequency dims, so ONNX shape inference cannot resolve them; a one-frame probe can.
    """
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    declared = {o.name for o in probe.graph.output}
    for name in sorted(names - declared):
        probe.graph.output.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        )
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        probe.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    feeds = {}
    for value in session.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in value.shape]
        feeds[value.name] = np.zeros(shape, dtype=np.float32)
    outputs = session.run(sorted(names), feeds)
    return {
        name: tuple(int(d) for d in out.shape)
        for name, out in zip(sorted(names), outputs)
    }


def to_streaming(model, prefix: str) -> tuple[object, list[StateSpec]]:
    """Rewrite one whole-sequence graph into a stateful per-call graph (see module doc)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    initializers = {i.name: i for i in graph.initializer}
    pad_sources = {n.input[0] for n in graph.node if n.op_type == "Pad"}
    source_shapes = _probe_shapes(model, pad_sources) if pad_sources else {}
    states: list[StateSpec] = []
    index = 0

    for node in list(graph.node):
        if node.op_type == "GRU":
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            hidden = int(attrs["hidden_size"])
            directions = (
                2 if attrs.get("direction", b"forward") == b"bidirectional" else 1
            )
            name, out = f"{prefix}_h{index}", f"{prefix}_h{index}_out"
            while len(node.input) < 6:
                node.input.append("")
            node.input[5] = name
            while len(node.output) < 2:
                node.output.append("")
            if not node.output[1]:
                node.output[1] = out
            else:  # keep the internal name; expose it under the state name via Identity
                graph.node.append(helper.make_node("Identity", [node.output[1]], [out]))
            shape = (directions, 1, hidden)
            graph.input.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            )
            graph.output.append(
                helper.make_tensor_value_info(out, TensorProto.FLOAT, shape)
            )
            states.append(StateSpec(name, out, shape))
            index += 1
        elif node.op_type == "Pad":
            source = node.input[0]
            convs = _consumers(graph, node.output[0])
            if len(convs) != 1 or convs[0].op_type != "Conv":
                raise ValueError(
                    f"{prefix}: Pad {node.name} is not feeding exactly one Conv"
                )
            weight = initializers.get(convs[0].input[1])
            if weight is None or len(weight.dims) != 4:
                raise ValueError(
                    f"{prefix}: Conv after {node.name} has no static 4-D weight"
                )
            context = int(weight.dims[2]) - 1  # kernel over time minus one
            source_shape = source_shapes[source]
            if len(source_shape) != 4 or source_shape[0] != 1:
                raise ValueError(
                    f"{prefix}: {source!r} is not [1, C, S, F]: {source_shape}"
                )
            shape = (1, source_shape[1], context, source_shape[3])
            name, out = f"{prefix}_ctx{index}", f"{prefix}_ctx{index}_out"
            concat = helper.make_node(
                "Concat",
                [name, source],
                [node.output[0]],
                axis=TIME_AXIS,
                name=f"{name}_concat",
            )
            starts = numpy_helper.from_array(
                np.array([-context], dtype="int64"), f"{name}_starts"
            )
            ends = numpy_helper.from_array(
                np.array([2**62], dtype="int64"), f"{name}_ends"
            )
            axes = numpy_helper.from_array(
                np.array([TIME_AXIS], dtype="int64"), f"{name}_axes"
            )
            graph.initializer.extend([starts, ends, axes])
            tail = helper.make_node(
                "Slice",
                [node.output[0], starts.name, ends.name, axes.name],
                [out],
                name=f"{name}_tail",
            )
            position = list(graph.node).index(node)
            graph.node.remove(node)
            graph.node.insert(position, tail)
            graph.node.insert(position, concat)
            graph.input.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            )
            graph.output.append(
                helper.make_tensor_value_info(out, TensorProto.FLOAT, shape)
            )
            states.append(StateSpec(name, out, shape))
            index += 1

    _prune_dead_nodes(graph)
    onnx.checker.check_model(model)
    return model, states


def _manifest_path(variant: str) -> Path:
    return model_dir(variant) / "stream.json"


def streaming_paths(variant: str) -> dict[str, Path]:
    return {g: model_dir(variant) / f"{g}.stream.onnx" for g in GRAPHS}


def provision(variant: str, download: bool = True) -> dict:
    """Make sure the streaming graphs for ``variant`` exist; return the manifest.

    The manifest holds the state specs, the parsed ``[df]``/``[deepfilternet]`` config and
    the source revision. Idempotent: a manifest with the current ``SURGERY_VERSION`` and
    archive checksum short-circuits. ``download=False`` raises if the archive is missing.
    """
    spec = VARIANTS[variant]
    manifest_file = _manifest_path(variant)
    paths = streaming_paths(variant)
    if manifest_file.exists() and all(p.exists() for p in paths.values()):
        manifest = json.loads(manifest_file.read_text())
        if (
            manifest.get("surgery") == SURGERY_VERSION
            and manifest.get("sha256") == spec.sha256
        ):
            return manifest
    archive = archive_path(variant)
    if not archive.exists():
        if not download:
            raise FileNotFoundError(
                f"{archive}: provision with hearing.enhance.dfn_graphs.provision({variant!r})"
            )
        download_archive(variant)
    else:
        _verify(archive.read_bytes(), spec.sha256, archive.name)
    import onnx

    graphs, config = read_archive(archive)
    manifest = {
        "variant": variant,
        "archive": spec.archive,
        "sha256": spec.sha256,
        "revision": REVISION,
        "surgery": SURGERY_VERSION,
        "df": dict(config["df"]),
        "deepfilternet": dict(config["deepfilternet"]),
        "model": config["train"].get("model", ""),
        "states": {},
    }
    model_dir(variant).mkdir(parents=True, exist_ok=True)
    for g in GRAPHS:
        model = onnx.load_from_string(graphs[g])
        model, states = to_streaming(model, g)
        manifest["states"][g] = [[s.name, s.out, list(s.shape)] for s in states]
        data = model.SerializeToString()
        with tempfile.NamedTemporaryFile(dir=paths[g].parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(data)
        os.replace(temporary, paths[g])
    with tempfile.NamedTemporaryFile("w", dir=manifest_file.parent, delete=False) as f:
        temporary = Path(f.name)
        json.dump(manifest, f, indent=2)
    os.replace(temporary, manifest_file)
    return manifest


def original_graphs(variant: str) -> dict[str, bytes]:
    """The untouched whole-sequence graphs (for equivalence tests)."""
    graphs, _ = read_archive(archive_path(variant))
    return graphs


def load_manifest(variant: str) -> dict:
    path = _manifest_path(variant)
    if not path.exists():
        raise FileNotFoundError(
            f"{path}: provision with hearing.enhance.dfn_graphs.provision({variant!r})"
        )
    return json.loads(path.read_text())


if __name__ == "__main__":
    import sys

    for key in sys.argv[1:] or VARIANTS:
        print(json.dumps(provision(key), indent=2))
