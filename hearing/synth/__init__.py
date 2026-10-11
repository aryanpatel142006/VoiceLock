"""Synthetic / Text mode: streaming ASR, TTS adapters and the ``synth`` stage.

Importing the package registers the ``synth`` stage (the server imports ``hearing.synth``
to discover stages).
"""

from hearing.synth import pipeline as _pipeline  # noqa: F401  (registers "synth")
