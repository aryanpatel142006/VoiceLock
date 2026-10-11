"""Fusion stages: combine the visual target decision with the audio path.

Importing this package registers ``gate``, ``mixer`` and ``split``. ``tiers`` belongs
to another task and registers itself when its module lands here.
"""

from hearing.fusion import (
    gate,  # registers "gate"
    mixer,  # registers "mixer" and "split"
)

__all__ = ["gate", "mixer"]
