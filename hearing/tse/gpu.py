"""One lock for torch GPU work across threads.

PyTorch's MPS backend shares one command queue per process and is not thread-safe: two threads
running models at once abort in Metal (``A command encoder is already encoding to this command
buffer``, seen Oct 5 2026 with Focus HQ, Light-ASD and the voice estimator in one session).
Every thread that runs a torch model in the session holds :data:`TORCH_LOCK` around the call:
the ``tse`` workers (Focus HQ, Enrolled), the vision worker's ASD and the voice estimator. On
one GPU the work is serial anyway, so the lock costs waiting, not throughput.
"""

from __future__ import annotations

import threading

TORCH_LOCK = threading.RLock()
