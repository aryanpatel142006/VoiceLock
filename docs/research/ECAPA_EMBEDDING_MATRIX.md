# ECAPA embedding matrix

`eval/extract_ecapa_matrix.py` prepares the overnight voice-drift experiment without
network access. The input directory is expected to contain WAV segments under
`clean/` and `corrupted/`; files are discovered recursively and each file stem is
used as its speaker label. Audio is converted to the pipeline format (mono,
16 kHz), split into complete crops, and passed through the frozen repository
ECAPA backend (`speechbrain/spkrec-ecapa-voxceleb`).

The command is intentionally local-only:

```bash
.venv/bin/python -c \
  "from hearing.tse.enrolled import download_model; download_model()"
.venv/bin/python -m eval.extract_ecapa_matrix \
  "$HEARING_FIXTURES_DIR/team/ecapa" \
  --output "$HEARING_RESULTS_DIR/ecapa-team.npz"
```

The preparation/download step is separate so the extraction command cannot
silently fetch weights overnight. The output NPZ contains `matrix_<crop>` and
`labels_<crop>` arrays for the default 1.0, 1.5, and 2.0 second crops. Matrix
entries are cosine distances (`1 - cosine_similarity`), the diagonal is exactly
zero, and labels include condition, speaker, file, and crop start time. A JSON
sidecar records sample rate, crop settings, segment count, and embedding count.
Use `--hop-s` to request overlapping crops; by default the hop equals the crop
duration.

No measurements are committed here: the matrices depend on the locally recorded
team fixtures and the frozen checkpoint installed on the overnight host.
