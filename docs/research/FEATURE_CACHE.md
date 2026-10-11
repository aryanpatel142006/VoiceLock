# Offline feature cache

`eval.cache_spectrograms` pre-computes the frame-level audio features used by
separator evaluations. It scans the generated fixture root
(`$HEARING_FIXTURES_DIR`, normally `eval/fixtures/generated`) for
`<set>/<fixture>/manifest.json`, then caches every WAV stem in
`eval/shards/feature_cache/<set>/<fixture>/`. The arrays and source audio are
gitignored because they contain participant voices; only `manifest.json` is
appropriate to commit when an overnight run is performed.

```bash
.venv/bin/python -m eval.cache_spectrograms
# or, for a non-default fixture/cache location:
.venv/bin/python -m eval.cache_spectrograms \
  --fixtures /path/to/generated --out /path/to/feature_cache
```

Each compressed NPZ contains:

| Array | Shape | Meaning |
|---|---|---|
| `stft` | `(frames, 257)` | Hann-windowed magnitude STFT (`n_fft=512`) |
| `filterbank` | `(frames, 40)` | Linear-power 40-band mel filterbank energies |
| `log_mel` | `(frames, 40)` | Natural log of filterbank energies |

Features use 16 kHz audio, a 512-sample analysis window, and a 160-sample
(10 ms) hop. Frames are centred with reflected padding, so a three-second
fixture has one frame per hop plus the centred boundary frames. The output
manifest records schema/parameters, source SHA-256, cache SHA-256, sample
rate, and frame count. Re-running with unchanged source hashes reuses existing
NPZ files; changed or missing source files are rebuilt. Consumers should check
the manifest parameters and hashes before using an overnight cache.

The implementation intentionally uses NumPy and SoundFile only; it does not
download models or call a network service. `stft` is magnitude (not complex)
because separator feature consumers need real-valued tensors and phase is not
part of this cache.
