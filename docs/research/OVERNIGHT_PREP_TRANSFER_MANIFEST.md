# Overnight preparation: public-input transfer manifests

`eval/prep_transfer_manifest.py` prepares a **plan only**. It never opens SSH, starts a
GPU job, copies media, or evaluates a shell command. A catalog is a local JSON file next
to its prepared public files:

```json
{
  "dataset": "public-demo",
  "revision": "rev-7",
  "license": {"spdx": "CC-BY-4.0", "url": "https://example.test/license"},
  "source_group": "speaker-a",
  "files": [
    {"path": "target.wav", "source_id": "utterance-1",
     "speaker_id": "speaker-a", "utterance_id": "utt-1",
     "interval_s": [0.0, 2.5], "sha256": "<optional declared digest>"}
  ]
}
```

Run from the repository root:

```bash
.venv/bin/python -m eval.prep_transfer_manifest \
  --output /tmp/t243-shard-0.json \
  --max-bytes 2147483648 \
  --shard-index 0 --shard-count 1 \
  /local/public/catalog.json
```

The output is portable JSON (`selective-hearing.prep-transfer/v1`). It contains verified
SHA-256 and byte counts, licence evidence, dataset revision, source group, stable source
IDs, and original speaker/utterance/interval references. Absolute local paths and private
metadata are deliberately omitted. Files are deduplicated by original bytes and sorted by
digest; shard assignment uses the digest, so repeated runs produce the same plan.

`available` counts all catalog entries successfully verified, while `planned` counts only
unique files selected for this shard and byte cap. `excluded` explains `other_shard`,
`byte_cap`, and `resume_complete` decisions. A prior output can be used as a deterministic
resume checkpoint:

```bash
.venv/bin/python -m eval.prep_transfer_manifest \
  --resume /tmp/t243-shard-0.json \
  --output /tmp/t243-shard-0-next.json \
  /local/public/catalog.json
```

The `transfer.commands` values are JSON argument arrays, not shell snippets. They use the
literal substitution tokens `CATALOG_ROOT` and `GPU_DEST`; Sol's runner must replace those
tokens in argv directly, validate the destination shard, and invoke `rsync` without a shell.
The manifest's hard maximum is 2 GiB per preparation batch. No transfer is performed by
this preparation task.
