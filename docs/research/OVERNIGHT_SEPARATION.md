# T-213 overnight separation / transcript recovery

## Status: blocked on saved output handoff

### T-289 dependency refresh: 2026-10-10 04:11 UTC

The orchestrator has assigned the missing saved-output bank to **T-289**; this
supersedes waiting for an unnamed Q-570 executor. A bounded read-only inspection
used the existing `~/.ssh/cm-hearing-jdr317` ControlMaster socket (plain batch SSH
could not authenticate):

```sh
ssh -o ControlPath=~/.ssh/cm-hearing-jdr317 -o ControlMaster=no \
  -o BatchMode=yes -o ConnectTimeout=8 jdr317@ilab1.cs.rutgers.edu \
  'find /common/home/jdr317/hearing-joseph-overnight/results -maxdepth 4 -type f \( -name "*.wav" -o -iname "*bank*" \); find /common/home/jdr317/hearing-joseph-overnight/models -maxdepth 3 -type d; squeue -j 447101 -h -o "%i %T %j"; date -u'
```

At **04:11:27 UTC**, job 447101 was `RUNNING hearing-lesser-overnight`.
There were **zero matching WAV/bank files** within that results-tree inspection.
`repo/docs/research/SEPARATOR_AB_GPU.md` was also absent on the remote checkout
and in this worktree. The inspected remote model tree contained MossFormer2,
ECAPA and `Systran/faster-whisper-base.en`, but no small.en directory. Locally,
`~/.cache/whisper` was absent and the HF hub cache had only grounding-dino-tiny.
Base.en is not substituted for the frozen small.en required by this task.
No allocation, model download, GPU step, or scoring run was started. The existing
04:11 inspection is an availability snapshot, not evidence that T-289 cannot
produce a bank later. First12 and expansion, including every ASR/quality metric,
remain **not measured**. T-215's frozen Replay12 manifest, authoritative global
case ledger and serialized access window are not present in this handoff; request
them from the orchestrator before any expansion. Existing stop/case caps stand.

Two regression fixes ensure an empty bank cannot claim successful measurement
and direct per-case failure execution creates its destination before checkpointing
the error. The tests inject a deterministic scoring failure and a dummy model
directory solely to test error persistence; neither produces quality evidence.

Implemented an offline saved-output scorer, not a GPU separation launcher. No
runtime settings, profiles, weights, demo configuration, or default was changed.
No GPU rental, cancellation, microphone, speaker, external ASR API, or model
download was used. The orchestrator handoff supersedes the original executor:
job 447008 is cancelled; codex-sol-b owns job 447101. T-213 must not launch,
cancel, or duplicate that allocation.

### Handoff refresh: 2026-10-10 03:52 UTC

T-213 remains parked pending Q-570 item 1: the executor's 30-minute real-pair
MossFormer2 versus SepFormer A/B. That run must save stream0/stream1, selected
and delivered WAVs plus actual selection JSON in ignored cluster/laptop-b
results storage. The local `overnight_grunt` tree still contains only diagnostic,
inventory and ready artifacts, with no saved waveform bank. The local
`~/.cache/whisper` directory is absent; no model download was attempted. Use an
existing frozen local small.en cache if supplied; otherwise WER remains not
measured. No remote executor was contacted and no cluster availability is
inferred from this local inspection. Stop at **2026-10-10 14:00 UTC**; no scoring
run was started during this refresh. First12 and expansion remain unmeasured.

At inspection on Oct 9, the shared `eval/results/overnight_grunt/ready/ready.jsonl`
contained **12 public_overnight cases**. The diagnostic `team-baseline.json`
contained six real condition summaries, but the shared overnight result tree
contained **zero WAV outputs**. These summaries are not saved waveforms and
cannot support new transcript measurements. The checked-in transcript audit
also warns about missing media/word alignments; the later shared prepared bank
is distinct from that earlier audit. No T-213 real ASR/separation scores have
been measured. All 12 prepared public cases and the approved actual team-pair
comparison remain unmeasured by this task.

The GPU executor must supply aligned stream0/stream1, actual selected waveform,
actual delivered mixer/AGC waveform, and actual selection JSON. Do not synthesize
these from the diagnostic summary. The refreshed handoff above supersedes the
previous multi-item GPU schedule: await the 30-minute pair comparison, then
score its saved bank. Preserve the first12 review and global 300-case cap.
This scorer does not start GPU jobs or manage those global allocations.

## Input contract (evaluation only)

A local JSON bank contains `asr_model: "small.en"` and `cases`. Each case has an
opaque safe `id`, paths `mix`, `target`, `interferer`, `stream0`, `stream1`,
`selected`, `delivered`, and `selection_json`, relative to the bank directory
(or absolute). All audio must be mono float32-compatible 16 kHz with exactly
matching lengths, already aligned; unequal lengths are errors, never silently
truncated. Supply `target_text`, `interferer_text` and `reference_status` when
available. Missing text yields null WER/ceiling; automatic team text must use
`automatic_unverified`. Selection JSON must be nonempty actual selector evidence.
Include separator/ECAPA weight hashes, source revision, device and config hashes
in that evidence. The scorer preserves it and hashes it, but does not authenticate
that the evidence came from the live wrapper.

Use a frozen **local faster-whisper small.en** model directory; no factory that
can fall back to a mock is called. Model files are hashed; local-files-only load,
device from hearing.device.pick (CUDA float16, otherwise CPU int8), English, beam 5, temperature 0, no VAD, and no previous-text conditioning
are fixed. Whisper small.en already has its licence in docs/MODELS.md; this task
introduces no model. The declared bank model identity still needs executor review
against the supplied model directory. No private transcripts belong in git.

## Reproduce

From this worktree (outputs in the shared ignored results directory):

```sh
.venv/bin/python -m eval.overnight_separation \
  --bank /absolute/shared/frozen-output-bank.json \
  --model /absolute/shared/frozen-small.en \
  --output /Users/joeroche/Developer/selective-hearing/eval/results/T-213-first12 \
  --limit 12 --seconds 14400
```

Review `summary.json` and the first 12 case results before expanding into a new
output directory. Bank size and limit cannot exceed 300. The upstream executor
must account for the global case count across other tasks. `--seconds` bounds
child scoring wall time (including model loading/transcription), at most four
hours; metadata hashing/preflight precedes that timer. Each case is a serialized
subprocess with a remaining-time timeout. There are no retries; failed cases
have JSON errors and process logs. Resume reuses measured/error checkpoints only
when bank, runner, model and input artifact hashes match. Changing limits changes
the artifact fingerprint; use a fresh destination when expanding. Unexpected
native process exits are checkpointed as errors. Summary lists all unmeasured IDs.

Successful cases copy seven waveform outputs and save transcripts plus JSON.
Source/audio/selection, model and runner hashes are retained. Partial waveform
copies may remain alongside an error result; only `status: measured` is valid.

## Metrics and limits

Raw mix, evaluation-only whole-clip oracle, actual selected and actual delivered
signals are distinct rows. Zero targets receive no target SI-SDR improvement or
oracle selection. Clean target/interferer ASR word errors provide the ASR ceiling;
target and interferer comparisons include substitutions, deletions, insertions
and WER. Interferer WER is a reference comparison, not an exact count of leaked
words when the voices share vocabulary.

Leak is joint least-squares interferer projection gain in dB. Identity accuracy
and wrong-speaker seconds use 100 ms clean-stem energy attribution windows,
excluding silent target/output and rank-deficient windows; this is an evaluation
attribution proxy, not ECAPA classifier accuracy. Target-absent RMS uses 100 ms
energy masks so zero crossings are not treated as absence. False-caption word
counts are provided for wholly target-absent cases; interval-level false captions
within mixed cases remain missing (requires timestamp-aligned words). No listening
assessment, live deadline result, GPU RTF, or quality claim follows from unit tests.

## Validation

Offline generated-signal tests cover edit counts, zero targets, silence masking,
wrong-speaker attribution, exact alignment, nonfinite audio, safe identifiers and
300-case overflow. These signals are correctness fixtures, not model measurements.
Full verify results are recorded in the worker report. Resume/native-exit and
real Whisper/GPU integration still require frozen artifact execution.

Inspection fingerprints (SHA256, shared artifacts, not committed media):

- `ready/ready.jsonl`: `39022c0360383b74f2aba5d9f031f2986ac5ed84c74f9d27a3f35f733d56c050`
- `diagnostic/team-baseline.json`: `8bc57b878b9fc8e83706a7c24f1f12849b6b3719988bdbfd82f73fd705248c7c`
