# Console unit tests in fast verification

Run `bash scripts/verify_fast.sh` for Python lint, fast Python tests, console
unit tests, and the frontend type check when present.

When Node is on `PATH`, fast verification runs exactly:

```sh
node --test web/console/tests/*.test.mjs
```

The output includes Node's pass/fail counts; full output is saved to
`.swarm-run/node.txt`. A failing console test makes verification exit non-zero.
When Node is absent, the step prints `SKIP: no node` and verification continues.

Use the `*.test.mjs` glob rather than `web/console/tests/`: the directory also
contains browser end-to-end scripts that are outside this unit-test step.
