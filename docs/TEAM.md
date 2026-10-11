# Team

VoiceLock was built at HackRU Fall 2026 by Aryan Patel, Joseph Roche and Ahmad Abbadi. Each of us writes our own section.

## Aryan Patel: what I worked on

- **Voice engine owner.** The Python service in `hearing/`: streaming noise suppression, target-speaker extraction from saved voice profiles, the identity gate and the fusion mixer, and the WebSocket session server that the app and the stage page talk to.
- **Orchestrating the build.** I ran the build from laptop-a: I planned the milestones and directed the AI coding agents (Claude Code with our swarm-control board) that wrote and tested much of the code, and reviewed their reports, numbers and listening results.
- **Choosing the compute.** I picked the Rutgers iLab GPU for the remote path (`scripts/gpu_up.sh`); the Mac stayed the stage backend because it measured better with saved profiles.
- **Voice and listening tests.** My voice is one of the two saved profiles in the team preset, and Joseph and I did the listening checks through the demo headphones (words understood, wrong-speaker intrusions, how audible the other voice is during overlap).
- **Demo, docs and pitch.** The demo runbook (`docs/DEMO.md`), the stage card (`docs/STAGE_CARD.md`), the claims we are willing to make (`docs/CLAIMS.md`), the README and the public VoiceLock repository, plus the pitch.
