# Web demo

A hosted version of VoiceLock that runs without the Python audio engine. It's built from this branch (`demo-site`) and deployed to Vercel.

## What a visitor gets

- The app itself, unchanged (`web/app`, `web/console`), plus a "What is this?" intro and About in the nav.
- **Listening:** the microphone is a recorded café clip of Ahmad and Joseph talking over each other. The output is the same clip after our fine-tuned separator kept only Ahmad. It was processed earlier on a laptop, frame-aligned to the mic and played 1.5 s behind, like the real engine. Focus, Quiet captions, the room dial, the A/B switch and SOS all work.
- **Talk to Me:** live. A Vercel function runs `hearing/server/comfort.py` with Gemini and ElevenLabs. Each visitor's profile stays in their own browser (`localStorage`) and is sent with each request, so nothing is stored on the server.
- **Who's speaking:** replays a recorded run of the fine-tuned speaker recognizer on clips of Aryan, then Joseph.

`demo/shim.js` does all of this in the browser by answering the app in the server's own protocol (`docs/CONTRACTS.md`). The telemetry, health and Who's speaking messages in `demo/data/` were recorded from the real server.

## Build and deploy

```bash
.venv/bin/python deploy/build.py           # writes deploy/site/
cd deploy/site && vercel deploy --prod     # first time: vercel link, then set the env vars below
```

The recorded clips live in `deploy/media/` (`room.wav`, `voice.wav`, `who.wav`) and are kept out of git like every other voice recording. The build stops if they're missing.

Environment variables on the Vercel project:

| Name | What it does |
|---|---|
| `GEMINI_API_KEY` | Talk to Me replies. Without it, replies fall back to a gentle default and the screen says so. |
| `ELEVENLABS_API_KEY` | Talk to Me's voice and hold-to-talk transcription. |
| `GEMINI_MODEL`, `GEMINI_TIMEOUT_S` | Optional, same as the local server. |
| `DEMO_CHAT_PER_MIN` | Optional. Chat turns per visitor IP per minute (default 12). |

Anyone with the link can spend these keys, so set a monthly cap on both the Gemini and ElevenLabs accounts.

## Testing locally

`deploy/site` is a static site plus one function. Serve it with anything that maps `/who` to `who.html` and `/api/comfort` to the FastAPI `app` in `api/comfort.py`, or run `vercel dev` inside `deploy/site`.
