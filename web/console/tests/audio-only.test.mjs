import test from "node:test";
import assert from "node:assert/strict";
import { audioOnly, helloVideo } from "../audio-only.js";

const team = { preset: { pins: { audio_only: true } } };
const track = { getVideoTracks: () => [{}] };

test("the team preset's audio_only pin turns the camera off", () => {
  assert.equal(audioOnly({ telemetry: team }), true);
  assert.equal(audioOnly({ telemetry: { preset: { pins: { audio_only: false } } } }), false);
  assert.equal(audioOnly({ telemetry: { preset: null } }), false);
  assert.equal(audioOnly({}), false);
});

test("?audio=1 turns it off without the preset", () => {
  assert.equal(audioOnly({ search: "?audio=1" }), true);
  assert.equal(audioOnly({ search: "?server=x&audio=true" }), true);
  assert.equal(audioOnly({ search: "?audio=0" }), false);
  assert.equal(audioOnly({ search: "?audio=0", telemetry: team }), true); // the preset wins: no camera
});

test("hello says video:false under audio only, whatever the stream", () => {
  assert.equal(helloVideo(true, track), false);
  assert.equal(helloVideo(false, track), true);
  assert.equal(helloVideo(false, null), false);
  assert.equal(helloVideo(false, { getVideoTracks: () => [] }), false);
});
