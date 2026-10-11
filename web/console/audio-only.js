// Audio-only judge page (T-168): the decided demo has no camera. Under the team preset (telemetry
// preset.pins.audio_only, config/demo-team.yaml) or with ?audio=1 the page never opens the camera,
// says video:false in hello and shows no face UI. The console page keeps its camera path.

/** True when the page must not capture video: `?audio=1` in the URL, or the team preset says audio only. */
export function audioOnly({ search = "", telemetry = null } = {}) {
  const flag = new URLSearchParams(search).get("audio");
  if (flag === "1" || flag === "true") return true;
  return Boolean(telemetry?.preset?.pins?.audio_only);
}

/** hello.video: a live camera track, and never under audio only. */
export function helloVideo(audioOnlyOn, stream) {
  return !audioOnlyOn && Boolean(stream?.getVideoTracks?.().length);
}
