"""T-129: HEARING_REMOTE_VIDEO=0 keeps camera frames on the laptop (a voice profile makes the
remote's vision unnecessary and frees its CPU); default on, as before."""

from hearing.server import remote
from hearing.server.remote import RemoteClient, RemoteStage


def test_default_forwards_video(monkeypatch):
    monkeypatch.delenv(remote.ENV_VIDEO, raising=False)
    assert remote.video_enabled() is True
    stage = RemoteStage(url="ws://127.0.0.1:9/ws/session")
    assert stage.client._hello["video"] is True
    stage.client.close()


def test_env_off_says_no_video_and_drops_frames(monkeypatch):
    monkeypatch.setenv(remote.ENV_VIDEO, "0")
    assert remote.video_enabled() is False
    stage = RemoteStage(url="ws://127.0.0.1:9/ws/session")
    assert stage.client._hello["video"] is False
    assert stage.client.send_video(1, 0.0, b"\xff\xd8") is False
    assert stage.client._queue.empty()
    stage.client.close()


def test_client_without_video_never_queues_frames():
    client = RemoteClient("ws://127.0.0.1:9/ws/session", video=False)
    assert client.send_video(1, 0.0, b"\xff\xd8") is False
    assert client._queue.empty()
    client.close()
