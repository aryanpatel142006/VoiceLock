"""T-121: the hello's mic label reaches the recording and session_report prints it."""

import json

from hearing.server.protocol import Hello, parse_message
from scripts.session_report import main


def test_hello_carries_the_device_labels():
    msg = parse_message({"type": "hello", "client": "judge", "input_device": "RODE VideoMic GO II",
                         "output_device": "USB-C Headphones"})
    assert isinstance(msg, Hello)
    assert (msg.input_device, msg.output_device) == ("RODE VideoMic GO II", "USB-C Headphones")
    assert Hello(type="hello").input_device is None  # older clients stay valid


def test_session_report_prints_the_mic(tmp_path, capsys):
    rows = [{"t": 0.1, "dir": "in", "event": {"type": "hello", "input_device": "RODE VideoMic GO II"}},
            {"t": 9.0, "dir": "in", "event": {"type": "hello", "input_device": None}}]  # Stop sends null
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    main(["session_report.py", str(tmp_path)])
    assert "mic: RODE VideoMic GO II" in capsys.readouterr().out
