"""``hearing serve --tls``: self-signed certificate, https + wss, QR banner, ``/connect``, ``/sdk``."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import ssl
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
import websockets
from cryptography import x509
from fastapi.testclient import TestClient

from hearing.cli import _tls_files
from hearing.server import tls
from hearing.server.app import create_app
from hearing.server.session import ServerConfig, default_console_dir


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARING_DEVICE", "cpu")
    monkeypatch.setenv("HEARING_FACES", "none")
    monkeypatch.setenv("HEARING_TLS_DIR", str(tmp_path / "tls"))


@pytest.fixture(scope="module")
def cert_pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("tls")
    return tls.ensure_cert(["192.168.1.20", "hearing.local", "localhost", "127.0.0.1"], d)


def _sans(path: Path) -> tuple[x509.Certificate, set[str]]:
    cert = x509.load_pem_x509_certificate(path.read_bytes())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    return cert, {str(v) for v in san.get_values_for_type(x509.IPAddress)} | set(
        san.get_values_for_type(x509.DNSName))


def test_cert_covers_lan_ip_and_hearing_local_and_is_a_ca(cert_pair):
    cert_path, key_path = cert_pair
    cert, names = _sans(cert_path)
    assert {"192.168.1.20", "hearing.local", "localhost", "127.0.0.1"} <= names
    assert cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca is True
    days = (cert.not_valid_after_utc - cert.not_valid_before_utc).days
    assert days <= 398  # Apple's limit for TLS server certificates
    assert cert.public_key().key_size == 2048
    assert key_path.stat().st_mode & 0o077 == 0


def test_cert_is_cached_and_regenerated_only_for_a_new_host(cert_pair, tmp_path):
    d = tmp_path / "c"
    first, _ = tls.ensure_cert(["10.0.0.5", "hearing.local"], d)
    pem = first.read_bytes()
    tls.ensure_cert(["10.0.0.5"], d)
    assert first.read_bytes() == pem  # cached: a phone trusts it once
    tls.ensure_cert(["172.20.10.2"], d)  # moved to a phone hotspot
    _, names = _sans(first)
    assert first.read_bytes() != pem
    assert {"10.0.0.5", "172.20.10.2", "hearing.local"} <= names  # the old hosts stay


def test_lan_ips_are_reachable_ipv4_only():
    for ip in tls.lan_ips():
        addr = ipaddress.IPv4Address(ip)
        assert not addr.is_loopback and not addr.is_link_local


def test_connect_info_and_banner(monkeypatch):
    monkeypatch.setattr(tls, "lan_ips", lambda: ["192.168.1.20"])
    monkeypatch.setattr(tls.socket, "gethostname", lambda: "Aryans-MacBook-Pro.local")
    info = tls.connect_info("0.0.0.0", 8443, True)
    assert info.urls == ["https://192.168.1.20:8443/simple", "https://Aryans-MacBook-Pro.local:8443/simple"]
    assert {"192.168.1.20", "Aryans-MacBook-Pro.local", "hearing.local"} <= set(tls.cert_hosts())
    assert tls.connect_info("127.0.0.1", 8000, False).urls == ["http://localhost:8000/simple"]
    text = tls.banner(info, Path("/x/cert.pem"))
    assert "https://192.168.1.20:8443/simple" in text and "/x/cert.pem" in text
    qr = tls.qr_ascii(info.primary)
    rows = qr.splitlines()
    assert len(rows) >= 12 and len({len(r) for r in rows}) == 1  # a square of half-block rows
    assert set(qr) <= set(" ▀▄█\n")


def test_tls_files_flags(tmp_path, cert_pair):
    cert, key = cert_pair
    ns = argparse.Namespace(tls=False, cert=None, key=None, host="127.0.0.1")
    assert _tls_files(ns) == {}
    ns.cert = cert
    assert _tls_files(ns) is None  # --cert without --key
    ns.key = key
    assert _tls_files(ns) == {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    out = _tls_files(argparse.Namespace(tls=True, cert=None, key=None, host="0.0.0.0"))
    assert Path(out["ssl_certfile"]).parent == tmp_path / "tls"


def test_connect_page_cert_download_sdk_and_console_alias(cert_pair):
    app = create_app(ServerConfig(console_dir=default_console_dir()))
    client = TestClient(app)
    assert client.get("/hearing-ca.crt").status_code == 404  # not serving TLS
    page = client.get("/connect").text
    assert "<svg" in page and "http://testserver/simple" in page and "plain http" in page
    app.state.connect = tls.ConnectInfo(["https://192.168.1.20:8443/simple"], True)
    app.state.cert_path = str(cert_pair[0])
    page = client.get("/connect").text
    assert "https://192.168.1.20:8443/simple" in page and "Certificate Trust Settings" in page
    crt = client.get("/hearing-ca.crt")
    assert crt.status_code == 200 and crt.text.startswith("-----BEGIN CERTIFICATE-----")
    assert crt.headers["content-type"].startswith("application/x-x509-ca-cert")
    sdk = client.get("/sdk/hearing-client.js")
    assert sdk.status_code == 200 and 'from "../console/protocol.js"' in sdk.text
    assert "export function sessionUrl" in client.get("/console/protocol.js").text
    assert "<title>" in client.get("/").text  # the console still owns /


def test_https_and_wss_on_an_ephemeral_port(cert_pair):
    cert, key = cert_pair
    app = create_app(ServerConfig(console_dir=default_console_dir()))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning",
                                           ssl_certfile=str(cert), ssl_keyfile=str(key)))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started
        port = server.servers[0].sockets[0].getsockname()[1]
        ctx = ssl.create_default_context(cafile=str(cert))  # the self-signed cert is its own root
        r = httpx.get(f"https://127.0.0.1:{port}/health", verify=ctx)
        assert r.json()["ok"] is True
        with pytest.raises(httpx.ConnectError):  # untrusted without the cert: why phones need the step
            httpx.get(f"https://127.0.0.1:{port}/health")

        async def session() -> dict:
            async with websockets.connect(f"wss://localhost:{port}/ws/session", ssl=ctx) as ws:
                await ws.send(json.dumps({"type": "hello", "sample_rate": 48000, "frame_ms": 20,
                                          "video": False, "client": "test"}))
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), 5))
                    if msg["type"] == "telemetry":
                        return msg

        assert "latency_ms" in asyncio.run(session())
    finally:
        server.should_exit = True
        thread.join(5)
