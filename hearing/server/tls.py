"""HTTPS for phones on the venue Wi-Fi: a cached self-signed certificate, the URLs to open, a QR code.

``getUserMedia`` (mic and camera) needs a secure context, so a phone can only use the app from
``https://<laptop-ip>:<port>``; plain http works on ``localhost`` only. ``hearing serve --tls``
calls :func:`ensure_cert` for every LAN IPv4 of the laptop plus ``hearing.local``/``localhost``
and hands the files to uvicorn (``ssl_certfile``/``ssl_keyfile``), which then serves https and wss.

The certificate is its own CA (``basicConstraints CA:TRUE`` + ``serverAuth``), RSA 2048, SHA-256,
397 days: the shape iOS accepts once the profile is installed and trusted, and Android once it is
added under "CA certificate". It is cached in ``~/.hearing/tls`` (``$HEARING_TLS_DIR``) and only
regenerated when a host is missing from it or it expires within 30 days, so a phone trusts it once.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import os
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

HOSTNAME = "hearing.local"
VALID_DAYS = 397  # under Apple's 398-day limit for TLS server certificates
RENEW_DAYS = 30


def tls_dir() -> Path:
    return Path(os.environ.get("HEARING_TLS_DIR") or Path.home() / ".hearing" / "tls")


def lan_ips() -> list[str]:
    """IPv4 addresses a phone on the same network can reach (no loopback, no link-local)."""
    found: set[str] = set()
    try:  # the interface the default route uses; connect() on UDP sends nothing
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            found.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    try:  # macOS/Linux: every configured interface (Wi-Fi and a phone hotspot or USB tether)
        out = subprocess.run(["ifconfig"], capture_output=True, text=True, timeout=2, check=False).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "inet":
                found.add(parts[1].removeprefix("addr:"))
    except (OSError, subprocess.SubprocessError):
        pass
    keep = []
    for ip in found:
        try:
            addr = ipaddress.IPv4Address(ip)
        except ValueError:
            continue
        if not (addr.is_loopback or addr.is_link_local or addr.is_unspecified):
            keep.append(ip)
    return sorted(keep, key=lambda ip: (not ipaddress.IPv4Address(ip).is_private, ip))


def mdns_name() -> str | None:
    """The name macOS answers on mDNS (``<host>.local``); ``hearing.local`` is only advertised if you do."""
    name = socket.gethostname()
    if not name or name == "localhost":
        return None
    return name if name.endswith(".local") else f"{name.split('.')[0]}.local"


def cert_hosts(extra: list[str] | None = None) -> list[str]:
    """Names and IPs the certificate must cover: LAN IPs, the mDNS name, ``hearing.local``, localhost."""
    hosts = [*lan_ips(), mdns_name() or "", HOSTNAME, "localhost", "127.0.0.1", *(extra or [])]
    return list(dict.fromkeys(h for h in hosts if h and h != "0.0.0.0"))


def _covers(cert_path: Path, hosts: list[str]) -> bool:
    from cryptography import x509

    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except (OSError, ValueError, x509.ExtensionNotFound):
        return False
    have = {str(v) for v in san.get_values_for_type(x509.IPAddress)}
    have |= set(san.get_values_for_type(x509.DNSName))
    left = cert.not_valid_after_utc - dt.datetime.now(dt.UTC)
    return set(hosts) <= have and left > dt.timedelta(days=RENEW_DAYS)


def ensure_cert(hosts: list[str], directory: Path | None = None) -> tuple[Path, Path]:
    """``(cert.pem, key.pem)`` covering every host, reusing the cached pair when it still does.

    A regenerated certificate keeps the old hosts too, so moving between networks (venue Wi-Fi,
    phone hotspot) does not invalidate a phone that already trusts it more than once.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    directory = Path(directory) if directory else tls_dir()
    cert_path, key_path = directory / "cert.pem", directory / "key.pem"
    if cert_path.is_file() and key_path.is_file() and _covers(cert_path, hosts):
        return cert_path, key_path
    hosts = list(dict.fromkeys([*hosts, *_old_hosts(cert_path)]))
    directory.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, f"{HOSTNAME} (selective hearing, self-signed)"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "hearing dev"),
    ])
    sans: list[x509.GeneralName] = []
    for h in hosts:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            sans.append(x509.DNSName(h))
    now = dt.datetime.now(dt.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=VALID_DAYS))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(
            digital_signature=True, key_encipherment=True, key_cert_sign=True, content_commitment=False,
            data_encipherment=False, key_agreement=False, crl_sign=False, encipher_only=False,
            decipher_only=False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    key_path.chmod(0o600)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


def _old_hosts(cert_path: Path) -> list[str]:
    from cryptography import x509

    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except (OSError, ValueError, x509.ExtensionNotFound):
        return []
    return [str(v) for v in san.get_values_for_type(x509.IPAddress)] + san.get_values_for_type(x509.DNSName)


@dataclass(frozen=True)
class ConnectInfo:
    """What a phone opens: ``urls`` best first (LAN IPs, then the laptop's mDNS name), ``tls`` on/off."""

    urls: list[str]
    tls: bool
    page: str = "/simple"

    @property
    def primary(self) -> str:
        return self.urls[0]


def connect_info(host: str, port: int, tls: bool, page: str = "/simple") -> ConnectInfo:
    scheme = "https" if tls else "http"
    if host in ("0.0.0.0", "", "::"):
        names = [*lan_ips(), *filter(None, [mdns_name()])]
    elif host in ("127.0.0.1", "localhost", "::1"):
        names = ["localhost"]
    else:
        names = [host]
    names = names or ["localhost"]
    return ConnectInfo([f"{scheme}://{n}:{port}{page}" for n in names], tls, page)


def qr_ascii(text: str) -> str:
    """The QR code as terminal text (two rows per line, half blocks); readable by phone cameras."""
    import qrcode

    qr = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_L)
    qr.add_data(text)
    qr.make(fit=True)
    m = qr.get_matrix()
    if len(m) % 2:
        m.append([False] * len(m[0]))
    glyph = {(False, False): " ", (True, False): "▀", (False, True): "▄", (True, True): "█"}
    # Light modules are the blocks (like ``qrencode -t UTF8``): right on the usual dark terminal.
    return "\n".join(
        "".join(glyph[(not top, not bot)] for top, bot in zip(m[r], m[r + 1], strict=True))
        for r in range(0, len(m), 2)
    )


def qr_svg(text: str) -> str:
    """The QR code as an inline ``<svg>`` element for the ``/connect`` page."""
    import qrcode
    import qrcode.image.svg

    img = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage, border=2)
    svg = img.to_string(encoding="unicode")
    return svg[svg.index("<svg"):]


def banner(info: ConnectInfo, cert_path: Path | None = None) -> str:
    """The text ``hearing serve`` prints at start: URLs, the QR of the first one, the trust step."""
    lines = ["", "Phones on this network: open (or scan)"]
    lines += [f"  {u}" for u in info.urls]
    lines += ["", qr_ascii(info.primary), ""]
    if info.tls:
        lines.append(f"Certificate: {cert_path}  (one-time trust step: docs/DEMO.md#phones;"
                     " the /connect page links it as /hearing-ca.crt)")
    else:
        lines.append("No --tls: a phone cannot open the mic over plain http (only localhost can).")
    return "\n".join(lines) + "\n"
