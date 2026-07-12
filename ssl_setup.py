# -*- coding: utf-8 -*-
"""このPC固有のSSL対策。yfinance を import する前に import すること。

Avast Antivirus が HTTPS 通信を傍受（SSL/TLSスキャン）しており、
Avast のルート証明書は Windows 証明書ストアにはあるが Python の certifi には
無いため、requests / curl_cffi の証明書検証が失敗する。
→ certifi + Avast 証明書を結合したCAバンドルを作り、環境変数で全ライブラリに渡す。
"""
import os
from pathlib import Path

import certifi

AVAST_PEM = Path(os.environ.get(
    "NODE_EXTRA_CA_CERTS",
    r"C:\ProgramData\Avast Software\Avast\wscert.pem",
))
BUNDLE = Path(__file__).parent / "data" / "cacert-combined.pem"


def setup() -> None:
    if not AVAST_PEM.exists():
        return  # Avast がいない環境では何もしない

    if not BUNDLE.exists() or BUNDLE.stat().st_mtime < AVAST_PEM.stat().st_mtime:
        BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        combined = Path(certifi.where()).read_bytes() + b"\n" + AVAST_PEM.read_bytes()
        BUNDLE.write_bytes(combined)

    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        os.environ[var] = str(BUNDLE)


setup()
