"""
preprocess.py — Argus Sentinel v4 (binary: clean / attack)

Chuẩn hoá một HTTP request thành chuỗi đầu vào của model.
File này là "hợp đồng" giữa train và inference: WAF (Python/Rust/Go) phải
tái hiện ĐÚNG các bước dưới đây, nếu không phân phối input sẽ lệch.

Định dạng:
    "{METHOD} {uri} [UA] {ua} [REF] {referer} [CK] {cookie} [BODY] {body}"

Các bước cho từng field:
    1. cắt thô  raw[: cap * 4]                (chặn chi phí CPU với payload lớn)
    2. URL-decode (unquote_plus, UTF-8 lossy), tối đa 2 vòng (chống double-encoding)
    3. lowercase
    4. chuỗi hex >= 16 ký tự   -> "HEX"      (session id, hash, token)
       chuỗi [a-z0-9] >= 32    -> "TOK"
    5. gộp whitespace, strip
    6. cắt [: cap]
Marker/placeholder viết HOA; nội dung đã lowercase nên attacker không giả mạo được.
"""
from __future__ import annotations

import re
from urllib.parse import unquote_plus, urlsplit

FIELD_CAPS = {"uri": 512, "ua": 160, "referer": 160, "cookie": 160, "body": 512}
MARKERS = ["[UA]", "[REF]", "[CK]", "[BODY]"]

_HEX = re.compile(r"[0-9a-f]{16,}")
_TOK = re.compile(r"[a-z0-9]{32,}")
_WS = re.compile(r"\s+")
_METHOD = re.compile(r"^[A-Z]{1,10}$")


def _url_decode(s: str, rounds: int = 2) -> str:
    for _ in range(rounds):
        d = unquote_plus(s, encoding="utf-8", errors="replace")
        if d == s:
            break
        s = d
    return s


def normalize_field(s: str | None, cap: int) -> str:
    if not s:
        return ""
    s = _url_decode(s[: cap * 4]).lower()
    s = _HEX.sub("HEX", s)
    s = _TOK.sub("TOK", s)
    s = _WS.sub(" ", s).strip()
    return s[:cap]


def strip_origin(uri: str) -> str:
    """'http://host:8080/a/b?x=1' -> '/a/b?x=1' (request line dạng absolute-form)."""
    if uri.startswith(("http://", "https://")):
        p = urlsplit(uri)
        uri = p.path + ("?" + p.query if p.query else "")
    return uri or "/"


def build_input(method: str, uri: str, ua: str = "", referer: str = "",
                cookie: str = "", body: str = "", caps: dict | None = None) -> str:
    c = caps or FIELD_CAPS
    m = (method or "").upper()
    m = m if _METHOD.match(m) else "OTHER"
    return (f"{m} {normalize_field(strip_origin(uri or '/'), c['uri'])}"
            f" [UA] {normalize_field(ua, c['ua'])}"
            f" [REF] {normalize_field(referer, c['referer'])}"
            f" [CK] {normalize_field(cookie, c['cookie'])}"
            f" [BODY] {normalize_field(body, c['body'])}")


def parse_raw_request(raw: str) -> dict:
    """Parse request HTTP/1.x dạng text thô -> {method, uri, headers(lowercase keys), body}."""
    raw = raw.replace("\r\n", "\n").lstrip("\n")
    head, _, body = raw.partition("\n\n")
    lines = head.split("\n")
    parts = lines[0].split(" ")
    method = parts[0] if parts else ""
    if len(parts) >= 3 and parts[-1].upper().startswith("HTTP/"):
        uri = " ".join(parts[1:-1])
    else:
        uri = " ".join(parts[1:])
    headers: dict[str, str] = {}
    for ln in lines[1:]:
        k, sep, v = ln.partition(":")
        if sep:
            k = k.strip().lower()
            headers[k] = (headers[k] + "; " + v.strip()) if k in headers else v.strip()
    return {"method": method, "uri": uri, "headers": headers, "body": body.strip("\n")}


def build_input_from_raw(raw: str, caps: dict | None = None) -> str:
    r = parse_raw_request(raw)
    h = r["headers"]
    return build_input(r["method"], r["uri"], h.get("user-agent", ""), h.get("referer", ""),
                       h.get("cookie", ""), r["body"], caps)
