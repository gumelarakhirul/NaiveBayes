"""Local PunyGuard server: persistent Naive Bayes plus Unicode homograph guard."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import json
import html
import ipaddress
import re
import unicodedata

import joblib

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
FINAL = joblib.load(ROOT / "models" / "final_url_model.joblib")
HOST, PORT = "127.0.0.1", 8000

CONFUSABLES = str.maketrans({
    # Cyrillic
    "а": "a", "с": "c", "е": "e", "і": "i", "ӏ": "l", "ј": "j",
    "о": "o", "р": "p", "ѕ": "s", "х": "x", "у": "y", "ԁ": "d",
    "ɡ": "g", "һ": "h", "к": "k", "м": "m", "п": "n", "т": "t",
    "ѵ": "v", "ԝ": "w", "ᴢ": "z",
    # Greek
    "α": "a", "β": "b", "ϲ": "c", "ε": "e", "ι": "i", "κ": "k",
    "ν": "v", "ο": "o", "ρ": "p", "τ": "t", "υ": "u", "χ": "x",
    # Armenian and Cherokee
    "օ": "o", "Ꭺ": "a", "ꭺ": "a", "Ᏼ": "b", "Ꮯ": "c", "Ꭰ": "d", "Ꭼ": "e",
    "Ꮐ": "g", "Ꮋ": "h", "Ꮖ": "i", "Ꭻ": "j", "Ꮶ": "k", "Ꮮ": "l",
    "Ꮇ": "m", "Ꮑ": "n", "Ꮎ": "o", "ꮎ": "o", "Ꮲ": "p", "Ꮪ": "s", "Ꭲ": "t",
    "Ꮜ": "u", "Ꮩ": "v", "Ꮤ": "w", "Ꭹ": "y",
})
PROTECTED_BRANDS = {
    "google", "gmail", "youtube", "apple", "microsoft", "microsoftonline",
    "paypal", "amazon", "facebook", "instagram", "netflix", "whatsapp",
    "telegram", "binance", "shopee", "tokopedia", "bukalapak", "mandiri",
    "bri", "bni", "dana", "ovo",
}
OFFICIAL_DOMAINS = {
    "google.com", "gmail.com", "youtube.com", "apple.com", "microsoft.com",
    "microsoftonline.com", "paypal.com", "amazon.com", "facebook.com",
    "instagram.com", "netflix.com", "whatsapp.com", "telegram.org",
    "binance.com", "shopee.co.id", "tokopedia.com", "bukalapak.com",
    "bankmandiri.co.id", "bri.co.id", "bni.co.id", "dana.id", "ovo.id",
}
SUSPICIOUS = {
    "login", "signin", "verify", "verification", "secure", "account",
    "update", "password", "wallet", "invoice", "suspended", "confirm",
}


def prepare_url(value: str) -> str:
    raw = unicodedata.normalize("NFKC", html.unescape(str(value).strip())).lower()
    parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
    host = (parsed.hostname or "").rstrip(".")
    path_query = f"{parsed.path}?{parsed.query}"
    tokens = []
    try:
        ipaddress.ip_address(host.strip("[]"))
        tokens.append("__host_ip__")
    except ValueError:
        pass
    if "@" in parsed.netloc:
        tokens.append("__at_symbol__")
    if host.count(".") >= 3:
        tokens.append("__many_subdomains__")
    if len(raw) >= 75:
        tokens.append("__long_url__")
    if host.startswith("xn--") or ".xn--" in host:
        tokens.append("__punycode__")
    if parsed.scheme == "http":
        tokens.append("__plain_http__")
    if any(word in path_query for word in SUSPICIOUS):
        tokens.append("__suspicious_path__")
    if re.search(r"%[0-9a-f]{2}", raw):
        tokens.append("__percent_encoding__")
    try:
        if parsed.port and parsed.port not in (80, 443):
            tokens.append("__unusual_port__")
    except ValueError:
        tokens.append("__invalid_port__")
    return raw + " " + " ".join(tokens)


def url_indicators(value: str) -> list[str]:
    return re.findall(r"__[a-z_]+__", prepare_url(value))


def normalize_host(value: str) -> str:
    candidate = unicodedata.normalize("NFKC", value.strip())
    if not candidate:
        raise ValueError("URL/domain tidak boleh kosong.")
    parsed = urlsplit(candidate if "://" in candidate else f"https://{candidate}")
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if hostname.startswith("www."):
        hostname = hostname[4:]
    if not hostname or " " in hostname:
        raise ValueError("Hostname tidak valid.")
    return hostname


def encode_idn(hostname: str) -> str:
    try:
        return ".".join(
            label.encode("idna").decode("ascii") for label in hostname.split(".")
        )
    except UnicodeError as error:
        raise ValueError("Hostname mengandung label Unicode yang tidak valid.") from error


def decode_idn(hostname: str) -> str:
    decoded = []
    for label in hostname.split("."):
        try:
            decoded.append(label.encode("ascii").decode("idna"))
        except UnicodeError:
            decoded.append(label)
    return ".".join(decoded)


def script_of(char: str) -> str | None:
    if not char.isalpha():
        return None
    name = unicodedata.name(char, "")
    for script in ("LATIN", "CYRILLIC", "GREEK", "ARMENIAN", "CHEROKEE"):
        if script in name:
            return script
    return "OTHER"


def skeleton(label: str) -> str:
    # Case first: Cherokee lowercase does not preserve the expected visual mapping.
    mapped = label.translate(CONFUSABLES).lower()
    decomposed = unicodedata.normalize("NFKD", mapped)
    return "".join(
        char for char in decomposed
        if char.isascii() and char.isalnum() and not unicodedata.combining(char)
    )


def homograph_analysis(hostname: str) -> dict:
    decoded = decode_idn(hostname)
    reasons, scripts_seen = [], set()
    for label in decoded.split(".")[:-1]:
        scripts = {script_of(char) for char in label} - {None}
        scripts_seen.update(scripts)
        if len(scripts) > 1:
            reasons.append("Mixed-script dalam satu label: " + ", ".join(sorted(scripts)))
        label_skeleton = skeleton(label)
        if label_skeleton in PROTECTED_BRANDS and any(
            not char.isascii() for char in label
        ):
            reasons.append(f"Unicode menyerupai merek '{label_skeleton}'")
    return {
        "decoded_domain": decoded,
        "is_homograph": bool(reasons),
        "reasons": list(dict.fromkeys(reasons)),
        "scripts": sorted(scripts_seen),
    }


def model_probability(value: str) -> float:
    matrix = FINAL["vectorizer"].transform([prepare_url(value)])
    return float(FINAL["model"].predict_proba(matrix)[0, 1])


def predict(value: str) -> dict:
    had_fullwidth = any(
        unicodedata.east_asian_width(char) in {"F", "W"}
        and ("FULLWIDTH" in unicodedata.name(char, "") or char == "．")
        for char in value
    )
    unicode_hostname = normalize_host(value)
    hostname = encode_idn(unicode_hostname)
    is_punycode = any(label.startswith("xn--") for label in hostname.split("."))
    probability = model_probability(value)
    homograph = homograph_analysis(hostname)
    indicators = url_indicators(value)
    if had_fullwidth:
        first_label = unicode_hostname.split(".")[0]
        if first_label in PROTECTED_BRANDS:
            homograph["reasons"].append(
                f"Karakter fullwidth menyamarkan merek '{first_label}'"
            )
            homograph["is_homograph"] = True

    # Conservative application policy: a confirmed protected-brand homograph is
    # phishing regardless of the statistical model score.
    if homograph["is_homograph"]:
        probability = max(probability, 0.99)
    official_exact = hostname in OFFICIAL_DOMAINS and not homograph["is_homograph"]
    if official_exact:
        probability = min(probability, 0.01)
    phishing = (not official_exact) and (
        homograph["is_homograph"] or probability >= FINAL["threshold"]
    )
    return {
        "input": value,
        "hostname": hostname,
        "decoded_domain": homograph["decoded_domain"],
        "punycode": is_punycode,
        "homograph": homograph["is_homograph"],
        "homograph_reasons": homograph["reasons"],
        "scripts": homograph["scripts"],
        "model": FINAL["name"],
        "url_indicators": indicators,
        "official_domain": official_exact,
        "probability": probability,
        "prediction": "phishing" if phishing else "legitimate",
        "interpretation": (
            "Hostname cocok persis dengan domain resmi yang dilindungi; "
            "tidak ada substitusi Unicode pada hostname."
            if official_exact else
            "Homograph Unicode terkonfirmasi. Domain diblokir oleh kebijakan "
            "konservatif meskipun skor model statistik lebih rendah."
            if homograph["is_homograph"] else
            "Pola hostname terindikasi phishing. Lakukan pemblokiran sementara "
            "dan verifikasi reputasi, DNS, sertifikat, serta isi halaman."
            if phishing else
            "Tidak ditemukan homograph merek yang dikenal dan pola model "
            "cenderung legitimate. Ini bukan jaminan keamanan absolut."
        ),
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def do_GET(self):
        if self.path == "/api/health":
            self.send_json({"status": "ready", "model": FINAL["name"],
                            "metrics": FINAL["metrics"]})
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 8192:
                raise ValueError("Ukuran request tidak valid.")
            payload = json.loads(self.rfile.read(length))
            self.send_json(predict(str(payload.get("url", ""))))
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, status=400)

    def send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"PunyGuard siap: http://{HOST}:{PORT}")
    print("Model dimuat sekali; tekan Ctrl+C untuk berhenti.")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
