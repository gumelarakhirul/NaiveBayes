import csv
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parent
LIMIT = 5_000

# IDNA-valid Cyrillic lookalikes. Replacing one character is enough to force an
# ASCII-compatible hostname label (xn--...) while retaining a recognizable form.
HOMOGLYPHS = {
    "a": "\u0430",
    "c": "\u0441",
    "e": "\u0435",
    "i": "\u0456",
    "j": "\u0458",
    "o": "\u043e",
    "p": "\u0440",
    "s": "\u0455",
    "x": "\u0445",
    "y": "\u0443",
}


def punycode_hostname(hostname: str) -> str | None:
    """Return a hostname with one label converted to an IDNA homograph."""
    labels = hostname.rstrip(".").split(".")
    # Do not mutate the public suffix; start with the left-most recognizable label.
    for label_index in range(max(0, len(labels) - 1)):
        label = labels[label_index]
        if label.lower().startswith("xn--"):
            continue
        for char_index, char in enumerate(label):
            replacement = HOMOGLYPHS.get(char.lower())
            if replacement is None:
                continue
            if char.isupper():
                replacement = replacement.upper()
            unicode_label = label[:char_index] + replacement + label[char_index + 1 :]
            try:
                encoded = unicode_label.encode("idna").decode("ascii")
            except UnicodeError:
                continue
            if encoded.startswith("xn--") and len(encoded) <= 63:
                labels[label_index] = encoded
                result = ".".join(labels)
                if len(result) <= 253:
                    return result
    return None


def transform_url(raw_url: str, default_scheme: str = "https") -> str | None:
    candidate = raw_url.strip()
    if not candidate:
        return None
    parsed = urlsplit(candidate if "://" in candidate else f"{default_scheme}://{candidate}")
    if not parsed.hostname:
        return None
    encoded_host = punycode_hostname(parsed.hostname)
    if not encoded_host:
        return None

    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((parsed.scheme, f"{userinfo}{encoded_host}{port}", parsed.path, parsed.query, parsed.fragment))


def write_legitimate() -> int:
    output = ROOT / "legitimate_punycode_5000.csv"
    seen: set[str] = set()
    with (ROOT / "tranco_6WNKX.csv").open(newline="", encoding="utf-8-sig") as source, output.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.reader(source)
        writer = csv.DictWriter(target, fieldnames=["url", "label", "source_url"])
        writer.writeheader()
        for row in reader:
            if len(row) < 2:
                continue
            source_url = row[1].strip()
            converted = transform_url(source_url)
            if converted and converted not in seen:
                writer.writerow({"url": converted, "label": "legitimate", "source_url": source_url})
                seen.add(converted)
                if len(seen) == LIMIT:
                    break
    return len(seen)


def write_phishing() -> int:
    output = ROOT / "phishing_punycode_5000.csv"
    seen: set[str] = set()
    with (ROOT / "verified_online.csv").open(newline="", encoding="utf-8-sig") as source, output.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=["url", "label", "source_url"])
        writer.writeheader()
        for row in reader:
            source_url = (row.get("url") or "").strip()
            converted = transform_url(source_url)
            if converted and converted not in seen:
                writer.writerow({"url": converted, "label": "phishing", "source_url": source_url})
                seen.add(converted)
                if len(seen) == LIMIT:
                    break
    return len(seen)


def main() -> None:
    legitimate_count = write_legitimate()
    phishing_count = write_phishing()
    if legitimate_count != LIMIT or phishing_count != LIMIT:
        raise RuntimeError(
            f"Insufficient convertible rows: legitimate={legitimate_count}, phishing={phishing_count}"
        )
    print(f"Created {legitimate_count} legitimate and {phishing_count} phishing Punycode URLs.")


if __name__ == "__main__":
    main()
