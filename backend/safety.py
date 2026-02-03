import re


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"
)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.\s]{2,40}\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b",
    re.IGNORECASE,
)


def redact_text(text: str) -> str:
    text = EMAIL_RE.sub("[redacted email]", text)
    text = PHONE_RE.sub("[redacted phone]", text)
    text = SSN_RE.sub("[redacted ssn]", text)
    text = ADDRESS_RE.sub("[redacted address]", text)
    return text


def trim_text(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def contains_disallowed_query(query: str) -> bool:
    lowered = query.lower()
    phrases = [
        "email address",
        "phone number",
        "home address",
        "contact info",
        "contact information",
        "list victims",
        "victim list",
        "dox",
        "doxxing",
        "personal address",
        "private address",
        "social security",
        "ssn",
    ]
    return any(p in lowered for p in phrases)
