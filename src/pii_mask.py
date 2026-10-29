import re

# Order matter karta hai: lamba/specific pattern pehle, generic baad mein.
# Aadhaar (12 digit) ko account number (9-18 digit) se pehle mask karna zaroori hai.
PATTERNS = [
    ("AADHAAR", re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)),
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE)),
    (
        "UPI",
        re.compile(
            r"\b[\w.\-]{2,256}@(?:ybl|okaxis|oksbi|okhdfcbank|okicici|paytm|upi|"
            r"axl|ibl|sbi|icici|hdfcbank|axisbank)\b",
            re.IGNORECASE,
        ),
    ),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+91[\s-]?|0)?[6-9]\d{9}(?!\d)")),
    ("ACCOUNT", re.compile(r"(?<!\d)\d{9,18}(?!\d)")),
]


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """Text mein PII ko placeholder se replace karta hai.

    Return: (masked_text, counts) jaise ("Mera [PHONE] hai", {"PHONE": 1})
    """
    counts: dict[str, int] = {}
    masked = text
    for label, pattern in PATTERNS:
        masked, n = pattern.subn(f"[{label}]", masked)
        if n:
            counts[label] = counts.get(label, 0) + n
    return masked, counts


if __name__ == "__main__":
    samples = [
        "Mera Aadhaar 2345 6789 0123 hai, kya ye safe hai?",
        "PAN ABCDE1234F se KYC update karna hai",
        "Account 123456789012 mein paise nahi aaye",
        "Mera number +91 9876543210 par OTP bhejo",
        "UPI ID rahul.sharma@okaxis pe payment fail hua",
        "IFSC SBIN0001234 wali branch se transfer karna",
        "Email: test.user@gmail.com par mail karo",
        "Mera card block ho gaya hai, kya karu?",
    ]
    for s in samples:
        masked, counts = mask_pii(s)
        print(f"IN : {s}")
        print(f"OUT: {masked}  {counts}")
        print()