from src.pii_mask import mask_pii


def test_aadhaar_masked():
    masked, counts = mask_pii("Aadhaar 2345 6789 0123 hai")
    assert "[AADHAAR]" in masked
    assert "2345" not in masked
    assert counts["AADHAAR"] == 1


def test_pan_masked():
    masked, _ = mask_pii("PAN ABCDE1234F se KYC karo")
    assert "[PAN]" in masked
    assert "ABCDE1234F" not in masked


def test_ifsc_masked():
    masked, _ = mask_pii("IFSC SBIN0001234 wali branch")
    assert "[IFSC]" in masked


def test_upi_masked():
    masked, _ = mask_pii("UPI rahul.sharma@okaxis pe payment fail")
    assert "[UPI]" in masked
    assert "rahul" not in masked


def test_phone_with_country_code_masked():
    masked, _ = mask_pii("Number +91 9876543210 par OTP bhejo")
    assert "[PHONE]" in masked
    assert "9876543210" not in masked


def test_account_number_masked():
    masked, _ = mask_pii("Account 123456789012 mein paise nahi aaye")
    assert "[ACCOUNT]" in masked


def test_normal_text_unchanged():
    text = "Mera card block ho gaya hai, kya karu?"
    masked, counts = mask_pii(text)
    assert masked == text
    assert counts == {}