from llm.eval.verifier import check_claim, claim_tokens, packet_haystack, verify_memo

PACKET = {
    "alert": {
        "order_id": "o_501",
        "amount": 899.99,
        "ts": "2026-02-11 03:40:00",
        "fired_rules": [{"id": "R01", "name": "credential change before order"}],
    },
    "account": {"user_id": "u_42", "tenure_days": 412, "n_orders_prior": 9},
    "events": [{"kind": "password_change", "ts": "2026-02-11 03:12:00", "device_id": "d_88121"}],
}


def test_supported_claim_passes():
    hay = packet_haystack(PACKET)
    c = check_claim("password_change at 2026-02-11 03:12:00 from device d_88121", hay)
    assert c["supported"], c


def test_planted_unsupported_claim_caught():
    hay = packet_haystack(PACKET)
    c = check_claim("3 prior chargebacks totalling $2,450.00 on device d_99999", hay)
    assert not c["supported"]
    assert "d_99999" in c["missing"]


def test_number_normalization():
    hay = packet_haystack(PACKET)
    assert check_claim("order amount $899.99 by user u_42", hay)["supported"]


def test_token_extraction_masks_timestamps():
    toks = claim_tokens("at 2026-02-11 03:12:00 exactly")
    assert "2026-02-11 03:12:00" in toks
    assert "2026" not in toks  # masked by the timestamp match


def test_verify_memo_citations():
    memo = {
        "signals_observed": ["tenure_days 412", "made-up device d_777"],
        "policy_citations": ["R01", "R99", "FP-1 §6.1"],
    }
    v = verify_memo(memo, PACKET)
    assert v["hallucinated"]
    assert v["n_unsupported_claims"] == 1
    assert v["n_invalid_citations"] == 1  # R99
    fired = [c for c in v["citations"] if c["rule_id"] == "R01"][0]
    assert fired["fired"] is True
