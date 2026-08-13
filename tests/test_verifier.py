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


def test_invented_short_count_does_not_match_digits_inside_other_values():
    hay = packet_haystack(PACKET)
    for invented in (7, 13):
        check = check_claim(f"there were {invented} linked accounts", hay)
        assert check["classification"] == "unsupported"
        assert check["missing"] == [str(invented)]


def test_digits_inside_longer_number_do_not_match():
    hay = packet_haystack({"account": {"tenure_days": 412}})
    assert check_claim("there were 12 prior orders", hay)["classification"] == "unsupported"


def test_legitimate_rounding_passes():
    packet = {"account": {"avg_amount_prior": 101.11857}}
    check = check_claim("average amount was $101.12", packet_haystack(packet))
    assert check["classification"] == "verbatim"


def test_visible_list_row_count_is_derived():
    packet = {
        "last_orders": [
            {"order_id": "o_101", "status": "approved"},
            {"order_id": "o_102", "status": "approved"},
            {"order_id": "o_103", "status": "declined"},
        ]
    }
    check = check_claim(
        "last_orders contains 2 approved orders: o_101 and o_102",
        packet_haystack(packet),
        packet=packet,
    )
    assert check["classification"] == "derived"
    assert check["missing"] == []


def test_uppercase_entity_id_is_checked():
    packet = {"device_id": "DEV_42"}
    assert check_claim("device DEV_42", packet_haystack(packet))["supported"]
    assert not check_claim("device DEV_43", packet_haystack(packet))["supported"]


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
