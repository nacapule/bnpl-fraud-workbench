from pathlib import Path

from analysis.followups import (
    case01_ato_address_bonus,
    case02_device_cooldown,
    case03_r08_threshold,
    case03_rejected_address_attach,
    case04_never_pay_candidate,
    case05_r03_suppression,
)

CASES = Path(__file__).resolve().parents[1] / "cases"


def _case(name: str) -> str:
    return (CASES / name).read_text()


def test_case01_followup_numbers() -> None:
    result = case01_ato_address_bonus()
    assert result == {
        "holdout_ato_orders": 50,
        "baseline_auto_declines": 1,
        "proposed_auto_declines": 13,
        "added_benign_auto_declines": 0,
    }
    text = _case("CASE-01-account-takeover.md")
    assert (
        f"**{result['baseline_auto_declines']} → {result['proposed_auto_declines']} "
        f"(of {result['holdout_ato_orders']})**" in text
    )


def test_case02_followup_numbers() -> None:
    result = case02_device_cooldown()
    assert result == {
        "devices_cooled": 46,
        "fraud_linked_devices": 46,
        "benign_linked_devices": 0,
    }
    text = _case("CASE-02-card-testing-stolen-card.md")
    assert f"**{result['devices_cooled']} devices**" in text
    assert f"**all {result['fraud_linked_devices']}**" in text


def test_case03_r08_followup_numbers() -> None:
    result = case03_r08_threshold()
    assert result == {
        "added_alerts": 4842,
        "added_benign_alerts": 4840,
        "added_fraud_alerts": 2,
        "added_alerts_per_day": 57.0,
    }
    text = _case("CASE-03-synthetic-ring-bustout.md")
    volume = (
        f"**{result['added_alerts']:,} added alerts "
        f"({result['added_alerts_per_day']:.1f}/day)**"
    )
    assert volume in text
    assert f"**{result['added_fraud_alerts']}** additional fraud" in text
    assert f"**{result['added_benign_alerts']:,}** benign" in text


def test_case03_address_attach_numbers() -> None:
    result = case03_rejected_address_attach()
    assert result == {
        "flagged_addresses": 3392,
        "addresses_with_benign_orders": 3382,
    }
    text = _case("CASE-03-synthetic-ring-bustout.md")
    assert f"**{result['addresses_with_benign_orders']:,} with benign orders**" in text


def test_case04_followup_numbers() -> None:
    result = case04_never_pay_candidate()
    assert result == {
        "holdout_never_pay_orders": 63,
        "currently_alerted_never_pay_orders": 1,
        "candidate_added_alerts": 1,
        "candidate_added_pattern": "P-PROMO",
        "benign_first_order_cap_rate": 0.263,
    }
    text = _case("CASE-04-neverpay-vs-hardship.md")
    assert (
        f"**{result['currently_alerted_never_pay_orders']} of "
        f"{result['holdout_never_pay_orders']}**" in text
    )
    assert f"a {result['candidate_added_pattern']} order" in text


def test_case05_followup_numbers() -> None:
    result = case05_r03_suppression()
    assert result == {
        "r03_only_alerts": 1303,
        "suppressed_alerts": 364,
        "suppressed_fraud_alerts": 0,
        "suppressed_alerts_per_day": 4.3,
    }
    text = _case("CASE-05-traveler-cleared.md")
    assert f"**{result['suppressed_alerts']} of {result['r03_only_alerts']:,}**" in text
    assert f"−{result['suppressed_alerts_per_day']:.1f} alerts/day" in text
