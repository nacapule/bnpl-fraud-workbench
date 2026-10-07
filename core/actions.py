"""Actions and their consequences in the replay (interface; implemented with the replay).

The world holds every attempted order with the outcomes it would have if
approved; policies act only in the replay, which applies these actions at their
times and records each consequence as ledger events (core.ledger), reviewer
minutes and customer friction. Vocabulary (the fraud policy's terms):

Checkout routing, decided at arrival from the as-of context with thresholds
fixed in advance (no within-day ranking):

* ``approve``: the order proceeds; its potential outcomes are realised.
* ``review``: the order proceeds and enters the review queue (reviewer minutes);
  fulfilment continues unless the reviewer holds or declines before shipment.
* ``auto_decline``: no order, no cash events, no reviewer minutes; records no
  fraud finding and blocks no account. A legitimate decline loses the merchant
  fee and an LTV proxy and counts as friction.

Analyst dispositions, applied when the review completes:

* ``clear``: nothing changes.
* ``hold``: runs the verification checks. Before shipment it pauses shipment and
  merchant settlement up to ``config/policy.yaml`` actions.hold_max_hours; a
  customer who verifies is released (shipment and the repayment schedule shift
  after the release), and one who does not respond is cancelled with the checkout
  payment refunded and no block (friction). After shipment nothing is paused and
  no response changes nothing; a failed check still declines and the loss stands.
* ``decline``: before shipment, voids the order (cash already moved stays and is
  refunded); after shipment the loss stands, the account is blocked and later
  orders are declined.
* ``escalate``: a decline that also blocks accounts linked by device or shipping
  address within 30-day windows, as known at the decision; extra minutes.

Verification checks the reviewer may use (at most two per order):
``contact`` and ``id_check``, with outcomes ``passed``, ``failed`` or
``no_response``. Outcome draws are keyed by the order's stable id, so every
policy sees the same draws. The memo drafter may suggest ``needs_check`` (a hold
with a named check); it never acts.

To be implemented here: the transition of an order's cash events, minutes and
friction under each action, and the policy state (core.asof.PolicyState) that
outcome-derived context columns are rebuilt from.
"""

from __future__ import annotations

from enum import StrEnum


class CheckoutRoute(StrEnum):
    APPROVE = "approve"
    REVIEW = "review"
    AUTO_DECLINE = "auto_decline"


class Disposition(StrEnum):
    CLEAR = "clear"
    HOLD = "hold"
    DECLINE = "decline"
    ESCALATE = "escalate"


MEMO_DISPOSITIONS = (*(d.value for d in Disposition), "needs_check")


class Check(StrEnum):
    CONTACT = "contact"
    ID_CHECK = "id_check"


class CheckOutcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NO_RESPONSE = "no_response"
