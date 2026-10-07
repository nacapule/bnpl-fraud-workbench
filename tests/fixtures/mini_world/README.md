# Mini world

A small world in the `core/world.py` contract for tests and development: 19
accounts, 3 merchants, 26 order attempts and every event and cash kind a natural
world produces (refunds come only from replay actions). `build.py` writes these
files; `tests/test_mini_world.py` checks they equal a fresh build, pass the
validator and carry the hand-checked cash and labels below, and
`tests/test_mini_world_mysql.py` loads them into MySQL (`db/load_world.py`).

Orders run from 2024-12-05 to 2025-06-10; outcomes are observed until
2025-06-30 23:59:59; the label horizon is 60 days.

| Account | Story | Net cash (cents) | Adjudicated label |
| --- | --- | ---: | --- |
| ana | Long-tenured; $84.50 repaid; $430 repaid after one installment bounced and was paid on retry | +423; +2,150 | 0 no finding |
| ben, cara | Household sharing an address and a tablet; ben is new and his first order is $1,299.99; both use FIRST10 | -6,500 (ben) | 0 (two linked accounts are not promotion abuse) |
| dan | Takeover: new device, password reset, new drop address, $899 order; owner reports it | -49,445 | 1 account takeover |
| eve | New account testing stolen cards: two processor declines, then a $650 approval; installments fail; unauthorized dispute lost | -53,500 | 1 third-party fraud at the dispute's resolution (one unmarked default is not never-pay) |
| fay | New customer with two orders three days apart ($520, $180) who pays nothing after either checkout | -28,600; -9,900 | 1 never-pay on both, known at the second default |
| gus | Returning customer in hardship: a zero-effort default after a repaid plan | -13,200 | 0 credit loss (no intent marker) |
| nr1-nr3 | Three new accounts on one phone, each with one $275-$410 order and nothing paid after checkout | -17,600; -15,125; -22,550 | 1 never-pay on all three, known at the third default |
| hal | Repays, then claims two delivered orders never arrived; both claims rejected | -1,125 on the first | 0 at the horizon, then 1 INR abuse |
| ivy | Parcel lost by the carrier; claim upheld; the merchant reimburses | -450 | 0, postponed until the dispute resolved |
| jay, kim | Customers of a jewelry merchant that stops shipping and closes; jay's order was never delivered, kim's was | -36,600 (jay) | 1 merchant bust-out (jay); 0 (kim) |
| lee | Travelling; orders from a French IP | +775 | 0 |
| pf1-pf3 | Three new accounts on one device take FIRST10 and never order again without it | about -300 each | 0 at the horizon, then 1 promotion abuse 90 days after the third use |
| mo | An order ten days before observation ends | -1,350 so far | none (unknown is not negative) |

Latent truth (episodes, account actors, order patterns and benign mimics) is in
`latent_*.csv`; nothing that makes decisions may read it or `labels.csv`.
