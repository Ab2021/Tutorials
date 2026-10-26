"""Offer comparison with equity scenarios and cost-of-living adjustment.

Run: python offer_compare.py
Standard library only. Compares a health-tech offer, a payer offer in a
lower-cost city, and a startup offer with options, under three share-price
scenarios.
"""

OFFERS = [
    {
        "name": "A health-tech, Boston",
        "base": 162_000,
        "bonus_rate": 0.10,
        "equity_per_year": 20_000,   # RSU grant 80k over 4 years
        "signing": 15_000,
        "match_rate": 0.04,
        "col_index": 100.0,
    },
    {
        "name": "B payer, Raleigh",
        "base": 148_000,
        "bonus_rate": 0.08,
        "equity_per_year": 0,
        "signing": 5_000,
        "match_rate": 0.06,
        "col_index": 90.0,
    },
    {
        "name": "C startup, Boston",
        "base": 150_000,
        "bonus_rate": 0.0,
        "equity_per_year": 0,        # options handled separately
        "signing": 0,
        "match_rate": 0.0,
        "col_index": 100.0,
    },
]

# Startup option terms.
OPTION_SHARES = 60_000
STRIKE = 3.00
SCENARIOS = [
    ("downside", 3.00, 0.60),
    ("base", 9.00, 0.30),
    ("upside", 18.00, 0.10),
]


def annual_cash(offer):
    return offer["base"] * (1 + offer["bonus_rate"]) + offer["base"] * offer["match_rate"]


def year_one(offer):
    return annual_cash(offer) + offer["equity_per_year"] + offer["signing"]


def four_year(offer):
    return 4 * annual_cash(offer) + 4 * offer["equity_per_year"] + offer["signing"]


def real(value, offer):
    return value / (offer["col_index"] / 100.0)


def option_value(price):
    return OPTION_SHARES * max(0.0, price - STRIKE)


def main():
    print("OFFER COMPARISON (NOMINAL USD)")
    print(f"  {'offer':24s} {'annual cash':>12s} {'year 1':>12s} {'4 years':>12s}")
    for o in OFFERS:
        print(f"  {o['name']:24s} {annual_cash(o):12,.0f} {year_one(o):12,.0f} {four_year(o):12,.0f}")

    print()
    print("COST-OF-LIVING ADJUSTED (BOSTON = 100)")
    for o in OFFERS:
        print(f"  {o['name']:24s} index {o['col_index']:5.1f}  "
              f"year 1 real {real(year_one(o), o):12,.0f}  "
              f"4y real {real(four_year(o), o):12,.0f}")

    print()
    print("STARTUP OPTIONS: VALUE BY SCENARIO")
    print(f"  {'scenario':10s} {'price':>7s} {'prob':>6s} {'value':>12s} {'per year':>12s} {'weighted':>12s}")
    weighted = 0.0
    for name, price, prob in SCENARIOS:
        value = option_value(price)
        weighted += value * prob
        print(f"  {name:10s} {price:7.2f} {prob:6.2f} {value:12,.0f} {value / 4:12,.0f} {value * prob:12,.0f}")
    print(f"  expected option value over 4 years: {weighted:,.0f}")

    startup = OFFERS[2]
    startup_4y = four_year(startup) + weighted
    print()
    print("STARTUP TOTAL WITH EXPECTED EQUITY")
    print(f"  cash over 4 years:            {four_year(startup):,.0f}")
    print(f"  expected equity:              {weighted:,.0f}")
    print(f"  total:                        {startup_4y:,.0f}")
    print(f"  health-tech offer for compare: {four_year(OFFERS[0]):,.0f}")
    print(f"  difference:                   {startup_4y - four_year(OFFERS[0]):,.0f}")

    print()
    print("GAP: INITIAL OFFER VS NEGOTIATED (HEALTH-TECH)")
    initial = 155_000
    initial_cash = initial * 1.10 + initial * 0.04
    initial_4y = 4 * initial_cash + 80_000 + 10_000
    print(f"  initial four-year value:    {initial_4y:,.0f}")
    print(f"  negotiated four-year value: {four_year(OFFERS[0]):,.0f}")
    print(f"  gain:                       {four_year(OFFERS[0]) - initial_4y:,.0f}")


if __name__ == "__main__":
    main()
