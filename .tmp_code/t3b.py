from datetime import date, timedelta

DATA_CUTOFF = date(2020, 12, 31)
FOLLOWUP_DAYS = 30

admissions = [
    {"id": "A1", "discharge": date(2020, 1, 10), "next_admit": date(2020, 1, 25), "died": None},
    {"id": "A2", "discharge": date(2020, 2, 1), "next_admit": None, "died": None},
    {"id": "A3", "discharge": date(2020, 11, 20), "next_admit": date(2020, 12, 15), "died": None},
    {"id": "A4", "discharge": date(2020, 12, 20), "next_admit": None, "died": None},
    {"id": "A5", "discharge": date(2020, 3, 5), "next_admit": None, "died": date(2020, 3, 20)},
]

print(f"{'ID':<4}{'Discharge':>12}{'Next':>12}{'Readmit30':>11}{'Observed':>10}  Note")
for a in admissions:
    observed = (a["discharge"] + timedelta(days=FOLLOWUP_DAYS)) <= DATA_CUTOFF
    if a["next_admit"] is not None:
        gap = (a["next_admit"] - a["discharge"]).days
        readmit = gap <= FOLLOWUP_DAYS
    else:
        readmit = False
    note = ""
    if a["died"] is not None:
        note = "death is a competing event"
    elif not observed:
        note = "administratively censored"
    print(f"{a['id']:<4}{a['discharge'].isoformat():>12}"
          f"{(a['next_admit'].isoformat() if a['next_admit'] else '-'):>12}"
          f"{str(readmit):>11}{str(observed):>10}  {note}")

eligible = [a for a in admissions if (a["discharge"] + timedelta(days=FOLLOWUP_DAYS)) <= DATA_CUTOFF and a["died"] is None]
print()
print(f"Admissions listed: {len(admissions)}")
print(f"Eligible with complete 30-day follow-up and alive at discharge: {len(eligible)}")
print("Rows without complete follow-up cannot contribute a negative label.")
