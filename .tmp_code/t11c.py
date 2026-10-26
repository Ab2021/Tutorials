LICENCE_CLASSES = {
    "MIT/Apache-2.0": {"commercial": True, "copyleft": False, "patent_grant": True},
    "GPL-3.0": {"commercial": True, "copyleft": True, "patent_grant": True},
    "AGPL-3.0": {"commercial": True, "copyleft": True, "patent_grant": True},
    "Proprietary": {"commercial": "contract", "copyleft": False, "patent_grant": "contract"},
}
print(f"{'Licence':<18}{'Commercial':>12}{'Copyleft':>10}{'Patent':>10}")
for name, terms in LICENCE_CLASSES.items():
    print(f"{name:<18}{str(terms['commercial']):>12}{str(terms['copyleft']):>10}{str(terms['patent_grant']):>10}")
print()
print("Copyleft affects distribution of derivative software, not internal use.")
print("AGPL can be triggered by network use; legal review is required.")
