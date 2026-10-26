ACCESS = {
    "GTEx processed":        ("open", "CC BY 4.0 for processed data; attribution", "minutes"),
    "GTEx raw sequence":     ("controlled", "dbGaP and AnVIL; research use", "weeks to months"),
    "TCGA clinical/RNA":     ("open", "GDC open tier; check terms", "hours"),
    "TCGA germline":         ("controlled", "dbGaP; institutional certification", "months"),
    "UK Biobank":            ("controlled", "application, fee, UKB-RAP", "weeks to months"),
    "CELLxGENE atlas":       ("open", "per-collection licence; often CC BY 4.0", "minutes"),
    "Tabula Sapiens raw":    ("controlled", "data use agreement; no redistribution", "weeks"),
    "GEO series":            ("open", "per-series terms", "minutes"),
    "Open Targets":          ("open", "attribution; release version", "minutes"),
}

print(f"{'Resource':<24}{'Tier':>12}  {'Latency':<18}Terms")
for name, (tier, terms, latency) in ACCESS.items():
    print(f"{name:<24}{tier:>12}  {latency:<18}{terms}")
print()
open_count = sum(1 for t, _, _ in ACCESS.values() if t == "open")
print(f"Open resources: {open_count}/{len(ACCESS)}")
print("Open processed data does not imply open raw data or open commercial rights.")
