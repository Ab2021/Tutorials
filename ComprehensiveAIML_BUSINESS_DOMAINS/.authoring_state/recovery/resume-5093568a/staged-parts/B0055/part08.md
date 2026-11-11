## 12. Quick Reference Card

| Item | Formula or rule | Illustrative value | Watch out for |
|---|---|---|---|
| Precision from rates | PPV = pi·TPR / (pi·TPR + (1−pi)·FPR) | 0.181 at pi=0.004, TPR=0.55, FPR=0.01 | Precision without prevalence is meaningless |
| Precision at capacity | TP / alerts at the capacity cut | 2,112/8,800 = 0.240 | Rounded inputs break reconciliation; use integer counts |
| Recall at capacity | TP / all mature frauds | 2,112/8,000 = 0.264 | Immature labels bias it down 19.5% at 60 days |
| Value per true positive | rho · L | 0.62 × 320 = 198.40 | rho unverified without a randomised hold-out |
| Break-even probability | (c_a + c_f) / (rho·L + c_f) | 16.10/200.50 = 0.0803 | Loss-ratio form gives 0.0751; state which you used |
| Net value per period | TP·rho·L − a·c_a − (a−TP)·c_f | 281,776 per week | Excludes fixed wage cost (70,169) |
| Marginal band value | n_band·(mp·rho·L − c_m − (1−mp)·c_f) | +17.63 at 8.8k→11k; −3.73 at 17.6k→21k | Use marginal cost 11.07, not average 14.00 |
| Capacity in minutes | alerts × minutes per alert ÷ FTE minutes | 8,800×9.2/60 = 1,349.3 h; 44.98 FTE | Alert counts hide a 35x spread in handle time |
| Calibration error | ECE = Σ (n_b/N)·\|p_b − o_b\| | 0.000470 (13.1% of base rate) | Population weighting hides the top band |
| Threshold portability | logit shift = ln(odds(pi_new)/odds(pi_old)) | −0.697 for 0.004→0.002 | Valid only for equal-dispersion models, not reweighted GBMs |
