print(f"{'Sampling rate':>14}{'Nyquist':>10}  Note")
for rate in (25, 100, 250, 500, 1000):
    print(f"{rate:>11} Hz{rate/2:>7.0f} Hz  filter content above {rate/2:.0f} Hz")
print()
print("A 100 Hz ECG preserves 0 to 50 Hz. QRS timing survives; fine morphology")
print("does not. Without an anti-aliasing filter, higher content folds into the band.")
