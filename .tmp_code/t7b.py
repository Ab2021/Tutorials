pixels = 100_000 * 100_000
bytes_rgb = 3
raw_gb = pixels * bytes_rgb / 1e9
print(f"Pixels: {pixels:,} ({pixels/1e9:.0f} gigapixels)")
print(f"Raw RGB: {raw_gb:.1f} GB")
for ratio in (8, 10, 20):
    print(f"Compressed at {ratio}:1 -> {raw_gb/ratio:.2f} GB")
print()
print("A 10-gigapixel slide is 10^10 pixels. Storage planning starts there.")
