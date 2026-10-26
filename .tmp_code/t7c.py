print(f"{'Magnification':>15}{'um/px':>8}{'Pixels per 25 mm':>18}{'Tiles at 256':>15}")
for mag, mpp in ((40, 0.25), (20, 0.5), (10, 1.0)):
    px = int(25_000 / mpp)
    tiles = (px // 256) ** 2
    print(f"{mag:>13}x{mpp:>8.2f}{px:>18,}{tiles:>15,}")
print()
print("Halving magnification quarters the pixel count and the tile count.")
