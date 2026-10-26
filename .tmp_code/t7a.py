W, H = 100_000, 100_000
tile = 256
for stride in (256, 128):
    nx = (W - tile) // stride + 1
    ny = (H - tile) // stride + 1
    print(f"stride={stride:>3} px  tiles={nx*ny:>8,}  overlap={(1-stride/tile):.0%}")
print()
tissue = 0.35
nx = (W - tile) // 256 + 1
ny = (H - tile) // 256 + 1
print(f"Full grid at stride 256: {nx*ny:,} tiles")
print(f"Tissue tiles at 35% tissue: {round(nx*ny*tissue):,}")
