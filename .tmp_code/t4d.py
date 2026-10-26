slides = [
    ("slide_01", 2, 1200),
    ("slide_02", 0, 900),
    ("slide_03", 45, 1500),
    ("slide_04", 0, 1100),
    ("slide_05", 1, 800),
]
image_positive = sum(1 for _, pos, total in slides if pos > 0)
n_images = len(slides)
total_tiles = sum(total for _, _, total in slides)
positive_tiles = sum(pos for _, pos, _ in slides)

print(f"{'Slide':<10}{'Positive tiles':>16}{'Total tiles':>13}{'Tile prevalence':>17}")
for name, pos, total in slides:
    print(f"{name:<10}{pos:>16}{total:>13}{pos/total:>17.1%}")
print()
print(f"Image-level prevalence: {image_positive/n_images:.1%} ({image_positive}/{n_images})")
print(f"Tile-level prevalence:  {positive_tiles/total_tiles:.1%} ({positive_tiles}/{total_tiles})")
print()
print("A model trained on uniformly sampled tiles sees a prevalence close to the")
print("tile-level value. A model evaluated at the image level sees the image-level")
print("value. Reporting one as the other overstates or understates the base rate.")
