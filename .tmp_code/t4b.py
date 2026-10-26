def cohen_kappa(matrix):
    n = sum(sum(row) for row in matrix)
    po = sum(matrix[i][i] for i in range(len(matrix))) / n
    row_totals = [sum(row) for row in matrix]
    col_totals = [sum(matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix))]
    pe = sum(row_totals[i] * col_totals[i] for i in range(len(matrix))) / (n * n)
    return (po - pe) / (1 - pe), po, pe

# rows: reader A absent/present; columns: reader B absent/present
matrix = [
    [812, 24],
    [38, 126],
]
k, po, pe = cohen_kappa(matrix)
n = sum(sum(row) for row in matrix)
print(f"Cases: {n}")
print(f"Observed agreement: {po:.3f}")
print(f"Chance agreement:   {pe:.3f}")
print(f"Cohen's kappa:      {k:.3f}")
print()
if k < 0.4:
    verdict = "poor"
elif k < 0.6:
    verdict = "moderate"
elif k < 0.8:
    verdict = "substantial"
else:
    verdict = "good"
print(f"Interpretation: {verdict} agreement on this sample.")
