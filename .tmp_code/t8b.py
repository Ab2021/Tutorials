def cohen_kappa(matrix):
    n = sum(sum(row) for row in matrix)
    po = sum(matrix[i][i] for i in range(len(matrix))) / n
    rows = [sum(row) for row in matrix]
    cols = [sum(matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix))]
    pe = sum(rows[i] * cols[i] for i in range(len(matrix))) / (n * n)
    return (po - pe) / (1 - pe), po

matrix = [
    [180, 20],
    [30, 70],
]
k, po = cohen_kappa(matrix)
print(f"Items: {sum(sum(r) for r in matrix)}")
print(f"Observed agreement: {po:.2f}")
print(f"Cohen's kappa: {k:.2f}")
