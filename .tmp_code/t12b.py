import random
random.seed(7)
teams = 50
trials = 2000
true_mean, sd = 0.70, 0.02
maxima = []
for _ in range(trials):
    scores = [random.gauss(true_mean, sd) for _ in range(teams)]
    maxima.append(max(scores))
expected_max = sum(maxima) / len(maxima)
print(f"Teams: {teams}")
print(f"True mean: {true_mean:.2f}, sd: {sd:.2f}")
print(f"Expected best of {teams}: {expected_max:.3f}")
print(f"Selection inflation: {expected_max - true_mean:+.3f}")
print()
print("With many teams and a noisy metric, the winner's score overstates the true skill.")
