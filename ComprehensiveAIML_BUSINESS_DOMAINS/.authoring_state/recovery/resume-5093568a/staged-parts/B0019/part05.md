
## 6. Visual / Geometric Intuition

### 6.1 The alignment lattice

Word error rate is the shortest path through a grid. Put the reference tokens along the top edge, left to right, and the hypothesis tokens down the left edge, top to bottom. A cell `(i, j)` is the minimum cost of matching the first `i` reference tokens against the first `j` hypothesis tokens. Moves:

```text
 diagonal (i-1, j-1) -> (i, j)   cost 0 if r_i == h_j   (match)
                                 cost 1 if r_i != h_j   (substitution)
 vertical (i, j-1)   -> (i, j)   cost 1                 (insertion of h_j)
 horizontal(i-1, j)  -> (i, j)   cost 1                 (deletion of r_i)
 start at (0,0) = 0, take min over the three incoming moves
 WER = value at (N, H) / N
```

For the eight-word reference "schedule the follow up call for monday morning" against the nine-word hypothesis "schedule follow up call on monday morning at nine", the bottom-right cell is 4 and the path is: the reference token `the` deleted, `for` substituted by `on`, then `at` and `nine` inserted. So `S = 1, D = 1, I = 2, N = 8`, `WER = 4/8 = 0.5000`.

The geometry explains two properties that are otherwise mysterious. First, the grid is `(N+1) × (H+1)` cells, so cost grows as the product of the two lengths — a 4,500-token reference against a 4,500-token hypothesis is 20.25 million cells; keeping the full table at one 8-byte integer per cell costs `20,250,000 × 8 / 10^6 = 162` MB. Second, only the last row plus one rolling row is needed for the score (`2 × 4501 × 8 bytes = 70.3` KB), which is why batch scorers stream the table but path-reporting scorers often do not.

### 6.2 Why insertions and deletions are mirror images on the grid

A deletion is an upward move with no horizontal progress; an insertion is a leftward move with no vertical progress. They are the same operation viewed from opposite axes, which is why any asymmetry in their treatment has to be introduced deliberately — WER weights them equally, WIP and WIL reweight insertions only, and a scorer that silently drops trailing insertions is effectively applying a position-dependent weight nobody documented.

This is also why the alignment is not unique. Whenever two paths tie, the reported `S`, `D` and `I` decompose differently while `E` stays fixed. A scorer that reports only `E` is safe; a scorer that reports the split must state its tie-breaking rule (prefer substitution, prefer deletion, prefer earlier match), because a diagnostics dashboard comparing splits across two scorer versions will otherwise show a "surge in insertions" that is a tie-breaking change.

### 6.3 The distribution shape of per-utterance error

Plot the twelve per-utterance rates as a histogram with bins of width 0.1:

```text
 0.00 - 0.10  #####                (E02 0.0000, E10 0.0000, E09 0.0714,
                                    E01 0.0769, E07 0.0909)
 0.10 - 0.20  ####                 (E03 0.1000, E12 0.1000, E08 0.1111,
                                    E04 0.1818)
 0.20 - 0.30  #                    (E11 0.2727)
 0.30 - 0.40  #                    (E05 0.3333)
 0.40 - 0.50  (empty)
 0.50 - 0.60  #                    (E06 0.5000)
```

Sorted, the twelve rates are 0, 0, 0.0714, 0.0769, 0.0909, 0.1000, 0.1000, 0.1111, 0.1818, 0.2727, 0.3333, 0.5000; the median is the mean of the sixth and seventh values, `(0.1000 + 0.1000)/2 = 0.1000`. The shape is not a bell curve; it is a spike near 0.08 with a long right tail. Consequences that follow immediately from the shape, not from opinion:

- The mean (0.1532) sits above the median (0.1000), so quoting a mean overstates the typical request and quoting a median understates the aggregate text burden.
- The tail is where the product damage is. E06 alone contributes 4 of 18 errors, 22 percent of all errors, from 6.3 percent of the audio.
- A sample of twelve supports a statement about the centre and almost nothing about the tail. Claiming "p95 utterance WER is 0.5" from this data is arithmetic on one observation.

### 6.4 Time-space diagram of a streaming session

```text
 t=0.0  speech starts
        |---- audio arrives ---->
 t=0.4  first partial emitted            first-token latency 0.40 s
 t=0.9  partial revised (word changed)   instability event
 t=2.1  partial grows
 t=4.6  last speech frame                end of speech
 t=4.9  endpoint closes (tau = 0.3 s)
 t=5.4  final segment emitted            finalisation 0.80 s after speech end
        i.e. 0.50 s decode + 0.30 s tau
```

Read the diagram as two separate contracts. The horizontal distance from speech onset to first partial is a responsiveness contract; the vertical gap after the last speech frame is a completion contract. A single "latency" number conflates them, and a system can be excellent on the first and unacceptable on the second — which is exactly the incident described in 8.11.

### 6.5 The cost surface: where a corpus gets expensive

Two axes determine evaluation cost: audio hours and annotation depth. A cheap point is "transcribe 20 hours with a single pass"; an expensive point is "double-transcribe 200 hours with adjudication on every disagreement". The middle is where most useful work happens:

```text
                annotation depth ->
                time-align   text-only    double + adjudicated
   audio hours
     10         cheap        cheap        moderate
     50         moderate     moderate     expensive
    200         expensive    expensive    prohibitive without sampling
```

The practical rule that this surface encodes: depth buys reliability about *what* the ground truth is, while hours buy reliability about *how variable* the system is. Budgeting all of one and none of the other produces the two classic failures — a precise measurement of a wrong reference, or a noisy measurement of a right one.

### 6.6 Confidence calibration geometry

A reliability diagram plots predicted confidence on the horizontal axis against observed correctness on the vertical. Perfect calibration is the diagonal. Binned data from an ILLUSTRATIVE 800-token sample:

| Bin centre | Tokens | Correct | Observed accuracy | Gap from bin centre |
|---|---:|---:|---:|---:|
| 0.10 | 120 | 6 | 0.050 | 0.050 |
| 0.30 | 90 | 18 | 0.200 | 0.100 |
| 0.50 | 60 | 30 | 0.500 | 0.000 |
| 0.70 | 40 | 26 | 0.650 | 0.050 |
| 0.90 | 490 | 465 | 0.949 | 0.049 |

Expected calibration error is the token-weighted mean absolute gap: `(120/800)(0.050) + (90/800)(0.100) + (60/800)(0.000) + (40/800)(0.050) + (490/800)(0.049) = 0.0075 + 0.01125 + 0 + 0.0025 + 0.03001 = 0.05126`.

The shape matters more than the scalar: the model is under-confident in the 0.30 bin (predicts 0.30, delivers 0.20) and nearly calibrated at 0.90. A routing rule that sends tokens below 0.6 to human review is therefore doing something different from what its author intended — it will catch 0.30-bin tokens that are already more often wrong than claimed, and it will pass 0.90-bin tokens that fail about one time in twenty, which at 490 tokens is 25 wrong tokens shipped.

**Takeaway.** Geometry buys three intuitions: error is a shortest path, per-utterance error is heavy-tailed rather than bell-shaped, and latency is a two-contract object that a single number cannot express.
