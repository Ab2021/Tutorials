"""Story bank coverage and selection, standard library only.

Run: python story_bank.py
Given stories tagged with the competencies they evidence strongly, reports the
coverage of each competency, identifies gaps, and selects a minimal subset of
stories that covers every competency (greedy set cover).
"""

COMPETENCIES = [
    "ownership",
    "influence",
    "developing_others",
    "ambiguity",
    "integrity",
    "prioritisation",
]

STORIES = {
    "S1 database lock delay": ["ownership", "prioritisation"],
    "S2 sponsor scope change": ["influence", "prioritisation"],
    "S3 mentoring two coordinators": ["developing_others"],
    "S4 audit finding": ["ownership", "integrity"],
    "S5 EDC migration": ["ambiguity", "prioritisation"],
    "S6 site conflict": ["influence"],
    "S7 query backlog": ["ownership", "prioritisation"],
    "S8 unblinding request": ["integrity"],
    "S9 endpoint definition": ["ambiguity", "influence"],
    "S10 training programme": ["developing_others", "influence"],
}


def coverage(stories, competencies):
    counts = {c: 0 for c in competencies}
    for tags in stories.values():
        for tag in tags:
            if tag in counts:
                counts[tag] += 1
    return counts


def greedy_cover(stories, competencies):
    uncovered = set(competencies)
    chosen = []
    while uncovered:
        best_name, best_gain = None, set()
        for name, tags in stories.items():
            gain = set(tags) & uncovered
            if len(gain) > len(best_gain):
                best_name, best_gain = name, gain
        if not best_gain:
            break
        chosen.append((best_name, sorted(best_gain)))
        uncovered -= best_gain
    return chosen, uncovered


def main():
    print("COVERAGE")
    counts = coverage(STORIES, COMPETENCIES)
    for comp in COMPETENCIES:
        bar = "#" * counts[comp]
        print(f"  {comp:18s} {counts[comp]:2d} {bar}")
    gaps = [c for c, n in counts.items() if n == 0]
    print(f"  gaps: {', '.join(gaps) if gaps else 'none'}")
    print(f"  stories: {len(STORIES)}")
    print(f"  mean tags per story: "
          f"{sum(len(v) for v in STORIES.values()) / len(STORIES):.1f}")

    chosen, uncovered = greedy_cover(STORIES, COMPETENCIES)
    print()
    print("MINIMAL COVERING SET (greedy)")
    for name, gain in chosen:
        print(f"  {name:32s} covers {', '.join(gain)}")
    print(f"  stories selected: {len(chosen)} of {len(STORIES)}")
    if uncovered:
        print(f"  UNCOVERED: {', '.join(sorted(uncovered))}")
    else:
        print("  all competencies covered")

    print()
    print("REDUNDANCY")
    for comp in COMPETENCIES:
        print(f"  {comp:18s} {counts[comp]} stories "
              f"({'thin' if counts[comp] < 2 else 'redundant'})")


if __name__ == "__main__":
    main()
