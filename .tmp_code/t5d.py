events = [
    ("e1", 10, 10),
    ("e2", 8, 1),
    ("e3", 12, 11),
    ("e4", 6, 0),
    ("e5", 15, 14),
]
total_windows = sum(w for _, w, d in events)
detected_windows = sum(d for _, w, d in events)
event_detected = sum(1 for _, w, d in events if d > 0)

print(f"{'Event':<6}{'Windows':>9}{'Detected':>10}{'Window sensitivity':>20}")
for name, w, d in events:
    print(f"{name:<6}{w:>9}{d:>10}{d/w:>20.0%}")
print()
print(f"Window-level sensitivity: {detected_windows/total_windows:.1%}")
print(f"Event-level sensitivity:  {event_detected/len(events):.1%}")
print()
print("A missed event with many windows can be hidden by window-level averaging.")
print("Report event-level sensitivity and the number of missed events.")
