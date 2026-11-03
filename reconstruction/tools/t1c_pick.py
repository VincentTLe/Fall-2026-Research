"""T1c: choose 7- and 8-camera Panoptic subsets with T1's rule (mid-height cameras, 0.5-1.5 m above
the hip; A = smallest largest azimuth gap, B and C = next best, each differing from A by >= 2 cameras),
plus 8 bunched = the 8 consecutive mid-height cameras spanning the narrowest arc."""
from itertools import combinations

AZ = {"00": 184.7, "03": 36.9, "05": 108.3, "08": 260.0, "11": 239.5, "12": 87.2, "16": 310.1, "18": 17.4,
      "20": 135.2, "21": 162.8, "22": 61.8, "23": 329.1, "27": 285.4, "29": 212.5, "30": 353.4}


def max_gap(cams):
    a = sorted(AZ[c] for c in cams)
    return max([b - x for x, b in zip(a, a[1:])] + [a[0] + 360 - a[-1]])


for n in (7, 8):
    ranked = sorted(combinations(sorted(AZ), n), key=lambda s: (max_gap(s), s))
    a = ranked[0]
    picks = [a]
    for s in ranked[1:]:
        if len(set(a) - set(s)) >= 2 and all(len(set(p) - set(s)) >= 1 for p in picks):
            picks.append(s)
        if len(picks) == 3:
            break
    for tag, s in zip(("", "_alt1", "_alt2"), picks):
        order = sorted(s, key=lambda c: AZ[c])
        print(f"s{n}{tag}|{' '.join(order)}|{max_gap(s):.0f}")
order = sorted(AZ, key=lambda c: AZ[c])
best = None
for i in range(len(order)):
    s = [order[(i + k) % len(order)] for k in range(8)]
    span = (AZ[s[-1]] - AZ[s[0]]) % 360
    if best is None or span < best[0]:
        best = (span, s)
print(f"c8bunch|{' '.join(best[1])}|span{best[0]:.0f}")
