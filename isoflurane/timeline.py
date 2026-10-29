"""One isoflurane session over time: dose, heart rate, breathing rate, perfusion index, skin temperature.

Lines join values at most 1 min apart; a value with no neighbour to join is a dot.
"""
import matplotlib.pyplot as plt
import numpy as np

import sessions
import panels  # after sessions, which puts the main folder on the path

SESSION = "26 Oct, 30 g"

v = sessions.vitals(SESSION)
stretches = sessions.epochs(v)
print(SESSION + ", median of the last 2 min at each dose (quartiles):")
for level, start, end in stretches:
    line = "  %g %% (%.1f-%.1f min):" % (level, start / 60, end / 60)
    for key, unit in [("hr", "beats/min"), ("breathing", "breaths/min"), ("pi", "%"), ("temp", "°C")]:
        s = sessions.steady(v, key, start, end)
        line += "  %s -" % key if s is None else "  %s %.3g (%.3g-%.3g) %s" % (key, *s, unit)
    print(line)

panels.figure_style()
rows = [("hr", "Heart rate\n(beats/min)"),
        ("breathing", "Breathing rate\n(breaths/min)"),
        ("pi", "Perfusion index\n(%)"),
        ("temp", "Skin temperature\n(°C)")]
fig, axes = plt.subplots(len(rows) + 1, 1, figsize=(6.3, 7.2), sharex=True, layout="constrained",
                         gridspec_kw={"height_ratios": [0.55] + [1] * len(rows)})

ax = axes[0]
when = [c[0] / 60 for c in v["changes"]] + [v["end"] / 60]
level = [c[1] for c in v["changes"]]
ax.stairs(level, when, color=panels.INK, linewidth=1.2, baseline=None)
ax.stairs(level, when, color=panels.HAIRLINE, fill=True, alpha=0.6)
for lvl, start, end in stretches:
    ax.text((start + end) / 120, lvl, "%g %%" % lvl, ha="center", va="bottom", color=panels.INK_2)
ax.set_ylabel("Isoflurane\n(%)")
ax.set_ylim(0, np.nanmax(level) * 1.35)
ax.set_yticks(sorted(set(lvl for lvl in level if not np.isnan(lvl))))
ax.grid(False)
panels.letter(ax, "A")

for i, (key, label) in enumerate(rows):
    ax = axes[i + 1]
    t, y = sessions.with_gaps(v, key)
    if len(y):
        ax.plot(t, y, color=panels.BLUE)
        alone = np.isnan(np.r_[np.nan, y[:-1]]) & np.isnan(np.r_[y[1:], np.nan])
        ax.plot(t[alone], y[alone], "o", color=panels.BLUE, markersize=2, markeredgewidth=0)
    else:
        ax.text(0.5, 0.5, "no values (no usable PPG)", transform=ax.transAxes, ha="center", color=panels.INK_2)
    ax.set_ylabel(label)
    panels.letter(ax, "BCDE"[i])

for ax in axes:
    for lvl, start, end in stretches[1:]:
        ax.axvline(start / 60, color=panels.AXIS, linewidth=0.5, zorder=0)
axes[-1].set_xlabel("Time (min)")
axes[-1].set_xlim(0, v["end"] / 60)
fig.align_ylabels(axes)

panels.save(fig, __file__, "timeline")
plt.show()
