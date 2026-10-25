"""Part 1: public vs private leaderboard. Scatter of our 21 submissions, of all teams, and rank shift vs number of submissions.

  python scripts/plot_part1_shakeup.py

Inputs: docs/experiments/part1_submissions.csv (ours), docs/experiments/part1_public_vs_private_teams.csv (51 teams, organiser baselines removed).
Output: docs/experiments/part1_public_vs_private.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID, GRAY, BLUE, ORANGE = "#1f2328", "#59636e", "#e3e6ea", "#a7adb5", "#1f6feb", "#d9730d"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

ours = pd.read_csv(ROOT / "docs/experiments/part1_submissions.csv")
teams = pd.read_csv(ROOT / "docs/experiments/part1_public_vs_private_teams.csv")
fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)

# A: our submissions
a = ax[0]
probe = ours["submission"].str.contains("probe")
a.scatter(ours.public[~probe], ours.private[~probe], s=46, color=BLUE, zorder=3, label="our submissions (19)")
a.scatter(ours.public[probe], ours.private[probe], s=46, facecolors="white", edgecolors=BLUE, linewidths=1.6, zorder=3, label="level probes x0.9 / x1.1")
sel = ours[ours.submission.str.contains("SELECTED")].iloc[0]
best = ours.loc[ours.private.idxmax()]
a.scatter([sel.public], [sel.private], s=120, color=ORANGE, zorder=4, label="v10 (selected, best public)")
sl, ic, *_ = stats.linregress(ours.public, ours.private)
xs = np.linspace(ours.public.min() - 0.0005, ours.public.max() + 0.0005, 50)
a.plot(xs, ic + sl * xs, color=GRAY, lw=1.5, zorder=1)
arrow = dict(arrowstyle="-", color=MUTED, lw=0.8)
a.annotate("v10 (selected): 0.12887 / 0.13864", (sel.public, sel.private), (-150, -60), textcoords="offset points", ha="left", color=INK, fontsize=9, arrowprops=arrow)
a.annotate("v9 (best private): 0.12854 / 0.13906", (best.public, best.private), (-250, -8), textcoords="offset points", ha="left", color=INK, fontsize=9, arrowprops=arrow)
r_all, rho_all = stats.pearsonr(ours.public, ours.private)[0], stats.spearmanr(ours.public, ours.private)[0]
late = ours.iloc[9:]
a.text(0.0, 1.02, f"all 21: Pearson {r_all:.2f}, Spearman {rho_all:.2f}   |   last 12 (public range 0.0017): Pearson {stats.pearsonr(late.public, late.private)[0]:.2f}, Spearman {stats.spearmanr(late.public, late.private)[0]:.2f}", transform=a.transAxes, va="bottom", fontsize=8.5, color=MUTED)
a.set_xlabel("public score"); a.set_ylabel("private score"); a.set_title("A. Our 21 submissions", loc="left", fontweight="bold", color=INK, pad=22)
a.grid(color=GRID, lw=0.8, zorder=0); a.legend(loc="lower right", frameon=False, fontsize=9)

# B: all teams, zoom on the competitive cluster
b = ax[1]
sl2, ic2, *_ = stats.linregress(teams.public, teams.private)
top = teams[teams.public_rank <= 25]
sd = top.res.std()
inside = teams[teams.public >= 0.1255]
b.scatter(inside.public, inside.private, s=34, color=GRAY, zorder=2, label=f"other teams ({len(inside) - 1} of {len(teams) - 1} shown)")
us = teams[teams.teamName == "Kohei Kudo"]
b.scatter(us.public, us.private, s=110, color=ORANGE, zorder=4, label="us (rank 14 public, 3 private)")
xs = np.linspace(0.1255, 0.1308, 50)
b.plot(xs, ic2 + sl2 * xs, color=MUTED, lw=1.4, zorder=1)
b.fill_between(xs, ic2 + sl2 * xs - sd, ic2 + sl2 * xs + sd, color=BLUE, alpha=0.10, zorder=0, label=f"trend +/- 1 SD of the top-25 residual ({sd:.4f})")
offs = {"overfitter...": (6, -12), "Lion is a Cat": (-8, 8), "Tevfik Doğukan Işık": (-122, -4), "Naz76": (6, 3), "SnorlaxCode": (-30, 8), "Manuel Rodríguez": (-40, 12), "kaiY": (-58, 8), "BenDavid313": (6, -10)}
for name, off in offs.items():
    r = teams[teams.teamName == name].iloc[0]
    b.annotate(f"{name} ({int(r.public_rank)}→{int(r.private_rank)})", (r.public, r.private), off, textcoords="offset points", fontsize=8, color=INK)
b.set_xlabel("public score"); b.set_ylabel("private score"); b.set_title("B. All teams (organiser baselines excluded)", loc="left", fontweight="bold", color=INK, pad=34)
rho = stats.spearmanr(teams.public, teams.private)[0]; rho25 = stats.spearmanr(top.public, top.private)[0]
b.text(0.0, 1.02, f"all 51 teams: Pearson {stats.pearsonr(teams.public, teams.private)[0]:.3f}, Spearman {rho:.2f}; public top 25 only: Spearman {rho25:.2f}\nprivate runs about {(teams.private - teams.public).mean():.4f} above public; labels: public rank → private rank", transform=b.transAxes, va="bottom", fontsize=8.5, color=MUTED)
b.grid(color=GRID, lw=0.8, zorder=0); b.legend(loc="lower right", frameon=False, fontsize=8.5)

# C: rank shift vs number of submissions
c = ax[2]
c.axhline(0, color=MUTED, lw=1)
c.scatter(teams.n_sub, teams["shift"], s=34, color=GRAY, zorder=2, label="other teams")
c.scatter(us.n_sub, us["shift"], s=110, color=ORANGE, zorder=4, label="us")
offs_c = {"overfitter...": (-20, 8), "Lion is a Cat": (6, -3), "Tevfik Doğukan Işık": (-118, 6), "Naz76": (6, -3), "Simone": (6, -3), "kaiY": (6, 4), "BenDavid313": (-34, -14), "CST": (-14, 8)}
for name, off in offs_c.items():
    r = teams[teams.teamName == name].iloc[0]
    c.annotate(name, (r.n_sub, r["shift"]), off, textcoords="offset points", fontsize=8, color=INK)
c.set_xlabel("number of submissions made"); c.set_ylabel("places moved, public rank − private rank (up is better)")
c.set_title("C. Who moved? Rank shift vs submissions made", loc="left", fontweight="bold", color=INK, pad=22)
c.text(0.0, 1.02, f"signed shift vs submissions: Spearman {stats.spearmanr(teams.n_sub, teams['shift'])[0]:.2f}   |   |shift| vs submissions: Spearman {stats.spearmanr(teams.n_sub, teams['shift'].abs())[0]:.2f}", transform=c.transAxes, va="bottom", fontsize=8.5, color=MUTED)
c.grid(color=GRID, lw=0.8, zorder=0); c.legend(loc="lower left", frameon=False, fontsize=9)
fig.savefig(ROOT / "docs/experiments/part1_public_vs_private.png", dpi=150, facecolor="white")
print("saved")
