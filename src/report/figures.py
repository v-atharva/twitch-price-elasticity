"""Figure generation. Every figure is regenerable from `make figures`.

Style follows the repo's dataviz conventions: recessive hairline grid, ink-token
text (never series-colored text), 2px lines, >=8px markers with a surface ring,
area washes at ~10% opacity, single-series charts carry no legend box.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# palette (light mode) — see PLAN.md / dataviz reference
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_1 = "#2a78d6"  # blue: the estimate
SERIES_2 = "#008300"  # green: known truth overlay (synthetic only)


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1)
    ax.grid(axis="y", color=GRID, linewidth=1, alpha=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)


def dose_response_figure(
    by_country: pd.DataFrame,  # country, dlogp, att, se
    epsilon: float,
    out_path: str | Path,
    *,
    subtitle: str,
) -> Path:
    """Per-country ATT vs price dose, WLS-through-origin elasticity line."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    ax.axvline(0, color=BASELINE, linewidth=1, zorder=1)

    xs = np.linspace(min(by_country["dlogp"].min() * 1.1, -0.05), 0.0, 50)
    ax.plot(xs, epsilon * xs, color=SERIES_1, linewidth=2, zorder=3)
    ax.errorbar(
        by_country["dlogp"], by_country["att"],
        yerr=1.96 * by_country["se"], fmt="none", ecolor=SERIES_1, elinewidth=1.5,
        alpha=0.55, capsize=3, zorder=4,
    )  # fmt: skip
    ax.scatter(
        by_country["dlogp"], by_country["att"],
        s=52, color=SERIES_1, edgecolors=SURFACE, linewidths=2, zorder=5,
    )  # fmt: skip
    for r in by_country.itertuples():
        ax.annotate(
            str(r.country), (r.dlogp, r.att), textcoords="offset points", xytext=(8, 6),
            fontsize=9, color=INK_2,
        )  # fmt: skip

    ax.set_xlabel("Price dose: log(new price / old price) at rollout", color=INK_2, fontsize=10)
    ax.set_ylabel("ATT on log paid subs", color=INK_2, fontsize=10)
    fig.suptitle(
        "Bigger price cuts, bigger subscription gains",
        x=0.07, y=0.97, ha="left", fontsize=13, fontweight=600, color=INK,
    )  # fmt: skip
    ax.set_title(subtitle, loc="left", fontsize=9, color=INK_2, pad=10)
    fig.tight_layout(rect=(0.01, 0.0, 1.0, 0.93))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def event_study_figure(
    event_study: pd.DataFrame,
    out_path: str | Path,
    *,
    title: str,
    subtitle: str,
    truth: pd.DataFrame | None = None,  # optional [rel_period, true_att] overlay
    ylabel: str = "ATT, log points",
) -> Path:
    """The money plot: dynamic effects around treatment with confidence band."""
    df = event_study.sort_values("rel_period")

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    # reference lines: zero effect, and the treatment boundary between t=-1 and t=0
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    ax.axvline(-0.5, color=BASELINE, linewidth=1, zorder=1)

    # confidence band as a wash, estimate as line + ringed dots
    ax.fill_between(
        df["rel_period"], df["ci_low"], df["ci_high"],
        color=SERIES_1, alpha=0.12, linewidth=0, zorder=2,
    )  # fmt: skip
    if truth is not None:
        tr = truth.sort_values("rel_period")
        ax.plot(
            tr["rel_period"], tr["true_att"],
            color=SERIES_2, linewidth=2, solid_joinstyle="round", zorder=3,
            label="true effect (known by construction)",
        )  # fmt: skip
    ax.plot(
        df["rel_period"], df["att"],
        color=SERIES_1, linewidth=2, solid_joinstyle="round", zorder=4,
        label="Callaway–Sant'Anna estimate",
    )  # fmt: skip
    ax.scatter(
        df["rel_period"], df["att"],
        s=42, color=SERIES_1, edgecolors=SURFACE, linewidths=2, zorder=5,
    )  # fmt: skip

    ax.set_xlabel("Months since local pricing arrived", color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.text(
        -0.25, 0.02, "← local pricing arrives", transform=ax.get_xaxis_transform(),
        ha="left", va="bottom", color=MUTED, fontsize=8.5,
    )  # fmt: skip

    if truth is not None:
        leg = ax.legend(
            loc="upper left", frameon=False, fontsize=9, handlelength=1.6, borderaxespad=0.2
        )
        for text in leg.get_texts():
            text.set_color(INK_2)

    fig.suptitle(title, x=0.065, y=0.97, ha="left", fontsize=13, fontweight=600, color=INK)
    ax.set_title(subtitle, loc="left", fontsize=9.5, color=INK_2, pad=10)
    fig.tight_layout(rect=(0.01, 0.0, 1.0, 0.94))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out
