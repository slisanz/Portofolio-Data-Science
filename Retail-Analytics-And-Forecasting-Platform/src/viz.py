from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#6A994E", "#E5C687", "#5DADE2"]

BG = "#0E1117"
TEXT = "#FAFAFA"


def set_style() -> None:
    sns.set_theme(style="darkgrid", context="notebook", palette=PALETTE)
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "grid.color": TEXT,
            "grid.alpha": 0.12,
            "legend.facecolor": BG,
            "legend.edgecolor": TEXT,
            "legend.labelcolor": TEXT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.figsize": (9, 5),
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "savefig.dpi": 120,
        }
    )


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def style_dark_axes(fig=None) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.text as mtext

    if fig is None:
        fig = plt.gcf()
    fig.patch.set_facecolor(BG)
    for txt in fig.findobj(mtext.Text):
        txt.set_color(TEXT)
    for ax in fig.axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, which="both")
        for spine in ax.spines.values():
            spine.set_color(TEXT)
            spine.set_alpha(0.3)
