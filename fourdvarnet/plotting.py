"""SciencePlots helpers for paper / report figures (Times New Roman + CJK)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Import registers scienceplots styles with matplotlib.
import scienceplots  # noqa: F401

# Latin body: Times New Roman. Chinese labels/captions: CJK serif/sans fallback.
# Windows typically has SimSun / Microsoft YaHei; keep DejaVu as last resort.
_CJK_SERIF = (
    "SimSun",
    "Noto Serif CJK SC",
    "Source Han Serif SC",
    "FangSong",
    "KaiTi",
)
_CJK_SANS = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "DengXian",
)


def _available_font(candidates: Sequence[str]) -> str | None:
    from matplotlib import font_manager as fm

    names = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in names:
            return name
    return None


def apply_science_style(*, fontsize: float = 10.0, chinese: bool = True) -> None:
    """Apply SciencePlots + Times New Roman; optional CJK fallback for Chinese text.

    Font policy (documented for report/paper figures):
    - Latin / math: Times New Roman (SciencePlots + IEEE-like).
    - Chinese glyphs: SimSun (serif) or Microsoft YaHei (sans) as fallback in
      ``font.serif`` / ``axes.unicode_minus=False`` so mixed CN/EN titles render.
    - DPI default 300 for manuscript / self-contained HTML embedding.
    """
    # Prefer IEEE/science; fall back if a style name is missing.
    for candidate in (["science", "ieee"], ["science"], ["seaborn-v0_8-whitegrid"]):
        try:
            plt.style.use(candidate)
            break
        except OSError:
            continue

    serif = ["Times New Roman", "Times", "DejaVu Serif"]
    sans = ["DejaVu Sans", "Arial"]
    if chinese:
        cjk_serif = _available_font(_CJK_SERIF)
        cjk_sans = _available_font(_CJK_SANS)
        if cjk_serif:
            serif = ["Times New Roman", cjk_serif, "Times", "DejaVu Serif"]
        if cjk_sans:
            sans = [cjk_sans, "DejaVu Sans", "Arial"]

    mpl.rcParams.update(
        {
            # SciencePlots may enable usetex; no system LaTeX on this host.
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": serif,
            "font.sans-serif": sans,
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "font.size": fontsize,
            "axes.titlesize": fontsize + 1,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            "legend.fontsize": fontsize - 1,
            "figure.titlesize": fontsize + 2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: mpl.figure.Figure, stem: Path, formats: Sequence[str] = ("png", "pdf")) -> list[Path]:
    """Save figure to stem.<ext> for each format; return written paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for ext in formats:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path)
        out.append(path)
    return out


def bar_compare(
    labels: Sequence[str],
    series: dict[str, Sequence[float]],
    *,
    ylabel: str,
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> mpl.figure.Figure:
    """Grouped bar chart for ablation / baseline comparison."""
    apply_science_style()
    x = np.arange(len(labels))
    n = max(len(series), 1)
    width = min(0.8 / n, 0.35)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for i, (name, vals) in enumerate(series.items()):
        offset = (i - (n - 1) / 2.0) * width
        ax.bar(x + offset, list(vals), width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels))
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def loss_curves(
    epochs: Sequence[int],
    train: Sequence[float],
    val: Sequence[float],
    *,
    title: str | None = None,
) -> mpl.figure.Figure:
    """Train/val loss vs epoch."""
    apply_science_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(epochs, train, label="Train", lw=1.6)
    ax.plot(epochs, val, label="Val", lw=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def mirror_to(paths: Iterable[Path], dest_dir: Path) -> list[Path]:
    """Copy already-written figure files into dest_dir (same filenames)."""
    import shutil

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for p in paths:
        p = Path(p)
        if not p.is_file():
            continue
        tgt = dest_dir / p.name
        shutil.copy2(p, tgt)
        copied.append(tgt)
    return copied
