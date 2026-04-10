#!/usr/bin/env python3
"""
Parse OralQA-ZH eval text -> plot category count pie chart.

Intended for paper submission (hi-res png + vector pdf).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


# --- cfg: match typical paper style ---
matplotlib.rcParams["axes.unicode_minus"] = False

# Prefer a CJK-capable sans font; fall back gracefully.
_font_candidates = [
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
]
_selected_font = _font_candidates[-1]
for _f in _font_candidates:
    try:
        # use FontProperties for robust lookup
        fp = fm.FontProperties(family=_f)
        fm.findfont(fp, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = ["sans-serif"]
        matplotlib.rcParams["font.sans-serif"] = [_f]
        _selected_font = _f
        break
    except Exception:
        continue

FONT_PROP = fm.FontProperties(family=_selected_font)

# Times New Roman (for English abbreviations in legend).
_roman_candidates = [
    "Times New Roman",
    "Times",
    "DejaVu Serif",
    "Liberation Serif",
]
_selected_roman = _roman_candidates[-1]
for _rf in _roman_candidates:
    try:
        fp = fm.FontProperties(family=_rf)
        fm.findfont(fp, fallback_to_default=False)
        _selected_roman = _rf
        break
    except Exception:
        continue
FONT_PROP_ROMAN = fm.FontProperties(family=_selected_roman)

# prefer direct font file binding (tends to remove CJK missing-glyph warnings)
_noto_ttc = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    if Path(_noto_ttc).exists():
        fm.fontManager.addfont(_noto_ttc)
        FONT_PROP = fm.FontProperties(fname=_noto_ttc)
except Exception:
    pass

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 600  # hi-res
plt.rcParams["savefig.bbox"] = "tight"


@dataclass(frozen=True)
class CatRow:
    category: str
    count: int
    score: float | None = None


_ROW_RE = re.compile(
    r"^\s*(\d+)\s+"  # idx
    r"(.+?)\s+"  # category (greedy handled by final fields)
    r"(\d+)\s+"  # count
    r"([0-9]+(?:\.[0-9]+)?)\s*$"  # score (float or int)
)


def _parse_rows(lines: Iterable[str], include_overall: bool) -> List[CatRow]:
    rows: List[CatRow] = []
    for line in lines:
        m = _ROW_RE.match(line)
        if not m:
            continue
        _idx, cat, count_s, score_s = m.groups()
        cat = cat.strip()
        if not include_overall and cat.lower() == "overall":
            continue
        try:
            count = int(count_s)
        except ValueError:
            continue
        try:
            score = float(score_s)
        except ValueError:
            score = None
        rows.append(CatRow(category=cat, count=count, score=score))

    # Keep original order as read; if duplicates exist, preserve first.
    seen = set()
    uniq: List[CatRow] = []
    for r in rows:
        if r.category in seen:
            continue
        seen.add(r.category)
        uniq.append(r)
    return uniq


def _make_colors(n: int) -> List[str]:
    # 中等清新风：清爽但不过浅，适合论文屏幕与打印。
    base = [
        "#5FA8D3",  # fresh blue
        "#59A14F",  # fresh green
        "#F28E8C",  # soft coral
        "#76B7B2",  # teal
        "#9C93D5",  # soft violet
        "#EDC948",  # warm yellow
        "#86BCB6",  # mint teal
        "#E39C6B",  # apricot
        "#A0CBE8",  # light azure
        "#8CD17D",  # light green
        "#B699D8",  # lilac
    ]
    if n <= len(base):
        return base[:n]
    # Extend using a qualitative cmap (rare, but keep robust).
    cmap = matplotlib.colormaps.get_cmap("tab20")
    return [matplotlib.colors.to_hex(cmap(i / max(1, n - 1))) for i in range(n)]


def plot_pie_from_rows(
    rows: List[CatRow],
    out_png: Path,
    out_pdf: Path,
    title: str | None = None,
) -> None:
    if not rows:
        raise ValueError("No category rows parsed; check input format.")

    cats = [r.category for r in rows]
    counts = [r.count for r in rows]
    total = sum(counts)

    # VIZ: pretty pie with legend showing counts.
    colors = _make_colors(len(rows))
    fig, ax = plt.subplots(figsize=(10.5, 7.2))

    def _autopct(pct: float) -> str:
        return f"{pct:.1f}%" if pct >= 1.0 else ""

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=None,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=_autopct,
        pctdistance=0.72,
        wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
    )

    for t in autotexts:
        t.set_color("white")
        t.set_weight("bold")
        t.set_fontsize(15)

    legend_cats = [_display_name(c) for c in cats]
    # Legend labels: remove parenthesized details (e.g. "(n=..., xx%)").
    legend_labels = [f"{c}" for c in legend_cats]
    legend = ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        # Move legend closer to the pie.
        bbox_to_anchor=(0.9, 0.5),
        # borderaxespad=0.0,
        frameon=True,
        prop=FONT_PROP_ROMAN,
        handlelength=2.5,
        handleheight=1.8,
        labelspacing=0.5,
        handletextpad=0.6,
        )
    # Make legend font size deterministic (avoid prop/rcParams overriding).
    for t in legend.get_texts():
        t.set_fontproperties(FONT_PROP_ROMAN)
        t.set_fontsize(12)

    # if title:
    #     ax.set_title(title, fontsize=13, pad=14, fontproperties=FONT_PROP)

    # ax.set_aspect("equal")
    # fig.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.15)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _default_oralqa_zh_rows() -> List[CatRow]:
    # hard-coded from the provided OralQA-ZH table (cnt + score)
    return [
        CatRow("牙体牙髓病学", 133, 63.9098),
        CatRow("牙周病学", 89, 80.8989),
        CatRow("口腔颌面外科学", 180, 60.0),
        CatRow("口腔修复学", 168, 51.1905),
        CatRow("口腔正畸学", 24, 25.0),
        CatRow("口腔黏膜病学", 54, 77.7778),
        CatRow("儿童口腔医学", 28, 60.7143),
        CatRow("口腔颌面医学影像诊断学", 12, 50.0),
        CatRow("口腔预防医学", 41, 58.5366),
        CatRow("口腔流行病学", 29, 72.4138),
        CatRow("口腔组织病理学", 40, 80.0),
    ]


# OralQA-ZH Abbrev mapping (for journal-style annotation)
CAT2ABBREV = {
    "儿童口腔医学": "PedDent",
    "口腔修复学": "Prosth",
    "口腔流行病学": "OralEpi",
    "口腔组织病理学": "OMFP",
    "口腔预防医学": "PrevDent",
    "口腔颌面医学影像诊断学": "OMFR",
    "口腔颌面外科学": "OMFS",
    "口腔黏膜病学": "OMD",
    "牙体牙髓病学": "Endo",
    "牙周病学": "Perio",
    "口腔正畸学": "Ortho",
}


def _display_name(cat: str) -> str:
    """Only use English abbreviations in figure annotations."""
    return CAT2ABBREV.get(cat, cat)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=str,
        required=False,
        default=None,
        help="Eval text file containing the category table.",
    )
    p.add_argument(
        "--out_base",
        type=str,
        default="oralqa_zh_category_count_pie",
        help="Output base name (without extension).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Where to save figure outputs.",
    )
    p.add_argument("--title", type=str, default="OralQA-ZH category count distribution")
    p.add_argument(
        "--include_overall",
        action="store_true",
        help="Include the 'Overall' row if present.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{args.out_base}.png"
    out_pdf = out_dir / f"{args.out_base}.pdf"

    if args.input is None:
        # IO/parse bypass: use benchmark snapshot (self-contained)
        rows = _default_oralqa_zh_rows()
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(str(input_path))
        # IO + parse
        lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        rows = _parse_rows(lines, include_overall=args.include_overall)

    # VIZ
    plot_pie_from_rows(rows, out_png=out_png, out_pdf=out_pdf, title=args.title)

    # done
    print(f"[OK] Saved: {out_png}")
    print(f"[OK] Saved: {out_pdf}")


if __name__ == "__main__":
    main()

