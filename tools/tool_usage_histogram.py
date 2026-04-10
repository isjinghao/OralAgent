#!/usr/bin/env python3
"""
统计请求日志 JSON 中各类工具的使用频次，并绘制期刊论文风格的直方图。
按模态对工具分组，同一模态使用相同颜色。
"""

import json
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch

# 工具名称 -> ID 映射（可手动修改 value）
TOOL_ID_MAP = {
    "OralGPT-Omni": "T15",
    "intraoral_image_region-level_abnormality_detection": "T02",
    "intraoral_image_teeth_number_and_type_detection": "T06",
    "intraoral_image_gingivitis_detection": "T03",
    "intraoral_image_image-level_abnormality_classification": "T01",
    "intraoral_image_malocclusion_issues_detection": "T05",
    "intraoral_image_fenestration_detection": "T04",
    "periapical_xray_disease_segmentation": "T07",
    "periapical_xray_disease_7_classification": "T08",
    "cephalometric_xray_landmark_detection": "T09",
    "cytopathology_cell_nucleus_segmentation": "T13",
    "cytopathology_cell_nucleus_grading": "T14",
    "histopathology_oscc_segmentation": "T12",
    "histopathology_osmf_oscc_5_classification": "T10",
    "histopathology_leukoplakia_oscc_3_classification": "T11",
}

# 模态 -> 颜色（期刊风格，区分度高）
MODALITY_COLORS = {
    "OralGPT-Omni": "#1f77b4",           # 蓝
    "intraoral_image": "#2ca02c",        # 绿
    "periapical_xray": "#ff7f0e",        # 橙
    "cephalometric_xray": "#d62728",     # 红
    "cytopathology": "#9467bd",          # 紫
    "histopathology": "#8c564b",         # 棕
    "other": "#7f7f7f",                  # 灰（未归类工具）
}

# 图例文字：模态 key -> 图例中显示的文字（可随意修改）
MODALITY_LEGEND_LABELS = {
    "OralGPT-Omni": "OralGPT-Omni",
    "intraoral_image": "Intraoral image",
    "periapical_xray": "Periapical X-ray",
    "cephalometric_xray": "Cephalometric X-ray",
    "cytopathology": "Cytopathology",
    "histopathology": "Histopathology",
    "other": "Other",
}

# 期刊论文风格：高 DPI、合适字体
FIG_DPI = 300
FIG_WIDTH_INCH = 6.0
FIG_HEIGHT_INCH = 5.0
FONT_SIZE = 10
TICK_SIZE = 9
TITLE_SIZE = 12

# 图例矩形位置（axes 坐标，0~1）：(x, y) 为图例锚点，配合 LEGEND_LOC 使用
LEGEND_ANCHOR_X = 0.8
LEGEND_ANCHOR_Y = 0.8
LEGEND_LOC = "center"  # 如 "center", "upper left", "upper right" 等
LEGEND_FONTSIZE = 10   # 图例文字字号，改大则图例整体变大
# x 轴刻度文字（T01、T02…）整体右移量（单位：英寸，正数向右）
X_TICK_LABEL_OFFSET_INCH = 0.1


def _get_modality(tool_name: str) -> str:
    """根据工具名称推断模态（用于着色）。"""
    for prefix in ("intraoral_image_", "periapical_xray_", "cephalometric_xray_", "cytopathology_", "histopathology_"):
        if tool_name.startswith(prefix):
            return prefix.rstrip("_")
    if tool_name == "OralGPT-Omni":
        return "OralGPT-Omni"
    return "other"


def load_tool_counts(log_dir: str) -> tuple[Counter, int]:
    """遍历 log_dir 下所有 JSON，统计每个工具名出现次数，并返回 case 数。"""
    log_path = Path(log_dir)
    if not log_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {log_dir}")

    counter = Counter()
    case_count = 0
    json_files = list(log_path.glob("*.json"))

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        case_count += 1

        events = data.get("stream_events") or []
        for ev in events:
            for msg in ev.get("messages") or []:
                # 仅统计 AI 消息中的 tool_calls，避免与 execute 消息重复计数
                for tc in msg.get("tool_calls") or []:
                    name = tc.get("name")
                    if name:
                        counter[name] += 1

    return counter, case_count


def _shorten_name(name: str, max_len: int = 36) -> str:
    """缩短工具名用于坐标轴，过长则截断并加省略号。"""
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def plot_tool_histogram(counter: Counter, save_path: str) -> None:
    """绘制工具使用频次直方图（竖版），按模态着色，期刊风格。"""
    if not counter:
        raise ValueError("No tool usage data to plot.")

    # 按数量从左到右降序排列（数量多的在左）
    ordered = counter.most_common()
    names = [n for n, _ in ordered]
    counts = [c for _, c in ordered]
    # 横轴显示工具 ID（字典中的 value）；若缺失则用 "?"
    labels = [TOOL_ID_MAP.get(n, "?") for n in names]
    modalities = [_get_modality(n) for n in names]
    colors = [MODALITY_COLORS.get(m, MODALITY_COLORS["other"]) for m in modalities]

    # 期刊风格
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": TICK_SIZE,
        "figure.dpi": FIG_DPI,
    })

    n_tools = len(names)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCH, FIG_HEIGHT_INCH))

    x_pos = range(n_tools)
    bars = ax.bar(x_pos, counts, color=colors, edgecolor="#2d2d2d", linewidth=1.0, width=0.72)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=0, ha="right")
    ax.set_ylabel("Tool Usage Count")
    ax.set_ylim(0, max(counts) * 1.06)

    # 图例：按模态（去重并保持顺序）
    seen = set()
    legend_handles = []
    for m in modalities:
        if m not in seen:
            seen.add(m)
            legend_handles.append(Patch(facecolor=MODALITY_COLORS.get(m, MODALITY_COLORS["other"]), edgecolor="black", label=MODALITY_LEGEND_LABELS.get(m, m)))
    if legend_handles:
        leg = ax.legend(handles=legend_handles, loc=LEGEND_LOC, bbox_to_anchor=(LEGEND_ANCHOR_X, LEGEND_ANCHOR_Y), fontsize=LEGEND_FONTSIZE, frameon=True, edgecolor="#cccccc", fancybox=False)
        leg.get_frame().set_linewidth(0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.grid(axis="y", linestyle="--", alpha=0.45, color="#888888")
    ax.tick_params(axis="both", length=4, width=1, colors="#333333")

    # # x、y 轴末端加箭头（axes 坐标）
    # ax.annotate("", xy=(1.02, 0), xycoords="axes fraction", xytext=(1, 0), arrowprops=dict(arrowstyle="->", color="black", lw=1))
    # ax.annotate("", xy=(0, 1.02), xycoords="axes fraction", xytext=(0, 1), arrowprops=dict(arrowstyle="->", color="black", lw=1))

    plt.tight_layout()
    # x 轴刻度文字整体右移（在 tight_layout 之后设置）
    if X_TICK_LABEL_OFFSET_INCH != 0:
        offset = mtransforms.ScaledTranslation(X_TICK_LABEL_OFFSET_INCH, 0, fig.dpi_scale_trans)
        for lbl in ax.get_xticklabels():
            lbl.set_transform(lbl.get_transform() + offset)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tool usage histogram from request JSON logs.")
    parser.add_argument(
        "log_dir",
        nargs="?",
        default="/home/jinghao/projects/OralGPT-Agent/OralAgent/logs/requests/20260313_200958/range_20260314_090000_145959",
        help="Directory containing request JSON files",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output figure path (default: <log_dir>/tool_usage_histogram.png)",
    )
    args = parser.parse_args()
    
    log_dir = args.log_dir
    out_path = args.output
    if out_path is None:
        out_path = str(Path(log_dir) / "tool_usage_histogram.png")

    counter, case_count = load_tool_counts(log_dir)
    n_kinds = len(counter)
    total_calls = sum(counter.values())
    avg_tools_per_case = total_calls / case_count if case_count else 0.0
    print(f"工具种类数: {n_kinds}")
    print(f"case 数: {case_count}")
    print(f"总调用次数: {total_calls}")
    print(f"平均每个 case 调用工具数量: {avg_tools_per_case:.4f}")
    print(f"总调用次数/case 数: {avg_tools_per_case:.4f}")
    print("各工具调用数量:")
    for name, cnt in counter.most_common():
        tid = TOOL_ID_MAP.get(name, "?")
        print(f"  [{tid}] {name}: {cnt}")
    print("\nTOOL_ID_MAP (edit IDs in script as needed):")
    for k, v in sorted(TOOL_ID_MAP.items(), key=lambda x: x[0]):
        print(f"  {k!r}: {v!r}")

    plot_tool_histogram(counter, out_path)


if __name__ == "__main__":
    main()
