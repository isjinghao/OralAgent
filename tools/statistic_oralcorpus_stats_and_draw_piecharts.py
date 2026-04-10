#!/usr/bin/env python3
"""
OralCorpus 语料统计与饼图绘制脚本
- 统计原始中/英文数据条数（按 Init_lang / 原版语言）
- 统计各 Subject 及对应书籍数量
- 统计全部内容的 token 数（GPT tokenizer），并按语料来源（CH/EN）划分用于饼图
- 输出两张饼图（Subject 仅英文；Token 分布带 M 单位、图例在饼内），适用于期刊论文

依赖: pip install tiktoken matplotlib
"""

import json
import os
from pathlib import Path
from collections import defaultdict

import tiktoken
import matplotlib.pyplot as plt
import matplotlib

# 期刊论文风格：使用 LaTeX 字体、高 DPI、白底
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "SimSun", "serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

# 论文友好的中等清新风：比马卡龙更稳重，比医疗高对比更柔和
PALETTE_FRESH_BALANCED = [
    "#5FA8D3",  # fresh sky blue (for largest first slice)
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
    "#F1CE63",  # sand yellow
]

CORPUS_ROOT = Path("/home/jinghao/projects/OralGPT-Agent/Corpus/OralCorpus")
CH_DIR = CORPUS_ROOT / "CH"
EN_DIR = CORPUS_ROOT / "EN"
OUTPUT_DIR = CORPUS_ROOT / "stats_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_tiktoken_encoder():
    """使用 GPT 常用 tokenizer (cl100k_base, 用于 GPT-3.5/4)"""
    return tiktoken.get_encoding("cl100k_base")


def repeat_palette(n, palette):
    """按需循环扩展调色板，保证类别数超过预设颜色时仍可绘图。"""
    if n <= 0:
        return []
    return [palette[i % len(palette)] for i in range(n)]


def iter_jsonl_records(dir_path, content_key="内容", lang_key="原版语言", subject_key="学科", subject_id_key="学科_ID"):
    """遍历目录下所有 .jsonl，逐行 yield (content, lang, subject, subject_id)。"""
    for p in sorted(dir_path.glob("*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = obj.get(content_key) or obj.get("Content") or ""
                lang = obj.get(lang_key) or obj.get("Init_lang") or ""
                subject = obj.get(subject_key) or obj.get("Subject") or ""
                subject_id = obj.get(subject_id_key) or obj.get("Subject_ID") or ""
                yield content, lang, subject, str(subject_id), p.name


def get_subject_per_book(dir_path):
    """每个 jsonl 文件取第一行的 subject 作为该书的分类。返回 {(subject_id, subject_name): [book_filename, ...]}"""
    subject_to_books = defaultdict(list)
    for p in sorted(dir_path.glob("*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                subject = obj.get("学科") or obj.get("Subject") or ""
                subject_id = obj.get("学科_ID") or obj.get("Subject_ID") or ""
                key = (str(subject_id), subject)
                if p.name not in subject_to_books[key]:
                    subject_to_books[key].append(p.name)
                break  # 一本书只取第一行
    return subject_to_books


def main():
    enc = load_tiktoken_encoder()

    # ---------- 1. 原始中/英文数据条数（按 Init_lang / 原版语言）----------
    lang_count = defaultdict(int)
    for _dir, content_key, lang_key in [
        (CH_DIR, "内容", "原版语言"),
        (EN_DIR, "Content", "Init_lang"),
    ]:
        if not _dir.exists():
            continue
        for _, lang, _, _, _ in iter_jsonl_records(_dir, content_key=content_key, lang_key=lang_key):
            if lang:
                lang_count[lang] += 1
            else:
                lang_count["Unknown"] += 1

    n_chinese = lang_count.get("Chinese", 0)
    n_english = lang_count.get("English", 0)
    n_other = sum(v for k, v in lang_count.items() if k not in ("Chinese", "English"))

    print("========== 1. 原始中/英文数据条数 ==========")
    print(f"  Chinese (原版语言/Init_lang=Chinese): {n_chinese}")
    print(f"  English (Init_lang=English):          {n_english}")
    if n_other:
        print(f"  Other/Unknown:                       {n_other}")
    print(f"  Total:                                {n_chinese + n_english + n_other}")

    # ---------- 2. 各 Subject 及对应书籍数量（一个 jsonl = 一本书）----------
    subject_books_ch = get_subject_per_book(CH_DIR) if CH_DIR.exists() else {}
    subject_books_en = get_subject_per_book(EN_DIR) if EN_DIR.exists() else {}

    # 仅英文部分：Subject 书籍数量（用于饼图）
    subject_book_count_en = dict()
    for key in sorted(subject_books_en.keys(), key=lambda x: (x[0], x[1])):
        books = subject_books_en[key]
        subject_book_count_en[key] = len(books)

    # 按书籍数量排序，便于饼图图例（图例只显示 subject 名称，不显示 ID）
    subject_sorted = sorted(subject_book_count_en.items(), key=lambda x: -x[1])
    # 将 Dental History, Oral Equipmentology, Oral Photography 合并为 Others（仅用于 pie2）
    OTHERS_NAMES = {"Dental History", "Oral Equipmentology", "Oral Photography"}
    non_others = [(sname, cnt) for (sid, sname), cnt in subject_sorted if sname not in OTHERS_NAMES]
    others_count = sum(cnt for (sid, sname), cnt in subject_sorted if sname in OTHERS_NAMES)
    subject_labels = non_others + ([("Others", others_count)] if others_count else [])
    subject_labels = sorted(subject_labels, key=lambda x: -x[1])

    print("\n========== 2. Subject 及对应书籍数量 (EN) ==========")
    for (sid, sname), cnt in subject_sorted:
        print(f"  [{sid}] {sname}: {cnt} 本书")
    print(f"  Total EN books: {sum(subject_book_count_en.values())}")

    # ---------- 3. Token 数量（全部语料，按 CH/EN 来源划分用于饼图）----------
    ch_tokens = 0
    en_tokens = 0
    ch_chars = []
    en_chars = []

    for _dir, content_key, is_ch in [(CH_DIR, "内容", True), (EN_DIR, "Content", False)]:
        if not _dir.exists():
            continue
        for content, _, _, _, _ in iter_jsonl_records(_dir, content_key=content_key):
            if not content:
                continue
            n = len(enc.encode(content))
            if is_ch:
                ch_tokens += n
                ch_chars.append(content)
            else:
                en_tokens += n
                en_chars.append(content)

    total_tokens = ch_tokens + en_tokens
    print("\n========== 3. Token 数量 (GPT cl100k_base) ==========")
    print(f"  CH 语料 token 数: {ch_tokens:,}")
    print(f"  EN 语料 token 数: {en_tokens:,}")
    print(f"  总 token 数:      {total_tokens:,}")

    # ---------- 4. 绘制饼图（pie1 已取消保存）----------
    # 图2：Subject 书籍数量分布（仅英文，图例不显示 ID）
    # 仅在比例 >= PIE2_PCT_MIN 的扇区上显示百分比，避免小扇区数字挤在一起
    PIE2_PCT_MIN = 2  # 低于此比例的类别不在饼上显示数字（图例中仍有类别名）

    def autopct_pie2(pct):
        return "%1.1f%%" % pct if pct >= PIE2_PCT_MIN else ""

    fig2, ax2 = plt.subplots(figsize=(7.2, 7.2))
    labels2 = [x[0] for x in subject_labels]
    sizes2 = [x[1] for x in subject_labels]
    colors2 = repeat_palette(len(sizes2), PALETTE_FRESH_BALANCED)
    wedges2, texts2, autotexts2 = ax2.pie(
        sizes2,
        labels=None,
        autopct=autopct_pie2,
        startangle=90,
        colors=colors2,
        pctdistance=0.6,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    # 图例放在图下方，横向排列；固定每行 2 个类别
    ncol2 = 2
    ax2.legend(
        wedges2,
        labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=ncol2,
        fontsize=10,
        frameon=True,
        columnspacing=0.9,
        handletextpad=0.35,
        labelspacing=0.35,
        borderpad=0.2,
    )
    legend2 = ax2.get_legend()
    if legend2 is not None:
        frame = legend2.get_frame()
        frame.set_edgecolor("0.8")
        frame.set_linewidth(0.8)
        frame.set_alpha(1.0)
    for t in autotexts2:
        t.set_fontsize(14.5)
        t.set_color("white")
        t.set_weight("bold")
    # ax2.set_title("Distribution of Books by Subject", fontsize=12)
    # 预留底部空间给图例，避免压缩饼图主体
    fig2.subplots_adjust(bottom=0.24)
    plt.savefig(OUTPUT_DIR / "pie2_subject_book_distribution.png", bbox_inches="tight")
    plt.close()

    # 图3：Token 数量分布（CH vs EN），中等清新配色，图例带 token 数量(M)
    # 两段图例文字位置可手动设置（饼图坐标系：圆心 (0,0)，半径约 1，可填小数如 0.5）
    PIE3_LABEL_POSITIONS = [(-0.44, -0.34), (0.5, 0.35)]  # (CH 文字, EN 文字) 的 (x, y)
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ch_m = ch_tokens / 1e6
    en_m = en_tokens / 1e6
    token_labels = [f"(Chinese {ch_m:.1f}M)", f"(English {en_m:.1f}M)"]
    token_sizes = [ch_tokens, en_tokens]
    colors3 = ["#6BB8D6", "#F28E8C"]  # CH: 清新蓝青, EN: 柔和珊瑚
    explode = (0.02, 0.02)
    wedges3, texts3, autotexts3 = ax3.pie(
        token_sizes,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors3,
        explode=explode,
        pctdistance=0.48,
        wedgeprops={"edgecolor": "white", "linewidth": 1.4},
    )
    for t in autotexts3:
        t.set_fontsize(10)
        t.set_color("white")
        t.set_weight("bold")
    # 图例文字与百分比同款样式，位置使用 PIE3_LABEL_POSITIONS
    for pos, label in zip(PIE3_LABEL_POSITIONS, token_labels):
        ax3.text(pos[0], pos[1], label, fontsize=10, color="white", weight="bold",
                 ha="center", va="center", multialignment="center")
    plt.savefig(OUTPUT_DIR / "pie3_token_distribution.png")
    plt.close()

    print(f"\n饼图已保存至: {OUTPUT_DIR}")
    print("  pie2_subject_book_distribution.png")
    print("  pie3_token_distribution.png")


if __name__ == "__main__":
    main()
