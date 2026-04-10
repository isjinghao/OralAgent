import os
import re
import argparse
from pathlib import Path

import cv2
import numpy as np


BOX_RE = re.compile(r"<box>\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*</box>")


def parse_box_string(boxes_str: str):
    """
    解析形如:
      <box>[132,137,601,585]</box>, <box>[2116,170,2728,655]</box>
    返回: [(x1,y1,x2,y2), ...]
    """
    if not boxes_str:
        return []
    matches = BOX_RE.findall(boxes_str)
    boxes = []
    for m in matches:
        x1, y1, x2, y2 = map(int, m)
        boxes.append((x1, y1, x2, y2))
    return boxes


def normalize_class_names(class_names_str: str | None, n: int):
    """
    支持:
      - None: 全部 unknown
      - "A,B,C": 按顺序对应每个 box
      - "A": 全部同一个类
    """
    if not class_names_str:
        return ["unknown"] * n
    raw = [s.strip() for s in class_names_str.split(",") if s.strip()]
    if not raw:
        return ["unknown"] * n
    if len(raw) == 1:
        return [raw[0]] * n
    # 长度不匹配：多的丢弃，不足的补 unknown
    out = []
    for i in range(n):
        if i < len(raw):
            out.append(raw[i])
        else:
            out.append("unknown")
    return out


def draw_boxes_on_image(image_path: Path, boxes, class_names, output_path: Path):
    """
    复用你现有脚本的画框风格:
      - 同类别固定颜色
      - 半透明实心框覆盖(alpha=0.2)
      - 粗边框(thickness=4)
      - 标签背景矩形 + 白色文字
    """
    # 高对比、色盲友好（尽量）的固定调色板（OpenCV 使用 BGR）
    palette_bgr = [
        (0, 255, 255),    # yellow
        (255, 0, 255),    # magenta
        (255, 255, 0),    # cyan
        (0, 165, 255),    # orange
        (0, 255, 0),      # green
        (0, 0, 255),      # red
        (255, 0, 0),      # blue
        (203, 192, 255),  # pink-ish
        (255, 255, 255),  # white
        (128, 0, 255),    # purple-ish
        (255, 128, 0),    # deep orange-ish
        (0, 128, 255),    # amber-ish
    ]

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"无法读取图片: {image_path}")

    # 画框：不做内部填充，只画更粗的边框
    for box, class_name in zip(boxes, class_names):
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = box

        # 为每种类别固定选择一种颜色（保持跨运行一致）
        idx = (hash(class_name) & 0xFFFFFFFF) % len(palette_bgr)
        color = palette_bgr[idx]
        color_outline = (0, 0, 0)

        # 先画黑色描边，再画彩色边框，保证在任何背景上都清晰
        cv2.rectangle(img, (x1, y1), (x2, y2), color_outline, thickness=32)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=16)

        label = f"{class_name}"
        font_scale = 4.0
        font_thickness = 8
        shadow_thickness = 18
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        # 让标签尽量画在框上方；整体略微左移/上移一点点
        label_block_dx = -10  # negative => move left
        label_block_dy = 8    # positive => move up
        y_top = y1 - text_h - baseline - label_block_dy
        if y_top < 0:
            y_top = y1 + text_h + baseline - label_block_dy
        y_bottom = y_top + text_h + baseline

        # 标签底：半透明黑底 + 彩色边条，兼顾可读性和类别提示
        pad_x = 24
        pad_y = 16
        box_x1 = max(0, x1 + label_block_dx)
        box_y1 = max(0, y_top - pad_y)
        box_x2 = box_x1 + text_w + pad_x + 10
        box_y2 = min(img.shape[0] - 1, y_bottom + pad_y)

        label_overlay = img.copy()
        cv2.rectangle(label_overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        img = cv2.addWeighted(label_overlay, 0.55, img, 0.45, 0)
        # 彩色边条（左侧）
        stripe_w = 24
        cv2.rectangle(img, (box_x1, box_y1), (box_x1 + stripe_w, box_y2), color, -1)
        # 黑色外描边
        cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), color_outline, 4)

        text_y = y_bottom - baseline // 2
        # 文字加黑色阴影描边，避免在亮色区域发虚
        cv2.putText(
            img,
            label,
            (box_x1 + 5 + stripe_w, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            shadow_thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img,
            label,
            (box_x1 + 5 + stripe_w, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    os.makedirs(output_path.parent, exist_ok=True)
    ok = cv2.imwrite(str(output_path), img)
    if not ok:
        raise RuntimeError(f"保存失败: {output_path}")
    print(f"✅ 已保存可视化结果: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="根据字符串 <box>[x1,y1,x2,y2]</box> 在图片上绘制框并保存。"
    )
    parser.add_argument("--image_path", default="/home/jinghao/LMUData/images/MMOral_OMNI/2361.jpg", help="输入图片路径")
    parser.add_argument(
        "--boxes_str",
        default="<box>[770,330,890,430]</box>, <box>[870,180,980,290]</box>",
        # default="<box>[109,346,578,784]</box>, <box>[181,647,673,1103]</box>, <box>[1983,200,2456,644]</box>, <box>[1924,536,2406,1027]</box>",
        # default=" <box>[177,524,639,961]</box>, <box>[2009,544,2501,989]</box>, <box>[2167,183,2639,601]</box>",
        help="框字符串，例如: '<box>[132,137,601,585]</box>, <box>[2116,170,2728,655]</box>'",
    )
    parser.add_argument(
        "--class_names",
        default='Caries',
        help="可选类别名(逗号分隔)。例如: 'tumor,normal' 或 'tumor'。若不提供则全部 unknown。",
    )
    parser.add_argument(
        "--output_path",
        default="/home/jinghao/projects/OralGPT-Agent/OralAgent/MMOral-Omni_2361_GPT-5_boxed.jpg",
        help="输出图片路径(可选)。默认: 与输入同目录，文件名追加 '_boxed.jpg'",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片: {image_path}")

    boxes = parse_box_string(args.boxes_str)
    if not boxes:
        raise ValueError("未解析到任何 box，请检查 boxes_str 格式是否符合 <box>[x1,y1,x2,y2]</box>。")

    class_names = normalize_class_names(args.class_names, len(boxes))

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = image_path.with_name(image_path.stem + "_boxed.jpg")

    draw_boxes_on_image(image_path, boxes, class_names, output_path)


if __name__ == "__main__":
    main()

