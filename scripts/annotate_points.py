#!/usr/bin/env python3
"""手工标点工具：在图片上点击选取正/负点提示，自动写入 CSV。

本地运行（需要图形界面 + matplotlib）：
    python annotate_points.py --img_dir 图片目录 [--mask_dir mask目录] --csv_out out.csv

操作：
    左键点击   = 正点（前景，label=1，绿色显示）
    右键点击   = 负点（背景，label=0，红色显示）
    u          = 撤销当前图最后一个点
    c          = 清空当前图所有点
    n / 空格   = 下一张（自动保存当前点到CSV）
    p          = 上一张
    q          = 退出并保存

CSV 格式（无表头，每行一个点）：
    image_name,x,y,label
    seq20307,1563.0,995.1,1
"""

import argparse
import csv
import os

import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def load_image_list(img_dir, mask_dir):
    """收集 jpg，可选配对 png mask。返回 [(img_path, mask_path or None, name), ...]"""
    items = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        # 只把 .jpg 当主图（避免把 mask png 当主图重复）
        if fname.lower().endswith(".png") and mask_dir is None:
            continue
        img_path = os.path.join(img_dir, fname)
        name = os.path.splitext(fname)[0]
        mask_path = None
        if mask_dir is not None:
            mp = os.path.join(mask_dir, fname.rsplit(".", 1)[0] + ".png")
            if os.path.exists(mp):
                mask_path = mp
        items.append((img_path, mask_path, name))
    return items


class Annotator:
    def __init__(self, items, csv_path):
        self.items = items
        self.csv_path = csv_path
        self.idx = 0
        # points[name] = [(x, y, label), ...]  在会话内持久，跨图切换不丢
        self.points = {name: [] for _, _, name in items}
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.scatter = None
        self.render()

    def render(self):
        self.ax.clear()
        img_path, mask_path, name = self.items[self.idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        self.ax.imshow(img)
        # mask 叠加（红色半透明）
        if mask_path is not None:
            m = np.array(Image.open(mask_path).convert("L")) > 0
            overlay = np.zeros((*m.shape, 4), dtype=np.float32)
            overlay[m] = [1, 0, 0, 0.4]
            self.ax.imshow(overlay)
        # 画已有点
        pts = self.points[name]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cs = ["g" if p[2] == 1 else "r" for p in pts]
            self.ax.scatter(xs, ys, c=cs, s=80, edgecolors="white", linewidths=1.5, zorder=5)
        n_pos = sum(1 for p in pts if p[2] == 1)
        n_neg = sum(1 for p in pts if p[2] == 0)
        self.ax.set_title(
            f"[{self.idx + 1}/{len(self.items)}] {name}  |  "
            f"正点(绿)={n_pos}  负点(红)={n_neg}  |  "
            f"左键=正 右键=负 u=撤销 c=清空 n=下一张 p=上一张 q=退出保存",
            fontsize=10, loc="left",
        )
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        name = self.items[self.idx][2]
        if event.button == 1:        # 左键 = 正点
            self.points[name].append((x, y, 1))
        elif event.button == 3:      # 右键 = 负点
            self.points[name].append((x, y, 0))
        self.render()

    def on_key(self, event):
        name = self.items[self.idx][2]
        if event.key == "u":         # 撤销最后一个点
            if self.points[name]:
                self.points[name].pop()
                self.render()
        elif event.key == "c":        # 清空当前图
            self.points[name] = []
            self.render()
        elif event.key in ("n", " "):  # 下一张
            self.save_csv()
            if self.idx < len(self.items) - 1:
                self.idx += 1
                self.render()
            else:
                print("已是最后一张。按 q 退出保存。")
        elif event.key == "p":       # 上一张
            self.save_csv()
            if self.idx > 0:
                self.idx -= 1
                self.render()
        elif event.key == "q":        # 退出并保存
            self.save_csv()
            plt.close(self.fig)

    def save_csv(self):
        """把当前所有点写入 CSV（每次切换都覆盖写一遍，幂等）。"""
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            for _, _, name in self.items:
                for x, y, label in self.points[name]:
                    w.writerow([name, f"{x:.1f}", f"{y:.1f}", label])

    def run(self):
        plt.show()


def main():
    p = argparse.ArgumentParser(description="手工标点工具，输出 prompt CSV")
    p.add_argument("--img_dir", required=True, help="图片目录（jpg）")
    p.add_argument("--mask_dir", default=None, help="mask 目录（可选，用于叠加显示）")
    p.add_argument("--csv_out", required=True, help="输出 CSV 路径")
    args = p.parse_args()

    items = load_image_list(args.img_dir, args.mask_dir)
    if not items:
        print(f"在 {args.img_dir} 没找到图片")
        return
    print(f"找到 {len(items)} 张图，CSV 输出: {args.csv_out}")
    print("左键=正点(绿)  右键=负点(红)  u=撤销  c=清空  n/空格=下一张  p=上一张  q=退出保存")
    ann = Annotator(items, args.csv_out)
    ann.run()
    print(f"已保存: {args.csv_out}")


if __name__ == "__main__":
    main()
