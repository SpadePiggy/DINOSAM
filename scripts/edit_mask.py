#!/usr/bin/env python3
"""手工涂抹编辑二值 mask 工具。

在图片上叠加当前 mask（红色半透明），左键拖动涂白（前景=255），
右键拖动涂黑（背景=0），保存为同尺寸二值 PNG。

本地运行（需图形界面 + matplotlib + numpy + PIL）：
    python edit_mask.py --image xxx.jpg --mask xxx.png [--out xxx_new.png]

操作：
    左键拖动   = 涂白（设为前景）
    右键拖动   = 涂黑（设为背景）
    [  /  ]    = 减小 / 增大笔刷
    滚轮        = 调整笔刷
    u          = 撤销上一笔
    r          = 重置为原始mask
    s          = 保存（二值 0/255，同尺寸）
    q          = 退出
"""

import argparse
import os
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


class MaskEditor:
    def __init__(self, image_path, mask_path, out_path, brush=15):
        self.image_path = image_path
        self.mask_path = mask_path
        self.out_path = out_path or mask_path
        self.brush = brush
        self.img = np.array(Image.open(image_path).convert("RGB"))
        H, W = self.img.shape[:2]
        self.H, self.W = H, W
        m = np.array(Image.open(mask_path).convert("L"))
        if m.shape != (H, W):
            m = np.array(Image.fromarray(m).resize((W, H), Image.NEAREST))
        self.orig_mask = (m > 127).copy()
        self.mask = self.orig_mask.copy()
        self.history = []  # 撤销栈，存每次笔画前的 mask 快照
        self.drawing = False
        self.button = None

        self.fig, self.ax = plt.subplots(1, 1, figsize=(11, 9))
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.render()

    def overlay(self):
        """红半透明叠加：mask=1 的区域显示红色"""
        ov = np.zeros((*self.mask.shape, 4), dtype=np.float32)
        ov[self.mask] = [1, 0, 0, 0.45]
        return ov

    def render(self):
        self.ax.clear()
        self.ax.imshow(self.img)
        self.ax.imshow(self.overlay())
        self.ax.set_title(
            f"{os.path.basename(self.mask_path)}  笔刷={self.brush}px  "
            f"前景={(~self.mask).sum() if False else int(self.mask.sum())}px  "
            f"| 左键涂白 右键涂黑 [/]笔刷 u撤销 r重置 s保存 q退出",
            fontsize=9, loc="left")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        # 画笔刷光圈
        self.fig.canvas.draw_idle()

    def stamp(self, x, y, value):
        """在 (x,y) 处画一个半径 brush 的圆盘，设为 value"""
        yy, xx = np.ogrid[:self.H, :self.W]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        sel = d2 <= self.brush ** 2
        self.mask[sel] = value

    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        self.drawing = True
        self.history.append(self.mask.copy())
        if len(self.history) > 50:
            self.history.pop(0)
        self.button = event.button
        val = 1 if event.button == 1 else 0
        self.stamp(event.xdata, event.ydata, val)
        self.render()

    def on_motion(self, event):
        if not self.drawing or event.inaxes != self.ax or event.xdata is None:
            return
        val = 1 if self.button == 1 else 0
        self.stamp(event.xdata, event.ydata, val)
        self.render()

    def on_release(self, event):
        self.drawing = False
        self.button = None

    def on_key(self, event):
        if event.key == "[":
            self.brush = max(1, self.brush - 2); self.render()
        elif event.key == "]":
            self.brush = min(200, self.brush + 2); self.render()
        elif event.key == "u":
            if self.history:
                self.mask = self.history.pop()
                self.render()
        elif event.key == "r":
            self.history.append(self.mask.copy())
            self.mask = self.orig_mask.copy()
            self.render()
        elif event.key == "s":
            self.save()
        elif event.key == "q":
            plt.close(self.fig)

    def save(self):
        arr = (self.mask.astype(np.uint8) * 255)
        Image.fromarray(arr).save(self.out_path)
        print(f"已保存: {self.out_path}  尺寸={arr.shape}  二值={set(np.unique(arr).tolist())}")

    def run(self):
        plt.show()


def main():
    p = argparse.ArgumentParser(description="手工涂抹编辑二值 mask")
    p.add_argument("--image", required=True, help="原图 jpg 路径")
    p.add_argument("--mask", required=True, help="要编辑的 mask png 路径")
    p.add_argument("--out", default=None, help="输出路径（默认覆盖 --mask）")
    p.add_argument("--brush", type=int, default=15, help="初始笔刷半径（像素）")
    args = p.parse_args()
    ed = MaskEditor(args.image, args.mask, args.out, args.brush)
    ed.run()


if __name__ == "__main__":
    main()
