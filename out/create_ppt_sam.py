"""生成 PPT1：SAM 主干方向技术汇报"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# 配色方案
C_TITLE = RGBColor(0x0F, 0x34, 0x60)
C_ACCENT = RGBColor(0xE9, 0x45, 0x60)
C_BODY = RGBColor(0x1A, 0x1A, 0x2E)
C_GRAY = RGBColor(0x66, 0x66, 0x66)
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_LIGHT_BG = RGBColor(0xF0, 0xF4, 0xF8)
C_BLUE = RGBColor(0x09, 0x84, 0xE3)
C_GREEN = RGBColor(0x00, 0xB8, 0x94)
C_PURPLE = RGBColor(0x53, 0x34, 0x83)
C_ORANGE = RGBColor(0xFF, 0x98, 0x00)
C_RED = RGBColor(0xD6, 0x30, 0x31)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
W = prs.slide_width
H = prs.slide_height


def add_bg(slide, color=C_LIGHT_BG):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, left, top, width, height, text, font_size=18,
                 color=C_BODY, bold=False, alignment=PP_ALIGN.LEFT, font_name="Microsoft YaHei"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_bullet_slide(slide, left, top, width, height, items, font_size=16, color=C_BODY):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Microsoft YaHei"
        p.space_after = Pt(6)
        p.level = 0
    return txBox


def add_rect(slide, left, top, width, height, fill_color, text="",
             font_size=12, font_color=C_WHITE, bold=False, border_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()
    if text:
        tf = shape.text_frame
        tf.word_wrap = True
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(font_size)
        p.font.color.rgb = font_color
        p.font.bold = bold
        p.font.name = "Microsoft YaHei"
    return shape


def add_arrow(slide, x1, y1, x2, y2, color=C_BODY):
    connector = slide.shapes.add_connector(1, x1, y1, x2, y2)  # 1 = straight
    connector.line.color.rgb = color
    connector.line.width = Pt(2)
    return connector


def add_table(slide, left, top, width, height, rows, cols, data, header_color=C_TITLE):
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    for i in range(rows):
        for j in range(cols):
            cell = table.cell(i, j)
            cell.text = data[i][j]
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12)
                p.font.name = "Microsoft YaHei"
                if i == 0:
                    p.font.bold = True
                    p.font.color.rgb = C_WHITE
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = header_color
                else:
                    p.font.color.rgb = C_BODY
    return table


# ============================================================
# 第 1 页：封面
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
add_bg(slide, RGBColor(0x0F, 0x34, 0x60))
add_text_box(slide, Inches(1.5), Inches(1.8), Inches(10), Inches(1.5),
             "DepthSam：SAM 主干方向", 44, C_WHITE, True, PP_ALIGN.CENTER)
add_text_box(slide, Inches(1.5), Inches(3.3), Inches(10), Inches(1),
             "基于 SAM ViT-B 的深度感知分割框架", 24, RGBColor(0xBB, 0xCC, 0xDD), False, PP_ALIGN.CENTER)
add_text_box(slide, Inches(1.5), Inches(4.5), Inches(10), Inches(0.8),
             "技术方案讨论 · 消融实验设计", 20, RGBColor(0x88, 0x99, 0xAA), False, PP_ALIGN.CENTER)
add_text_box(slide, Inches(1.5), Inches(5.8), Inches(10), Inches(0.5),
             "2026.06", 16, RGBColor(0x66, 0x77, 0x88), False, PP_ALIGN.CENTER)

# ============================================================
# 第 2 页：问题定义与核心思路
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "问题定义与核心思路", 32, C_TITLE, True)
# 左侧：问题
add_rect(slide, Inches(0.8), Inches(1.4), Inches(5.5), Inches(2.2),
         RGBColor(0xFF, 0xF3, 0xE0), border_color=C_ORANGE)
add_text_box(slide, Inches(1.0), Inches(1.5), Inches(5.1), Inches(0.4),
             "❶ 单目深度估计的困难", 18, C_ORANGE, True)
add_bullet_slide(slide, Inches(1.0), Inches(2.0), Inches(5.1), Inches(1.5), [
    "• 本质上是欠定问题——同一张图像可以对应无穷多深度解",
    "• 密集逐像素回归需要大量标注数据",
    "• 现有方法缺乏对实例边界的感知",
], 14)

# 右侧：思路
add_rect(slide, Inches(6.8), Inches(1.4), Inches(5.8), Inches(2.2),
         RGBColor(0xE8, 0xF8, 0xF5), border_color=C_GREEN)
add_text_box(slide, Inches(7.0), Inches(1.5), Inches(5.4), Inches(0.4),
             "❷ 核心思路：掩码条件化深度回归", 18, C_GREEN, True)
add_bullet_slide(slide, Inches(7.0), Inches(2.0), Inches(5.4), Inches(1.5), [
    "• 将深度预测条件化于分割掩码",
    "• 从密集回归 → 逐实例标量预测",
    "• 掩码界定空间范围，解码器专注深度判别",
], 14)

# 下方：关键设计决策
add_text_box(slide, Inches(0.8), Inches(4.0), Inches(10), Inches(0.5),
             "关键设计决策（消融出发点）", 22, C_ACCENT, True)
add_rect(slide, Inches(0.8), Inches(4.6), Inches(3.6), Inches(2.2),
         C_WHITE, border_color=C_BLUE)
add_text_box(slide, Inches(1.0), Inches(4.7), Inches(3.2), Inches(2.0),
             "冻结 SAM 编码器\n\n保留 SA-1B 大规模分割预训练先验\n消融隔离：解码侧修改可独立评估", 13, C_BODY)

add_rect(slide, Inches(4.7), Inches(4.6), Inches(3.6), Inches(2.2),
         C_WHITE, border_color=C_BLUE)
add_text_box(slide, Inches(4.9), Inches(4.7), Inches(3.2), Inches(2.0),
             "两阶段分离训练\n\n先掩码后深度，避免多任务干扰\n每阶段可独立评估、独立消融", 13, C_BODY)

add_rect(slide, Inches(8.6), Inches(4.6), Inches(3.9), Inches(2.2),
         C_WHITE, border_color=C_BLUE)
add_text_box(slide, Inches(8.8), Inches(4.7), Inches(3.5), Inches(2.0),
             "梯度截断（detach）\n\n深度分支不回传梯度到掩码分支\n分割误差与深度误差解耦", 13, C_BODY)

# ============================================================
# 第 3 页：整体架构
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "DepthSam 整体架构", 32, C_TITLE, True)

# 架构图 - 用 shapes 构建
# 输入
add_rect(slide, Inches(0.5), Inches(2.5), Inches(1.2), Inches(0.7),
         RGBColor(0xE8, 0xF4, 0xF8), "输入图像", 12, C_TITLE, True, C_TITLE)

# 编码器
add_rect(slide, Inches(2.2), Inches(2.3), Inches(2.0), Inches(1.1),
         RGBColor(0xFF, 0xEA, 0xA7), "SAM ViT-B/16\n图像编码器\n🔒 冻结", 12, C_RED, True, C_RED)

# 分支标签
add_text_box(slide, Inches(5.0), Inches(1.2), Inches(2), Inches(0.4),
             "掩码分支 (Phase 1 可训练)", 13, C_BLUE, True)

# 掩码分支
add_rect(slide, Inches(5.0), Inches(1.6), Inches(1.8), Inches(0.7),
         RGBColor(0xDF, 0xF9, 0xFB), "PromptEncoder\n点提示 → 嵌入", 11, C_BLUE, False, C_BLUE)

add_rect(slide, Inches(7.2), Inches(1.6), Inches(1.8), Inches(0.7),
         RGBColor(0xDF, 0xF9, 0xFB), "MaskDecoder\nTwoWayTransformer", 11, C_BLUE, False, C_BLUE)

add_rect(slide, Inches(9.5), Inches(1.3), Inches(1.3), Inches(0.5),
         RGBColor(0x55, 0xEF, 0xC4), "分割掩码", 11, RGBColor(0x00, 0x69, 0x5C), True, C_GREEN)
add_rect(slide, Inches(9.5), Inches(2.0), Inches(1.3), Inches(0.5),
         RGBColor(0x81, 0xEC, 0xEC), "IoU 预测", 11, RGBColor(0x00, 0x62, 0x66), True)

# detach 标注
add_rect(slide, Inches(7.5), Inches(2.8), Inches(2.5), Inches(0.5),
         RGBColor(0xFF, 0xF3, 0xE0), "low_res_masks (detach)", 11, C_ACCENT, True, C_ACCENT)

# 深度分支
add_text_box(slide, Inches(5.0), Inches(3.7), Inches(3), Inches(0.4),
             "深度分支 (Phase 2 可训练)", 13, C_PURPLE, True)

add_rect(slide, Inches(5.0), Inches(4.1), Inches(1.8), Inches(0.7),
         RGBColor(0xF0, 0xE6, 0xFF), "Outer PE\n掩码 → 密集嵌入", 11, C_PURPLE, False, C_PURPLE)

add_rect(slide, Inches(7.2), Inches(4.1), Inches(2.0), Inches(0.7),
         RGBColor(0xF0, 0xE6, 0xFF), "DepthMaskDecoder\ndepth_token + MLP", 11, C_PURPLE, False, C_PURPLE)

add_rect(slide, Inches(9.7), Inches(3.8), Inches(1.3), Inches(0.5),
         RGBColor(0xFF, 0xEA, 0xA7), "深度标量", 11, RGBColor(0x6C, 0x51, 0x00), True)
add_rect(slide, Inches(9.7), Inches(4.5), Inches(1.3), Inches(0.5),
         RGBColor(0xFA, 0xB1, 0xA0), "辅助掩码", 11, RGBColor(0x6B, 0x2D, 0x00), True)

# 输出维度标注
add_text_box(slide, Inches(4.3), Inches(2.7), Inches(1.2), Inches(0.3),
             "(B,256,64,64)", 10, C_GRAY)

# 下方：关键参数
add_text_box(slide, Inches(0.5), Inches(5.4), Inches(12), Inches(0.5),
             "关键参数", 18, C_TITLE, True)
data = [
    ["组件", "维度", "初始化", "说明"],
    ["SAM ViT-B/16", "输出 (B,256,64,64)", "SA-1B 预训练", "全程冻结，不参与任何阶段训练"],
    ["prompt_embed_dim", "256", "-", "提示编码维度，贯穿整个解码管线"],
    ["mask_in_chans", "16", "-", "掩码输入通道数（outer PE）"],
    ["TwoWayTransformer", "d=2, h=8, mlp=2048", "SAM 权重", "掩码和深度分支各一个"],
]
add_table(slide, Inches(0.5), Inches(5.9), Inches(12), Inches(1.3), 5, 4, data)

# ============================================================
# 第 4 页：DepthMaskDecoder 详细结构
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "DepthMaskDecoder — 与 SAM MaskDecoder 的三处修改", 28, C_TITLE, True)

# 左侧：SAM MaskDecoder
add_rect(slide, Inches(0.5), Inches(1.3), Inches(5.5), Inches(5.5),
         C_WHITE, border_color=C_BLUE)
add_text_box(slide, Inches(0.7), Inches(1.4), Inches(5.1), Inches(0.5),
             "SAM MaskDecoder", 20, C_BLUE, True)

add_rect(slide, Inches(0.8), Inches(2.0), Inches(2.2), Inches(0.5),
         RGBColor(0xDF, 0xF9, 0xFB), "iou_token (1×256)", 11, C_BLUE, False, C_BLUE)
add_rect(slide, Inches(3.2), Inches(2.0), Inches(2.5), Inches(0.5),
         RGBColor(0xDF, 0xF9, 0xFB), "mask_tokens (4×256)", 11, C_BLUE, False, C_BLUE)
add_rect(slide, Inches(1.2), Inches(2.8), Inches(4.0), Inches(0.6),
         RGBColor(0x74, 0xB9, 0xFF), "TwoWayTransformer (d=2, h=8)", 12, C_WHITE, True)
add_rect(slide, Inches(0.8), Inches(3.7), Inches(2.5), Inches(0.5),
         RGBColor(0x81, 0xEC, 0xEC), "IoU Prediction Head", 11, RGBColor(0x00, 0x62, 0x66), True)
add_rect(slide, Inches(3.5), Inches(3.7), Inches(2.2), Inches(0.5),
         RGBColor(0xDF, 0xF9, 0xFB), "4×HyperMLP", 11, C_BLUE, False, C_BLUE)
add_rect(slide, Inches(0.8), Inches(4.5), Inches(2.0), Inches(0.5),
         RGBColor(0x81, 0xEC, 0xEC), "IoU (B,1)", 11, RGBColor(0x00, 0x62, 0x66), True)
add_rect(slide, Inches(3.5), Inches(4.5), Inches(2.2), Inches(0.5),
         RGBColor(0x55, 0xEF, 0xC4), "4 Masks (B,4,H,W)", 11, RGBColor(0x00, 0x69, 0x5C), True)

# 右侧：DepthMaskDecoder
add_rect(slide, Inches(6.5), Inches(1.3), Inches(6.3), Inches(5.5),
         C_WHITE, border_color=C_PURPLE)
add_text_box(slide, Inches(6.7), Inches(1.4), Inches(5.9), Inches(0.5),
             "DepthMaskDecoder", 20, C_PURPLE, True)

add_rect(slide, Inches(6.8), Inches(2.0), Inches(2.5), Inches(0.5),
         RGBColor(0xF0, 0xE6, 0xFF), "depth_token (1×256)", 11, C_PURPLE, False, C_PURPLE)
add_rect(slide, Inches(9.5), Inches(2.0), Inches(2.5), Inches(0.5),
         RGBColor(0xF0, 0xE6, 0xFF), "mask_tokens (1×256)", 11, C_PURPLE, False, C_PURPLE)
add_rect(slide, Inches(7.2), Inches(2.8), Inches(4.5), Inches(0.6),
         RGBColor(0xA2, 0x9B, 0xFE), "TwoWayTransformer (d=2, h=8)", 12, C_WHITE, True)
add_rect(slide, Inches(6.8), Inches(3.7), Inches(2.5), Inches(0.5),
         RGBColor(0xFF, 0xEA, 0xA7), "Depth Head (MLP 3层)", 11, RGBColor(0x6C, 0x51, 0x00), True)
add_rect(slide, Inches(9.5), Inches(3.7), Inches(2.8), Inches(0.5),
         RGBColor(0xF0, 0xE6, 0xFF), "1×HyperMLP", 11, C_PURPLE, False, C_PURPLE)
add_rect(slide, Inches(6.8), Inches(4.5), Inches(2.5), Inches(0.5),
         RGBColor(0xFF, 0xEA, 0xA7), "Depth (B,)", 11, RGBColor(0x6C, 0x51, 0x00), True)
add_rect(slide, Inches(9.5), Inches(4.5), Inches(2.8), Inches(0.5),
         RGBColor(0xFA, 0xB1, 0xA0), "1 Mask (B,1,H,W)", 11, RGBColor(0x6B, 0x2D, 0x00), True)

# 三处修改
add_rect(slide, Inches(6.8), Inches(5.3), Inches(5.7), Inches(1.3),
         RGBColor(0xFF, 0xF3, 0xE0), border_color=C_ORANGE)
add_text_box(slide, Inches(7.0), Inches(5.4), Inches(5.3), Inches(1.1),
             "三处关键修改：\n❶ iou_token → depth_token + depth_prediction_head (MLP: 256→256→1)\n❷ 4 mask_tokens → 1 mask_token + 1 HyperMLP\n❸ IoU head → Depth head (3层 MLP)", 12, RGBColor(0xE6, 0x51, 0x00))

# ============================================================
# 第 5 页：权重初始化策略
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "从 SAM 到 DepthSam 的权重初始化映射", 28, C_TITLE, True)

data = [
    ["SAM 源组件", "DepthSam 目标", "迁移方式", "说明"],
    ["prompt_encoder", "outer_prompt_encoder", "完全复制", "结构完全一致"],
    ["mask_decoder.transformer", "depth_decoder.transformer", "完全复制*", "*dual_attn 模式跳过，随机初始化"],
    ["iou_token (1×256)", "depth_token (1×256)", "直接复制", "形状一致"],
    ["mask_tokens[0] (取第1个)", "mask_tokens (1×256)", "取第1个", "SAM 有 4 个，只取第 1 个"],
    ["output_upscaling", "output_upscaling", "完全复制", "两层 ConvTranspose2d 结构一致"],
    ["hyper_mlps[0]", "hyper_mlps[0]", "完全复制", "SAM 有 4 个，只取第 1 个"],
    ["（无对应）", "depth_prediction_head", "随机初始化", "MLP: 256→256→1，唯一新增组件"],
]
add_table(slide, Inches(0.5), Inches(1.3), Inches(12.3), Inches(2.8), 8, 4, data)

add_text_box(slide, Inches(0.5), Inches(4.5), Inches(12), Inches(0.5),
             "设计意图", 20, C_ACCENT, True)
add_bullet_slide(slide, Inches(0.5), Inches(5.1), Inches(12), Inches(2), [
    "• 使深度分支初始即理解如何注意 SAM 图像特征和提示嵌入",
    "• 仅 depth_prediction_head 需要从头学习新的「特征→深度」映射",
    "• 消融问题：dual_attn 模式下 Transformer 随机初始化 vs TwoWay 从 SAM 迁移，哪个更好？",
    "  → 这是一个重要的消融变量（--depth_transformer_type twoway|dual_attn）",
], 15)

# ============================================================
# 第 6 页：迭代提示精炼
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "迭代提示精炼机制", 32, C_TITLE, True)

# 流程图
rounds = [
    ("迭代轮 t=0", RGBColor(0xE8, 0xF4, 0xF8), C_BLUE),
    ("迭代轮 t=1", RGBColor(0xF0, 0xE6, 0xFF), C_PURPLE),
    ("最终轮 t=n-1", RGBColor(0xE8, 0xF8, 0xF5), C_GREEN),
]
for i, (label, bg_c, fg_c) in enumerate(rounds):
    y = Inches(1.3 + i * 1.7)
    add_rect(slide, Inches(0.5), y, Inches(12.3), Inches(1.4), bg_c, border_color=fg_c)
    add_text_box(slide, Inches(0.7), y + Emu(Inches(0.1).emu), Inches(2), Inches(0.3),
                 label, 13, fg_c, True)
    if i < 2:
        steps = ["初始提示 P₀" if i == 0 else "P₁ (增长)",
                 "forward_mask →\nmasks, low_res",
                 "误差分析\npred vs GT",
                 "采样修正点\n+1正 +1负",
                 "P₁ = P₀+Δ" if i == 0 else "→ P₂"]
    else:
        steps = ["P_final",
                 "forward_mask\n→ final masks",
                 "loss = Σ/n_sub\n(dice+iou)",
                 "detach →\nforward_depth",
                 "深度标量"]
    for j, step in enumerate(steps):
        x = Inches(0.8 + j * 2.45)
        add_rect(slide, x, y + Emu(Inches(0.4).emu), Inches(2.0), Inches(0.8),
                 C_WHITE, step, 10, C_BODY, False, fg_c)

# 消融参数
add_text_box(slide, Inches(0.5), Inches(6.0), Inches(12), Inches(0.5),
             "可消融参数", 18, C_ACCENT, True)
data2 = [
    ["参数", "默认值", "消融范围", "影响"],
    ["n_sub_iterations", "1（不迭代）", "1, 2, 3, 4", "提示精炼轮数，更多轮 = 更精确掩码"],
    ["mask_prob", "0.5", "0.0, 0.25, 0.5, 0.75, 1.0", "是否将上一轮 low_res 作为 mask_input"],
    ["max_points", "3", "1, 3, 5, 7", "初始点提示数量（GT 连通域质心）"],
]
add_table(slide, Inches(0.5), Inches(6.4), Inches(12.3), Inches(1.0), 4, 4, data2)

# ============================================================
# 第 7 页：两阶段训练策略
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "两阶段训练策略", 32, C_TITLE, True)

# Phase 1
add_rect(slide, Inches(0.5), Inches(1.3), Inches(5.8), Inches(5.5),
         C_WHITE, border_color=C_BLUE)
add_text_box(slide, Inches(0.7), Inches(1.4), Inches(5.4), Inches(0.5),
             "Phase 1：掩码训练", 20, C_BLUE, True)

add_rect(slide, Inches(0.8), Inches(2.0), Inches(2.5), Inches(0.4),
         RGBColor(0xFF, 0xEA, 0xA7), "🔒 Image Encoder", 11, C_RED, True)
add_rect(slide, Inches(3.5), Inches(2.0), Inches(2.5), Inches(0.4),
         RGBColor(0xFF, 0xEA, 0xA7), "🔒 深度分支", 11, C_RED, True)
add_rect(slide, Inches(0.8), Inches(2.6), Inches(2.5), Inches(0.4),
         RGBColor(0x55, 0xEF, 0xC4), "✏️ PromptEncoder", 11, RGBColor(0x00, 0x69, 0x5C), True)
add_rect(slide, Inches(3.5), Inches(2.6), Inches(2.5), Inches(0.4),
         RGBColor(0x55, 0xEF, 0xC4), "✏️ MaskDecoder", 11, RGBColor(0x00, 0x69, 0x5C), True)

add_text_box(slide, Inches(0.8), Inches(3.3), Inches(5.2), Inches(1.5),
             "损失函数：\nL = L_dice(masks, GT) + MSE(IoU_pred, IoU_true)\n\n优化器：AdamW, lr=1e-5\n调度器：StepLR(step=10, γ=0.5)\n迭代模式：多轮 forward + 误差采点", 13, C_BODY)

# Phase 2
add_rect(slide, Inches(7.0), Inches(1.3), Inches(5.8), Inches(5.5),
         C_WHITE, border_color=C_PURPLE)
add_text_box(slide, Inches(7.2), Inches(1.4), Inches(5.4), Inches(0.5),
             "Phase 2：深度训练", 20, C_PURPLE, True)

add_rect(slide, Inches(7.3), Inches(2.0), Inches(2.5), Inches(0.4),
         RGBColor(0xFF, 0xEA, 0xA7), "🔒 Image Encoder", 11, C_RED, True)
add_rect(slide, Inches(10.0), Inches(2.0), Inches(2.5), Inches(0.4),
         RGBColor(0xFF, 0xEA, 0xA7), "🔒 掩码分支", 11, C_RED, True)
add_rect(slide, Inches(7.3), Inches(2.6), Inches(2.5), Inches(0.4),
         RGBColor(0x55, 0xEF, 0xC4), "✏️ Outer PE", 11, RGBColor(0x00, 0x69, 0x5C), True)
add_rect(slide, Inches(10.0), Inches(2.6), Inches(2.5), Inches(0.4),
         RGBColor(0x55, 0xEF, 0xC4), "✏️ DepthDecoder", 11, RGBColor(0x00, 0x69, 0x5C), True)

add_text_box(slide, Inches(7.3), Inches(3.3), Inches(5.2), Inches(1.5),
             "损失函数：\nL = λ × MSE(depth_pred, depth_GT)\n\n关键操作：\n1. 加载 Phase 1 检查点\n2. 重新 init_depth_from_sam()\n3. 掩码分支在 no_grad 下运行", 13, C_BODY)

# 底部：训练配置消融
add_text_box(slide, Inches(0.5), Inches(5.2), Inches(12), Inches(0.5),
             "训练配置消融选项", 18, C_ACCENT, True)
data3 = [
    ["消融维度", "选项", "默认值", "关注点"],
    ["训练模式", "phase1 / phase2 / both", "both", "两阶段 vs 联合训练的效果差异"],
    ["学习率", "1e-4 / 1e-5 / 5e-6", "1e-5", "AdamW 学习率"],
    ["Epochs", "30 / 50 / 80", "50", "两阶段各自的训练轮数"],
    ["depth_loss_weight λ", "0.5 / 1.0 / 2.0", "1.0", "深度损失权重"],
]
add_table(slide, Inches(0.5), Inches(5.7), Inches(12.3), Inches(1.5), 5, 4, data3)

# ============================================================
# 第 8 页：★ 消融实验总体设计（SAM 方向）
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, RGBColor(0x16, 0x21, 0x3E))
add_text_box(slide, Inches(0.8), Inches(0.3), Inches(10), Inches(0.7),
             "★ SAM 方向消融实验矩阵", 32, C_WHITE, True)
add_text_box(slide, Inches(0.8), Inches(0.9), Inches(10), Inches(0.4),
             "控制变量法：每次只变一个维度，其余保持默认", 16, RGBColor(0xBB, 0xCC, 0xDD))

# 消融维度卡片
dims = [
    ("维度 A\n深度分支 Transformer", "--depth_transformer_type", "twoway (SAM) ↔ dual_attn (PAM+CAM)",
     "TwoWay 从 SAM 迁移权重\nDualAttn 随机初始化\n→ 迁移学习 vs 新架构", C_BLUE),
    ("维度 B\n迭代提示精炼", "--n_sub_iterations", "1 / 2 / 3 / 4",
     "更多迭代 = 更好掩码 = 更好深度？\n训练开销 vs 精度收益", C_GREEN),
    ("维度 C\n掩码传入策略", "--mask_prob", "0.0 / 0.25 / 0.5 / 0.75 / 1.0",
     "上一轮掩码作为先验的概率\n过高可能导致依赖累积误差", C_PURPLE),
    ("维度 D\n初始点数量", "--max_points", "1 / 3 / 5 / 7",
     "GT 连通域质心采样\n多点 vs 信息冗余", C_ORANGE),
    ("维度 E\n训练策略", "--train_phase + λ", "两阶段 / 联合 / 不同 λ",
     "分离训练 vs 多任务学习\n深度权重的最佳配比", C_ACCENT),
]

for i, (title, param, options, note, color) in enumerate(dims):
    row = i // 3
    col = i % 3
    x = Inches(0.5 + col * 4.2)
    y = Inches(1.5 + row * 2.8)
    add_rect(slide, x, y, Inches(3.9), Inches(2.5), RGBColor(0x1E, 0x2D, 0x4A), border_color=color)
    add_text_box(slide, x + Emu(Inches(0.15).emu), y + Emu(Inches(0.1).emu),
                 Inches(3.6), Inches(0.5), title, 13, color, True)
    add_text_box(slide, x + Emu(Inches(0.15).emu), y + Emu(Inches(0.6).emu),
                 Inches(3.6), Inches(0.3), param, 11, RGBColor(0xFF, 0xEA, 0xA7))
    add_text_box(slide, x + Emu(Inches(0.15).emu), y + Emu(Inches(0.9).emu),
                 Inches(3.6), Inches(0.3), "选项: " + options, 10, RGBColor(0xBB, 0xCC, 0xDD))
    add_text_box(slide, x + Emu(Inches(0.15).emu), y + Emu(Inches(1.3).emu),
                 Inches(3.6), Inches(1.0), note, 10, RGBColor(0x99, 0xAA, 0xBB))

# ============================================================
# 第 9 页：消融实验具体方案
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "消融实验具体方案", 32, C_TITLE, True)

data4 = [
    ["实验编号", "消融维度", "变量", "默认基线", "评估指标", "预期发现"],
    ["A1 vs A2", "深度 Transformer", "twoway vs dual_attn", "twoway", "Depth MAE/R², Mask Dice",
     "迁移学习是否优于新架构"],
    ["B1-B4", "迭代次数", "n_sub=1,2,3,4", "n_sub=1", "Mask Dice/IoU, Depth MAE",
     "迭代精炼的边际收益递减点"],
    ["C1-C5", "掩码传入概率", "prob=0,0.25,0.5,0.75,1", "prob=0.5", "Mask Dice, Depth MAE",
     "最优先验注入比例"],
    ["D1-D4", "初始点数", "pts=1,3,5,7", "pts=3", "Mask Dice/IoU",
     "信息量 vs 冗余的最优点"],
    ["E1-E3", "训练策略", "两阶段/联合/λ变化", "两阶段,λ=1", "Depth MAE, Mask Dice",
     "多任务学习是否优于分离训练"],
]
add_table(slide, Inches(0.3), Inches(1.3), Inches(12.7), Inches(2.5), 6, 6, data4)

# 实验优先级
add_text_box(slide, Inches(0.5), Inches(4.2), Inches(12), Inches(0.5),
             "建议实验优先级", 20, C_ACCENT, True)
add_bullet_slide(slide, Inches(0.5), Inches(4.8), Inches(12), Inches(2.5), [
    "🔴 P0（必做）：A1 vs A2 — depth_transformer_type 是架构层面的核心决策",
    "🔴 P0（必做）：B1-B4 — 迭代提示精炼是方法论的核心贡献之一",
    "🟡 P1（推荐）：C1-C5 — mask_prob 影响训练稳定性",
    "🟡 P1（推荐）：E1-E3 — 两阶段 vs 联合训练的对比是论文常见消融",
    "🟢 P2（可选）：D1-D4 — 初始点数影响较小，可作为补充",
], 14)

# ============================================================
# 第 10 页：讨论点
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
             "与老师讨论的开放问题", 32, C_TITLE, True)

questions = [
    ("架构选择", [
        "1. 冻结 SAM 编码器的前提是否过强？是否应该尝试微调最后几层？",
        "2. depth_prediction_head 只有 3 层 MLP (256→256→1)，容量是否足够？",
        "3. 深度预测为标量值——是否考虑过密集深度图预测？",
    ]),
    ("训练策略", [
        "4. 两阶段训练 vs 端到端联合训练的消融如何设计？",
        "5. Phase 2 重新 init_depth_from_sam() 是否必要？直接续训如何？",
        "6. StepLR(step=10, γ=0.5) 是否过于激进？CosineAnnealing 是否更好？",
    ]),
    ("消融设计", [
        "7. dual_attn_transformer 模式跳过权重迁移——是否影响公平对比？",
        "8. 迭代精炼的训练/推理一致性：训练时迭代 n 轮，推理时是否也必须？",
        "9. SAM 编码器 vs DINOv3 编码器作为最顶层消融，实验如何设计？",
    ]),
]
for i, (title, items) in enumerate(questions):
    y = Inches(1.3 + i * 2.0)
    add_rect(slide, Inches(0.5), y, Inches(12.3), Inches(1.8),
             C_WHITE, border_color=[C_BLUE, C_PURPLE, C_ACCENT][i])
    add_text_box(slide, Inches(0.7), y + Emu(Inches(0.05).emu), Inches(3), Inches(0.4),
                 title, 16, [C_BLUE, C_PURPLE, C_ACCENT][i], True)
    add_bullet_slide(slide, Inches(0.7), y + Emu(Inches(0.4).emu), Inches(11.8), Inches(1.3),
                     items, 13)

# 保存
output_path = "/home/admin/workspace/SEU/DINOSAM/out/ppt-sam-backbone.pptx"
prs.save(output_path)
print(f"已保存: {output_path}")
