"""生成 PPT2：DINOv3 主干方向技术汇报——重点消融实验设计"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# 配色
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
C_DARK = RGBColor(0x16, 0x21, 0x3E)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def add_bg(slide, color=C_LIGHT_BG):
    bg = slide.background; fill = bg.fill; fill.solid(); fill.fore_color.rgb = color


def tb(slide, l, t, w, h, text, sz=18, c=C_BODY, b=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text; p.font.size = Pt(sz)
    p.font.color.rgb = c; p.font.bold = b; p.font.name = "Microsoft YaHei"; p.alignment = align
    return box


def bullets(slide, l, t, w, h, items, sz=16, c=C_BODY):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item; p.font.size = Pt(sz); p.font.color.rgb = c
        p.font.name = "Microsoft YaHei"; p.space_after = Pt(6)
    return box


def rect(slide, l, t, w, h, fill, text="", sz=12, fc=C_WHITE, b=False, border=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if border: s.line.color.rgb = border; s.line.width = Pt(1.5)
    else: s.line.fill.background()
    if text:
        tf = s.text_frame; tf.word_wrap = True; tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        p = tf.paragraphs[0]; p.text = text; p.font.size = Pt(sz)
        p.font.color.rgb = fc; p.font.bold = b; p.font.name = "Microsoft YaHei"
    return s


def tbl(slide, l, t, w, h, data, hc=C_TITLE):
    rows, cols = len(data), len(data[0])
    ts = slide.shapes.add_table(rows, cols, l, t, w, h)
    table = ts.table
    for i in range(rows):
        for j in range(cols):
            cell = table.cell(i, j); cell.text = data[i][j]
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12); p.font.name = "Microsoft YaHei"
                if i == 0:
                    p.font.bold = True; p.font.color.rgb = C_WHITE
                    cell.fill.solid(); cell.fill.fore_color.rgb = hc
                else: p.font.color.rgb = C_BODY
    return table


# ============================================================
# 第 1 页：封面
# ============================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(s, C_PURPLE)
tb(s, Inches(1.5), Inches(1.8), Inches(10), Inches(1.5),
   "DINOSAM：DINOv3 主干方向", 44, C_WHITE, True, PP_ALIGN.CENTER)
tb(s, Inches(1.5), Inches(3.3), Inches(10), Inches(1),
   "以 DINOv3 ViT-B/16 + 可训练适配器替代 SAM 图像编码器", 22,
   RGBColor(0xDD, 0xCC, 0xEE), False, PP_ALIGN.CENTER)
tb(s, Inches(1.5), Inches(4.5), Inches(10), Inches(0.8),
   "适配器消融实验设计 · 技术方案讨论", 20, RGBColor(0xBB, 0xAA, 0xCC), False, PP_ALIGN.CENTER)
tb(s, Inches(1.5), Inches(5.8), Inches(10), Inches(0.5),
   "2026.06", 16, RGBColor(0x99, 0x88, 0xAA), False, PP_ALIGN.CENTER)

# ============================================================
# 第 2 页：为什么换编码器？
# ============================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(s)
tb(s, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
   "动机：为什么用 DINOv3 替换 SAM 编码器？", 28, C_TITLE, True)

# SAM vs DINOv3 对比
rect(s, Inches(0.5), Inches(1.3), Inches(5.8), Inches(2.5), C_WHITE, border=C_BLUE)
tb(s, Inches(0.7), Inches(1.4), Inches(5.4), Inches(0.4), "SAM ViT-B/16", 18, C_BLUE, True)
bullets(s, Inches(0.7), Inches(1.9), Inches(5.4), Inches(1.8), [
    "✅ SA-1B 大规模分割监督预训练",
    "✅ 强分割空间先验",
    "❌ 表征受限于分割任务本身的监督信号",
    "❌ 可能缺乏通用语义理解能力",
], 13)

rect(s, Inches(7.0), Inches(1.3), Inches(5.8), Inches(2.5), C_WHITE, border=C_PURPLE)
tb(s, Inches(7.2), Inches(1.4), Inches(5.4), Inches(0.4), "DINOv3 ViT-B/16", 18, C_PURPLE, True)
bullets(s, Inches(7.2), Inches(1.9), Inches(5.4), Inches(1.8), [
    "✅ 自监督预训练，通用视觉表征",
    "✅ 多下游任务迁移性能优越",
    "✅ RoPE 位置编码，LayerScale",
    "❌ 输出维度 768 ≠ SAM 内部 256，需要对齐",
], 13)

# 核心问题
rect(s, Inches(0.5), Inches(4.2), Inches(12.3), Inches(1.2),
     RGBColor(0xFF, 0xF3, 0xE0), border=C_ORANGE)
tb(s, Inches(0.7), Inches(4.3), Inches(11.9), Inches(1.0),
   "核心实验问题：对于「分割+深度」联合任务，分割监督的 SAM 先验 vs 自监督的 DINOv3 通用表征，哪个更有效？\n→ 这是整个消融实验体系的最顶层维度，两个 PPT 对应的两个方向分别回答这个问题的两侧", 14, RGBColor(0xE6, 0x51, 0x00))

# 技术挑战
tb(s, Inches(0.5), Inches(5.8), Inches(12), Inches(0.4), "特征对齐的技术挑战", 18, C_ACCENT, True)
d = [
    ["差异", "SAM 编码器", "DINOv3 编码器"],
    ["输出维度", "256", "768 → 需要投影层"],
    ["图像缩放", "保持宽高比 + padding", "直接 resize 到 1024² (可能形变)"],
    ["归一化参数", "SAM 专有 pixel_mean/std", "ImageNet mean/std ×255"],
    ["位置编码", "Learned positional embedding", "RoPE (base=100)"],
]
tbl(s, Inches(0.5), Inches(6.2), Inches(12.3), Inches(1.2), d)

# ============================================================
# 第 3 页：整体架构
# ============================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(s)
tb(s, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
   "DINOv3 方向整体架构", 32, C_TITLE, True)

# DINOv3MonaEncoder 大框
rect(s, Inches(0.5), Inches(1.3), Inches(6.0), Inches(2.8),
     RGBColor(0xFF, 0xF8, 0xE1), border=C_ORANGE)
tb(s, Inches(0.7), Inches(1.4), Inches(5.5), Inches(0.4),
   "DINOv3MonaEncoder", 16, C_ORANGE, True)

rect(s, Inches(0.8), Inches(1.9), Inches(2.0), Inches(0.8),
     RGBColor(0xFF, 0xEA, 0xA7), "DINOv3 ViT-B/16\n🔒 冻结", 11, C_RED, True, C_RED)
rect(s, Inches(3.2), Inches(1.9), Inches(1.2), Inches(0.8),
     RGBColor(0x55, 0xEF, 0xC4), "Adapter\n×2 串联", 11, RGBColor(0x00, 0x69, 0x5C), True, C_GREEN)
rect(s, Inches(4.8), Inches(1.9), Inches(1.2), Inches(0.8),
     RGBColor(0x74, 0xB9, 0xFF), "Linear\n768→256", 11, C_WHITE, True, C_BLUE)

tb(s, Inches(0.8), Inches(3.0), Inches(5.5), Inches(0.8),
   "关键设计：替换对下游分支完全透明\n掩码/深度分支结构与 SAM 方案完全一致", 12, C_BODY)

# 右侧：输出 → 分支
rect(s, Inches(7.0), Inches(1.5), Inches(2.5), Inches(0.6),
     RGBColor(0xDF, 0xF9, 0xFB), "掩码分支 (共享)", 12, C_BLUE, True, C_BLUE)
rect(s, Inches(7.0), Inches(2.5), Inches(2.5), Inches(0.6),
     RGBColor(0xF0, 0xE6, 0xFF), "深度分支 (共享)", 12, C_PURPLE, True, C_PURPLE)

rect(s, Inches(10.0), Inches(1.5), Inches(1.2), Inches(0.6),
     RGBColor(0x55, 0xEF, 0xC4), "掩码", 12, RGBColor(0x00, 0x69, 0x5C), True)
rect(s, Inches(10.0), Inches(2.5), Inches(1.2), Inches(0.6),
     RGBColor(0xFF, 0xEA, 0xA7), "深度", 12, RGBColor(0x6C, 0x51, 0x00), True)

# DINOv3 配置表
tb(s, Inches(0.5), Inches(4.5), Inches(12), Inches(0.4), "DINOv3 ViT-B/16 骨干配置", 18, C_TITLE, True)
d2 = [
    ["参数", "值", "参数", "值"],
    ["patch_size", "16", "embed_dim", "768"],
    ["depth", "12", "num_heads", "12"],
    ["ffn_ratio", "4 (MLP=3072)", "pos_embed", "RoPE (base=100)"],
    ["layerscale_init", "1×10⁻⁵", "norm_layer", "LayerNorm BF16"],
    ["n_storage_tokens", "4", "drop_path_rate", "0.0"],
]
tbl(s, Inches(0.5), Inches(5.0), Inches(12.3), Inches(2.0), d2)

# ============================================================
# 第 4 页：★★★ 适配器消融设计逻辑
# ============================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(s, C_DARK)
tb(s, Inches(0.8), Inches(0.3), Inches(10), Inches(0.7),
   "★ 适配器消融设计逻辑", 32, C_WHITE, True)
tb(s, Inches(0.8), Inches(0.9), Inches(10), Inches(0.4),
   "核心问题：对于稠密预测迁移，何种归纳偏置最有效？", 18, RGBColor(0xFF, 0xEA, 0xA7))

# 三种适配器卡片
adapters = [
    ("FCAdapter\n通道基线", "纯全连接瓶颈\n768→96→768",
     "无空间操作\n仅通道变换\n→ 测量纯通道变换的下限",
     C_BLUE, "控制组"),
    ("MoNA\n多尺度卷积", "瓶颈 + MonaOp\n3×3 + 5×5 + 7×7 DWConv",
     "局部多尺度空间信息\n三种尺度均值聚合\n→ 评估局部空间信息的贡献",
     C_GREEN, "默认方案"),
    ("DualAttnAdapter\n双重注意力", "PAM + CAM\n位置注意力 + 通道注意力",
     "全局长程依赖\n无瓶颈压缩\n→ 评估全局上下文的价值",
     C_PURPLE, "注意力方案"),
]
for i, (name, mech, note, color, role) in enumerate(adapters):
    x = Inches(0.5 + i * 4.2)
    rect(s, x, Inches(1.6), Inches(3.9), Inches(3.6), RGBColor(0x1E, 0x2D, 0x4A), border=color)
    tb(s, x + Inches(0.15), Inches(1.7), Inches(3.6), Inches(0.6), name, 15, color, True)
    rect(s, x + Inches(0.15), Inches(2.4), Inches(1.0), Inches(0.3),
         color, role, 10, C_WHITE, True)
    tb(s, x + Inches(0.15), Inches(2.8), Inches(3.6), Inches(0.5), mech, 12, RGBColor(0xDD, 0xDD, 0xDD))
    tb(s, x + Inches(0.15), Inches(3.5), Inches(3.6), Inches(1.5), note, 11, RGBColor(0xBB, 0xCC, 0xDD))

# 消融逻辑箭头
tb(s, Inches(4.0), Inches(5.4), Inches(5.3), Inches(0.5),
   "FC (纯通道) →  +局部多尺度空间(MoNA) →  换为全局注意力(DualAttn)", 13, C_ACCENT, True, PP_ALIGN.CENTER)

# 共性
tb(s, Inches(0.5), Inches(6.0), Inches(12), Inches(0.4),
   "共性：相同的瓶颈压缩比(768/8=96)、残差连接结构、零初始化策略", 14, RGBColor(0x88, 0x99, 0xAA), False, PP_ALIGN.CENTER)
tb(s, Inches(0.5), Inches(6.4), Inches(12), Inches(0.4),
   "唯一变量：核心变换机制（无/局部卷积/全局注意力）", 14, RGBColor(0xFF, 0xEA, 0xA7), True, PP_ALIGN.CENTER)

# ============================================================
# 第 5 页：MoNA 适配器详细结构
# ============================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(s)
tb(s, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
   "MoNA — 多尺度卷积适配器（默认方案）", 28, C_TITLE, True)
tb(s, Inches(0.8), Inches(1.0), Inches(10), Inches(0.4),
   "核心假设：稠密预测任务受益于多尺度局部空间信息的注入", 16, C_GREEN, True)

# 数据流
steps = [
    ("输入 x\n(B, N, 768)", RGBColor(0xE8, 0xF4, 0xF8), C_TITLE),
    ("仿射归一化\nLN(x)·γ + x·γₓ", RGBColor(0xDF, 0xE6, 0xE9), C_BODY),
    ("下投影\n768 → 96", RGBColor(0x74, 0xB9, 0xFF), C_WHITE),
    ("Reshape\n→ (B,96,H,W)", RGBColor(0xDF, 0xE6, 0xE9), C_BODY),
    ("MonaOp\n3/5/7 DWConv", RGBColor(0x55, 0xEF, 0xC4), RGBColor(0x00, 0x69, 0x5C)),
    ("GELU + Drop", RGBColor(0xDF, 0xE6, 0xE9), C_BODY),
    ("上投影\n96 → 768", RGBColor(0x74, 0xB9, 0xFF), C_WHITE),
    ("+ 残差\n→ 输出", RGBColor(0xA2, 0x9B, 0xFE), C_WHITE),
]
for i, (text, bg, fg) in enumerate(steps):
    rect(s, Inches(0.3 + i * 1.6), Inches(1.6), Inches(1.4), Inches(0.9), bg, text, 10, fg, False, fg)

# MonaOp 细节
rect(s, Inches(0.5), Inches(2.9), Inches(12.3), Inches(2.5), C_WHITE, border=C_GREEN)
tb(s, Inches(0.7), Inches(3.0), Inches(5), Inches(0.4), "MonaOp 内部 — 多尺度深度卷积模块", 16, C_GREEN, True)

convs = [
    ("DWConv 3×3", "groups=96, pad=1", "捕获细粒度局部特征"),
    ("DWConv 5×5", "groups=96, pad=2", "捕获中等尺度局部特征"),
    ("DWConv 7×7", "groups=96, pad=3", "捕获粗粒度局部特征"),
]
for i, (name, param, desc) in enumerate(convs):
    y = Inches(3.5 + i * 0.6)
    rect(s, Inches(0.8), y, Inches(2.0), Inches(0.45), RGBColor(0x55, 0xEF, 0xC4), name, 11, RGBColor(0x00, 0x69, 0x5C), True, C_GREEN)
    tb(s, Inches(3.0), y, Inches(2.5), Inches(0.45), param, 11, C_GRAY)
    tb(s, Inches(5.5), y, Inches(3), Inches(0.45), desc, 11, C_BODY)

tb(s, Inches(9.0), Inches(3.5), Inches(3.5), Inches(1.5),
   "融合方式：\n(conv3 + conv5 + conv7) / 3\n+ identity\n+ 1×1 Conv projector\n\n初始化：\nγ = 1×10⁻⁶ (几乎为零)\nγₓ = 1.0 (保留 identity)\n→ 初始时 ≈ 恒等映射", 12, C_BODY)

# 参数量
tb(s, Inches(0.5), Inches(5.6), Inches(12), Inches(0.4), "关键参数", 16, C_TITLE, True)
d3 = [
    ["参数", "值", "说明"],
    ["bottleneck_dim", "96 (768/8)", "瓶颈维度"],
    ["γ (gamma)", "1×10⁻⁶ 初始化", "控制适配器初始输出接近零"],
    ["γₓ (gammax)", "1.0 初始化", "保留输入 identity"],
    ["Dropout", "0.1", "正则化"],
    ["串联数量", "2 (adapter1 + adapter2)", "两个 Mona 串联"],
]
tbl(s, Inches(0.5), Inches(6.0), Inches(12.3), Inches(1.3), d3)

# ============================================================
# 第 6 页：FCAdapter 与 DualAttnAdapter
# ============================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(s)
tb(s, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
   "FCAdapter 与 DualAttnAdapter 详细结构", 28, C_TITLE, True)

# FCAdapter
rect(s, Inches(0.5), Inches(1.2), Inches(5.8), Inches(2.8), C_WHITE, border=C_BLUE)
tb(s, Inches(0.7), Inches(1.3), Inches(5.4), Inches(0.4),
   "FCAdapter — 纯全连接基线", 18, C_BLUE, True)

fc_steps = [
    ("LN·γ+x·γₓ", RGBColor(0xDF, 0xE6, 0xE9)),
    ("768→96", RGBColor(0x74, 0xB9, 0xFF)),
    ("GELU", RGBColor(0xFA, 0xB1, 0xA0)),
    ("Drop 0.1", RGBColor(0xDF, 0xE6, 0xE9)),
    ("96→768", RGBColor(0x74, 0xB9, 0xFF)),
    ("+残差", RGBColor(0xA2, 0x9B, 0xFE)),
]
for i, (text, bg) in enumerate(fc_steps):
    rect(s, Inches(0.7 + i * 0.85), Inches(1.9), Inches(0.75), Inches(0.5),
         bg, text, 9, C_BODY if bg != RGBColor(0x74, 0xB9, 0xFF) else C_WHITE, False)

tb(s, Inches(0.7), Inches(2.6), Inches(5.4), Inches(1.2),
   "与 MoNA 的唯一区别：移除了 MonaOp (3/5/7 DWConv)\n仅保留通道瓶颈变换\n→ 直接测量多尺度空间卷积的边际贡献\n\n共享：仿射归一化、γ/γₓ 初始化、Dropout、残差", 12, C_BODY)

# DualAttnAdapter
rect(s, Inches(7.0), Inches(1.2), Inches(5.8), Inches(5.5), C_WHITE, border=C_PURPLE)
tb(s, Inches(7.2), Inches(1.3), Inches(5.4), Inches(0.4),
   "DualAttnAdapter — PAM + CAM 并行", 18, C_PURPLE, True)

# PAM
rect(s, Inches(7.3), Inches(1.9), Inches(2.5), Inches(2.2), RGBColor(0xE8, 0xF4, 0xF8), border=C_BLUE)
tb(s, Inches(7.4), Inches(2.0), Inches(2.3), Inches(0.3), "PAM 位置注意力", 12, C_BLUE, True)
bullets(s, Inches(7.4), Inches(2.3), Inches(2.3), Inches(1.6), [
    "Q,K: Conv1×1 C→C/8",
    "V: Conv1×1 C→C",
    "A = softmax(Qᵀ×K)",
    "out = α·(V×Aᵀ)",
    "α 初始化为 0",
], 10, C_BODY)

# CAM
rect(s, Inches(10.0), Inches(1.9), Inches(2.5), Inches(2.2), RGBColor(0xF0, 0xE6, 0xFF), border=C_PURPLE)
tb(s, Inches(10.1), Inches(2.0), Inches(2.3), Inches(0.3), "CAM 通道注意力", 12, C_PURPLE, True)
bullets(s, Inches(10.1), Inches(2.3), Inches(2.3), Inches(1.6), [
    "无参数化 Q/K/V",
    "E = Q×K (B,C,C)",
    "E' = max(E)-E (稳定)",
    "out = β·softmax(E')·Q",
    "β 初始化为 0",
], 10, C_BODY)

# 融合
rect(s, Inches(7.3), Inches(4.3), Inches(5.2), Inches(0.5),
     RGBColor(0xFF, 0xEA, 0xA7), "输出 = x + (PAM(X_2d) + CAM(X_2d)) / 2", 12, RGBColor(0x6C, 0x51, 0x00), True)

tb(s, Inches(7.3), Inches(5.0), Inches(5.2), Inches(1.5),
   "与 MoNA/FC 的结构差异：\n• 无仿射归一化（直接操作原始特征）\n• 无瓶颈压缩（PAM 的 QK 降维仅限注意力计算）\n• α/β=0 初始化 vs γ=1e-6 初始化\n• 参数量和计算复杂度更高——消融目标是归纳偏置类型", 12, C_BODY)

# 底部：结构对比表
tb(s, Inches(0.5), Inches(4.5), Inches(5.5), Inches(0.4), "三种适配器结构对比", 16, C_TITLE, True)
d4 = [
    ["属性", "MoNA", "FCAdapter", "DualAttn"],
    ["空间建模", "局部多尺度3/5/7", "无", "全局自注意力"],
    ["通道建模", "线性瓶颈", "线性瓶颈", "通道自注意力"],
    ["瓶颈维度", "96", "96", "无瓶颈"],
    ["零初始化", "γ=10⁻⁶", "γ=10⁻⁶", "α=β=0"],
]
tbl(s, Inches(0.5), Inches(4.9), Inches(5.8), Inches(1.5), d4)

# ============================================================
# 第 7 页：深度分支 Transformer 消融
# ============================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(s)
tb(s, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
   "深度分支 Transformer 消融（--depth_transformer_type）", 28, C_TITLE, True)
tb(s, Inches(0.8), Inches(1.0), Inches(10), Inches(0.4),
   "HTML 文档中未提及的消融维度——源码中发现的额外选项", 16, C_ACCENT, True)

# TwoWay vs DualAttn
rect(s, Inches(0.5), Inches(1.6), Inches(5.8), Inches(3.5), C_WHITE, border=C_BLUE)
tb(s, Inches(0.7), Inches(1.7), Inches(5.4), Inches(0.4), "TwoWayTransformer（默认）", 18, C_BLUE, True)
bullets(s, Inches(0.7), Inches(2.2), Inches(5.4), Inches(2.5), [
    "SAM 原生结构，2 层 TwoWayAttentionBlock",
    "每层：token self-attn → token→image cross-attn → MLP → image→token cross-attn",
    "权重从 SAM mask_decoder.transformer 迁移",
    "d=2, h=8, mlp_dim=2048",
    "",
    "优势：继承 SAM 已学到的注意力模式",
    "风险：SAM 学到的注意力模式是否适合深度任务？",
], 12)

rect(s, Inches(7.0), Inches(1.6), Inches(5.8), Inches(3.5), C_WHITE, border=C_PURPLE)
tb(s, Inches(7.2), Inches(1.7), Inches(5.4), Inches(0.4), "DualAttnTransformer（新增）", 18, C_PURPLE, True)
bullets(s, Inches(7.2), Inches(2.2), Inches(5.4), Inches(2.5), [
    "PAM+CAM 替换 token self-attention",
    "每层：PAM+CAM增强image → token→image cross-attn → MLP → image→token cross-attn",
    "随机初始化（与 SAM Transformer 结构不兼容）",
    "d=2, h=8, mlp_dim=2048",
    "",
    "优势：全局空间+通道注意力增强图像特征",
    "风险：随机初始化 vs 迁移权重的差距",
], 12)

# 对比表
tb(s, Inches(0.5), Inches(5.5), Inches(12), Inches(0.4), "关键对比", 16, C_TITLE, True)
d5 = [
    ["维度", "TwoWayTransformer", "DualAttnTransformer"],
    ["结构", "SAM 原生，token self-attention", "PAM+CAM 增强 image features"],
    ["权重初始化", "从 SAM mask_decoder 迁移", "完全随机初始化（跳过迁移）"],
    ["与适配器关系", "独立于编码器适配器", "可与编码器适配器中的 DualAttn 搭配"],
    ["实验价值", "基线：迁移学习的上限", "探索：新架构是否能超越迁移"],
]
tbl(s, Inches(0.5), Inches(5.9), Inches(12.3), Inches(1.3), d5)

# ============================================================
# 第 8 页：训练策略差异
# ============================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(s)
tb(s, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
   "DINOv3 方向训练策略 — 与 SAM 方案的差异", 28, C_TITLE, True)

# Phase 1 对比
rect(s, Inches(0.5), Inches(1.3), Inches(12.3), Inches(2.8), C_WHITE, border=C_BLUE)
tb(s, Inches(0.7), Inches(1.4), Inches(5), Inches(0.4), "Phase 1 对比", 18, C_BLUE, True)

d6 = [
    ["", "SAM 方案", "DINOv3 方案"],
    ["冻结组件", "Image Encoder + 深度分支", "DINOv3 骨干 + SAM ViT-B + 深度分支"],
    ["可训练组件", "PromptEncoder + MaskDecoder", "Adapter1&2 + Projection + PE + MD"],
    ["编码器可训练部分", "无", "适配器 + 投影层 ← 核心区别"],
    ["预处理", "ResizeLongestSide + padding", "直接 resize 到 1024²"],
]
tbl(s, Inches(0.7), Inches(1.9), Inches(11.8), Inches(1.8), d6)

# Phase 2
rect(s, Inches(0.5), Inches(4.5), Inches(12.3), Inches(1.5), C_WHITE, border=C_PURPLE)
tb(s, Inches(0.7), Inches(4.6), Inches(5), Inches(0.4), "Phase 2：两种方案完全一致", 16, C_PURPLE, True)
bullets(s, Inches(0.7), Inches(5.0), Inches(11.5), Inches(0.8), [
    "• 冻结所有 Phase 1 参数（DINOv3 方案中适配器也被冻结）",
    "• 仅 outer_prompt_encoder + DepthMaskDecoder 可训练",
    "• 损失函数、优化器、调度器完全一致 → 保证消融公平性",
], 13)

# 消融意义
rect(s, Inches(0.5), Inches(6.2), Inches(12.3), Inches(1.0),
     RGBColor(0xFF, 0xF3, 0xE0), border=C_ORANGE)
tb(s, Inches(0.7), Inches(6.3), Inches(11.8), Inches(0.8),
   "消融意义：Phase 1 的差异 = 编码器表征能力的差异\nPhase 2 一致 → 深度性能差异可完全归因于编码器特征质量 + 适配器设计", 14, RGBColor(0xE6, 0x51, 0x00))

# ============================================================
# 第 9 页：★★★ 完整消融实验矩阵
# ============================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(s, C_DARK)
tb(s, Inches(0.8), Inches(0.3), Inches(10), Inches(0.7),
   "★ DINOv3 方向完整消融实验矩阵", 32, C_WHITE, True)
tb(s, Inches(0.8), Inches(0.9), Inches(10), Inches(0.4),
   "三个层次的消融：编码器级 → 适配器级 → 解码器级", 16, RGBColor(0xFF, 0xEA, 0xA7))

# 层次 1：编码器
rect(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(1.2),
     RGBColor(0x1E, 0x2D, 0x4A), border=C_ACCENT)
tb(s, Inches(0.7), Inches(1.6), Inches(2), Inches(0.3), "层次 1：编码器", 14, C_ACCENT, True)
tb(s, Inches(0.7), Inches(1.9), Inches(11.8), Inches(0.6),
   "--encoder: sam ↔ dinov3\n最顶层消融 — 两个 PPT 方向的对比即是此维度的实验", 12, RGBColor(0xBB, 0xCC, 0xDD))

# 层次 2：适配器
rect(s, Inches(0.5), Inches(2.9), Inches(12.3), Inches(2.0),
     RGBColor(0x1E, 0x2D, 0x4A), border=C_GREEN)
tb(s, Inches(0.7), Inches(3.0), Inches(3), Inches(0.3), "层次 2：适配器（DINOv3 专有）", 14, C_GREEN, True)

adapters_data = [
    ["实验", "适配器", "核心变换", "评估重点"],
    ["F1", "MoNA (默认)", "多尺度 DWConv 3/5/7", "局部空间信息的贡献"],
    ["F2", "FCAdapter", "纯通道 MLP", "基线：无空间操作的下限"],
    ["F3", "DualAttnAdapter", "PAM + CAM 全局注意力", "全局上下文 vs 局部卷积"],
]
tbl(s, Inches(0.7), Inches(3.4), Inches(11.8), Inches(1.3), adapters_data, C_GREEN)

# 层次 3：解码器
rect(s, Inches(0.5), Inches(5.1), Inches(12.3), Inches(1.5),
     RGBColor(0x1E, 0x2D, 0x4A), border=C_PURPLE)
tb(s, Inches(0.7), Inches(5.2), Inches(3), Inches(0.3), "层次 3：深度分支 Transformer", 14, C_PURPLE, True)

dec_data = [
    ["实验", "Transformer", "权重初始化", "评估重点"],
    ["G1", "TwoWay (默认)", "SAM 迁移", "迁移学习的效果"],
    ["G2", "DualAttn", "随机初始化", "新架构 vs 迁移学习"],
]
tbl(s, Inches(0.7), Inches(5.6), Inches(11.8), Inches(0.8), dec_data, C_PURPLE)

# 交叉消融
tb(s, Inches(0.5), Inches(6.8), Inches(12), Inches(0.5),
   "交叉消融：F1-F3 × G1-G2 = 6 组实验 + 共享迭代参数消融 B/C/D（SAM PPT 中的维度）", 13, RGBColor(0xFF, 0xEA, 0xA7), True, PP_ALIGN.CENTER)

# ============================================================
# 第 10 页：消融实验具体方案表
# ============================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(s)
tb(s, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
   "消融实验具体方案 & 优先级", 28, C_TITLE, True)

d7 = [
    ["编号", "层次", "变量", "基线", "指标", "优先级"],
    ["SAM vs DINOv3", "编码器", "sam ↔ dinov3 (MoNA)", "sam", "Dice/IoU + MAE/R²", "🔴 P0"],
    ["F1 vs F2 vs F3", "适配器", "mona ↔ fc ↔ dual_attn", "mona", "Dice/IoU + MAE/R²", "🔴 P0"],
    ["G1 vs G2", "Depth Trans.", "twoway ↔ dual_attn", "twoway", "Depth MAE/R²", "🔴 P0"],
    ["F×G 交叉", "适配器×Trans.", "3 adapter × 2 trans", "mona+twoway", "全部指标", "🟡 P1"],
    ["B1-B4", "迭代次数", "n_sub=1,2,3,4", "n_sub=1", "Dice + MAE", "🟡 P1"],
    ["C1-C5", "掩码传入", "prob=0~1.0", "prob=0.5", "Dice + MAE", "🟢 P2"],
]
tbl(s, Inches(0.3), Inches(1.2), Inches(12.7), Inches(2.5), d7)

# 实验数量估算
rect(s, Inches(0.5), Inches(4.0), Inches(12.3), Inches(1.5), C_WHITE, border=C_ORANGE)
tb(s, Inches(0.7), Inches(4.1), Inches(3), Inches(0.4), "实验工作量估算", 16, C_ORANGE, True)
bullets(s, Inches(0.7), Inches(4.5), Inches(11.8), Inches(0.9), [
    "P0 必做实验：编码器对比(2) + 适配器消融(3) + Transformer消融(2) = 7 组",
    "P1 推荐实验：交叉消融(6) + 迭代消融(4) = 10 组",
    "P2 可选实验：掩码概率(5) + 其他超参 = ~8 组",
    "每组实验 = Phase1(50 epoch) + Phase2(50 epoch)，总计 25 组起",
], 13)

# 建议的实验顺序
tb(s, Inches(0.5), Inches(5.8), Inches(12), Inches(0.4), "建议实验顺序", 16, C_ACCENT, True)
bullets(s, Inches(0.5), Inches(6.2), Inches(12), Inches(1.2), [
    "Step 1：先跑 SAM 基线 (twoway) 作为所有对比的 anchor",
    "Step 2：跑 DINOv3 + MoNA (twoway) 确认编码器替换的基本效果",
    "Step 3：三种适配器消融 (F1-F3)，固定 twoway",
    "Step 4：最佳适配器 × dual_attn Transformer (G2)",
    "Step 5：最佳配置上跑迭代消融 (B1-B4)",
], 13)

# ============================================================
# 第 11 页：讨论点
# ============================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(s)
tb(s, Inches(0.8), Inches(0.4), Inches(10), Inches(0.7),
   "与老师讨论的开放问题", 32, C_TITLE, True)

qs = [
    ("适配器设计", [
        "1. DualAttnAdapter 没有瓶颈压缩，参数量与 MoNA/FC 不同——是否影响公平对比？",
        "2. 是否需要增加更多适配器类型（如 LoRA、Spatial Adapter）扩展消融维度？",
        "3. 适配器串联 2 个是否最优？1 个 / 3 个的消融是否有必要？",
    ]),
    ("架构决策", [
        "4. DINOv3 直接 resize (不保持宽高比) 是否导致形变影响性能？",
        "5. depth_transformer_type=dual_attn 跳过权重迁移——是否应设计兼容的迁移方案？",
        "6. 编码器适配器中的 DualAttn 与深度分支的 DualAttnTransformer 是否有协同效应？",
    ]),
    ("实验方法论", [
        "7. 消融实验是否需要多次随机种子求均值和标准差？",
        "8. 交叉消融 (F×G) 的 6 组实验是否过多？哪些组合可以跳过？",
        "9. 论文中应如何组织这些消融结果——分层展示还是单一大表？",
    ]),
]
for i, (title, items) in enumerate(qs):
    y = Inches(1.3 + i * 2.0)
    rect(s, Inches(0.5), y, Inches(12.3), Inches(1.8), C_WHITE, border=[C_GREEN, C_PURPLE, C_ORANGE][i])
    tb(s, Inches(0.7), y + Inches(0.05), Inches(3), Inches(0.4),
       title, 16, [C_GREEN, C_PURPLE, C_ORANGE][i], True)
    bullets(s, Inches(0.7), y + Inches(0.4), Inches(11.8), Inches(1.3), items, 13)

# ============================================================
# 第 12 页：两方向消融实验全景图
# ============================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(s, C_DARK)
tb(s, Inches(0.8), Inches(0.3), Inches(10), Inches(0.7),
   "★ 两方向消融实验全景图", 32, C_WHITE, True)
tb(s, Inches(0.8), Inches(0.9), Inches(10), Inches(0.4),
   "SAM 方向 + DINOv3 方向 = 完整消融实验体系", 16, RGBColor(0xFF, 0xEA, 0xA7))

# 层次树
levels = [
    ("最顶层", "编码器选择", "--encoder sam | dinov3", "跨两个 PPT 方向的对比", C_ACCENT, Inches(1.5)),
    ("编码器级", "适配器类型 (DINOv3)", "--adapter_type mona|fc|dual_attn", "三种归纳偏置消融", C_GREEN, Inches(2.3)),
    ("解码器级", "深度 Transformer", "--depth_transformer_type twoway|dual_attn", "迁移学习 vs 新架构", C_PURPLE, Inches(3.1)),
    ("训练级", "迭代策略", "--n_sub 1~4, --mask_prob 0~1", "提示精炼的效果", C_BLUE, Inches(3.9)),
    ("训练级", "训练配置", "--train_phase, --lr, --depth_loss_weight", "训练策略的影响", C_ORANGE, Inches(4.7)),
]
for title, name, param, desc, color, y in levels:
    rect(s, Inches(0.5), y, Inches(1.8), Inches(0.6), color, title, 11, C_WHITE, True)
    rect(s, Inches(2.5), y, Inches(2.5), Inches(0.6), RGBColor(0x1E, 0x2D, 0x4A), name, 12, color, True, color)
    tb(s, Inches(5.2), y, Inches(4), Inches(0.6), param, 11, RGBColor(0xBB, 0xCC, 0xDD))
    tb(s, Inches(9.5), y, Inches(3.5), Inches(0.6), desc, 11, RGBColor(0x99, 0xAA, 0xBB))

# 统计
rect(s, Inches(0.5), Inches(5.6), Inches(12.3), Inches(1.5),
     RGBColor(0x1E, 0x2D, 0x4A), border=RGBColor(0xFF, 0xEA, 0xA7))
tb(s, Inches(0.7), Inches(5.7), Inches(11.8), Inches(1.3),
   "实验总量统计：\n• P0 必做：7 组（编码器2 + 适配器3 + Transformer2）\n• P1 推荐：14 组（交叉6 + 迭代4 + 训练策略4）\n• P2 可选：~8 组（超参调优）\n• 总计约 25-30 组实验，每组 100 epoch（Phase1+Phase2）", 13, RGBColor(0xDD, 0xDD, 0xDD))

# 保存
output_path = "/home/admin/workspace/SEU/DINOSAM/out/ppt-dino-backbone.pptx"
prs.save(output_path)
print(f"已保存: {output_path}")
