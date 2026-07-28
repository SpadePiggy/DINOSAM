# model_DINO 实验变体对比报告

> 数据来源：`/data3/LXT/DINOv3-SAM/model_DINO/` 下各子目录的训练 config（`config.txt` / `config.log`）、评估脚本（`run_eval_*.sh` / `launch_*.sh`）与评估日志（`eval_iter*/log.txt`、`eval_*.log`）。
> 生成日期：2026-07-17

---

## 1. 变体总览

共 20 个模型目录，分两大系列 + 2 个辅助目录：

| 系列 | 目录 | 训练时间 | 说明 |
|---|---|---|---|
| **OPT 系列**（单尺度 adapter 消融） | `{mona, fc, dual_attn}_{iter8, simple, bd, bd5}_OPT` 共 12 个 | 2026-06-19 ~ 06-29 | 对比 3 种 DINOv3→SAM 特征适配器 × 4 种训练变体，均在 `0_sam` 上训练 |
| **DPT 系列**（多层特征融合 + 层选择消融） | `DPT_{0sam, 1724}[_layersXXXX]_bs8_iter8_ep50` 共 8 个 | 2026-06-30 ~ 07-08 | `dpt_simple` 多层融合 adapter，消融 `dpt_layers` 层选择，分别在 `0_sam` / `1724` 上训练 |
| 辅助 | `DINO/` | 2026-05-11 | 早期 baseline，仅存 checkpoint 与 eval 可视化，**无 config / log 记录** |
| 辅助 | `test/` | 2026-06-19 | 冒烟测试（quick_test），不参与对比 |

**注意事项（数据完整性）：**
- `*_bd5_OPT`（boundary loss w=5）三个目录 `train.log` 均为 0 字节、`logs/` 为空、无 checkpoint —— **训练未成功启动，无任何结果**。
- `*_bd_OPT` 的 non-iter 评估日志（1724_test）只完成了 Phase 1，Phase 2 加载后日志截断，**无 P2 结果**。
- `*_simple_OPT` 与 `*_bd_OPT`（non-iter）的评估用的是 `1724_test`，但训练集是 `0_sam_train`，属于**跨数据集评估**，与 DPT_1724 系列（1724 训练）不可直接比较。
- `*_iter8_OPT` 使用早期版本评估脚本，日志中**无 Boundary IoU / Precision / Recall** 输出。

---

## 2. 训练配置对比

### 2.1 公共参数（所有变体一致）

| 参数 | 值 |
|---|---|
| encoder | `dinov3`（ViT-B/16，`dinov3_vitb16_pretrain_lvd1689m`） |
| SAM checkpoint | `sam_vit_b_01ec64.pth`（ViT-B） |
| train_phase | both（Phase1 mask → Phase2 depth 两阶段） |
| epochs | 50 |
| lr | 1e-5 |
| n_sub_iterations（训练） | 8 |
| depth_loss_weight | 1.0 |
| max_points（训练） | 3 |
| mask_prob（训练） | 0.5 |

### 2.2 各变体差异参数

| 模型 | adapter_type | dpt_layers | depth_transformer | boundary_loss | batch_size | 训练集（样本数） |
|---|---|---|---|---|---|---|
| DPT_0sam_bs8_iter8_ep50 | dpt_simple | **2,5,8,11**（默认） | twoway | — | 8 | 0_sam（1345） |
| DPT_0sam_layers0369 | dpt_simple | **0,3,6,9** | twoway | — | 8 | 0_sam（1345） |
| DPT_0sam_layers36911 | dpt_simple | **3,6,9,11** | twoway | — | 8 | 0_sam（1345） |
| DPT_0sam_layers_all | dpt_simple | **0–11 全部 12 层** | twoway | — | 8 | 0_sam（1345） |
| DPT_1724_bs8_iter8_ep50 | dpt_simple | 2,5,8,11 | twoway | — | 8 | 1724（880） |
| DPT_1724_layers0369 | dpt_simple | 0,3,6,9 | twoway | — | 8 | 1724（880） |
| DPT_1724_layers36911 | dpt_simple | 3,6,9,11 | twoway | — | 8 | 1724（880） |
| DPT_1724_layers_all | dpt_simple | 0–11 全部 | twoway | — | 8 | 1724（880） |
| mona_iter8_OPT | mona | —（单层） | twoway | — | 4 | 0_sam（1345） |
| fc_iter8_OPT | fc | — | twoway | — | 4 | 0_sam（1345） |
| dual_attn_iter8_OPT | dual_attn | — | twoway | — | 4 | 0_sam（1345） |
| mona_simple_OPT | mona | — | **simple** | — | 4 | 0_sam（1345） |
| fc_simple_OPT | fc | — | **simple** | — | 4 | 0_sam（1345） |
| dual_attn_simple_OPT | dual_attn | — | **simple** | — | 4 | 0_sam（1345） |
| mona_bd_OPT | mona | — | twoway | **w=1.0, width=3** | 4 | 0_sam（1345） |
| fc_bd_OPT | fc | — | twoway | **w=1.0, width=3** | 4 | 0_sam（1345） |
| dual_attn_bd_OPT | dual_attn | — | twoway | **w=1.0, width=3** | 4 | 0_sam（1345） |
| mona/fc/dual_attn_bd5_OPT | 同上 | — | twoway | w=5.0, width=3 | 4 | ⚠️ 训练未启动 |

### 2.3 评估配置

| 参数 | 值 |
|---|---|
| max_points（评估） | 8 |
| mask_prob（评估） | 0.0 |
| n_sub_iterations | 1（non-iter / iter1）或 8（iter / iter8） |
| batch_size | 8 |
| 测试集 | 0_sam_test（343 样本）/ 1724_test（220 样本） |

评估分别加载 `phase1_best.pt`（仅 mask 训练）与 `phase2_best.pt`（深度训练后），下表分割指标除注明外均取 **Phase 2** 模型（P1/P2 分割差异普遍 ≤0.005），深度指标取 Phase 2（Phase 1 深度头未训练，MAE≈50–70、R² 为负，无参考意义）。

---

## 3. 实验结果

### 3.1 0_sam_test · 迭代评估（n_sub=8）

| 模型 | Dice | IoU | BoundIoU | Precision | Recall | Depth MAE | Depth RMSE | Depth R² |
|---|---|---|---|---|---|---|---|---|
| **DPT 3,6,9,11** | **0.9637** | **0.9310** | 0.2049 | 0.9637 | 0.9660 | 7.06 | 9.53 | 0.9744 |
| DPT 0,3,6,9 | 0.9593 | 0.9250 | **0.2091** | 0.9647 | 0.9592 | **6.10** | 8.73 | 0.9785 |
| DPT 2,5,8,11（默认） | 0.9587 | 0.9238 | 0.2040 | 0.9555 | 0.9662 | 6.18 | **8.70** | **0.9786** |
| DPT 全部 12 层 | 0.9554 | 0.9181 | 0.1917 | 0.9482 | 0.9681 | 7.40 | 11.25 | 0.9643 |
| mona_iter8 | 0.9500 | 0.9061 | — | — | — | 7.83 | 13.06 | 0.9519 |
| dual_attn_iter8 | 0.9462 | 0.9000 | — | — | — | 8.30 | 12.75 | 0.9542 |
| fc_iter8 | 0.9428 | 0.8940 | — | — | — | 7.75 | 11.71 | 0.9613 |
| mona_bd | 0.9243 | 0.8623 | 0.0725 | 0.8694 | 0.9916 | 7.23 | 10.29 | 0.9701 |
| fc_bd | 0.9217 | 0.8580 | 0.0724 | 0.8662 | 0.9903 | 8.61 | 13.73 | 0.9468 |
| dual_attn_bd | 0.9146 | 0.8464 | 0.0604 | 0.8523 | 0.9926 | 11.57 | 18.81 | 0.9002 |

### 3.2 0_sam_test · 单次点击（n_sub=1）

n_sub=1 时 P1 与 P2 分割结果完全相同（同一初始 prompt，分割头未变）。

| 模型 | Dice | IoU | BoundIoU | Depth MAE | Depth R² |
|---|---|---|---|---|---|
| **DPT 0,3,6,9** | **0.8687** | **0.8045** | **0.1297** | **5.80** | 0.9806 |
| DPT 2,5,8,11 | 0.8653 | 0.7980 | 0.1158 | 5.98 | 0.9783 |
| DPT 全部 12 层 | 0.8642 | 0.7969 | 0.1173 | 6.77 | 0.9668 |
| DPT 3,6,9,11 | 0.8641 | 0.7996 | 0.1243 | 6.08 | **0.9820** |
| dual_attn_iter8 | 0.8590 | 0.7867 | — | 7.73 | 0.9613 |
| mona_iter8 | 0.8506 | 0.7736 | — | 7.28 | 0.9552 |
| fc_iter8 | 0.8383 | 0.7538 | — | 7.51 | 0.9637 |

### 3.3 1724_test · 迭代评估（n_sub=8）

| 模型 | Dice | IoU | BoundIoU | Precision | Recall | Depth MAE | Depth RMSE | Depth R² |
|---|---|---|---|---|---|---|---|---|
| DPT 0,3,6,9 | **0.9307** | **0.8755** | **0.2335** | 0.9376 | 0.9321 | 10.55 | 16.28 | 0.9243 |
| DPT 3,6,9,11 | 0.9308 | 0.8752 | 0.2309 | 0.9266 | 0.9426 | **10.19** | **14.92** | **0.9364** |
| DPT 2,5,8,11（默认） | 0.9292 | 0.8728 | 0.2304 | 0.9373 | 0.9291 | 11.17 | 18.92 | 0.8978 |
| DPT 全部 12 层 | 0.9231 | 0.8638 | 0.2212 | 0.9320 | 0.9261 | 11.75 | 17.71 | 0.9103 |
| dual_attn_simple † | 0.8659 | 0.7725 | 0.1013 | 0.8710 | 0.8851 | 39.52 | 49.67 | 0.2950 |
| fc_simple † | 0.8629 | 0.7686 | 0.0999 | 0.8610 | 0.8905 | 40.38 | 51.05 | 0.2555 |
| mona_simple † | 0.8548 | 0.7573 | 0.0961 | 0.9120 | 0.8333 | 40.31 | 49.38 | 0.3033 |

† simple 系列在 `0_sam_train` 上训练、在 `1724_test` 上评估（跨数据集），与 DPT_1724 不可直接对比。其深度指标 iter/non-iter 完全相同（深度分支不依赖迭代 prompt）。

### 3.4 1724_test · 单次点击（n_sub=1）

| 模型 | Dice | IoU | BoundIoU | Depth MAE | Depth R² |
|---|---|---|---|---|---|
| **DPT 0,3,6,9** | **0.8976** | **0.8257** | 0.1811 | 10.75 | 0.9198 |
| DPT 2,5,8,11 | 0.8953 | 0.8232 | **0.1813** | 10.78 | 0.9175 |
| DPT 全部 12 层 | 0.8951 | 0.8220 | 0.1792 | 11.50 | 0.9137 |
| DPT 3,6,9,11 | 0.8950 | 0.8211 | 0.1716 | **9.99** | **0.9402** |
| mona_simple † | 0.7501 | 0.6303 | 0.0612 | 40.31 | 0.3033 |
| fc_simple † | 0.7330 | 0.6153 | 0.0612 | 40.38 | 0.2555 |
| dual_attn_simple † | 0.7284 | 0.6121 | 0.0600 | 39.52 | 0.2950 |
| fc_bd †‡（仅 P1） | 0.7144 | 0.5937 | 0.0449 | — | — |
| dual_attn_bd †‡（仅 P1） | 0.7006 | 0.5813 | 0.0486 | — | — |
| mona_bd †‡（仅 P1） | 0.6767 | 0.5571 | 0.0397 | — | — |

† 跨数据集评估（0_sam 训练 → 1724_test）。 ‡ bd 系列 non-iter 日志在 Phase 2 处截断，仅有 Phase 1 分割结果。

### 3.5 Phase 1 → Phase 2 对比（两阶段训练有效性）

以 n_sub=8 为例（各自的同域测试集）：

| 模型 | Dice P1→P2 | Depth MAE P1→P2 | Depth R² P1→P2 |
|---|---|---|---|
| DPT_0sam 2,5,8,11 | 0.9585 → 0.9587（+0.0002） | 55.36 → 6.18 | -0.26 → 0.979 |
| DPT_0sam 3,6,9,11 | 0.9639 → 0.9637（-0.0002） | 54.16 → 7.06 | -0.10 → 0.974 |
| DPT_1724 3,6,9,11 | 0.9309 → 0.9308（-0.0001） | 49.95 → 10.19 | -0.05 → 0.936 |
| mona_iter8 | 0.9499 → 0.9500（+0.0001） | 52.81 → 7.83 | -0.12 → 0.952 |
| DPT_0sam 全部 12 层 | 0.9587 → 0.9554（-0.0033） | 68.19 → 7.40 | -0.95 → 0.964 |
| DPT_0sam 0,3,6,9 | 0.9619 → 0.9593（-0.0026） | 70.28 → 6.10 | -1.08 → 0.979 |

Phase 2 深度训练后分割指标基本无损（多数 |Δ|≤0.003，最大 -0.0033），深度 MAE 从 ~50–70 降至 6–11，R² 由负转 0.90+，两阶段策略成立。

---

## 4. 结论

1. **DPT 多层融合 > 单层 OPT adapter**（0_sam_test, n_sub=8, P2）：DPT 系列 Dice 0.955–0.964 全面高于 mona/fc/dual_attn 的 0.943–0.950，深度 MAE 6.1–7.4 也优于 7.7–8.3。注意 DPT 系列 bs=8、OPT 系列 bs=4，不是严格受控对比。
2. **层选择消融**：
   - 分割最优：**3,6,9,11**（0_sam Dice 0.9637 / IoU 0.9310）；1724 上 0,3,6,9 与 3,6,9,11 并列第一。
   - 深度最优：0_sam 上 **0,3,6,9**（MAE 6.10，iter1 时 5.80）；1724 上 **3,6,9,11**（MAE 10.19 / R² 0.9364）。
   - **全部 12 层在两个数据集上分割与深度均垫底** —— 层数多不等于好，选 4 层即可。
   - 默认 2,5,8,11 各项均被 0,3,6,9 或 3,6,9,11 小幅超越。
3. **boundary loss（bd, w=1）负收益**：相对 iter8 基线，Dice 下降约 0.02–0.03（mona 0.9500→0.9243），Recall≈0.99 / Precision≈0.87，呈系统性过分割；Boundary IoU（0.06–0.07）远低于无 boundary loss 的 DPT 系列（0.19–0.23），未达到改善边界的目的。bd5（w=5）训练未成功启动，无数据。
4. **simple depth transformer**：仅有跨数据集评估结果（0_sam→1724），深度 R² 仅 0.26–0.30、MAE≈40，远差于同为跨域下限的 twoway+bd 系列无法对比（后者无 P2 数据），但可确认 simple 结构在域偏移下深度几乎失效；缺少同域对照，无法下严格结论。
5. **迭代 prompt 收益显著**：n_sub 1→8 时所有模型 Dice 提升 0.03–0.10（0_sam：~0.86→~0.96；1724：~0.90→~0.93）。
6. **数据集难度**：1724 整体比 0_sam 难（同构型模型 Dice 低 ~0.03，深度 MAE 高 ~4）。
7. **OPT 三种 adapter 排序**（0_sam, n_sub=8）：mona（0.9500）> dual_attn（0.9462）> fc（0.9428）；n_sub=1 时 dual_attn 反超 mona（0.8590 vs 0.8506）。

## 附：原始文件索引

| 内容 | 路径模式 |
|---|---|
| DPT 训练 config / log | `DPT_*/config.txt`、`DPT_*/log.txt` |
| DPT 评估 log | `DPT_*/eval_iter{1,8}/log.txt` |
| OPT 训练 config / log | `*_OPT/config.log`、`*_OPT/train.log` |
| OPT 评估 log | `*_iter8_OPT/eval_{iter8,noiter}/eval.log`、`*_{simple,bd}_OPT/eval_{iter,noniter}.log`、`mona_bd_OPT/eval_masks_*/…` |
| 评估启动脚本 | `launch_all_evals.sh`、各目录 `run_eval_*.sh` |
