# Training Speed Optimization — Design Spec

Date: 2026-06-19
Target: 方案 A — 安全修复，低风险高回报
Reviewed: 2026-06-19 (round 1: general + harness), 2026-06-19 (round 2: re-review)

## Goal

在不改变两阶段训练框架的前提下，修复关键 bug 并减少训练中的冗余计算，预计提速 2-4x。

## Non-Goals

- 不改动训练流程（两阶段、epoch 数、optimizer、scheduler）
- 不改动数据集和 loss 函数
- 不引入 mixed precision、不增大 batch_size
- 不改动 DINO encoder 路径

## Assumptions

- 同 batch 内所有图像原始尺寸相同（由 `collate_fn` 中 `torch.stack` 保证）
- 此假设对 `preprocess`、`forward_mask`、`forward_depth` 的 `input_size` 均适用

## Coordinate Space Contract

**`forward_mask` 和 `forward_depth` 使用相同的坐标约定：**

| 参数 | 空间 | 说明 |
|------|------|------|
| `point_coords` / `depth_point_coords` | **原始图像空间** | 与 `original_sizes` 对应，`(B, N, 2)` |
| `forward_mask` 内部 | 逐样本 `apply_coords_torch` 变换到 model-input 空间 | 已有逻辑，不变 |
| `forward_depth` 内部 | 逐样本 `apply_coords_torch` 变换到 model-input 空间 | **新增**，对齐 `forward_mask` |

**所有调用方**必须传入原始图像空间的坐标，不得预变换。这是 `forward_mask` 当前的约定，`forward_depth` 将与之对齐。

---

## Change 1: Fix `preprocess` — bypass broken `apply_image_torch`

**File:** `dinosam/model/depth_sam.py` — `preprocess` method (line 134)

**Problem:** SAM's `apply_image_torch` uses `image.shape[0]` (batch size) and `image.shape[1]` (channels) as H and W to compute the resize target.

**Fix:**
```python
def preprocess(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    oldh, oldw = x.shape[2], x.shape[3]
    newh, neww = ResizeLongestSide.get_preprocess_shape(oldh, oldw, self.img_size)
    x = F.interpolate(x, size=(newh, neww), mode="bilinear", align_corners=False,
                      antialias=True)
    input_size = (newh, neww)
    x = (x - self.sam.pixel_mean.unsqueeze(0)) / self.sam.pixel_std.unsqueeze(0)
    padh = self.img_size - newh
    padw = self.img_size - neww
    x = F.pad(x, (0, padw, 0, padh))
    return x, input_size
```

**Note:** This fix only affects the SAM encoder path. The DINO path (`dinov3_encoder.forward`) does its own `F.interpolate` to `(1024, 1024)` and is unaffected.

---

## Change 2: Fix `original_sizes[0]` bug — align `forward_depth` coordinate contract

### 2a. Affected call sites (all remove `apply_coords_torch` pre-transform)

| File | Line | Current code (buggy) | Fix |
|------|------|---------------------|-----|
| `trainer.py:train_one_epoch_phase2` | 188-191 | `original_sizes[0].tolist()` → `apply_coords_torch` for batch | Pass `point_coords` directly |
| `trainer.py:validate_phase2` | 264-267 | Same pattern | Pass `point_coords` directly |
| `iterative.py:iterative_train_step` | 90-93 | Same pattern | Pass `point_coords` directly |

### 2b. `forward_depth` — add per-sample coordinate transform

**File:** `dinosam/model/depth_sam.py` — `forward_depth` method (line 224)

Inside the per-sample loop (before `outer_prompt_encoder` call), add:
```python
if depth_point_coords is not None:
    orig_h, orig_w = original_sizes[i].tolist()
    pc = self.transform.apply_coords_torch(
        depth_point_coords[i:i+1], (orig_h, orig_w)
    )
    pl = depth_point_labels[i:i+1]
    points_i = (pc, pl)
```

### 2c. `iterative_depth_step` — scale 256×256 coords to original image space

**File:** `dinosam/train/iterative.py` — `iterative_depth_step` (line 267-275)

The iterative prompt generator produces coords in **256×256 low-res mask space**. These must be scaled to original image space before passing to `forward_depth` (matching `iterative_mask_step`'s pattern at lines 168-176):
```python
# Scale 256x256 coords to original image space
lr_h, lr_w = low_res_masks.shape[-2:]  # 256, 256
scale_x = original_sizes[:, 1].float().view(B, 1, 1) / lr_w
scale_y = original_sizes[:, 0].float().view(B, 1, 1) / lr_h
new_coords_scaled = new_coords.clone()
new_coords_scaled[:, :, 0] = new_coords[:, :, 0] * scale_x.squeeze(-1)
new_coords_scaled[:, :, 1] = new_coords[:, :, 1] * scale_y.squeeze(-1)
cur_depth_coords = torch.cat([cur_depth_coords, new_coords_scaled], dim=1)
```

**Note:** This is a bug fix -- the current code passes 256×256 coords directly, which is already incorrect (the PromptEncoder expects model-input-space coords). The fix makes it correct AND compatible with the new coordinate contract.

---

## Change 3: Batch `forward_mask`

**File:** `dinosam/model/depth_sam.py` — `forward_mask` method (line 158)

**Fix:** Batch `prompt_encoder` and `mask_decoder` calls (from B calls to 1). Per-sample `apply_coords_torch` and `postprocess_masks` remain sequential.

**Key logic:**
```python
# Per-sample coord transform → batch
pc_list = []
for i in range(B):
    orig_h, orig_w = original_sizes[i].tolist()
    pc = self.transform.apply_coords_torch(point_coords[i:i+1], (orig_h, orig_w))
    pc_list.append(pc)
pc = torch.cat(pc_list, dim=0)  # (B, N, 2)

# Batched prompt_encoder + mask_decoder (1 call instead of B)
sparse_emb, dense_emb = self.sam.prompt_encoder(
    points=(pc, point_labels), boxes=None, masks=mask_input,
)
low_res_masks, iou_pred = self.sam.mask_decoder(
    image_embeddings=image_embeddings, image_pe=inner_pe,
    sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb,
    multimask_output=False,
)

# Postprocessing stays per-sample (original_size varies)
```

**Interface change:** Add `return_fullres: bool = True`. When `False`, returns `None` for `masks`.

---

## Change 4: Batch `forward_depth`

**File:** `dinosam/model/depth_sam.py` — `forward_depth` method (line 224)

**Fix:** Same batching pattern. Includes the per-sample `apply_coords_torch` from Change 2b.

---

## Change 5: Skip redundant mask loss in Phase 2 training

**File:** `dinosam/train/trainer.py` — `train_one_epoch_phase2` (line 144)

**Fix:**
```python
image_embeddings, input_size = model.encode_images(images)

# Non-iterative path:
low_res_masks, iou_pred, masks_fullres = model.forward_mask(
    ..., return_fullres=False,  # skip postprocess on non-log batches
)

depth_pred, decoder_masks = model.forward_depth(
    image_embeddings=image_embeddings,
    low_res_masks=low_res_masks,
    input_size=input_size,
    original_sizes=original_sizes,
    depth_point_coords=point_coords,      # raw, no pre-transform (Change 2a)
    depth_point_labels=point_labels,
)

depth_loss = depth_mse_loss(depth_pred, depth_labels)
loss = args.depth_loss_weight * depth_loss

# Only compute mask loss for logging, on log-interval batches
is_log_batch = (batch_idx + 1) % args.log_interval == 0
if is_log_batch:
    masks_fullres_for_log, _, _ = model.forward_mask(
        ..., return_fullres=True,
    )  # or reuse masks_fullres if already computed
    inner_mask_loss = dice_loss(masks_fullres_for_log, gt_masks)
    decoder_mask_loss = dice_loss(decoder_masks, gt_masks)
    mask_val = inner_mask_loss.item()
    n_mask_batches += 1
else:
    mask_val = 0.0
```

---

## Change 6: Apply same optimization to `validate_phase2`

**File:** `dinosam/train/trainer.py` — `validate_phase2` (line 241)

**Fix:** Use `return_fullres=False`, pass `point_coords` directly (Change 2a), remove mask loss computation. The `original_sizes[0]` bug at line 264-266 is also fixed by Change 2a.

---

## Files Changed

| File | Changes |
|------|---------|
| `dinosam/model/depth_sam.py` | Change 1: `preprocess` rewrite. Change 2b: per-sample coord transform in `forward_depth`. Change 3: `forward_mask` batching + `return_fullres`. Change 4: `forward_depth` batching. Update `depth_point_coords` docstring to "original image space". |
| `dinosam/train/trainer.py` | Change 2a: remove `apply_coords_torch` pre-transform (L188, L264). Change 5: conditional mask loss in `train_one_epoch_phase2`. Change 6: `return_fullres=False` in `validate_phase2`. |
| `dinosam/train/iterative.py` | Change 2a: remove `apply_coords_torch` pre-transform (L90). Change 2c: scale iterative coords to original image space in `iterative_depth_step`. |

## NOT Changed (verified: `return_fullres` defaults to `True`)

| File | Reason |
|------|--------|
| `dinosam/evaluate.py` | Uses `forward_mask`/`forward_depth` with defaults (no pre-transform today; if it had one, it would be removed per Change 2a). No behavior change. |
| `dinosam/model/dinov3_encoder.py` | DINO path bypasses `preprocess` entirely, unaffected. |

---

## Verification Checklist

1. **Batch size invariance:** After fix, train with batch_size=4, run `evaluate.py` with batch_size=1 and batch_size=4 — IoU/Dice match within 1e-4
2. **Batched vs per-sample equivalence:** `forward_mask`(B=4) vs 4×`forward_mask`(B=1) — `low_res_masks`, `iou_pred` match within 1e-6. Same for `forward_depth`: `depth_pred`, `decoder_masks` match within 1e-6
3. **`original_sizes[0]` fix:** Construct a batch with different original sizes (bypass DataLoader, manually stack), verify depth coords are transformed per-sample and results match per-sample evaluation
4. **Phase 1 regression:** Train Phase 1, verify mask loss curve identical to before
5. **Phase 2 regression:** Train Phase 2, verify depth loss curve identical to before
6. **DINO encoder:** Train DINO model, evaluate at batch_size=1 and batch_size=4 — metrics match (DINO path already correct, verify no regression)
7. **Speed measurement:** Measure Phase 2 wall-clock time/epoch (using existing `time.time()` instrumentation in trainer.py) before and after changes. Target: 30-50% reduction. (Phase 1 time unchanged since `forward_depth` is not called.)
