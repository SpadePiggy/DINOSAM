# Mask Loss Type Selection Design

**Date:** 2026-06-29  
**Status:** APPROVED  
**Scope:** Phase 1 mask training only

---

## Overview

Add a command-line argument `--mask_loss_type` to control the loss function used for Phase 1 mask branch training in DINOSAM. Supports three modes: `dice` (default, backward compatible), `bce`, and `dice_bce` (equal-weight combination).

## Motivation

Currently DINOSAM uses only Dice Loss for mask training. While Dice Loss handles class imbalance well for small targets, combining it with BCE Loss often improves segmentation quality by providing stable pixel-level gradients. This change enables experimentation with different loss configurations without code modifications.

## Design Decisions

### 1. Function Naming (Review Finding: HIGH)

**Decision:** Use `compute_mask_loss` as the dispatcher function name.

**Rationale:** Avoids naming collision with the local variable `mask_loss` used at all call sites. Prevents fragile shadowing where `mask_loss = mask_loss(...)` would work syntactically but confuse readers and future maintainers.

### 2. File Organization (Review Finding: LOW)

**Decision:** Place `bce_loss` in a new file `dinosam/loss/bce.py`, and move the dispatcher to `dinosam/loss/__init__.py`.

**Rationale:** Keeping `bce_loss` in `dice.py` violates principle of least surprise. A new file is cleaner and follows the existing pattern where `depth_loss.py` contains depth-specific losses.

### 3. Scope Boundaries (Review Finding: MEDIUM)

**Decision:** Only modify Phase 1 mask training losses. Explicitly exclude:
- Phase 2 depth decoder mask monitoring losses
- `iterative_train_step` (joint mask+depth training, not used in Phase 1)
- Evaluation metrics in `evaluate.py`

**Rationale:** Phase 2 mask losses are monitoring-only (no gradients), so they don't affect training. `iterative_train_step` is a joint training path not used in Phase 1. Changing these would expand scope beyond the stated goal.

### 4. Boundary Loss Interaction (Review Finding: MEDIUM)

**Decision:** Document the interaction; do not modify `boundary_loss` behavior.

**Rationale:** When `--mask_loss_type bce` or `dice_bce` is used with `--boundary_loss`, BCE appears twice (once globally, once inside boundary_loss). This is acceptable because:
- boundary_loss's BCE term is localized to boundary regions with distance weighting
- The user can adjust `--boundary_loss_weight` to compensate
- Modifying boundary_loss internals would expand scope

**Documentation:** Add inline comment at call sites explaining the interaction.

### 5. Loss Scale (Review Finding: MEDIUM)

**Decision:** Use hardcoded 1:1 weighting for `dice_bce` mode; document that tuning may be needed.

**Rationale:** User requested fixed equal weighting. Dice loss returns values in [0,1], while BCE is unbounded but typically 0.3-0.7 for segmentation. The 1:1 ratio is a reasonable starting point; users can add `--bce_weight` in future work if needed.

### 6. Backward Compatibility (Review Finding: HIGH)

**Decision:** Default value `--mask_loss_type dice` must produce bit-identical results to current code.

**Implementation:** The dispatcher's `dice` branch calls the existing `dice_loss(pred, target)` directly with no wrapper or modification.

**Verification:** Add regression test comparing `compute_mask_loss(pred, target, "dice")` against `dice_loss(pred, target)`.

---

## Implementation

### File Changes

#### 1. `dinosam/loss/bce.py` (NEW)

```python
import torch
from torch.nn import functional as F


def bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy loss for segmentation.

    Args:
        pred: (B, 1, H, W) predicted logits (before sigmoid)
        target: (B, 1, H, W) binary ground truth

    Returns:
        Scalar BCE loss value
    """
    return F.binary_cross_entropy_with_logits(pred, target)
```

#### 2. `dinosam/loss/__init__.py` (MODIFY)

```python
import torch

from .dice import dice_loss, boundary_loss, compute_iou
from .bce import bce_loss
from .depth_loss import depth_mse_loss


def compute_mask_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "dice",
) -> torch.Tensor:
    """Compute mask loss for Phase 1 training.

    Args:
        pred: (B, 1, H, W) predicted logits (before sigmoid)
        target: (B, 1, H, W) binary ground truth
        loss_type: One of "dice", "bce", or "dice_bce"

    Returns:
        Scalar loss value

    Note:
        When combined with boundary_loss, BCE may appear twice (once here,
        once inside boundary_loss). Adjust --boundary_loss_weight accordingly.
    """
    if loss_type == "dice":
        return dice_loss(pred, target)
    elif loss_type == "bce":
        return bce_loss(pred, target)
    elif loss_type == "dice_bce":
        return dice_loss(pred, target) + bce_loss(pred, target)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


__all__ = [
    "dice_loss",
    "boundary_loss",
    "compute_iou",
    "bce_loss",
    "compute_mask_loss",
    "depth_mse_loss",
]
```

#### 3. `dinosam/train/trainer.py` (MODIFY)

**Import change (line 14):**
```python
from dinosam.loss import dice_loss, boundary_loss, compute_iou, compute_mask_loss
```

**Add argparse argument (after line 538):**
```python
parser.add_argument("--mask_loss_type", type=str, default="dice",
                    choices=["dice", "bce", "dice_bce"],
                    help="Phase 1 mask loss: 'dice' (default), 'bce', or 'dice_bce' (1:1 sum)")
```

**Replace in `train_one_epoch_phase1` (line 64):**
```python
mask_loss = compute_mask_loss(masks_fullres, gt_masks, loss_type=args.mask_loss_type)
```

**Replace in `validate_phase1` (line 142):**
```python
mask_loss = compute_mask_loss(masks_fullres, gt_masks, loss_type=args.mask_loss_type)
```

#### 4. `dinosam/train/iterative.py` (MODIFY)

**Import change (line 6):**
```python
from dinosam.loss import dice_loss, boundary_loss, compute_iou, compute_mask_loss
```

**Replace in `iterative_mask_step` (line 152):**
```python
mask_loss = compute_mask_loss(masks_fullres, gt_masks, loss_type=args.mask_loss_type)
```

**Note:** `iterative_train_step` (line 53) and `iterative_depth_step` (lines 234, 258) are intentionally NOT modified (Phase 2 scope).

#### 5. `tests/test_mask_loss.py` (NEW)

```python
import torch
from dinosam.loss import dice_loss, bce_loss, compute_mask_loss


def test_compute_mask_loss_dice_equivalent():
    """Regression: dice mode must match dice_loss exactly."""
    pred = torch.randn(2, 1, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > 0.5).float()

    expected = dice_loss(pred, target)
    actual = compute_mask_loss(pred, target, "dice")

    assert torch.allclose(actual, expected), "dice mode must be bit-identical to dice_loss"


def test_compute_mask_loss_bce():
    """BCE mode uses binary_cross_entropy_with_logits."""
    pred = torch.randn(2, 1, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > 0.5).float()

    expected = bce_loss(pred, target)
    actual = compute_mask_loss(pred, target, "bce")

    assert torch.allclose(actual, expected)


def test_compute_mask_loss_dice_bce():
    """dice_bce mode is sum of dice and bce."""
    pred = torch.randn(2, 1, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > 0.5).float()

    expected = dice_loss(pred, target) + bce_loss(pred, target)
    actual = compute_mask_loss(pred, target, "dice_bce")

    assert torch.allclose(actual, expected)


def test_compute_mask_loss_invalid_type():
    """Invalid loss_type raises ValueError."""
    pred = torch.randn(2, 1, 64, 64)
    target = torch.zeros(2, 1, 64, 64)

    try:
        compute_mask_loss(pred, target, "invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown loss_type" in str(e)


if __name__ == "__main__":
    test_compute_mask_loss_dice_equivalent()
    test_compute_mask_loss_bce()
    test_compute_mask_loss_dice_bce()
    test_compute_mask_loss_invalid_type()
    print("All tests passed.")
```

#### 6. `README.md` (MODIFY)

**Add to Key Arguments table (after line 150):**

```markdown
| `--mask_loss_type` | `dice` | Phase 1 mask loss: `dice`, `bce`, or `dice_bce` |
```

---

## Usage Examples

### Default (Dice only, backward compatible)
```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --train_phase 1
```

### BCE only
```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --train_phase 1 \
    --mask_loss_type bce
```

### Dice + BCE combination
```bash
python -m dinosam.train.trainer \
    --sam_checkpoint /path/to/sam_vit_b.pth \
    --train_phase 1 \
    --mask_loss_type dice_bce \
    --boundary_loss \
    --boundary_loss_weight 0.5  # Reduced to compensate for BCE overlap
```

---

## Testing

Run the regression tests:
```bash
python tests/test_mask_loss.py
```

Or with pytest:
```bash
pytest tests/test_mask_loss.py -v
```

Expected output: All 4 tests pass, confirming backward compatibility and correctness.

---

## Out of Scope

The following are explicitly NOT included in this design:

1. **Phase 2 mask losses** — These are monitoring-only (no gradients), so they don't affect training
2. **`iterative_train_step`** — Joint mask+depth training path not used in Phase 1
3. **Configurable BCE weight** — User requested fixed 1:1; can be added later if needed
4. **Boundary loss modification** — Interaction documented but behavior unchanged
5. **Evaluation metrics** — `evaluate.py` uses inline Dice/IoU computation, not loss functions

---

## Review Findings Addressed

| Severity | Finding | Resolution |
|----------|---------|------------|
| HIGH | `mask_loss` name collision | Renamed to `compute_mask_loss` |
| HIGH | No regression test | Added `tests/test_mask_loss.py` |
| MEDIUM | `bce_loss` in `dice.py` | Moved to `dinosam/loss/bce.py` |
| MEDIUM | Phase 2 exclusion unclear | Explicitly documented in "Out of Scope" |
| MEDIUM | Boundary loss interaction | Documented with inline comments |
| MEDIUM | Loss scale mismatch | Documented; 1:1 is starting point |
| LOW | README update needed | Added to Key Arguments table |
| LOW | `__init__.py` exports | Updated with `compute_mask_loss` and `bce_loss` |

---

## Verification Checklist

Before merging:

- [ ] `python tests/test_mask_loss.py` passes all 4 tests
- [ ] Training with `--mask_loss_type dice` produces identical loss curves to before
- [ ] Training with `--mask_loss_type bce` converges (loss decreases)
- [ ] Training with `--mask_loss_type dice_bce` converges
- [ ] `--mask_loss_type invalid_value` is rejected by argparse
- [ ] README.md includes the new argument in the table
