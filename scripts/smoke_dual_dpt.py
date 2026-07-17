"""Dual DPT layers smoke checks (requires real SAM + DINOv3 checkpoints).

1. 开关关闭时 state_dict 键集合与单头版逐字节一致（回归断言）
2. 双头模式 forward 输出 {"mask","depth"} 两个 (B,256,64,64)
3. 双头模式 freeze_depth_branch / phase2 解冻语义正确

Usage:
    python scripts/smoke_dual_dpt.py \
        --sam_checkpoint /data3/LXT/data/checkpoint/sam_vit_b_01ec64.pth \
        --dinov3_checkpoint <path-to-dinov3-vitb16.pth>
"""
import argparse

import torch

from segment_anything.build_sam import sam_model_registry
from dinosam.model import DepthSam


def build(args, dpt_layers_depth=None):
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
    return DepthSam(sam, encoder_type="dinov3",
                    dinov3_checkpoint=args.dinov3_checkpoint,
                    adapter_type="dpt_simple",
                    dpt_layers=[2, 5, 8, 11],
                    dpt_layers_depth=dpt_layers_depth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--dinov3_checkpoint", type=str, required=True)
    args = parser.parse_args()

    # 1. 开关关闭：键集合回归（无任何 *_depth 键）
    single = build(args)
    single_keys = set(single.state_dict().keys())
    depth_keys = {k for k in single_keys
                  if k.startswith("dinov3_encoder.") and "_depth." in k}
    assert not depth_keys, f"single-head model leaked depth keys: {depth_keys}"
    print(f"[1/3] single-head state_dict clean: {len(single_keys)} keys, no *_depth")

    # 2. 双头模式：键集合 = 单头 ∪ 四个 *_depth 模块；forward 输出 shape
    dual = build(args, dpt_layers_depth=[5, 8, 11])
    dual_keys = set(dual.state_dict().keys())
    extra = dual_keys - single_keys
    assert dual_keys >= single_keys, "dual mode must be a superset of single-head keys"
    prefixes = {k.split(".")[1] for k in extra}
    assert prefixes == {"dpt_head_depth", "adapter1_depth",
                        "adapter2_depth", "projection_depth"}, prefixes
    print(f"[2/3] dual-head adds exactly the 4 depth modules ({len(extra)} keys)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dual.to(device).eval()
    with torch.no_grad():
        emb, input_size = dual.encode_images(
            torch.rand(1, 3, 512, 512, device=device) * 255)
    assert isinstance(emb, dict) and set(emb) == {"mask", "depth"}, type(emb)
    assert emb["mask"].shape == (1, 256, 64, 64), emb["mask"].shape
    assert emb["depth"].shape == (1, 256, 64, 64), emb["depth"].shape
    print(f"[2/3] dual forward ok: mask/depth both (1,256,64,64), input_size={input_size}")

    # 3. 冻结/解冻语义
    dual.freeze_image_encoder()
    dual.freeze_depth_branch()  # phase1
    enc = dual.dinov3_encoder
    depth_params = [p for m in (enc.dpt_head_depth, enc.adapter1_depth,
                                enc.adapter2_depth, enc.projection_depth)
                    for p in m.parameters()]
    assert all(not p.requires_grad for p in depth_params), "phase1 must freeze *_depth"
    # phase2 解冻（复刻 run_phase2 的独立段）
    for p in dual.parameters():
        p.requires_grad = False
    for m in (enc.dpt_head_depth, enc.adapter1_depth,
              enc.adapter2_depth, enc.projection_depth):
        for p in m.parameters():
            p.requires_grad = True
    trainable = sum(p.numel() for p in dual.parameters() if p.requires_grad)
    assert trainable == sum(p.numel() for p in depth_params)
    print(f"[3/3] freeze/unfreeze ok: phase2 *_depth trainable params = {trainable:,}")

    print("\nAll dual-DPT smoke checks passed.")


if __name__ == "__main__":
    main()
