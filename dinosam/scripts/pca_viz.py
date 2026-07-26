"""PCA visualization CLI for DINOSAM inference images."""

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms as T
from segment_anything.build_sam import sam_model_registry

from dinosam.model import DepthSam
from dinosam.viz.pca import visualize_model_features


def main():
    parser = argparse.ArgumentParser(description="PCA visualization on inference images")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="DINOSAM model checkpoint path")
    parser.add_argument("--sam_checkpoint", type=str, required=True,
                        help="SAM ViT-B checkpoint path")
    parser.add_argument("--image", type=str, nargs="+", required=True,
                        help="Input image path(s)")
    parser.add_argument("--output_dir", type=str, default="./pca_output",
                        help="Output directory (default: ./pca_output)")
    parser.add_argument("--dpt_layers", type=str, default="2,5,8,11",
                        help="Comma-separated DPT layer indices (default: 2,5,8,11)")
    parser.add_argument("--mask", action="store_true",
                        help="Use model-predicted mask for foreground-only PCA fitting")
    parser.add_argument("--encoder", type=str, default="sam",
                        choices=["sam", "dinov3", "resnet50", "resnet101"],
                        help="Image encoder type (default: sam)")
    parser.add_argument("--dinov3_checkpoint", type=str, default=None,
                        help="DINOv3 ViT-B/16 checkpoint (required when --encoder dinov3)")
    parser.add_argument("--adapter_type", type=str, default="mona",
                        help="Feature adapter type (default: mona)")
    parser.add_argument("--img_size", type=int, default=1024,
                        help="Image size for preprocessing (default: 1024)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse dpt_layers from comma-separated string
    dpt_layers = [int(x.strip()) for x in args.dpt_layers.split(",")]

    # Load SAM and create DepthSam model
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
    model = DepthSam(
        sam,
        encoder_type=args.encoder,
        dinov3_checkpoint=args.dinov3_checkpoint,
        adapter_type=args.adapter_type,
        dpt_layers=dpt_layers,
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.to(device)
    model.eval()

    preprocess = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
    ])

    os.makedirs(args.output_dir, exist_ok=True)

    for img_path in args.image:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        save_dir = os.path.join(args.output_dir, img_name)
        os.makedirs(save_dir, exist_ok=True)

        # Load and save original image
        pil_img = Image.open(img_path).convert("RGB")
        pil_img.save(os.path.join(save_dir, "original.png"))

        # Preprocess to tensor: (3, H, W) float [0, 1]
        img_tensor = preprocess(pil_img)

        # Optional mask prediction
        mask_tensor = None
        if args.mask:
            # Build batch tensors for forward_mask
            input_batch = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
            with torch.no_grad():
                image_embeddings, input_size = model.encode_images(input_batch)
                orig_h, orig_w = pil_img.size[1], pil_img.size[0]
                original_sizes = torch.tensor([[orig_h, orig_w]], device=device)
                # Center point prompt as default
                center_coords = torch.tensor([[[orig_w / 2, orig_h / 2]]], device=device)
                center_labels = torch.tensor([[1]], device=device)
                _, _, masks_fullres = model.forward_mask(
                    image_embeddings=image_embeddings,
                    point_coords=center_coords,
                    point_labels=center_labels,
                    original_sizes=original_sizes,
                    input_size=input_size,
                )
                mask_tensor = (torch.sigmoid(masks_fullres) > 0.5).float()

            # Save predicted mask
            mask_np = (mask_tensor[0, 0].cpu().numpy() * 255).astype("uint8")
            Image.fromarray(mask_np).save(os.path.join(save_dir, "mask.png"))

        # Run PCA visualization
        # visualize_model_features expects (B, 3, H, W) raw [0, 255] range
        img_raw = img_tensor.unsqueeze(0) * 255.0  # (1, 3, H, W) [0, 255]
        visualize_model_features(
            model=model,
            images=img_raw,
            device=device,
            save_dir=save_dir,
            mask=mask_tensor,
        )

        print(f"Processed: {img_path} -> {save_dir}")

    print(f"\nDone. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
