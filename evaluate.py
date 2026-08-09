from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from settings import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    NUM_WORKERS,
    OUTPUT_DIR,
    TEST_IMAGE_DIR,
    TEST_MASK_DIR,
)
from unet_core import (
    StretchMarkDataset,
    create_model,
    denormalize_image,
    make_overlay,
    segmentation_metrics,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = CHECKPOINT_DIR / "best_unet.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("먼저 python train.py를 실행해 best_unet.pt를 만드세요.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("image_size", 512))
    threshold = float(checkpoint.get("threshold", 0.5))
    encoder_name = str(checkpoint.get("encoder_name", "resnet34"))

    dataset = StretchMarkDataset(
        TEST_IMAGE_DIR, TEST_MASK_DIR, train=False, image_size=image_size
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = create_model(encoder_name, use_imagenet_weights=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    count = 0
    output_dir = OUTPUT_DIR / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, masks, names in tqdm(loader, desc="Test"):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            metrics = segmentation_metrics(logits, masks, threshold)
            batch_size = images.size(0)
            for key, value in metrics.items():
                totals[key] += value * batch_size
            count += batch_size

            probabilities = torch.sigmoid(logits).cpu().numpy()
            masks_np = masks.cpu().numpy()
            for index, name in enumerate(names):
                image_rgb = denormalize_image(images[index])
                gt = masks_np[index, 0] >= 0.5
                pred = probabilities[index, 0] >= threshold
                comparison = np.concatenate(
                    [image_rgb, make_overlay(image_rgb, gt), make_overlay(image_rgb, pred)],
                    axis=1,
                )
                cv2.imwrite(
                    str(output_dir / f"{Path(name).stem}_original_gt_pred.jpg"),
                    cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR),
                )

    averages = {key: value / max(count, 1) for key, value in totals.items()}
    report = (
        f"Test images: {count}\n"
        f"Dice: {averages['dice']:.4f}\n"
        f"IoU: {averages['iou']:.4f}\n"
        f"Precision: {averages['precision']:.4f}\n"
        f"Recall: {averages['recall']:.4f}\n"
    )
    print("\n" + report)
    (output_dir / "metrics.txt").write_text(report, encoding="utf-8")
    print(f"결과 저장: {output_dir}")


if __name__ == "__main__":
    main()
