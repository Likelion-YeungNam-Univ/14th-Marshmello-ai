from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from settings import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOPPING_PATIENCE,
    ENCODER_NAME,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    NUM_WORKERS,
    OUTPUT_DIR,
    THRESHOLD,
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    USE_IMAGENET_WEIGHTS,
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
    WEIGHT_DECAY,
)
from unet_core import (
    CombinedLoss,
    StretchMarkDataset,
    create_model,
    denormalize_image,
    ensure_directories,
    make_overlay,
    segmentation_metrics,
    set_seed,
)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    sample_count = 0

    progress = tqdm(loader, desc="Train" if training else "Validation", leave=False)

    for images, masks, _ in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.size(0)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_function(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

        metrics = segmentation_metrics(logits, masks, threshold=THRESHOLD)
        totals["loss"] += float(loss.item()) * batch_size
        for key, value in metrics.items():
            totals[key] += value * batch_size
        sample_count += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{metrics['dice']:.4f}",
        )

    return {key: value / max(sample_count, 1) for key, value in totals.items()}


@torch.no_grad()
def save_validation_previews(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    max_images: int = 8,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0

    for images, masks, names in loader:
        images = images.to(device)
        logits = model(images)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        masks_np = masks.numpy()

        for index, name in enumerate(names):
            image_rgb = denormalize_image(images[index])
            predicted_mask = probabilities[index, 0] >= THRESHOLD
            true_mask = masks_np[index, 0] >= 0.5

            predicted_overlay = make_overlay(image_rgb, predicted_mask)
            true_overlay = make_overlay(image_rgb, true_mask)

            comparison = np.concatenate(
                [image_rgb, true_overlay, predicted_overlay], axis=1
            )
            cv2.imwrite(
                str(output_dir / f"{Path(name).stem}_original_gt_pred.jpg"),
                cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR),
            )
            saved += 1
            if saved >= max_images:
                return


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_history.csv"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_dice"] for row in history], label="train dice")
    plt.plot(epochs, [row["val_dice"] for row in history], label="val dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "dice_curve.png", dpi=150)
    plt.close()


def main() -> None:
    set_seed()
    ensure_directories([CHECKPOINT_DIR, OUTPUT_DIR])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    train_dataset = StretchMarkDataset(
        TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, train=True, image_size=IMAGE_SIZE
    )
    val_dataset = StretchMarkDataset(
        VAL_IMAGE_DIR, VAL_MASK_DIR, train=False, image_size=IMAGE_SIZE
    )

    print(f"Train: {len(train_dataset)}장")
    print(f"Validation: {len(val_dataset)}장")
    if len(train_dataset) < 20:
        print("주의: 현재 데이터 수는 코드 동작 확인용에 가깝습니다. 실제 성능 평가는 어렵습니다.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    try:
        model = create_model(ENCODER_NAME, USE_IMAGENET_WEIGHTS).to(device)
    except Exception as error:
        print("\n모델 생성 중 오류가 발생했습니다.")
        print(error)
        print(
            "인터넷에서 ImageNet 가중치를 받지 못한 경우 settings.py의 "
            "USE_IMAGENET_WEIGHTS를 False로 바꾸고 다시 실행하세요."
        )
        raise

    loss_function = CombinedLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_dice = -1.0
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, float]] = []
    best_path = CHECKPOINT_DIR / "best_unet.pt"
    last_path = CHECKPOINT_DIR / "last_unet.pt"

    for epoch in range(1, EPOCHS + 1):
        train_result = run_epoch(
            model, train_loader, loss_function, device, optimizer=optimizer
        )
        with torch.no_grad():
            val_result = run_epoch(
                model, val_loader, loss_function, device, optimizer=None
            )

        scheduler.step(val_result["dice"])
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_result["loss"],
            "train_dice": train_result["dice"],
            "train_iou": train_result["iou"],
            "val_loss": val_result["loss"],
            "val_dice": val_result["dice"],
            "val_iou": val_result["iou"],
            "val_precision": val_result["precision"],
            "val_recall": val_result["recall"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train loss={train_result['loss']:.4f}, dice={train_result['dice']:.4f} | "
            f"val loss={val_result['loss']:.4f}, dice={val_result['dice']:.4f}, "
            f"iou={val_result['iou']:.4f} | lr={current_lr:.2e}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "encoder_name": ENCODER_NAME,
            "image_size": IMAGE_SIZE,
            "threshold": THRESHOLD,
            "epoch": epoch,
            "val_dice": val_result["dice"],
        }
        torch.save(checkpoint, last_path)

        if val_result["dice"] > best_dice:
            best_dice = val_result["dice"]
            best_epoch = epoch
            no_improvement = 0
            torch.save(checkpoint, best_path)
            print(f"  최고 모델 저장: {best_path}")
        else:
            no_improvement += 1

        if no_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping: {EARLY_STOPPING_PATIENCE} epoch 동안 개선 없음")
            break

    training_output_dir = OUTPUT_DIR / "training"
    save_history(history, training_output_dir)

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    save_validation_previews(
        model,
        val_loader,
        device,
        OUTPUT_DIR / "validation_previews",
    )

    print("\n학습 완료")
    print(f"최고 epoch: {best_epoch}")
    print(f"최고 validation Dice: {best_dice:.4f}")
    print(f"모델: {best_path}")
    print(f"그래프: {training_output_dir}")
    print(f"검증 미리보기: {OUTPUT_DIR / 'validation_previews'}")


if __name__ == "__main__":
    main()
