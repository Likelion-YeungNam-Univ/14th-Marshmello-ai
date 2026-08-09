from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from locator_core import (
    AnatomyDataset,
    AnatomyLoss,
    create_locator_model,
    locator_metrics,
    set_seed,
)
from locator_settings import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    ENCODER_NAME,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    LOCATOR_CHECKPOINT_DIR,
    LOCATOR_CHECKPOINT_PATH,
    LOCATOR_OUTPUT_DIR,
    LOCATOR_TRAIN_ABDOMEN_DIR,
    LOCATOR_TRAIN_IMAGE_DIR,
    LOCATOR_TRAIN_NAVEL_DIR,
    LOCATOR_VAL_ABDOMEN_DIR,
    LOCATOR_VAL_IMAGE_DIR,
    LOCATOR_VAL_NAVEL_DIR,
    NAVEL_LOSS_WEIGHT,
    NUM_WORKERS,
    USE_IMAGENET_WEIGHTS,
    WEIGHT_DECAY,
)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    metric_sums = {
        "abdomen_dice": 0.0,
        "navel_distance_px": 0.0,
        "navel_distance_normalized": 0.0,
    }
    batches = 0

    progress = tqdm(loader, desc="Train" if training else "Validation")
    for images, targets, _names in progress:
        images = images.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_function(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        metrics = locator_metrics(logits.detach(), targets)
        total_loss += float(loss.item())
        for key in metric_sums:
            metric_sums[key] += metrics[key]
        batches += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            abdomen_dice=f"{metrics['abdomen_dice']:.3f}",
            navel_px=f"{metrics['navel_distance_px']:.1f}",
        )

    if batches == 0:
        raise RuntimeError("데이터 배치가 없습니다.")

    return total_loss / batches, {key: value / batches for key, value in metric_sums.items()}


def main() -> None:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = AnatomyDataset(
        LOCATOR_TRAIN_IMAGE_DIR,
        LOCATOR_TRAIN_ABDOMEN_DIR,
        LOCATOR_TRAIN_NAVEL_DIR,
        train=True,
        image_size=IMAGE_SIZE,
    )
    val_dataset = AnatomyDataset(
        LOCATOR_VAL_IMAGE_DIR,
        LOCATOR_VAL_ABDOMEN_DIR,
        LOCATOR_VAL_NAVEL_DIR,
        train=False,
        image_size=IMAGE_SIZE,
    )

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

    model = create_locator_model(
        encoder_name=ENCODER_NAME,
        use_imagenet_weights=USE_IMAGENET_WEIGHTS,
    ).to(device)
    loss_function = AnatomyLoss(navel_weight=NAVEL_LOSS_WEIGHT)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    LOCATOR_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history_dir = LOCATOR_OUTPUT_DIR / "training"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / "history.csv"

    print("=" * 64)
    print("복부 영역 + 배꼽 위치 자동 모델 학습")
    print(f"장치: {device}")
    print(f"Train: {len(train_dataset)}장")
    print(f"Validation: {len(val_dataset)}장")
    print("모델 출력 0번 채널: 복부 영역")
    print("모델 출력 1번 채널: 배꼽 주변 원형 마스크")
    print("=" * 64)

    best_val_loss = float("inf")
    no_improvement = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_metrics = run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, loss_function, device, optimizer=None
        )

        print(
            f"train loss={train_loss:.4f}, abdomen dice={train_metrics['abdomen_dice']:.4f}, "
            f"navel distance={train_metrics['navel_distance_px']:.1f}px"
        )
        print(
            f"val   loss={val_loss:.4f}, abdomen dice={val_metrics['abdomen_dice']:.4f}, "
            f"navel distance={val_metrics['navel_distance_px']:.1f}px"
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_abdomen_dice": train_metrics["abdomen_dice"],
                "val_abdomen_dice": val_metrics["abdomen_dice"],
                "train_navel_distance_px": train_metrics["navel_distance_px"],
                "val_navel_distance_px": val_metrics["navel_distance_px"],
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": IMAGE_SIZE,
                    "encoder_name": ENCODER_NAME,
                    "abdomen_threshold": 0.5,
                    "best_val_loss": best_val_loss,
                    "val_abdomen_dice": val_metrics["abdomen_dice"],
                    "val_navel_distance_px": val_metrics["navel_distance_px"],
                },
                LOCATOR_CHECKPOINT_PATH,
            )
            print(f"최고 모델 저장: {LOCATOR_CHECKPOINT_PATH}")
        else:
            no_improvement += 1

        if no_improvement >= EARLY_STOPPING_PATIENCE:
            print("Validation 성능이 개선되지 않아 조기 종료합니다.")
            break

    with history_path.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = list(history_rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)

    print("\n학습 완료")
    print(f"모델: {LOCATOR_CHECKPOINT_PATH.resolve()}")
    print(f"기록: {history_path.resolve()}")
    print("다음 단계: python predict_locator.py")


if __name__ == "__main__":
    main()
