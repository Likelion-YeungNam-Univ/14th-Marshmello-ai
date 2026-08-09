from pathlib import Path

# 이 파일만 수정하면 대부분의 학습 설정을 바꿀 수 있습니다.
ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"

TRAIN_IMAGE_DIR = DATASET_DIR / "train" / "images"
TRAIN_MASK_DIR = DATASET_DIR / "train" / "masks"
VAL_IMAGE_DIR = DATASET_DIR / "val" / "images"
VAL_MASK_DIR = DATASET_DIR / "val" / "masks"
TEST_IMAGE_DIR = DATASET_DIR / "test" / "images"
TEST_MASK_DIR = DATASET_DIR / "test" / "masks"

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
OUTPUT_DIR = ROOT_DIR / "outputs"
NEW_IMAGES_DIR = ROOT_DIR / "new_images"

# 처음에는 512로 시작하세요. GPU 메모리가 부족하면 384 또는 256으로 낮추세요.
IMAGE_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15
NUM_WORKERS = 0  # Windows에서는 처음에 0이 가장 안전합니다.

ENCODER_NAME = "resnet34"
USE_IMAGENET_WEIGHTS = True
THRESHOLD = 0.5
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
