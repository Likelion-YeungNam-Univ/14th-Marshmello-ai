import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

print("=" * 60)
print("U-Net 개발 환경 확인")
print("=" * 60)
print("Python:", sys.version.split()[0])
print("실행 경로:", sys.executable)
print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Albumentations:", A.__version__)
print("segmentation-models-pytorch:", smp.__version__)
print("GPU 사용 가능:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 이름:", torch.cuda.get_device_name(0))
else:
    print("현재 CPU 모드입니다.")
print("프로젝트 경로:", Path(__file__).resolve().parent)
print("=" * 60)
print("정상: 모든 필수 라이브러리를 불러왔습니다.")
