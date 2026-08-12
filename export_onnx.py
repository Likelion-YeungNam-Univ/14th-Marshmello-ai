import torch

from settings import CHECKPOINT_DIR
from unet_core import create_model

checkpoint_path = CHECKPOINT_DIR / "best_unet.pt"
onnx_path = CHECKPOINT_DIR / "best_unet.onnx"

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

image_size = checkpoint["image_size"]
encoder_name = checkpoint["encoder_name"]

model = create_model(
    encoder_name,
    use_imagenet_weights=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dummy_input = torch.randn(
    1, 3, image_size, image_size
)

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {0: "batch"},
        "logits": {0: "batch"},
    },
    opset_version=17,
)

print(f"ONNX 저장 완료: {onnx_path}")