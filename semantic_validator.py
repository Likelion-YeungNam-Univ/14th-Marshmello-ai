from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image


DEFAULT_POSITIVE_PROMPTS = (
    "a close-up photo of a human abdomen with the navel visible",
    "a clinical photo of a human abdomen",
    "a frontal photo of a human torso focused on the abdomen",
    "a photo of a person's bare stomach and navel",
    "a photo of a pregnant human belly",
)

DEFAULT_NEGATIVE_PROMPTS = (
    "a photo of a cat",
    "a photo of a dog",
    "a photo of an animal",
    "a photo of a pet",
    "a close-up photo of a human face",
    "a photo of a human arm or leg",
    "a photo of hands or feet",
    "a photo of a room or indoor scene",
    "a photo of furniture or bedding",
    "a photo of food",
    "a landscape photo",
    "a screenshot of an app or website",
    "an illustration or cartoon",
)


class SemanticStatus(str, Enum):
    VALID = "valid"
    REVIEW = "review"
    INVALID = "invalid_input"


@dataclass(frozen=True)
class SemanticValidationResult:
    status: SemanticStatus
    target_probability: float
    positive_similarity: float
    negative_similarity: float
    top_positive_prompt: str
    top_negative_prompt: str
    reason_code: str
    warning: str

    @property
    def can_continue(self) -> bool:
        return self.status != SemanticStatus.INVALID


class SemanticInputValidator:
    """Zero-shot target-image gate using OpenCLIP.

    This validator intentionally runs before the task-specific abdomen locator.
    A segmentation model can be overconfident on out-of-distribution images, so
    locator confidence alone is not a reliable way to decide whether an image is
    actually a human-abdomen photograph.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        positive_prompts: Sequence[str] = DEFAULT_POSITIVE_PROMPTS,
        negative_prompts: Sequence[str] = DEFAULT_NEGATIVE_PROMPTS,
        reject_probability: float = 0.40,
        valid_probability: float = 0.65,
        comparison_scale: float = 30.0,
    ) -> None:
        if not (0.0 <= reject_probability < valid_probability <= 1.0):
            raise ValueError(
                "semantic thresholds must satisfy "
                "0 <= reject_probability < valid_probability <= 1"
            )
        if comparison_scale <= 0:
            raise ValueError("comparison_scale must be > 0")
        if not positive_prompts or not negative_prompts:
            raise ValueError("positive_prompts and negative_prompts must not be empty")

        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                "semantic input validation requires open_clip_torch. "
                "Install it with: pip install -r requirements_semantic.txt"
            ) from exc

        self.device = torch.device(device)
        self.model_name = model_name
        self.pretrained = pretrained
        self.reject_probability = float(reject_probability)
        self.valid_probability = float(valid_probability)
        self.comparison_scale = float(comparison_scale)
        self.positive_prompts = tuple(positive_prompts)
        self.negative_prompts = tuple(negative_prompts)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        all_prompts = self.positive_prompts + self.negative_prompts
        text_tokens = self.tokenizer(list(all_prompts)).to(self.device)
        with torch.inference_mode():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.text_features = text_features

    def predict(self, image_bgr: np.ndarray) -> SemanticValidationResult:
        if image_bgr is None or image_bgr.size == 0:
            return SemanticValidationResult(
                status=SemanticStatus.INVALID,
                target_probability=0.0,
                positive_similarity=-1.0,
                negative_similarity=1.0,
                top_positive_prompt="",
                top_negative_prompt="",
                reason_code="UNREADABLE_IMAGE",
                warning="이미지를 읽을 수 없습니다.",
            )

        if image_bgr.ndim == 2:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            similarities = (image_features @ self.text_features.T).squeeze(0)

        positive_count = len(self.positive_prompts)
        positive_similarities = similarities[:positive_count]
        negative_similarities = similarities[positive_count:]

        positive_index = int(torch.argmax(positive_similarities).item())
        negative_index = int(torch.argmax(negative_similarities).item())
        positive_similarity = float(positive_similarities[positive_index].item())
        negative_similarity = float(negative_similarities[negative_index].item())

        margin = positive_similarity - negative_similarity
        target_probability = float(
            torch.sigmoid(torch.tensor(margin * self.comparison_scale)).item()
        )

        if target_probability < self.reject_probability:
            status = SemanticStatus.INVALID
            reason_code = "NOT_TARGET_IMAGE"
            warning = (
                "사람의 복부 사진으로 판단되지 않습니다"
                f"(target_probability={target_probability:.3f})."
            )
        elif target_probability < self.valid_probability:
            status = SemanticStatus.REVIEW
            reason_code = "SEMANTIC_IMAGE_AMBIGUOUS"
            warning = (
                "사람의 복부 사진인지 확신하기 어렵습니다"
                f"(target_probability={target_probability:.3f})."
            )
        else:
            status = SemanticStatus.VALID
            reason_code = ""
            warning = ""

        return SemanticValidationResult(
            status=status,
            target_probability=target_probability,
            positive_similarity=positive_similarity,
            negative_similarity=negative_similarity,
            top_positive_prompt=self.positive_prompts[positive_index],
            top_negative_prompt=self.negative_prompts[negative_index],
            reason_code=reason_code,
            warning=warning,
        )


def draw_semantic_overlay(
    image_bgr: np.ndarray,
    result: SemanticValidationResult,
) -> np.ndarray:
    canvas = image_bgr.copy()
    height = canvas.shape[0]
    status = result.status.value.upper()
    line1 = f"SEMANTIC INPUT: {status}  p(target)={result.target_probability:.3f}"
    line2 = f"negative: {result.top_negative_prompt[:64]}"

    color = (0, 200, 0) if result.status == SemanticStatus.VALID else (0, 0, 255)
    for text, y in ((line1, 32), (line2, 62)):
        y = min(y, max(18, height - 8))
        cv2.putText(
            canvas,
            text,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas
