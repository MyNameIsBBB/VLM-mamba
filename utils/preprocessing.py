from __future__ import annotations

from typing import Iterable

from torch import Tensor
from torchvision import transforms
from transformers import AutoTokenizer


class MultilingualTokenizer:
    def __init__(
        self,
        model_name: str,
        max_length: int,
        use_fast: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.sep_token is not None:
                self.tokenizer.pad_token = self.tokenizer.sep_token
            else:
                raise ValueError(
                    f"Tokenizer for {model_name} does not define a pad/eos/sep token."
                )

    def encode(self, text: str, max_length: int | None = None) -> dict[str, Tensor]:
        encoded = self.batch_encode([text], max_length=max_length)
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
        }

    def batch_encode(self, texts: Iterable[str], max_length: int | None = None) -> dict[str, Tensor]:
        encoded = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length or self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }


def build_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )