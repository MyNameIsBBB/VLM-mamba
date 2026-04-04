from __future__ import annotations

import re
from typing import Iterable

import torch
from torch import Tensor
from torchvision import transforms


class SimpleTokenizer:
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        unk_token_id: int = 1,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
    ) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self._token_pattern = re.compile(r"[A-Za-z0-9']+|[^\w\s]", flags=re.UNICODE)

    def _tokenize(self, text: str) -> list[str]:
        return self._token_pattern.findall(text.lower())

    def _token_to_id(self, token: str) -> int:
        bucket_count = max(self.vocab_size - 4, 1)
        return 4 + (hash(token) % bucket_count)

    def encode(self, text: str, max_length: int) -> dict[str, Tensor]:
        tokens = [self.bos_token_id]
        tokens.extend(self._token_to_id(token) for token in self._tokenize(text))
        tokens.append(self.eos_token_id)
        tokens = tokens[:max_length]
        attention_length = len(tokens)

        if attention_length < max_length:
            tokens.extend([self.pad_token_id] * (max_length - attention_length))

        attention_mask = [1] * attention_length + [0] * max(max_length - attention_length, 0)
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[:max_length], dtype=torch.long),
        }

    def batch_encode(self, texts: Iterable[str], max_length: int) -> dict[str, Tensor]:
        encoded = [self.encode(text, max_length=max_length) for text in texts]
        input_ids = torch.stack([item["input_ids"] for item in encoded], dim=0)
        attention_mask = torch.stack([item["attention_mask"] for item in encoded], dim=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def build_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )