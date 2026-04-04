from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from utils.preprocessing import SimpleTokenizer


class ImageTextPairDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        annotations_path: str | Path,
        image_root: str | Path,
        tokenizer: SimpleTokenizer,
        image_transform: Callable[[Image.Image], Tensor],
        max_length: int,
    ) -> None:
        self.annotations_path = Path(annotations_path)
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict[str, str]]:
        with self.annotations_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        image_path = self.image_root / sample["image"]
        image = Image.open(image_path).convert("RGB")
        encoded = self.tokenizer.encode(sample["text"], max_length=self.max_length)
        return {
            "image": self.image_transform(image),
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "text": sample["text"],
            "image_path": str(image_path),
        }