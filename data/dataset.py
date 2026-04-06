from __future__ import annotations

from typing import Iterator

from datasets import load_dataset
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info


class ThaiCOCODataset(IterableDataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        split: str = "train",
        transform: callable | None = None,
        dataset_name: str = "patomp/thai-mscoco-2014-captions",
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform

    def _stream_examples(self) -> Iterator[dict[str, object]]:
        return iter(load_dataset(self.dataset_name, split=self.split, streaming=True))

    def __iter__(self) -> Iterator[dict[str, Tensor | str]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        for index, example in enumerate(self._stream_examples()):
            if index % num_workers != worker_id:
                continue

            image = example["image"].convert("RGB")
            if not isinstance(image, Image.Image):
                raise TypeError("Expected a PIL image from the Hugging Face dataset stream.")

            caption = str(example["th_sentences_raw"][0])
            if self.transform is not None:
                image = self.transform(image)

            yield {"image": image, "caption": caption}