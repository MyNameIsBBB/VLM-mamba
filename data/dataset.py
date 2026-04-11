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
        use_all_captions: bool = True,
        shuffle_buffer_size: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.use_all_captions = use_all_captions
        self.shuffle_buffer_size = shuffle_buffer_size

    def _stream_examples(self) -> Iterator[dict[str, object]]:
        dataset = load_dataset(self.dataset_name, split=self.split, streaming=True)
        if self.shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=42)
        return iter(dataset)

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

            captions = [str(caption) for caption in example["th_sentences_raw"] if str(caption).strip()]
            if not captions:
                continue

            image_id = int(example.get("cocoid", index))
            captions_to_use = captions if self.use_all_captions else captions[:1]
            for caption_index, caption in enumerate(captions_to_use):
                image_value = image.copy()
                if self.transform is not None:
                    image_value = self.transform(image_value)

                yield {
                    "image": image_value,
                    "caption": caption,
                    "image_id": image_id,
                    "caption_index": caption_index,
                }