from __future__ import annotations

from typing import Iterator

from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info


class ThaiCOCODataset(IterableDataset[dict[str, str]]):
    def __init__(
        self,
        split: str = "train",
        dataset_name: str = "patomp/thai-mscoco-2014-captions",
        use_all_captions: bool = True,
        shuffle_buffer_size: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.use_all_captions = use_all_captions
        self.shuffle_buffer_size = shuffle_buffer_size

    def _stream_examples(self) -> Iterator[dict[str, object]]:
        dataset = load_dataset(self.dataset_name, split=self.split, streaming=True)
        if self.shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=42)
        return iter(dataset)

    def __iter__(self) -> Iterator[dict[str, str]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        for index, example in enumerate(self._stream_examples()):
            if index % num_workers != worker_id:
                continue

            captions = [str(caption) for caption in example["th_sentences_raw"] if str(caption).strip()]
            if not captions:
                continue

            captions_to_use = captions if self.use_all_captions else captions[:1]
            for caption in captions_to_use:
                yield {
                    "caption": caption,
                }