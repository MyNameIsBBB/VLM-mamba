from __future__ import annotations

import warnings
from typing import Iterable

from huggingface_hub import hf_hub_download
from torch import Tensor
from torchvision import transforms
from transformers import AutoTokenizer, CamembertTokenizer


class MultilingualTokenizer:
    def __init__(
        self,
        model_name: str,
        max_length: int,
        use_fast: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = self._load_tokenizer(model_name=model_name, use_fast=use_fast)

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.sep_token is not None:
                self.tokenizer.pad_token = self.tokenizer.sep_token
            else:
                raise ValueError(
                    f"Tokenizer for {model_name} does not define a pad/eos/sep token."
                )

    def _load_tokenizer(self, model_name: str, use_fast: bool):
        try:
            return AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
        except (ImportError, TypeError, ValueError) as error:
            if "wangchanberta" not in model_name.lower() and "PyTuple" not in str(error):
                raise

            warnings.warn(
                "AutoTokenizer failed for this SentencePiece-based checkpoint. Falling back to direct CamembertTokenizer loading.",
                stacklevel=2,
            )
            vocab_file = hf_hub_download(repo_id=model_name, filename="sentencepiece.bpe.model")
            return CamembertTokenizer(
                vocab_file=vocab_file,
                bos_token="<s>",
                eos_token="</s>",
                sep_token="</s>",
                cls_token="<s>",
                unk_token="<unk>",
                pad_token="<pad>",
                mask_token="<mask>",
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


class SimpleTokenizer(MultilingualTokenizer):
    def __init__(
        self,
        vocab_size: int | None = None,
        pad_token_id: int | None = None,
        unk_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        model_name: str = "airesearch/wangchanberta-base-att-spm-uncased",
        max_length: int = 64,
        use_fast: bool = False,
    ) -> None:
        del vocab_size, pad_token_id, unk_token_id, bos_token_id, eos_token_id
        super().__init__(model_name=model_name, max_length=max_length, use_fast=use_fast)


def build_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )